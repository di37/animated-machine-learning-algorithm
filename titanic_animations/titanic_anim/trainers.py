"""Training loops that yield a TrainingFrame after every real unit of classifier training work.

Each strategy is a generator of ``(step_value, fitted_model, rows_seen)``, one item per real piece
of training: a ``partial_fit``, a refit, more trees. ``training_steps`` times each step, scores the
model after it, and yields an immutable frame together with the live model, so a dashboard draws
exactly what training has just produced. The generator's return value is the final fitted model.

A model is scored on what it has learned from. Training accuracy and loss count only the
``rows_seen`` training passengers (a data-growth model has not seen the rest yet), and every step
is also scored on the validation passengers, which no model trains on.

Scores are checked as they are read: a model that gives NaN or infinite scores, or labels other
than 0 and 1, raises animkit's ``NumericalError`` instead of drawing nonsense.
"""

from collections.abc import Generator, Iterator
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.base import ClassifierMixin, clone

from animkit.data import read_only
from animkit.trainers import CUMULATIVE_STRATEGIES, NumericalError, growth_sizes
from titanic_anim.config import CLASSES, MIN_GROWTH_ROWS, RANDOM_STATE
from titanic_anim.data import TitanicSplits, read_only_labels
from titanic_anim.frames import (
    KIND_COEFFICIENT,
    KIND_IMPORTANCE,
    KIND_SENSITIVITY,
    KIND_SEPARATION,
    FeatureWeights,
    TrainingFrame,
    TrainingPoint,
)
from titanic_anim.metrics import PROBABILITY_CLIP, score
from titanic_anim.models import (
    SCORE_MARGIN,
    STRATEGY_DATA_GROWTH,
    STRATEGY_INCREMENTAL,
    STRATEGY_PARAM_SWEEP,
    STRATEGY_WARM_START,
    ClassifierSpec,
)

__all__ = ["NumericalError", "TrainingStep", "checked_labels", "planned_steps", "survival_scores",
           "train_silently", "training_frames", "training_steps"]

FitStream = Iterator[tuple[float, ClassifierMixin, int]]
CLASS_LABELS = np.array(CLASSES)


def survival_scores(model, X: np.ndarray, score_kind: str) -> np.ndarray:
    """Each passenger's score: P(survive) from ``predict_proba``, or the signed margin from ``decision_function``.

    Args:
        model: A fitted classifier.
        X: Standardised passengers, one per row.
        score_kind: ``SCORE_PROBABILITY`` or ``SCORE_MARGIN``: which score to read.

    Returns:
        One read-only score per passenger.

    Raises:
        NumericalError: If any score is NaN or infinite.
    """
    raw = model.decision_function(X) if score_kind == SCORE_MARGIN else model.predict_proba(X)[:, 1]
    values = np.asarray(raw, dtype=float)
    if not np.isfinite(values).all():
        raise NumericalError(f"{type(model).__name__} produced non-finite scores; "
                             "try a smaller learning rate or stronger regularisation")
    return read_only(values)


def checked_labels(model, X: np.ndarray) -> np.ndarray:
    """The model's predicted labels, refused unless every one is 0 or 1.

    Args:
        model: A fitted classifier.
        X: Standardised passengers, one per row.

    Returns:
        One read-only label per passenger.

    Raises:
        NumericalError: If a label is not 0 or 1.
    """
    labels = read_only_labels(model.predict(X))
    if not np.isin(labels, CLASSES).all():
        raise NumericalError(f"{type(model).__name__} predicted labels outside {CLASSES}")
    return labels


def rows_with_both_classes(y: np.ndarray) -> int:
    """How many leading rows it takes to have seen a survivor and a non-survivor.

    A classifier cannot be fit on one class, so the smallest training subset must reach this far.

    Args:
        y: The training labels, in order.

    Returns:
        The number of leading rows that hold both outcomes.

    Raises:
        ValueError: If the labels hold only one outcome.
    """
    first = [int(np.argmax(y == label)) for label in CLASSES if (y == label).any()]
    if len(first) < len(CLASSES):
        raise ValueError("training labels need both survivors and non-survivors")
    return max(first) + 1


def growth_plan(spec: ClassifierSpec, splits: TitanicSplits) -> tuple[int, ...]:
    """The training-set sizes a data-growth model is refit on, smallest first.

    animkit's ``growth_sizes`` turns the spec's fractions into row counts. The smallest holds at least
    ``MIN_GROWTH_ROWS`` passengers, and at least one of each outcome.

    Args:
        spec: The classifier; its ``steps`` are fractions of the training set.
        splits: The splits; only the training labels are read.

    Returns:
        Distinct row counts in ascending order, the last being every training passenger.
    """
    minimum = max(MIN_GROWTH_ROWS, rows_with_both_classes(splits.y_train))
    return growth_sizes(spec.steps, splits.n_train, minimum=minimum)


def _display(spec: ClassifierSpec, value: float) -> float:
    """A step's value as the x-axis shows it (log10 C, for example), via the spec's ``step_transform``."""
    return float(spec.step_transform(value)) if spec.step_transform else float(value)


def planned_steps(spec: ClassifierSpec, splits: TitanicSplits) -> tuple[float, ...]:
    """The x-axis value of every frame, known before training starts.

    Args:
        spec: The classifier and its schedule.
        splits: The splits; a data-growth schedule depends on how many training passengers there are.

    Returns:
        One value per frame: epochs in fractions (1/8, 2/8, ...) for ``partial_fit`` training, row
        counts for data growth, and the schedule itself (through ``step_transform``) for the others.

    Raises:
        ValueError: If the spec's strategy is not one of the four.
    """
    if spec.strategy == STRATEGY_INCREMENTAL:
        chunks = spec.chunks_per_epoch
        return tuple(i / chunks for i in range(1, len(spec.steps) * chunks + 1))
    if spec.strategy == STRATEGY_DATA_GROWTH:
        return tuple(float(n) for n in growth_plan(spec, splits))
    if spec.strategy in (STRATEGY_WARM_START, STRATEGY_PARAM_SWEEP):
        return tuple(_display(spec, v) for v in spec.steps)
    raise ValueError(f"Unknown training strategy '{spec.strategy}'")


def _incremental_fits(spec: ClassifierSpec, splits: TitanicSplits) -> FitStream:
    """``partial_fit`` chunk by chunk, epoch after epoch; the first epoch walks the rows in order.

    Each item's step is the epochs done so far (1.125 is one epoch and an eighth); later epochs
    reshuffle the rows.
    """
    model = spec.build()
    rng = np.random.default_rng(RANDOM_STATE)
    for epoch in range(len(spec.steps)):
        # The split already shuffled the rows, so epoch 0 walks them in order: after k chunks
        # the model has seen exactly the first rows, and training accuracy is scored on those.
        order = np.arange(splits.n_train) if epoch == 0 else rng.permutation(splits.n_train)
        seen = 0
        for chunk, rows in enumerate(np.array_split(order, spec.chunks_per_epoch)):
            model.partial_fit(splits.X_train[rows], splits.y_train[rows], classes=CLASS_LABELS)
            seen += len(rows)
            rows_seen = seen if epoch == 0 else splits.n_train
            yield epoch + (chunk + 1) / spec.chunks_per_epoch, model, rows_seen


def _warm_start_fits(spec: ClassifierSpec, splits: TitanicSplits) -> FitStream:
    """One ensemble, grown: each step raises its member count and fits only the new members."""
    model = spec.build()
    for n_members in spec.steps:
        model.set_params(**{spec.param_name: int(n_members)})
        model.fit(splits.X_train, splits.y_train)
        yield float(n_members), model, splits.n_train


def _param_sweep_fits(spec: ClassifierSpec, splits: TitanicSplits) -> FitStream:
    """A fresh model for each value of the swept hyper-parameter, fit on every training passenger."""
    template = spec.build()
    for value in spec.steps:
        model = clone(template).set_params(**{spec.param_name: value})
        model.fit(splits.X_train, splits.y_train)
        yield _display(spec, value), model, splits.n_train


def _data_growth_fits(spec: ClassifierSpec, splits: TitanicSplits) -> FitStream:
    """A fresh model for each training-set size, fit on that many leading passengers."""
    template = spec.build()
    for rows in growth_plan(spec, splits):
        model = clone(template).fit(splits.X_train[:rows], splits.y_train[:rows])
        yield float(rows), model, rows


_FIT_STREAMS = {
    STRATEGY_INCREMENTAL: _incremental_fits,
    STRATEGY_WARM_START: _warm_start_fits,
    STRATEGY_PARAM_SWEEP: _param_sweep_fits,
    STRATEGY_DATA_GROWTH: _data_growth_fits,
}


def class_separation(model) -> np.ndarray:
    """How far apart Naive Bayes' two bell curves sit, feature by feature.

    The gap is (survivors' mean − non-survivors' mean) / pooled standard deviation, the pooled
    variance being the mean of the two classes' variances. A large gap means the feature tells the
    outcomes apart; its sign says which outcome has the higher values.

    Args:
        model: A fitted ``GaussianNB``.

    Returns:
        One gap per feature, in standard deviations.
    """
    pooled = np.sqrt(model.var_.mean(axis=0))
    return (model.theta_[1] - model.theta_[0]) / pooled


def extract_weights(model, names: tuple[str, ...]) -> FeatureWeights | None:
    """A per-feature view of what the model has learned, when the model exposes one.

    Linear models give their coefficients (log-odds per standard deviation), trees and ensembles their
    feature importances, the MLP the mean |weight| leaving each input (its sensitivity), and Naive
    Bayes its ``class_separation``.

    Args:
        model: A fitted classifier.
        names: The feature names, in column order.

    Returns:
        The values and what kind they are, or None for a model with no per-feature view (an RBF-kernel
        SVM, KNN).
    """
    if hasattr(model, "coef_"):  # an RBF-kernel SVC raises on coef_, so hasattr is False
        values, kind = np.ravel(model.coef_), KIND_COEFFICIENT
    elif hasattr(model, "feature_importances_"):
        values, kind = model.feature_importances_, KIND_IMPORTANCE
    elif hasattr(model, "coefs_"):
        values, kind = np.abs(model.coefs_[0]).mean(axis=1), KIND_SENSITIVITY
    elif hasattr(model, "theta_"):
        values, kind = class_separation(model), KIND_SEPARATION
    else:
        return None
    return FeatureWeights(names=names, values=read_only(values), kind=kind)


def training_loss(y: np.ndarray, scores: np.ndarray, score_kind: str) -> float:
    """The loss the model minimises, averaged over the passengers.

    A probability model minimises the log loss, −[y log p + (1 − y) log(1 − p)], with p clipped away
    from 0 and 1 so the log stays finite. A margin model (the SVM) minimises the hinge loss,
    max(0, 1 − y·f(x)) with y as ±1, which is zero only for a passenger on the right side by at least
    the margin.

    Args:
        y: The true labels, 0 or 1.
        scores: The model's scores for the same passengers.
        score_kind: ``SCORE_PROBABILITY`` or ``SCORE_MARGIN``: which loss to compute.

    Returns:
        The mean loss per passenger.
    """
    if score_kind == SCORE_MARGIN:
        signed = np.where(y == 1, 1.0, -1.0)
        return float(np.mean(np.maximum(0.0, 1.0 - signed * scores)))
    p = np.clip(scores, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def _x_range(spec: ClassifierSpec, steps: tuple[float, ...]) -> tuple[float, float]:
    """The learning curve's x-axis span: from 0 for epochs, otherwise from the first step."""
    start = 0.0 if spec.strategy == STRATEGY_INCREMENTAL else steps[0]
    return start, steps[-1]


def _score_model(model, spec: ClassifierSpec, splits: TitanicSplits, rows_seen: int):
    """Training accuracy and loss on the rows the model has seen, plus validation results."""
    X_seen, y_seen = splits.X_train[:rows_seen], splits.y_train[:rows_seen]
    train_accuracy = float(np.mean(checked_labels(model, X_seen) == y_seen))
    train_loss = training_loss(y_seen, survival_scores(model, X_seen, spec.score_kind), spec.score_kind)
    val_pred = checked_labels(model, splits.X_val)
    val_score = survival_scores(model, splits.X_val, spec.score_kind)
    val_scores = score(splits.y_val, val_pred, val_score, probabilities=spec.score_kind != SCORE_MARGIN)
    return train_accuracy, train_loss, val_pred, val_score, val_scores


@dataclass(frozen=True)
class TrainingStep:
    """One step of training: the frame to draw and the model that produced it.

    Attributes:
        frame: The immutable snapshot of the model's scores after this step.
        model: The live estimator. A warm-start or ``partial_fit`` model keeps changing, so read it
            before advancing the loop.
    """

    frame: TrainingFrame
    # The live estimator. Warm-start models keep changing, so read it before advancing the loop.
    model: ClassifierMixin


def training_steps(
    spec: ClassifierSpec, splits: TitanicSplits
) -> Generator[TrainingStep, None, ClassifierMixin]:
    """Trains a classifier along its schedule, yielding a step after every real piece of work.

    Each step is timed, then the model is scored on the training passengers it has seen and on the
    validation passengers. The generator's return value is the final fitted model.

    Args:
        spec: The classifier and its schedule.
        splits: The train and validation passengers.

    Yields:
        A ``TrainingStep`` per planned step, its frame holding the history of every step so far.

    Raises:
        ValueError: If the spec's strategy is not one of the four.
        NumericalError: If the model gives non-finite scores or labels other than 0 and 1.
    """
    if spec.strategy not in _FIT_STREAMS:
        raise ValueError(f"Unknown training strategy '{spec.strategy}'")
    steps = planned_steps(spec, splits)
    span = _x_range(spec, steps)
    fits = _FIT_STREAMS[spec.strategy](spec, splits)
    cumulative = spec.strategy in CUMULATIVE_STRATEGIES
    history: tuple[TrainingPoint, ...] = ()
    fit_seconds, model = 0.0, None
    for index in range(len(steps)):
        started = perf_counter()
        step, model, rows_seen = next(fits)  # the actual training work happens here
        step_seconds = perf_counter() - started
        fit_seconds += step_seconds
        train_accuracy, train_loss, val_pred, val_score, val_scores = _score_model(model, spec, splits, rows_seen)
        history = history + (TrainingPoint(step, train_accuracy, val_scores.accuracy, train_loss),)
        frame = TrainingFrame(
            step=step,
            step_index=index,
            total_steps=len(steps),
            x_range=span,
            history=history,
            val_pred=val_pred,
            val_score=val_score,
            val_scores=val_scores,
            weights=extract_weights(model, splits.feature_names),
            fit_seconds=fit_seconds,
            model_fit_seconds=fit_seconds if cumulative else step_seconds,
        )
        yield TrainingStep(frame=frame, model=model)
    return model


def training_frames(
    spec: ClassifierSpec, splits: TitanicSplits
) -> Generator[TrainingFrame, None, ClassifierMixin]:
    """The frames of ``training_steps``, without the models; the generator's return value is the final model.

    Args:
        spec: The classifier and its schedule.
        splits: The train and validation passengers.

    Yields:
        The ``TrainingFrame`` of each step.
    """
    stream = training_steps(spec, splits)
    while True:
        try:
            step = next(stream)
        except StopIteration as stop:
            return stop.value
        yield step.frame


def train_silently(spec: ClassifierSpec, splits: TitanicSplits) -> tuple[ClassifierMixin, TrainingFrame]:
    """Runs the same training loop without drawing anything.

    The inference-only phase uses it, so the model it classifies with is the one the training
    animation would have shown.

    Args:
        spec: The classifier and its schedule.
        splits: The train and validation passengers.

    Returns:
        The final fitted model and the final training frame.
    """
    stream = training_frames(spec, splits)
    final = None
    while True:
        try:
            final = next(stream)
        except StopIteration as stop:
            return stop.value, final
