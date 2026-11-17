"""Training loops that yield a TrainingFrame after every real unit of training work.

Each strategy is a generator of ``(step_value, fitted_model, rows_seen)``; ``training_frames``
wraps it, scores the model, and yields an immutable frame for the dashboard to draw. The
generator's return value is the final fitted model.

Each frame carries two timings, both of the model's own work only (scoring and drawing excluded):
``fit_seconds`` is all fitting so far, and ``model_fit_seconds`` is what it cost to produce the
model shown. They differ for refitting strategies: a sweep's model is one fresh refit, while an
incremental or warm-start model is the product of every step so far.
"""

from collections.abc import Generator, Iterator
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.base import RegressorMixin, clone

from animkit.trainers import CUMULATIVE_STRATEGIES, NumericalError, growth_sizes
from house_anim.config import RANDOM_STATE, TRAIN_EVAL_SAMPLE
from house_anim.data import HousingSplits, read_only
from house_anim.frames import (
    KIND_COEFFICIENT,
    KIND_IMPORTANCE,
    KIND_SENSITIVITY,
    FeatureWeights,
    TrainingFrame,
    TrainingPoint,
)
from house_anim.metrics import score
from house_anim.models import (
    STRATEGY_DATA_GROWTH,
    STRATEGY_INCREMENTAL,
    STRATEGY_PARAM_SWEEP,
    STRATEGY_WARM_START,
    ModelSpec,
)

FitStream = Iterator[tuple[float, RegressorMixin, int]]


def checked_predict(model, X: np.ndarray) -> np.ndarray:
    """``model.predict(X)``, refusing NaN or infinite predictions.

    A diverging learning rate shows up as non-finite predictions; catching it here gives a clear
    error instead of a blank chart.

    Args:
        model: A fitted regressor.
        X: Scaled features, one row per house.

    Returns:
        The predictions, in $100k, as a read-only array.

    Raises:
        NumericalError: If any prediction is NaN or infinite.
    """
    predictions = np.asarray(model.predict(X), dtype=float)
    if not np.isfinite(predictions).all():
        raise NumericalError(
            f"{type(model).__name__} produced non-finite predictions; "
            "try a smaller learning rate or stronger regularisation"
        )
    return read_only(predictions)


def _display(spec: ModelSpec, value: float) -> float:
    """A schedule value as the x axis shows it (for example -log10 alpha), via the spec's ``step_transform``."""
    return float(spec.step_transform(value)) if spec.step_transform else float(value)


def planned_steps(spec: ModelSpec, splits: HousingSplits) -> tuple[float, ...]:
    """The x-axis value of every frame, known before training starts.

    Knowing them up front lets the learning curve fix its x range before the first frame.

    Args:
        spec: The model being trained.
        splits: The data; its training row count sets a data-growth schedule's sizes.

    Returns:
        One value per frame: fractional epochs for incremental training (1/16, 2/16, ...),
        row counts for data growth, and the (transformed) parameter for warm start and sweeps.

    Raises:
        ValueError: If the spec names an unknown strategy.
    """
    if spec.strategy == STRATEGY_INCREMENTAL:
        chunks = spec.chunks_per_epoch
        return tuple(i / chunks for i in range(1, len(spec.steps) * chunks + 1))
    if spec.strategy == STRATEGY_DATA_GROWTH:
        return tuple(float(n) for n in growth_sizes(spec.steps, splits.n_train))
    if spec.strategy in (STRATEGY_WARM_START, STRATEGY_PARAM_SWEEP):
        return tuple(_display(spec, v) for v in spec.steps)
    raise ValueError(f"Unknown training strategy '{spec.strategy}'")


def _incremental_fits(spec: ModelSpec, splits: HousingSplits) -> FitStream:
    """``partial_fit`` on one chunk of the training rows per step, reshuffled after the first epoch."""
    model = spec.build()
    rng = np.random.default_rng(RANDOM_STATE)
    for epoch in range(len(spec.steps)):
        # The split already shuffled the rows, so epoch 0 walks them in order: after k chunks
        # the model has seen exactly the first rows, and training error is scored on those.
        order = np.arange(splits.n_train) if epoch == 0 else rng.permutation(splits.n_train)
        seen = 0
        for chunk, rows in enumerate(np.array_split(order, spec.chunks_per_epoch)):
            model.partial_fit(splits.X_train[rows], splits.y_train[rows])
            seen += len(rows)
            rows_seen = seen if epoch == 0 else splits.n_train
            yield epoch + (chunk + 1) / spec.chunks_per_epoch, model, rows_seen


def _warm_start_fits(spec: ModelSpec, splits: HousingSplits) -> FitStream:
    """Raises the ensemble's member count and refits; with ``warm_start`` only the new members are fitted."""
    model = spec.build()
    for n_members in spec.steps:
        model.set_params(**{spec.param_name: int(n_members)})
        model.fit(splits.X_train, splits.y_train)
        yield float(n_members), model, splits.n_train


def _param_sweep_fits(spec: ModelSpec, splits: HousingSplits) -> FitStream:
    """A fresh model per step, refitted with the parameter set to the schedule's next value."""
    template = spec.build()
    for value in spec.steps:
        model = clone(template).set_params(**{spec.param_name: value})
        model.fit(splits.X_train, splits.y_train)
        yield _display(spec, value), model, splits.n_train


def _data_growth_fits(spec: ModelSpec, splits: HousingSplits) -> FitStream:
    """A fresh model per step, fitted on the first n training rows for a growing n (a learning curve)."""
    template = spec.build()
    for rows in growth_sizes(spec.steps, splits.n_train):
        model = clone(template).fit(splits.X_train[:rows], splits.y_train[:rows])
        yield float(rows), model, rows


_FIT_STREAMS = {
    STRATEGY_INCREMENTAL: _incremental_fits,
    STRATEGY_WARM_START: _warm_start_fits,
    STRATEGY_PARAM_SWEEP: _param_sweep_fits,
    STRATEGY_DATA_GROWTH: _data_growth_fits,
}


def extract_weights(model, names: tuple[str, ...]) -> FeatureWeights | None:
    """Per-feature view of what the model has learned, when the model exposes one.

    Args:
        model: A fitted regressor.
        names: The feature names, in column order.

    Returns:
        Linear models' coefficients, trees' impurity importances, or an MLP's mean absolute
        first-layer weight per input (a rough sensitivity); None for a model with none (KNN).
    """
    if hasattr(model, "coef_"):
        values, kind = np.ravel(model.coef_), KIND_COEFFICIENT
    elif hasattr(model, "feature_importances_"):
        values, kind = model.feature_importances_, KIND_IMPORTANCE
    elif hasattr(model, "coefs_"):
        values, kind = np.abs(model.coefs_[0]).mean(axis=1), KIND_SENSITIVITY
    else:
        return None
    return FeatureWeights(names=names, values=read_only(values), kind=kind)


def _x_range(spec: ModelSpec, steps: tuple[float, ...]) -> tuple[float, float]:
    """The learning curve's x range: from the first step (0 epochs for incremental training) to the last."""
    start = 0.0 if spec.strategy == STRATEGY_INCREMENTAL else steps[0]
    return start, steps[-1]


def _score_model(model, splits: HousingSplits, rows_seen: int):
    """Training RMSE on (a sample of) the rows the model has seen, plus validation results."""
    eval_rows = min(rows_seen, TRAIN_EVAL_SAMPLE)
    train_pred = checked_predict(model, splits.X_train[:eval_rows])
    val_pred = checked_predict(model, splits.X_val)
    return score(splits.y_train[:eval_rows], train_pred).rmse, val_pred, score(splits.y_val, val_pred)


@dataclass(frozen=True)
class TrainingStep:
    """One step of a training loop: the snapshot to draw and the model it was taken from.

    Attributes:
        frame: The immutable snapshot (scores, predictions, weights, timings).
        model: The live estimator, for views that need the model itself (equations, diagrams).
    """

    frame: TrainingFrame
    # The live estimator. Warm-start models keep changing, so read it before advancing the loop.
    model: RegressorMixin


def training_steps(
    spec: ModelSpec, splits: HousingSplits
) -> Generator[TrainingStep, None, RegressorMixin]:
    """Trains one model step by step, scoring it after every real unit of training work.

    After each step it scores the model on (a sample of) the training rows it has seen and on the
    whole validation set, and adds the step to the learning-curve history.

    Args:
        spec: The model and its schedule.
        splits: The data.

    Returns:
        The final fitted model, as the generator's return value.

    Yields:
        One ``TrainingStep`` per planned step. A warm-start model keeps changing as the loop
        goes on, so read a step's model before advancing.

    Raises:
        ValueError: If the spec names an unknown strategy.
        NumericalError: If the model's predictions stop being finite.
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
        train_rmse, val_pred, val_scores = _score_model(model, splits, rows_seen)
        history = history + (TrainingPoint(step, train_rmse, val_scores.rmse),)
        frame = TrainingFrame(
            step=step,
            step_index=index,
            total_steps=len(steps),
            x_range=span,
            history=history,
            val_pred=val_pred,
            val_scores=val_scores,
            weights=extract_weights(model, splits.feature_names),
            fit_seconds=fit_seconds,
            model_fit_seconds=fit_seconds if cumulative else step_seconds,
        )
        yield TrainingStep(frame=frame, model=model)
    return model


def training_frames(
    spec: ModelSpec, splits: HousingSplits
) -> Generator[TrainingFrame, None, RegressorMixin]:
    """The frames of training_steps; the generator's return value is the final model.

    Args:
        spec: The model and its schedule.
        splits: The data.

    Returns:
        The final fitted model, as the generator's return value.

    Yields:
        One ``TrainingFrame`` per planned step.
    """
    stream = training_steps(spec, splits)
    while True:
        try:
            step = next(stream)
        except StopIteration as stop:
            return stop.value
        yield step.frame


def train_silently(spec: ModelSpec, splits: HousingSplits) -> tuple[RegressorMixin, TrainingFrame]:
    """Run the same training loop without drawing; returns (model, final frame).

    The inference-only run uses it, so the model it predicts with is the one the training
    animation would have ended on.

    Args:
        spec: The model and its schedule.
        splits: The data.

    Returns:
        The final fitted model and the last training frame.
    """
    stream = training_frames(spec, splits)
    final = None
    while True:
        try:
            final = next(stream)
        except StopIteration as stop:
            return stop.value, final
