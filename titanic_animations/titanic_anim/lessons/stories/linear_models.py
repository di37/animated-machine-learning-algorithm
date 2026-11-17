"""Two score-only linear classifiers trained online: the perceptron and a linear SVM (hinge loss, SGD).

Neither outputs a probability. The maps shade the signed score θ₀ + θ·z and the boundary is score 0;
the SVM also shows its margin, the band where |score| < 1. A frozen copy of the scikit-learn model
is kept for every frame, so each frame's map and the final check call the real model at that moment.

The perceptron changes its weights only when it gets a passenger wrong, moving them towards the right
answer. The linear SVM minimises the hinge loss, which charges nothing for a passenger already on its own
side with |score| ≥ 1, plus an L2 penalty on the weights. Both learn with ``partial_fit``, one chunk of
passengers or one epoch per frame. A perceptron never settles on data it cannot separate, so its lesson
keeps the epoch with the best validation accuracy; the SVM keeps its last epoch.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier

from animkit.data import read_only
from animkit.lessons.lesson import Story
from titanic_anim.config import RANDOM_STATE
from titanic_anim.lessons import maps
from titanic_anim.lessons.data import PassengerData
from titanic_anim.lessons.lesson import (
    MARGIN,
    ClassLesson,
    ClassSnapshot,
    accuracy,
    best_validation,
    make_snapshot,
    signed,
    verified_score,
)
from titanic_anim.lessons.lesson_diagrams import LinearDiagram, weight_scale
from titanic_anim.lessons.stories.common import Standardiser, WeightsPanel, z_line

CHUNK = 60  # the perceptron's first epoch is shown every 60 passengers
PERCEPTRON_EPOCHS = 15
SVM_EPOCHS = 30
# A light penalty keeps real weight on age and fare. With STRONG_ALPHA the SVM ignores them and
# splits on sex alone; the final caption reports that model's validation score too.
SVM_ALPHA = 3e-4
STRONG_ALPHA = 1e-2


@dataclass(frozen=True)
class LinearState:
    """A linear model after one frame of training.

    Attributes:
        epochs: Passes over the training data so far (a fraction during the perceptron's first epoch).
        theta: The bias and the age, fare and woman weights, on the scaled inputs.
        mistakes: Training passengers on the wrong side of the boundary.
        inside_margin: Training passengers with |score| < 1.
        fitted: A frozen copy of the scikit-learn model after this frame.
    """

    epochs: float
    theta: np.ndarray  # bias, age, fare, woman
    mistakes: int  # training passengers on the wrong side of the boundary
    inside_margin: int  # training passengers with |score| < 1
    fitted: Any  # a frozen copy of the scikit-learn model after this frame


def _capture(model, epochs: float, z: np.ndarray, y: np.ndarray) -> LinearState:
    """Records the model after a frame: its weights, its mistakes, the margin count, and a frozen copy."""
    scores = model.decision_function(z)
    return LinearState(epochs, read_only(np.concatenate([model.intercept_, model.coef_[0]])),
                       int(np.sum((scores > 0) != y)), int(np.sum(np.abs(scores) < 1.0)), copy.deepcopy(model))


def _scorer(state: LinearState, standardise: Standardiser) -> Callable[[np.ndarray], np.ndarray]:
    """The frozen model of one state, as a scorer: raw feature rows in, signed scores out."""
    return lambda x: state.fitted.decision_function(standardise(x))


def _train(model, z: np.ndarray, y: np.ndarray, schedule: list[tuple[slice, float]]) -> list[LinearState]:
    """One ``partial_fit`` per scheduled (rows, epochs done after them), each captured as a state."""
    states = []
    for rows, epochs in schedule:
        model.partial_fit(z[rows], y[rows], classes=np.array([0, 1]))
        states.append(_capture(model, epochs, z, y))
    return states


def _epoch_label(epochs: float) -> str:
    """'Epoch 3', or 'Epoch 0.25' part-way through the first."""
    return f"Epoch {epochs:.0f}" if float(epochs).is_integer() else f"Epoch {epochs:.2f}"


class MarginWeights(WeightsPanel):
    """Weight bars; for the SVM, the maps also get the dashed ±1 margin lines."""

    def __init__(self, *args, margin: bool, data: PassengerData, **kwargs):
        """Sets up the weight bars, and whether to draw the margin.

        Args:
            *args: Passed on to ``WeightsPanel``.
            margin: Whether to draw the ±1 margin lines on the maps (the SVM).
            data: The passengers, for the maps' grid.
            **kwargs: Passed on to ``WeightsPanel``.
        """
        super().__init__(*args, **kwargs)
        self._margin = margin
        self._data = data

    def overlay_training(self, axes, snapshot: ClassSnapshot) -> None:
        """Draws the dashed score = −1 and score = +1 lines on both maps, for the SVM only.

        Args:
            axes: The two maps, women's first.
            snapshot: The moment being drawn.
        """
        if not self._margin:
            return
        for sex, ax in enumerate(axes):
            for level in (-1.0, 1.0):
                maps.draw_boundary(ax, self._data, snapshot.surface[sex], MARGIN, level=level, width=1.4,
                                   dashed=True)


def _explainer(final: LinearState, standardise: Standardiser, name: str):
    """The worked prediction for a linear lesson's kept state; ``name`` is the model in words."""
    def explain(features: np.ndarray):
        """Works one passenger's score through: scale, weighted sum, sign.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            The verified explanation, checked against the kept epoch's real model.
        """
        z = standardise(features)
        worked = float(final.theta[0] + z @ final.theta[1:])  # the weights shown in the panel
        lines = [z_line(features, z),
                 f"score = θ₀ + Σ θ·input = {final.theta[0]:+.2f} "
                 f"{' '.join(f'{w * v:+.2f}' for w, v in zip(final.theta[1:], z))} = {signed(worked)}",
                 f"score {'>' if worked > 0 else '≤'} 0 → predicted: {'survived' if worked > 0 else 'did not survive'}",
                 f"a {name} gives a score, not a probability"]
        model_value = float(final.fitted.decision_function(z[None, :])[0])  # the real model, that epoch
        return verified_score(lines, worked, model_value, MARGIN,
                              call=f"{type(final.fitted).__name__}.decision_function(x), {_epoch_label(final.epochs).lower()}")
    return explain


def _lesson(data: PassengerData, story: Callable[[tuple[ClassSnapshot, ...], int], Story],
            states: list[LinearState], final_index: int | None, standardise: Standardiser, name: str,
            margin: bool, config: dict[str, str], note: Callable[[int, LinearState], str]) -> ClassLesson:
    """Builds a score-only linear lesson from its states; ``final_index`` None keeps the best on validation."""
    snapshots = tuple(make_snapshot(_scorer(s, standardise), data, MARGIN, label=_epoch_label(s.epochs),
                                    progress=s.epochs, note=note(i, s), detail=s)
                      for i, s in enumerate(states))
    chosen = best_validation(snapshots) if final_index is None else final_index
    final = states[chosen]
    limit = max(float(np.max(np.abs(s.theta))) for s in states) * 1.25
    panel = MarginWeights(limit, lambda s: s.theta, "Weights on the scaled inputs", final.theta, standardise,
                          note=lambda s: (f"{s.mistakes} misclassified · {s.inside_margin} inside the margin"
                                          if margin else f"{s.mistakes} training passengers misclassified"),
                          margin=margin, data=data)
    return ClassLesson(story=story(snapshots, chosen), data=data, snapshots=snapshots, final_index=chosen,
                       score=_scorer(final, standardise), kind=MARGIN,
                       explain=_explainer(final, standardise, name), panel=panel, config=config,
                       diagram=LinearDiagram("perceptron" if name == "perceptron" else "svm", standardise,
                                             lambda s: s.theta, final.theta, weight_scale(s.theta for s in states)))


def _perceptron_story(snapshots: tuple[ClassSnapshot, ...], chosen: int) -> Story:
    """The perceptron's narration; ``chosen`` is the kept epoch's snapshot."""
    best = snapshots[chosen]
    return Story(
        key="perceptron", title="Perceptron",
        method="the original learning rule: fix the weights only after a mistake",
        objective=("The perceptron reads passengers one at a time. When it gets one wrong it nudges its "
                   "weights towards the right answer; when it is right it does nothing. Watch the "
                   "boundary jump around as mistakes come in."),
        initial=(f"After the first {CHUNK} passengers: a rough first boundary. The shading is a score, not "
                 "a probability: blue means the 'survived' side, red the other."),
        training=("Each frame is real online training: the first epoch every 60 passengers, then one frame "
                  "per epoch. The data cannot be split perfectly, so the boundary never settles."),
        final=(f"Validation picked {best.label.lower()}: train accuracy {best.train_accuracy:.0%}, validation "
               f"{best.val_accuracy:.0%}. A perceptron keeps moving, so the best epoch is kept."),
        inference=("An unseen passenger: add bias + weight × input with the kept epoch's weights. A positive "
                   "score means 'survived'; its size is not a probability."),
        takeaway="The perceptron learns from its mistakes alone; on data that cannot be separated it never stops moving.",
        limitation=("it gives no probabilities and no margin, and on overlapping classes its weights keep "
                    "oscillating, so the result depends on when you stop."),
        progress_label="epochs (passes over the training data)")


def build_perceptron(data: PassengerData) -> ClassLesson:
    """Builds the perceptron lesson: its first epoch every ``CHUNK`` passengers, then one frame per epoch.

    Args:
        data: The passengers.

    Returns:
        The lesson, with the epoch of highest validation accuracy kept as the final model.
    """
    standardise = Standardiser.fit(data.x_train)
    z, y, n = standardise(data.x_train), data.y_train, data.n_train
    edges = [*range(0, n, CHUNK), n]
    schedule = [(slice(a, b), b / n) for a, b in zip(edges, edges[1:])]
    schedule += [(slice(0, n), float(epoch)) for epoch in range(2, PERCEPTRON_EPOCHS + 1)]
    states = _train(Perceptron(random_state=RANDOM_STATE), z, y, schedule)

    def note(i: int, s: LinearState) -> str:
        """The training callout for one frame.

        Args:
            i: The frame's index.
            s: The model after the frame.

        Returns:
            The epoch, and how many training passengers are now on the wrong side.
        """
        return (f"{_epoch_label(s.epochs)}: weights changed only on mistakes\n"
                f"{s.mistakes} of {n} training passengers now on the wrong side")

    return _lesson(data, _perceptron_story, states, None, standardise, "perceptron", False,
                   {"model": "Perceptron", "inputs": "standardised age, log fare, woman",
                    "schedule": f"epoch 1 in chunks of {CHUNK}, then {PERCEPTRON_EPOCHS} epochs",
                    "selection": "highest validation accuracy", "random_state": str(RANDOM_STATE)}, note)


def _strong_penalty_accuracy(z: np.ndarray, y: np.ndarray, zv: np.ndarray, yv: np.ndarray) -> float:
    """Validation accuracy of the same SVM with the strong penalty ``STRONG_ALPHA``, for the final caption."""
    model = SGDClassifier(loss="hinge", alpha=STRONG_ALPHA, random_state=RANDOM_STATE)
    for _ in range(SVM_EPOCHS):
        model.partial_fit(z, y, classes=np.array([0, 1]))
    return accuracy(model.decision_function(zv), yv, MARGIN)


def build_linear_svm(data: PassengerData) -> ClassLesson:
    """Builds the linear SVM lesson: one frame per epoch of SGD on the hinge loss.

    Args:
        data: The passengers.

    Returns:
        The lesson, with the last epoch kept as the final model.
    """
    standardise = Standardiser.fit(data.x_train)
    z, y, n = standardise(data.x_train), data.y_train, data.n_train
    model = SGDClassifier(loss="hinge", alpha=SVM_ALPHA, random_state=RANDOM_STATE)
    states = _train(model, z, y, [(slice(0, n), float(epoch)) for epoch in range(1, SVM_EPOCHS + 1)])
    strong = _strong_penalty_accuracy(z, y, standardise(data.x_val), data.y_val)

    def note(i: int, s: LinearState) -> str:
        """The training callout for one epoch.

        Args:
            i: The frame's index.
            s: The model after the epoch.

        Returns:
            The epoch, and how many training passengers are still inside the margin.
        """
        return (f"{_epoch_label(s.epochs)}: hinge loss pushes every passenger to\n"
                f"score ≥ 1 on its own side · {s.inside_margin} still inside the margin")

    def story(snapshots: tuple[ClassSnapshot, ...], chosen: int) -> Story:
        """The SVM's narration, which also reports the strong-penalty model's validation accuracy.

        Args:
            snapshots: The lesson's snapshots.
            chosen: The index of the kept snapshot.

        Returns:
            The story.
        """
        last = snapshots[chosen]
        return Story(
            key="linear_svm", title="Linear SVM",
            method="hinge loss with an L2 penalty, trained by stochastic gradient descent",
            objective=("A support vector machine wants more than the right side of the boundary: it wants every "
                       "passenger at least a margin away from it. Watch the boundary and its dashed margin lines "
                       "move as training repeats over the passengers."),
            initial=("After one pass: the solid line is score 0, the dashed lines are scores −1 and +1. The "
                     "shading is a score, not a probability."),
            training=(f"Each frame is one real epoch of SGD on hinge loss + {SVM_ALPHA:g} × ½|weights|². "
                      "Passengers inside the margin, or on the wrong side, are the ones that move the boundary."),
            final=(f"After {SVM_EPOCHS} epochs: train accuracy {last.train_accuracy:.0%}, validation "
                   f"{last.val_accuracy:.0%}. A stronger penalty (α = {STRONG_ALPHA:g}) ignores age and fare, "
                   f"splits on sex alone, and scores {strong:.0%}."),
            inference=("An unseen passenger: the score is bias + weight × input; its sign is the prediction. "
                       "|score| < 1 means it falls inside the margin."),
            takeaway="A linear SVM looks for the widest margin it can afford; the penalty decides how much detail it keeps.",
            limitation=("it gives a score, not a probability, and a straight boundary cannot capture patterns "
                        "such as 'young boys survived'."),
            progress_label="epochs (passes over the training data)")

    return _lesson(data, story, states, len(states) - 1, standardise, "linear SVM", True,
                   {"model": "SGDClassifier(loss='hinge')", "alpha": f"{SVM_ALPHA:g}",
                    "comparison_alpha": f"{STRONG_ALPHA:g} (validation {strong:.3f})",
                    "inputs": "standardised age, log fare, woman", "epochs": str(SVM_EPOCHS),
                    "random_state": str(RANDOM_STATE)}, note)
