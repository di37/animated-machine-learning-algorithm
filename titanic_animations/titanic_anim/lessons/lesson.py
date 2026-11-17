"""The classification lesson contract: snapshots of a real model's score map, and the final model.

A model scores a passenger either with a probability (P(survive), threshold 0.5) or, for the
perceptron and linear SVM, with a signed score that is not a probability (threshold 0). The kind
travels with the lesson so no frame can label a score as a probability.

A story module trains its model and keeps a ``ClassSnapshot`` at each moment worth showing: the model's
score on every map cell, its accuracy (and log loss) on the training and validation passengers, and which
of the shown passengers it gets wrong. Every number comes from calling the real model at that moment.
The snapshots, the final model, the narration (an animkit ``Story``) and a side panel make up a
``ClassLesson``: what animkit's director plays and the Titanic ``Scene`` draws. ``verified_score`` wraps
animkit's ``verified``, so a worked prediction that does not reproduce the model's own output is refused
before it can be shown with a ✓.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from matplotlib.axes import Axes

from animkit.data import read_only
from animkit.lessons.lesson import Explanation, Story, verified
from titanic_anim.data import read_only_flags
from titanic_anim.lessons.data import Passenger, PassengerData

PROBABILITY = "probability"
MARGIN = "margin"
KINDS = (PROBABILITY, MARGIN)
LOSS_FLOOR = 1e-12

Scorer = Callable[[np.ndarray], np.ndarray]  # (n, 3) features -> P(survive) or a signed score


def threshold(kind: str) -> float:
    """The score a passenger must beat to be predicted to survive.

    Args:
        kind: ``PROBABILITY`` or ``MARGIN``.

    Returns:
        0.5 for a probability, 0 for a signed score.
    """
    return 0.5 if kind == PROBABILITY else 0.0


def accuracy(scores: np.ndarray, labels: np.ndarray, kind: str) -> float:
    """The share of passengers whose prediction is right.

    Args:
        scores: The model's score for each passenger.
        labels: Their true labels, 1 = survived.
        kind: What the scores are, ``PROBABILITY`` or ``MARGIN``; it sets the threshold.

    Returns:
        A share from 0 to 1: a score above the threshold counts as "survived".
    """
    return float(np.mean((np.asarray(scores) > threshold(kind)) == np.asarray(labels)))


def log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """The mean log loss (cross-entropy) of predicted survival probabilities.

    Each passenger costs -log of the probability the model gave their real outcome, so a confident wrong
    answer costs the most. Probabilities are clipped to ``LOSS_FLOOR`` from 0 and 1, so the cost stays finite.

    Args:
        probabilities: The model's P(survive) for each passenger.
        labels: Their true labels, 1 = survived.

    Returns:
        The mean cost, in nats; 0 is perfect, and log 2 ≈ 0.693 is a flat 50% guess.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), LOSS_FLOOR, 1.0 - LOSS_FLOOR)
    y = np.asarray(labels)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def percent(value: float) -> str:
    """A share as a percentage with one decimal.

    Args:
        value: A share from 0 to 1.

    Returns:
        For example '81.1%'.
    """
    return f"{value:.1%}"


def signed_small(value: float) -> str:
    """A signed weight, push, log term or log-odds step, as shown on screen.

    Args:
        value: The number.

    Returns:
        '+0.31' as a rule, '-0.004' when it is small, and '0.000' (never '-0.00' or '+0.00') for a value
        that rounds to zero.
    """
    if abs(value) >= 0.01:
        return f"{value:+.2f}"
    return f"{value:+.3f}" if round(abs(value), 3) else "0.000"


def signed(value: float) -> str:
    """A signed score with two decimals.

    Args:
        value: The score.

    Returns:
        For example '+1.23' or '-0.40'.
    """
    return f"{value:+.2f}"


def outcome(label: int) -> str:
    """A label in words.

    Args:
        label: 1 for a survivor, anything else for a passenger who died.

    Returns:
        'survived' or 'did not survive'.
    """
    return "survived" if label == 1 else "did not survive"


@dataclass(frozen=True)
class ClassSnapshot:
    """The model at one moment of training, as one frame of the video shows it.

    Attributes:
        label: The frame's name in the header, such as "Step 12 of 60" or "k = 8".
        progress: Where the frame sits on the metric chart's x axis: a step, an epoch, a count of leaves,
            trees, neighbours or passengers.
        surface: The model's score on every map cell, shape (2, fares, ages).
        train_accuracy: The share of training passengers it gets right.
        val_accuracy: The share of validation passengers it gets right.
        train_log_loss: Its log loss on the training passengers; NaN for score-only models.
        val_log_loss: Its log loss on the validation passengers; NaN for score-only models.
        wrong: For each training passenger in display order, whether the model gets them wrong.
        note: The teaching note in the main plot's corner.
        detail: The story's own record of the model at this moment (its weights, tree, counts ...), which
            its panel and diagram draw from.
    """

    label: str
    progress: float
    surface: np.ndarray  # (2, fares, ages) scores on both maps
    train_accuracy: float
    val_accuracy: float
    train_log_loss: float  # NaN for score-only models
    val_log_loss: float
    wrong: np.ndarray  # misclassified flags for the displayed training passengers
    note: str
    detail: Any = None


def make_snapshot(score: Scorer, data: PassengerData, kind: str, *, label: str, progress: float,
                  note: str, detail: Any = None) -> ClassSnapshot:
    """Scores the model at one moment on the training, validation and map passengers, and records the frame.

    Args:
        score: The model at this moment: (n, 3) feature rows in, one score per row out.
        data: The passengers.
        kind: What ``score`` returns, ``PROBABILITY`` or ``MARGIN``; log loss is measured only for a probability.
        label: The frame's name in the header.
        progress: Where the frame sits on the metric chart's x axis.
        note: The teaching note in the main plot's corner.
        detail: The story's own record of the model at this moment, for its panel and diagram.

    Returns:
        The snapshot, with read-only arrays.
    """
    train, val = score(data.x_train), score(data.x_val)
    shown = data.display
    probability = kind == PROBABILITY
    return ClassSnapshot(
        label=label,
        progress=float(progress),
        surface=read_only(np.asarray(score(data.grid)).reshape(data.surface_shape)),
        train_accuracy=accuracy(train, data.y_train, kind),
        val_accuracy=accuracy(val, data.y_val, kind),
        train_log_loss=log_loss(train, data.y_train) if probability else float("nan"),
        val_log_loss=log_loss(val, data.y_val) if probability else float("nan"),
        wrong=read_only_flags((np.asarray(train)[shown] > threshold(kind)) != data.y_train[shown]),
        note=note,
        detail=detail,
    )


def verified_score(lines: list[str], worked: float, model_value: float, kind: str,
                   call: str | None = None) -> Explanation:
    """The worked score must reproduce the model's own output before the ✓ is printed.

    A thin wrapper of animkit's ``verified`` that formats the check line for the lesson's kind of score.

    Args:
        lines: The worked prediction, one line each; the check line is added after them.
        worked: The score the worked lines arrive at.
        model_value: The model's own score for the same passenger.
        kind: ``PROBABILITY`` (shown as a percentage) or ``MARGIN`` (shown signed).
        call: What ``model_value`` came from, when it is not the scikit-learn default
            (``predict_proba`` for a probability, ``decision_function`` for a score).

    Returns:
        The explanation: the lines, the check line, and the prediction.

    Raises:
        LessonError: If ``worked`` and ``model_value`` disagree.
    """
    if kind == PROBABILITY:
        return verified(lines, worked, model_value, show=percent, call=call or "model.predict_proba(x)[survived]")
    return verified(lines, worked, model_value, show=signed, call=call or "model.decision_function(x)")


def best_validation(snapshots: tuple[ClassSnapshot, ...]) -> int:
    """Highest validation accuracy; ties go to the lower validation log loss, then the earlier frame.

    Args:
        snapshots: The lesson's snapshots, in training order.

    Returns:
        The index of the snapshot to keep as the final model.
    """
    def rank(index: int) -> tuple[float, float, int]:
        """The sort key of one snapshot: smaller is better.

        Args:
            index: The snapshot's index.

        Returns:
            Minus its validation accuracy, its validation log loss (infinite when there is none), and
            the index itself.
        """
        snap = snapshots[index]
        loss = snap.val_log_loss if np.isfinite(snap.val_log_loss) else np.inf
        return (-snap.val_accuracy, loss, index)
    return min(range(len(snapshots)), key=rank)


class ClassPanel(Protocol):
    """Side-panel drawing for a classifier lesson; overlays get both maps (women, men).

    Each story has its own panel: weight bars, bell curves, the tree, the network and so on. animkit's
    director calls ``draw_training`` and ``draw_inference`` on the panel's own axes; the Titanic scene
    calls ``overlay_training`` and ``overlay_inference`` to add extras on the two maps (most panels add none).
    """

    def draw_training(self, ax: Axes, snapshot: "ClassSnapshot") -> None:
        """Draws the model at one moment of training in the side panel.

        Args:
            ax: The side panel's axes.
            snapshot: The moment to draw; its ``detail`` is the story's own record of the model.
        """
        ...

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the final model working on the unseen passenger in the side panel.

        Args:
            ax: The side panel's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        ...

    def overlay_training(self, axes: tuple[Axes, ...], snapshot: "ClassSnapshot") -> None:
        """Adds the panel's extras, if any, to the two maps during training.

        Args:
            axes: The two maps, women's first.
            snapshot: The moment being drawn.
        """
        ...

    def overlay_inference(self, axes: tuple[Axes, ...], query: Passenger, t: float) -> None:
        """Adds the panel's extras, if any, to the two maps during the prediction.

        Args:
            axes: The two maps, women's first.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        ...


@dataclass(frozen=True)
class ClassLesson:
    """One classifier lesson: what animkit's director plays and the Titanic scene draws.

    Building one checks it: the worked prediction for the unseen passenger is run at once, so a check
    that does not add up fails before any frame is rendered.

    Attributes:
        story: Every word the video says about the algorithm (animkit's ``Story``).
        data: The passengers.
        snapshots: The model at each moment of training, in order.
        final_index: The snapshot kept as the final model.
        score: The final model: (n, 3) feature rows in, one score per row out.
        kind: What ``score`` returns, ``PROBABILITY`` or ``MARGIN``.
        explain: Works one passenger's prediction through, line by line, ending in the verified check.
        panel: The side panel that draws the model's inner workings.
        config: The model's settings as text, for the summary CSV; stored read-only.
        frame_weights: Each training snapshot's relative share of the training chapter's frames; None
            shares them equally.
        diagram: An animkit ``Diagram`` of how the algorithm computes, shown above the panel; None when the
            panel already is one.
    """

    story: Story
    data: PassengerData
    snapshots: tuple[ClassSnapshot, ...]
    final_index: int
    score: Scorer
    kind: str
    explain: Callable[[np.ndarray], Explanation]
    panel: ClassPanel
    config: Mapping[str, str]
    frame_weights: tuple[float, ...] | None = None
    diagram: Any = None  # algorithm diagram shown above the panel; None when the panel already is one

    def __post_init__(self):
        """Freezes ``config`` and refuses a lesson that cannot be shown truthfully.

        Raises:
            ValueError: If the kind is unknown, there are no snapshots, or ``final_index`` points at none.
            LessonError: If the worked prediction for the unseen passenger does not reproduce the model.
        """
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not self.snapshots:
            raise ValueError("a lesson needs at least one snapshot")
        if not 0 <= self.final_index < len(self.snapshots):
            raise ValueError("final_index must point at a snapshot")
        self.explain(self.data.query.features)  # a worked check that does not add up fails now, not mid-render

    @property
    def final(self) -> ClassSnapshot:
        """The snapshot kept as the final model.

        Returns:
            ``snapshots[final_index]``.
        """
        return self.snapshots[self.final_index]

    def score_one(self, features: np.ndarray) -> float:
        """The final model's score for one passenger.

        Args:
            features: One (age, log fare, female) row.

        Returns:
            Their P(survive), or their signed score for a score-only model.
        """
        return float(self.score(np.asarray(features, dtype=float).reshape(1, -1))[0])
