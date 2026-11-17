"""The Algorithm Race for classifiers: lockstep training, standings by validation error, a test exam.

The nine classifiers have schedules of different lengths: 14 depths for the tree, 80 chunks for
SGD. animkit's ``lockstep`` advances them together, so at tick k of n every model is k/n of the way
through its own schedule: the models are compared at the same share of their training, not after
the same number of steps. Each tick, they are ranked by validation error. When the race ends,
every final model sits one exam on the test passengers, which no model has seen.

A model keeps one colour slot whatever its rank (``colour_slots``), so the eye can follow it
up and down the leaderboard.
"""

from collections.abc import Generator, Sequence
from dataclasses import dataclass

import numpy as np

from animkit.data import read_only
from animkit.lockstep import Lane, lockstep
from animkit.panels import format_step
from animkit.race import colour_slots as shared_colour_slots
from titanic_anim.data import TitanicSplits
from titanic_anim.metrics import score
from titanic_anim.models import SCORE_MARGIN, ClassifierSpec
from titanic_anim.trainers import checked_labels, survival_scores, training_steps

# Eight validated hues for nine models: SGD shares Logistic Regression's hue (both are logistic
# models) and is told apart by texture (hatched bar, dashed curve) as well as its name label.
SIBLING_HUES = {"sgd": "logistic_regression"}


def colour_slots(specs: Sequence[ClassifierSpec]) -> tuple[tuple[int, bool], ...]:
    """Each model's palette slot and texture; SGD shares Logistic Regression's hue and is hatched.

    Args:
        specs: The racing classifiers, in order.

    Returns:
        ``(slot, textured)`` per model, in the same order.
    """
    return shared_colour_slots(specs, SIBLING_HUES)


@dataclass(frozen=True)
class RaceEntry:
    """One model's standing at one tick.

    Attributes:
        key: The model's key.
        short_title: Its name on the dashboard.
        slot: Its fixed colour slot: colour follows the model, never its rank.
        textured: Whether it is told from a sibling by texture (hatched bar, dashed curve).
        val_error: The share of validation passengers it misclassifies now.
        history_progress: The share of its own schedule done at each step so far.
        history_error: Its validation error at each of those steps.
        step_text: Where it is in its schedule, e.g. "40 trees".
        finished: Whether it has finished its schedule.
    """

    key: str
    short_title: str
    slot: int  # fixed colour slot: colour follows the model, never its rank
    textured: bool
    val_error: float  # share of validation passengers misclassified
    history_progress: np.ndarray  # share of the model's own schedule at each recorded step
    history_error: np.ndarray
    step_text: str
    finished: bool

    @property
    def val_accuracy(self) -> float:
        """The share of validation passengers the model classifies correctly.

        Returns:
            1 − ``val_error``.
        """
        return 1.0 - self.val_error


@dataclass(frozen=True)
class RaceFrame:
    """The whole race at one tick.

    Attributes:
        tick: This tick, from 1.
        ticks: How many ticks the race has.
        entries: Every model's standing, in the order the models were given.
    """

    tick: int
    ticks: int
    entries: tuple[RaceEntry, ...]  # in the order the models were given

    @property
    def standings(self) -> tuple[RaceEntry, ...]:
        """The entries ranked by validation error.

        Returns:
            The entries, lowest error first.
        """
        return tuple(sorted(self.entries, key=lambda e: e.val_error))

    @property
    def leader(self) -> RaceEntry:
        """The model with the lowest validation error.

        Returns:
            The first of ``standings``.
        """
        return self.standings[0]

    @property
    def is_final(self) -> bool:
        """Whether this is the race's last tick.

        Returns:
            True on the last tick.
        """
        return self.tick == self.ticks


@dataclass(frozen=True)
class ExamResult:
    """One model's score on the test passengers, after the race.

    Attributes:
        key: The model's key.
        short_title: Its name on the dashboard.
        slot: Its colour slot.
        test_accuracy: The share of test passengers it classifies correctly.
        test_auc: The area under its ROC curve on the test passengers.
    """

    key: str
    short_title: str
    slot: int
    test_accuracy: float
    test_auc: float


def _entry(style: tuple[int, bool], lane: Lane) -> RaceEntry:
    """One model's standing at this tick, read from its lane's latest training frame."""
    frame = lane.frame
    points = len(frame.history)
    return RaceEntry(
        key=lane.spec.key,
        short_title=lane.spec.short_title,
        slot=style[0],
        textured=style[1],
        val_error=1.0 - frame.val_scores.accuracy,
        history_progress=read_only(np.arange(1, points + 1) / frame.total_steps),
        history_error=read_only(1.0 - np.array([p.val_accuracy for p in frame.history])),
        step_text=lane.spec.step_format.format(format_step(frame.step)),
        finished=lane.finished,
    )


def final_exam(lanes: Sequence[Lane], splits: TitanicSplits) -> tuple[ExamResult, ...]:
    """Scores every finished model once on the held-out test passengers, best first.

    Args:
        lanes: Each model's lane at the race's last tick.
        splits: The splits; only the test passengers are scored.

    Returns:
        One result per model, highest test accuracy first.
    """
    results = []
    for (slot, _), lane in zip(colour_slots([lane.spec for lane in lanes]), lanes):
        kind = lane.spec.score_kind
        scores = score(splits.y_test, checked_labels(lane.model, splits.X_test),
                       survival_scores(lane.model, splits.X_test, kind), probabilities=kind != SCORE_MARGIN)
        results.append(ExamResult(lane.spec.key, lane.spec.short_title, slot, scores.accuracy, scores.auc))
    return tuple(sorted(results, key=lambda r: -r.test_accuracy))


def race_frames(
    specs: Sequence[ClassifierSpec], splits: TitanicSplits, ticks: int
) -> Generator[RaceFrame, None, tuple[ExamResult, ...]]:
    """Trains every model in lockstep, yielding the race at each tick.

    The generator's return value is the final exam, ``final_exam`` of the last tick's models.

    Args:
        specs: The racing classifiers.
        splits: The splits: training and validation drive the race, test is the exam.
        ticks: How many frames the race has; at tick k every model is k/ticks through its schedule.

    Yields:
        A ``RaceFrame`` per tick.
    """
    styles = colour_slots(specs)
    last = None
    for snapshot in lockstep(specs, splits, ticks, step_fn=training_steps):
        last = snapshot
        yield RaceFrame(
            tick=snapshot.tick,
            ticks=snapshot.ticks,
            entries=tuple(_entry(style, lane) for style, lane in zip(styles, snapshot.lanes)),
        )
    return final_exam(last.lanes, splits)
