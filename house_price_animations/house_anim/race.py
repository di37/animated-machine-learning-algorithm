"""The Algorithm Race: every model trains in lockstep; standings by validation error, then a test exam.

At each tick every model is equally far through its own schedule (see ``lockstep``), and the
standings rank them by validation RMSE: the validation houses are there to judge a model while it
trains. The test houses stay unseen until the very end, when each finished model sits one final
exam on them.

This module makes the race's immutable snapshots; ``race_dashboard`` draws them.
"""

from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from animkit.race import colour_slots as colour_slots_with
from house_anim.config import TARGET_TO_THOUSANDS
from house_anim.data import HousingSplits, read_only
from house_anim.lockstep import Lane, lockstep
from house_anim.metrics import score
from house_anim.models import ModelSpec
from house_anim.panels import format_step
from house_anim.trainers import checked_predict

# The palette has eight validated hues. A model listed here shares its sibling's hue and is told
# apart by texture (hatched bar, dashed curve) as well as its name label.
SIBLING_HUES = {"ridge": "lasso"}


def colour_slots(specs: Sequence[ModelSpec],
                 siblings: Mapping[str, str] = SIBLING_HUES) -> tuple[tuple[int, bool], ...]:
    """(palette slot, textured) per model, in order; colour follows the model, never its rank.

    animkit's ``colour_slots``, with this project's sibling pairs by default.

    Args:
        specs: The models, in the order they are drawn.
        siblings: Model key -> the key whose hue it borrows (it is then drawn textured).

    Returns:
        One (palette slot, textured) pair per model.
    """
    return colour_slots_with(specs, siblings)


@dataclass(frozen=True)
class RaceEntry:
    """One model's lane at one tick of the race.

    Attributes:
        key: The model's key.
        short_title: Its name on the lane and the leaderboard.
        slot: Its fixed colour slot in the palette.
        textured: Whether it borrows a sibling's hue and is drawn hatched and dashed.
        val_rmse_k: Its validation RMSE now, in $k.
        history_progress: The share of its own schedule done at each step so far (0 to 1).
        history_val_k: Its validation RMSE at each of those steps, in $k.
        step_text: How far it has trained, such as ``"40 trees"``.
        fit_seconds: All its fitting so far, in seconds.
        finished: Whether it has reached the end of its schedule.
    """

    key: str
    short_title: str
    slot: int  # fixed colour slot: colour follows the model, never its rank
    textured: bool
    val_rmse_k: float
    history_progress: np.ndarray  # share of the model's own schedule at each recorded step
    history_val_k: np.ndarray
    step_text: str
    fit_seconds: float
    finished: bool


@dataclass(frozen=True)
class RaceFrame:
    """Every model's lane at one tick.

    Attributes:
        tick: This tick, from 1.
        ticks: How many ticks the race has.
        entries: One entry per model, in the order the models were given (not by rank).
    """

    tick: int
    ticks: int
    entries: tuple[RaceEntry, ...]  # in the order the models were given

    @property
    def standings(self) -> tuple[RaceEntry, ...]:
        """The entries ranked by validation RMSE, lowest first.

        Returns:
            The entries, best first.
        """
        return tuple(sorted(self.entries, key=lambda e: e.val_rmse_k))

    @property
    def leader(self) -> RaceEntry:
        """The entry with the lowest validation RMSE.

        Returns:
            The first of ``standings``.
        """
        return self.standings[0]

    @property
    def is_final(self) -> bool:
        """Whether this is the race's last tick.

        Returns:
            True on the last tick, when every model has finished its schedule.
        """
        return self.tick == self.ticks


@dataclass(frozen=True)
class ExamResult:
    """One model's score on the final exam: the test houses it never saw while training.

    Attributes:
        key: The model's key.
        short_title: Its name on the podium.
        slot: Its colour slot.
        test_rmse_k: Test RMSE, in $k.
        test_r2: Test R².
    """

    key: str
    short_title: str
    slot: int
    test_rmse_k: float
    test_r2: float


def _entry(style: tuple[int, bool], lane: Lane) -> RaceEntry:
    """One lane's entry, in $k, with its colour style and its learning curve against schedule share."""
    frame = lane.frame
    points = len(frame.history)
    return RaceEntry(
        key=lane.spec.key,
        short_title=lane.spec.short_title,
        slot=style[0],
        textured=style[1],
        val_rmse_k=frame.val_scores.rmse * TARGET_TO_THOUSANDS,
        history_progress=read_only(np.arange(1, points + 1) / frame.total_steps),
        history_val_k=read_only(np.array([p.val_rmse for p in frame.history]) * TARGET_TO_THOUSANDS),
        step_text=lane.spec.step_format.format(format_step(frame.step)),
        fit_seconds=frame.fit_seconds,
        finished=lane.finished,
    )


def final_exam(lanes: Sequence[Lane], splits: HousingSplits) -> tuple[ExamResult, ...]:
    """Score every finished model once on the held-out test houses, best first.

    Args:
        lanes: The last tick's lanes, one finished model each.
        splits: The data; only its test rows are used.

    Returns:
        One result per model, lowest test RMSE first.
    """
    results = []
    for (slot, _), lane in zip(colour_slots([lane.spec for lane in lanes]), lanes):
        scores = score(splits.y_test, checked_predict(lane.model, splits.X_test))
        results.append(ExamResult(lane.spec.key, lane.spec.short_title, slot,
                                  scores.rmse * TARGET_TO_THOUSANDS, scores.r2))
    return tuple(sorted(results, key=lambda r: r.test_rmse_k))


def race_frames(
    specs: Sequence[ModelSpec], splits: HousingSplits, ticks: int
) -> Generator[RaceFrame, None, tuple[ExamResult, ...]]:
    """Runs the race: one frame per lockstep tick, then the final exam.

    Args:
        specs: The models to race.
        splits: The data they train on.
        ticks: How many frames the race has.

    Returns:
        The final exam's results, best first, as the generator's return value.

    Yields:
        One ``RaceFrame`` per tick.
    """
    styles = colour_slots(specs)
    last = None
    for snapshot in lockstep(specs, splits, ticks):
        last = snapshot
        yield RaceFrame(
            tick=snapshot.tick,
            ticks=snapshot.ticks,
            entries=tuple(_entry(style, lane) for style, lane in zip(styles, snapshot.lanes)),
        )
    return final_exam(last.lanes, splits)
