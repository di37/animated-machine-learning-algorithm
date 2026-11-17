"""Multi-model scenes: the classifier race and the survival landscapes.

Each scene builds its figure, opens animkit's sinks (the live window, the video file or both),
plays its frames as training produces them, holds the last picture so it can be read, and closes
the sinks, even when the run is stopped part way.
"""

from pathlib import Path

from animkit.config import FINAL_HOLD_SECONDS
from animkit.runner import RunOptions, finish, open_sinks_or_close, play
from titanic_anim.data import TitanicSplits
from titanic_anim.landscape import LANDSCAPE_MODEL_KEYS, LandscapeFrame, build_survival_grid, landscape_frames
from titanic_anim.landscape_dashboard import LandscapeDashboard
from titanic_anim.models import MODEL_SPECS, get_spec
from titanic_anim.panels import percent
from titanic_anim.race import ExamResult, race_frames
from titanic_anim.race_dashboard import RaceDashboard

RACE_TICKS = 60
RACE_TWEEN_FRAMES = 3  # in-between frames per tick so leaderboard bars glide to their new rank
RESULTS_HOLD_SECONDS = 5.0
LANDSCAPE_TICKS = 60
LANDSCAPE_FRAMES_PER_TICK = 2

RACE_FILE = "algorithm_race"
LANDSCAPE_FILE = "survival_landscape"


def _play_race(stream, dashboard: RaceDashboard, sinks) -> tuple[ExamResult, ...]:
    """Draws each race frame ``RACE_TWEEN_FRAMES`` times, easing the bars; returns the stream's exam results."""
    blends = [(i + 1) / RACE_TWEEN_FRAMES for i in range(RACE_TWEEN_FRAMES)]
    while True:
        try:
            frame = next(stream)
        except StopIteration as stop:
            return stop.value
        for blend in blends:
            dashboard.render(frame, blend)
            sinks.emit()


def animate_race(splits: TitanicSplits, options: RunOptions,
                 keep_open: bool = False) -> tuple[tuple[ExamResult, ...], Path | None]:
    """Plays the Algorithm Race with all nine classifiers, then shows the exam's podium.

    Args:
        splits: The passengers.
        options: Where the frames go: live window, video file, frame rate and so on.
        keep_open: Whether to leave the live window open afterwards.

    Returns:
        The exam results, best first, and the video's path (None when nothing was saved).
    """
    specs = tuple(MODEL_SPECS.values())
    dashboard = RaceDashboard(specs, splits)
    sinks, path = open_sinks_or_close(dashboard, options, RACE_FILE)
    try:
        exam = _play_race(race_frames(specs, splits, RACE_TICKS), dashboard, sinks)
        sinks.hold(FINAL_HOLD_SECONDS / 2)
        dashboard.show_results(exam, n_test=len(splits.y_test))
        sinks.hold(RESULTS_HOLD_SECONDS)
    finally:
        finish(dashboard, sinks, keep_open=keep_open)
    return exam, path


def animate_landscape(splits: TitanicSplits, options: RunOptions,
                      keep_open: bool = False) -> tuple[LandscapeFrame, Path | None]:
    """Plays the survival landscapes of the ``LANDSCAPE_MODEL_KEYS`` models.

    Args:
        splits: The passengers.
        options: Where the frames go: live window, video file, frame rate and so on.
        keep_open: Whether to leave the live window open afterwards.

    Returns:
        The last frame, and the video's path (None when nothing was saved).
    """
    specs = tuple(get_spec(key) for key in LANDSCAPE_MODEL_KEYS)
    grid = build_survival_grid(splits)
    dashboard = LandscapeDashboard(specs, grid)
    sinks, path = open_sinks_or_close(dashboard, options, LANDSCAPE_FILE)
    try:
        frames = landscape_frames(specs, splits, grid, LANDSCAPE_TICKS)
        final, _ = play(frames, dashboard.render, sinks, hold_frames=LANDSCAPE_FRAMES_PER_TICK)
        sinks.hold(FINAL_HOLD_SECONDS * 1.5)
    finally:
        finish(dashboard, sinks, keep_open=keep_open)
    return final, path


def format_exam(exam: tuple[ExamResult, ...], n_test: int) -> str:
    """The exam results as a console table.

    Args:
        exam: Every model's exam result, best first.
        n_test: How many test passengers sat the exam.

    Returns:
        A title line and one ranked line per model.
    """
    lines = [f"Final exam on {n_test:,} unseen test passengers"]
    lines += [f"{rank:>2}. {r.short_title:<20} test accuracy {percent(r.test_accuracy):>6}   AUC {r.test_auc:.3f}"
              for rank, r in enumerate(exam, start=1)]
    return "\n".join(lines)
