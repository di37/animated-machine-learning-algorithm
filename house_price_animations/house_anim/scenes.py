"""Multi-model scenes: the Algorithm Race and the learning-California maps.

Each scene builds its dashboard, opens the sinks (animkit's live window and video writer), plays
the frame stream into them, holds the last picture, and closes the sinks even if something fails.
The race draws ``RACE_TWEEN_FRAMES`` pictures per tick so the leaderboard bars glide; the map
holds each tick for ``MAP_FRAMES_PER_TICK`` video frames.
"""

from pathlib import Path

from house_anim.config import FINAL_HOLD_SECONDS
from house_anim.data import HousingSplits
from house_anim.geo_grid import build_geo_grid
from house_anim.map_dashboard import MapDashboard
from house_anim.map_scene import MAP_MODEL_KEYS, MapFrame, map_frames
from house_anim.models import MODEL_SPECS, get_spec
from house_anim.race import ExamResult, race_frames
from house_anim.race_dashboard import RaceDashboard
from house_anim.runner import RunOptions, finish, open_sinks_or_close, play

RACE_TICKS = 60
RACE_TWEEN_FRAMES = 3  # in-between frames per tick so leaderboard bars glide to their new rank
RESULTS_HOLD_SECONDS = 5.0
MAP_TICKS = 60
MAP_FRAMES_PER_TICK = 2
MAP_GRID_STEP_DEG = 0.06
MAP_MASK_RADIUS_DEG = 0.12

RACE_FILE = "algorithm_race"
MAP_FILE = "california_map"


def _play_race(stream, dashboard: RaceDashboard, sinks) -> tuple[ExamResult, ...]:
    """Draws each race frame several times as the bars glide; returns the stream's final exam."""
    blends = [(i + 1) / RACE_TWEEN_FRAMES for i in range(RACE_TWEEN_FRAMES)]
    while True:
        try:
            frame = next(stream)
        except StopIteration as stop:
            return stop.value
        for blend in blends:
            dashboard.render(frame, blend)
            sinks.emit()


def animate_race(splits: HousingSplits, options: RunOptions,
                 keep_open: bool = False) -> tuple[tuple[ExamResult, ...], Path | None]:
    """Records the Algorithm Race: every model, then the final-exam podium.

    Args:
        splits: The data.
        options: Live window, video file, format, fps and output folder.
        keep_open: Leave the live window open at the end (for the last thing shown).

    Returns:
        The final exam's results (best first) and the video's path (None when not saving).
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


def animate_map(splits: HousingSplits, options: RunOptions,
                keep_open: bool = False) -> tuple[MapFrame, Path | None]:
    """Records the map scene: the ``MAP_MODEL_KEYS`` models repainting California as they train.

    Args:
        splits: The data.
        options: Live window, video file, format, fps and output folder.
        keep_open: Leave the live window open at the end.

    Returns:
        The last map frame and the video's path (None when not saving).
    """
    specs = tuple(get_spec(key) for key in MAP_MODEL_KEYS)
    grid = build_geo_grid(splits, step_deg=MAP_GRID_STEP_DEG, radius_deg=MAP_MASK_RADIUS_DEG)
    dashboard = MapDashboard(specs, grid)
    sinks, path = open_sinks_or_close(dashboard, options, MAP_FILE)
    try:
        frames = map_frames(specs, splits, grid, MAP_TICKS)
        final, _ = play(frames, dashboard.render, sinks, hold_frames=MAP_FRAMES_PER_TICK)
        sinks.hold(FINAL_HOLD_SECONDS * 1.5)
    finally:
        finish(dashboard, sinks, keep_open=keep_open)
    return final, path


def format_exam(exam: tuple[ExamResult, ...], n_test: int) -> str:
    """The final exam as a console table, for after the race.

    Args:
        exam: The results, best first.
        n_test: How many test houses the exam used.

    Returns:
        A title line and one ranked line per model.
    """
    lines = [f"Final exam on {n_test:,} unseen test houses"]
    lines += [f"{rank:>2}. {r.short_title:<18} test RMSE ${r.test_rmse_k:6.1f}k   R² {r.test_r2:.3f}"
              for rank, r in enumerate(exam, start=1)]
    return "\n".join(lines)
