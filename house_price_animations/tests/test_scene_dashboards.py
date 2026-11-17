from dataclasses import replace

import numpy as np
import pytest

from house_anim.geo_grid import build_geo_grid
from house_anim.map_dashboard import MapDashboard
from house_anim.map_scene import map_frames
from house_anim.models import get_spec
from house_anim.race import race_frames
from house_anim.race_dashboard import RaceDashboard, ease


def race_specs():
    return (
        get_spec("linear_regression"),
        replace(get_spec("decision_tree"), steps=(1, 2, 3)),
        replace(get_spec("random_forest"), steps=(1, 2, 3)),
    )


def drain(generator):
    frames = []
    while True:
        try:
            frames.append(next(generator))
        except StopIteration as stop:
            return tuple(frames), stop.value


def test_ease_is_smooth_and_pinned_at_the_ends():
    assert ease(0.0) == 0.0 and ease(1.0) == 1.0
    assert ease(0.25) < 0.25 < 0.75 < ease(0.75)


def test_race_bars_settle_on_rank_order(splits):
    frames, exam = drain(race_frames(race_specs(), splits, ticks=3))
    dashboard = RaceDashboard(race_specs(), splits)
    for frame in frames:
        for blend in (0.5, 1.0):
            dashboard.render(frame, blend)
    dashboard.figure.canvas.draw()

    final = frames[-1]
    rank_of = {entry.key: rank for rank, entry in enumerate(final.standings)}
    for entry, position in zip(final.entries, dashboard.bar_positions):
        assert position == pytest.approx(rank_of[entry.key])
    assert final.leader.short_title in dashboard.status_text.get_text()


def test_race_bars_move_part_way_mid_tween(splits):
    frames, _ = drain(race_frames(race_specs(), splits, ticks=3))
    dashboard = RaceDashboard(race_specs(), splits)
    dashboard.render(frames[0], 1.0)
    before = np.array(dashboard.bar_positions)
    dashboard.render(frames[-1], 0.5)
    middle = np.array(dashboard.bar_positions)
    dashboard.render(frames[-1], 1.0)
    after = np.array(dashboard.bar_positions)
    moved = ~np.isclose(before, after)
    if moved.any():
        assert np.all((middle[moved] - before[moved]) * (after[moved] - before[moved]) > 0)


def test_race_results_card_lists_the_podium(splits):
    frames, exam = drain(race_frames(race_specs(), splits, ticks=2))
    dashboard = RaceDashboard(race_specs(), splits)
    dashboard.render(frames[-1], 1.0)
    dashboard.show_results(exam, n_test=len(splits.y_test))
    dashboard.figure.canvas.draw()
    assert dashboard.results_text.get_visible()
    assert exam[0].short_title in dashboard.results_text.get_text()


def test_map_dashboard_paints_each_panel(splits):
    grid = build_geo_grid(splits, step_deg=0.5, radius_deg=0.6)
    specs = (get_spec("linear_regression"), replace(get_spec("decision_tree"), steps=(1, 2)))
    dashboard = MapDashboard(specs, grid)
    final = None
    for frame in map_frames(specs, splits, grid, ticks=2):
        dashboard.render(frame)
        final = frame
    dashboard.figure.canvas.draw()

    assert len(dashboard.images) == 2
    painted = dashboard.images[0].get_array()
    assert np.isfinite(painted[grid.inside]).all()
    assert "depth 2" in dashboard.step_texts[1].get_text()
    assert "100%" in dashboard.status_text.get_text() and final.is_final


def test_textured_entries_get_hatched_bars_and_dashed_curves(splits):
    specs = (get_spec("lasso"), get_spec("ridge"))
    small = tuple(replace(spec, steps=spec.steps[:2]) for spec in specs)
    frames, _ = drain(race_frames(small, splits, ticks=2))
    dashboard = RaceDashboard(small, splits)
    dashboard.render(frames[-1], 1.0)
    bars = dashboard.leaderboard.bars
    assert bars[0].get_hatch() is None and bars[1].get_hatch() == "///"
    assert dashboard.lanes.lines[1].get_linestyle() == "--"
    assert bars[0].get_facecolor() == bars[1].get_facecolor()


def test_runaway_errors_are_capped_on_both_axes(splits):
    frames, _ = drain(race_frames(race_specs(), splits, ticks=2))
    huge = replace(frames[0], entries=tuple(
        replace(e, val_rmse_k=900.0, history_val_k=e.history_val_k * 0 + 900.0) if i == 0 else e
        for i, e in enumerate(frames[0].entries)))
    dashboard = RaceDashboard(race_specs(), splits)
    dashboard.render(huge, 1.0)
    assert dashboard.lanes.axes[0].get_ylim()[1] <= 250.0 * 1.0001
    x_max = dashboard.leaderboard.ax.get_xlim()[1]
    assert x_max <= 250.0 * 1.25 + 1e-6
    assert dashboard.leaderboard.bars[0].get_width() < x_max
    assert "900" in dashboard.leaderboard.value_texts[0].get_text()
