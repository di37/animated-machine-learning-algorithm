from dataclasses import replace

import numpy as np
import pytest

from titanic_anim.landscape import LANDSCAPE_MODEL_KEYS, build_survival_grid, landscape_frames
from titanic_anim.landscape_dashboard import LandscapeDashboard
from titanic_anim.models import MODEL_SPECS, get_spec
from titanic_anim.race import colour_slots, race_frames
from titanic_anim.race_dashboard import RaceDashboard
from titanic_anim.scenes import format_exam


def quick(spec):
    return spec if spec.strategy == "data_growth" else replace(spec, steps=spec.steps[:3])


def race_specs():
    return tuple(quick(spec) for spec in MODEL_SPECS.values())


def run_race(splits, ticks):
    stream, frames = race_frames(race_specs(), splits, ticks), []
    while True:
        try:
            frames.append(next(stream))
        except StopIteration as stop:
            return frames, stop.value


def test_sgd_shares_logistic_hue_with_texture():
    styles = dict(zip(MODEL_SPECS, colour_slots(tuple(MODEL_SPECS.values()))))
    assert styles["sgd"] == (styles["logistic_regression"][0], True)
    plain = [slot for key, (slot, textured) in styles.items() if not textured]
    assert sorted(plain) == list(range(8))  # eight validated hues, each used once


def test_race_ranks_by_validation_error_and_ends_with_an_exam(splits):
    frames, exam = run_race(splits, ticks=4)
    final = frames[-1]
    assert final.is_final and all(e.finished for e in final.entries)
    errors = [e.val_error for e in final.standings]
    assert errors == sorted(errors)
    assert final.leader.val_accuracy == pytest.approx(1 - errors[0])
    accuracies = [r.test_accuracy for r in exam]
    assert accuracies == sorted(accuracies, reverse=True) and len(exam) == len(MODEL_SPECS)
    text = format_exam(exam, n_test=len(splits.y_test))
    assert text.splitlines()[0] == f"Final exam on {len(splits.y_test)} unseen test passengers"


def test_race_dashboard_glides_and_shows_the_podium(splits):
    frames, exam = run_race(splits, ticks=3)
    dashboard = RaceDashboard(race_specs(), splits)
    dashboard.render(frames[0], blend=0.5)
    dashboard.render(frames[-1], blend=1.0)
    ranks = {e.key: r for r, e in enumerate(frames[-1].standings)}
    assert dashboard.bar_positions == tuple(float(ranks[e.key]) for e in frames[-1].entries)
    dashboard.show_results(exam, n_test=len(splits.y_test))
    dashboard.figure.canvas.draw()
    assert dashboard.results_text.get_visible() and "1st" in dashboard.results_text.get_text()
    assert "leader:" in dashboard.status_text.get_text()


def test_survival_grid_holds_one_typical_passenger_per_cell(splits):
    grid = build_survival_grid(splits, ages=(5, 30, 60))
    assert grid.shape == (6, 3)
    assert grid.groups[0] == "1st class women" and grid.groups[-1] == "3rd class men"
    raw = grid.features * splits.feature_scale + splits.feature_mean
    np.testing.assert_allclose(raw[:3, 2], [5, 30, 60])  # ages run along each row
    assert (raw[:, 6:] == 0).all()  # Southampton
    with pytest.raises(ValueError):
        grid.to_image(np.zeros(5))


def test_landscape_frames_repaint_only_models_that_trained(splits):
    specs = tuple(quick(get_spec(key)) for key in LANDSCAPE_MODEL_KEYS)
    grid = build_survival_grid(splits, ages=range(0, 81, 10))
    frames = list(landscape_frames(specs, splits, grid, ticks=6))
    assert frames[-1].is_final and len(frames[-1].panels) == len(LANDSCAPE_MODEL_KEYS)
    for panel in frames[-1].panels:
        assert panel.probabilities.shape == (grid.shape[0] * grid.shape[1],)
        assert ((0 <= panel.probabilities) & (panel.probabilities <= 1)).all()
    # Three-step schedules over six ticks: every other frame reuses the cached panel.
    assert frames[0].panels[0] is frames[1].panels[0]
    dashboard = LandscapeDashboard(specs, grid)
    dashboard.render(frames[-1])
    dashboard.figure.canvas.draw()
    assert "val accuracy" in dashboard.step_texts[0].get_text()
