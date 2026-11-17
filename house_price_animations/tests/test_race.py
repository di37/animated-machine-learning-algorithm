from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from house_anim.config import TARGET_TO_THOUSANDS
from house_anim.models import get_spec
from house_anim.race import final_exam, race_frames
from house_anim.trainers import checked_predict


def race_specs():
    return (
        replace(get_spec("linear_regression")),
        replace(get_spec("decision_tree"), steps=(1, 2, 3, 4)),
        replace(get_spec("gradient_boosting"), steps=(5, 10, 20, 40)),
    )


def drain(generator):
    frames = []
    while True:
        try:
            frames.append(next(generator))
        except StopIteration as stop:
            return tuple(frames), stop.value


def test_race_streams_a_frame_per_tick_and_returns_exam(splits):
    frames, exam = drain(race_frames(race_specs(), splits, ticks=5))
    assert len(frames) == 5
    assert frames[-1].is_final and not frames[0].is_final
    assert len(exam) == 3


def test_entries_keep_registry_order_and_fixed_colour_slots(splits):
    frames, _ = drain(race_frames(race_specs(), splits, ticks=4))
    for frame in frames:
        assert [e.key for e in frame.entries] == ["linear_regression", "decision_tree", "gradient_boosting"]
        assert [e.slot for e in frame.entries] == [0, 1, 2]


def test_standings_sort_by_validation_error(splits):
    frames, _ = drain(race_frames(race_specs(), splits, ticks=4))
    for frame in frames:
        errors = [e.val_rmse_k for e in frame.standings]
        assert errors == sorted(errors)
        assert frame.leader == frame.standings[0]


def test_entry_history_is_in_thousands_and_ends_at_current_value(splits):
    frames, _ = drain(race_frames(race_specs(), splits, ticks=4))
    entry = frames[-1].entries[2]
    assert entry.history_val_k[-1] == pytest.approx(entry.val_rmse_k)
    assert entry.history_progress[-1] == pytest.approx(1.0)
    assert np.all(np.diff(entry.history_progress) > 0)
    assert "trees" in entry.step_text


def test_race_frames_are_frozen(splits):
    frames, _ = drain(race_frames(race_specs(), splits, ticks=3))
    with pytest.raises(FrozenInstanceError):
        frames[0].tick = 7
    assert not frames[0].entries[0].history_val_k.flags.writeable


def test_final_exam_scores_each_model_on_the_test_set(splits):
    frames, exam = drain(race_frames(race_specs(), splits, ticks=3))
    assert [r.test_rmse_k for r in exam] == sorted(r.test_rmse_k for r in exam)
    assert {r.key for r in exam} == {"linear_regression", "decision_tree", "gradient_boosting"}
    assert all(0.0 < r.test_rmse_k for r in exam)


def test_final_exam_matches_direct_scoring(splits):
    from house_anim.lockstep import lockstep

    final_tick = tuple(lockstep(race_specs(), splits, ticks=2))[-1]
    exam = final_exam(final_tick.lanes, splits)
    lane = final_tick.lanes[0]
    predictions = checked_predict(lane.model, splits.X_test)
    expected = np.sqrt(np.mean((predictions - splits.y_test) ** 2)) * TARGET_TO_THOUSANDS
    assert next(r for r in exam if r.key == lane.spec.key).test_rmse_k == pytest.approx(expected)


def test_colour_slots_stay_within_the_palette_and_ridge_borrows_lasso_hue():
    from house_anim.models import MODEL_SPECS
    from house_anim.race import colour_slots

    specs = tuple(MODEL_SPECS.values())
    slots = colour_slots(specs)
    assert len(set(slots)) == len(specs)
    by_key = dict(zip((s.key for s in specs), slots))
    assert by_key["ridge"] == (by_key["lasso"][0], True)
    assert max(slot for slot, _ in slots) < 8
