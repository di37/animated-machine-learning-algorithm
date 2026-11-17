from dataclasses import FrozenInstanceError, replace

import pytest

from house_anim.lockstep import lockstep
from house_anim.models import get_spec


def specs():
    return (
        replace(get_spec("decision_tree"), steps=(1, 2, 3)),
        replace(get_spec("gradient_boosting"), steps=tuple(range(5, 65, 5))),
    )


def test_one_snapshot_per_tick_with_a_lane_per_model(splits):
    ticks = tuple(lockstep(specs(), splits, ticks=6))
    assert [t.tick for t in ticks] == [1, 2, 3, 4, 5, 6]
    assert all(len(t.lanes) == 2 for t in ticks)
    assert [lane.spec.key for lane in ticks[0].lanes] == ["decision_tree", "gradient_boosting"]


def test_lanes_advance_in_proportion_to_their_schedules(splits):
    ticks = tuple(lockstep(specs(), splits, ticks=6))
    tree_steps = [t.lanes[0].frame.step_index + 1 for t in ticks]
    boosting_steps = [t.lanes[1].frame.step_index + 1 for t in ticks]
    assert tree_steps == [1, 1, 2, 2, 3, 3]
    assert boosting_steps == [2, 4, 6, 8, 10, 12]


def test_every_lane_finishes_on_the_last_tick(splits):
    final = tuple(lockstep(specs(), splits, ticks=4))[-1]
    assert all(lane.finished for lane in final.lanes)
    assert all(lane.progress == 1.0 for lane in final.lanes)
    assert final.is_final


def test_lane_exposes_a_usable_model(splits):
    first = next(lockstep(specs(), splits, ticks=3))
    assert first.lanes[0].model.predict(splits.X_val).shape == splits.y_val.shape


def test_snapshots_are_frozen(splits):
    first = next(lockstep(specs(), splits, ticks=3))
    with pytest.raises(FrozenInstanceError):
        first.tick = 9


@pytest.mark.parametrize("ticks", [0, -1])
def test_ticks_must_be_positive(splits, ticks):
    with pytest.raises(ValueError, match="ticks"):
        next(lockstep(specs(), splits, ticks=ticks))


def test_needs_at_least_one_model(splits):
    with pytest.raises(ValueError, match="at least one"):
        next(lockstep((), splits, ticks=3))
