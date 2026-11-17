"""The tracker follows leaves frame to frame: unchanged, changed, replaced, gone, carried, live."""

import numpy as np

from manimkit.timeline import CARRIED, Tracker
from tests.test_leaves import RED, square, text


def keyed(leaf, key):
    leaf.key = key
    return leaf


def test_a_leaf_that_draws_the_same_stays_one_life():
    log = Tracker().begin(3)
    for frame in range(3):
        log.frame(frame, [square()])
    chapter = log.finish()
    assert len(chapter.lives) == 1 and len(chapter.lives[0].states) == 1
    assert list(chapter.orders) == [0]  # the order never changed


def test_a_moved_leaf_is_the_same_life_with_one_state_per_change():
    log = Tracker().begin(3)
    for frame in range(3):
        log.frame(frame, [square(frame * 5)])
    life = log.finish().lives[0]
    assert [frame for frame, _ in life.states] == [0, 1, 2] and life.end is None


def test_a_leaf_that_jumps_too_far_or_changes_kind_is_replaced():
    tracker = Tracker(max_step_px=20)
    log = tracker.begin(3)
    log.frame(0, [square(0)])
    log.frame(1, [square(100)])  # too far
    log.frame(2, [keyed(text(), ("k", 0))])  # something else under the same key
    lives = log.finish().lives
    assert [life.start for life in lives] == [0, 1, 2]
    assert lives[0].end == 1 and lives[0].replaced and lives[1].end == 2


def test_a_leaf_missing_from_a_frame_has_gone():
    log = Tracker().begin(2)
    log.frame(0, [square(), keyed(square(fill=RED), ("k", 1))])
    log.frame(1, [square()])
    lives = log.finish().lives
    assert lives[1].end == 1 and not lives[1].replaced
    np.testing.assert_array_equal(log.finish().orders[1], [lives[0].id])


def test_whatever_is_still_on_screen_is_carried_into_the_next_chapter():
    tracker = Tracker()
    first = tracker.begin(1)
    first.frame(0, [square()])
    life = first.finish().lives[0]
    second = tracker.begin(1)
    second.frame(0, [square()])
    carried = second.finish().lives
    assert carried == [life] and life.start == CARRIED and life.states[0][0] == CARRIED


def test_live_leaves_start_a_new_life_whenever_they_change():
    log = Tracker().begin(3, live=lambda key: key[0] == "k")
    for frame in range(3):
        log.frame(frame, [square(frame)])
    lives = log.finish().lives
    assert len(lives) == 3 and all(life.live for life in lives)


def test_two_leaves_with_one_key_are_told_apart_by_their_order():
    log = Tracker().begin(1)
    log.frame(0, [square(), square(50)])
    assert len(log.finish().lives) == 2


def test_orders_are_kept_only_when_they_change():
    log = Tracker().begin(3)
    log.frame(0, [square(), keyed(square(fill=RED), ("k", 1))])
    log.frame(1, [square(), keyed(square(fill=RED), ("k", 1))])
    log.frame(2, [keyed(square(fill=RED), ("k", 1)), square()])
    assert sorted(log.finish().orders) == [0, 2]
