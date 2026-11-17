"""The player runs a chapter's tracks frame by frame and ends on the last recorded picture."""

import numpy as np
import pytest

from manimkit.motion import Planner
from manimkit.recorder import record
from manimkit.timeline import Tracker
from tests.conftest import agg_pixels, figure
from tests.test_recorder import blurred

FRAMES = 80
MOTION = 60


def sliding_scene():
    """A figure whose painter at time t slides a dot across, grows a line and shows a label late."""
    fig = figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_gid("main")

    def paint(t: float) -> None:
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.scatter([0.1 + 0.8 * t], [0.5], s=80, color="#2a78d6")
        if t > 0:
            ax.plot([0.1, 0.1 + 0.8 * t], [0.2, 0.2], color="#e34948", linewidth=2)
        if t > 0.6:
            ax.text(0.5, 0.8, "label", ha="center", fontsize=12)

    return fig, paint


def play(player, fig, paint, frames=FRAMES, motion=MOTION, tracker=None):
    tracker = tracker or Tracker()
    log = tracker.begin(frames)
    for number in range(motion):
        paint(number / max(motion - 1, 1))
        log.frame(number, record(fig)[0])
    chapter = log.finish()
    written = []
    count = player.play(chapter, Planner(chapter).tracks(), lambda data, repeat: written.append((data, repeat)))
    return chapter, written, count


def test_every_frame_is_written_and_still_frames_are_repeated(player):
    fig, paint = sliding_scene()
    _, written, count = play(player, fig, paint)
    assert count == FRAMES and sum(repeat for _, repeat in written) == FRAMES
    assert written[-1][1] > 1  # the hold after the motion is one frame written again
    assert all(len(data) == 160 * 90 * 4 for data, _ in written)


def test_the_chapter_ends_on_the_last_recorded_picture(player):
    fig, paint = sliding_scene()
    chapter, written, _ = play(player, fig, paint)
    assert player.mismatches(chapter) == []  # every shape is its life's last state, nothing still moving
    last = np.frombuffer(written[-1][0], dtype=np.uint8).reshape(90, 160, 4)[..., :3].astype(float)
    paint(1.0)
    assert np.abs(blurred(last) - blurred(agg_pixels(fig))).mean() < 0.5  # the dot alone moves the mean by ~2


def test_a_shape_left_mid_move_is_caught(player):
    fig, paint = sliding_scene()
    chapter, _, _ = play(player, fig, paint)
    moved = next(life for life in chapter.lives if life.end is None and life.leaf.kind == "path")
    shape = player.shape(moved.id)
    shape.points = shape.points + 0.01
    assert player.mismatches(chapter) == [f"life {moved.id} is not drawn as its last state"]


def test_an_animation_running_past_the_chapter_is_refused(player):
    from manimkit.motion import APPEAR, FADE_IN, Track
    fig, paint = sliding_scene()
    paint(1.0)
    log = Tracker().begin(5)
    log.frame(0, record(fig)[0])
    chapter = log.finish()
    tracks = [Track(life.id, APPEAR, 0, 9, life.leaf, FADE_IN) for life in chapter.lives]  # runs to frame 9 of 5
    with pytest.raises(RuntimeError, match="still running"):
        player.play(chapter, tracks, lambda data, repeat: None)


def test_in_between_the_dot_moves_smoothly(player):
    fig, paint = sliding_scene()
    _, written, _ = play(player, fig, paint)
    frames = [np.frombuffer(data, dtype=np.uint8).reshape(90, 160, 4) for data, repeat in written
              for _ in range(repeat)]
    dot = [np.flatnonzero((frame[45, :, 0] < 120) & (frame[45, :, 2] > 150)) for frame in frames[:MOTION]]
    centres = [cols.mean() for cols in dot if len(cols)]
    assert len(set(np.round(centres))) > MOTION // 3  # it slides through many places, not jumping to the end
    assert all(b >= a - 0.5 for a, b in zip(centres, centres[1:]))


def test_the_next_chapter_starts_from_whatever_is_on_screen(player):
    fig, paint = sliding_scene()
    tracker = Tracker()
    play(player, fig, paint, tracker=tracker)
    shapes = player.on_screen()
    chapter, written, _ = play(player, fig, lambda t: paint(1.0), frames=10, motion=1, tracker=tracker)
    assert all(life.start == -1 for life in chapter.lives)  # nothing new: everything carried over
    assert player.on_screen() == shapes and len(written) == 2  # one drawn frame, then nine repeats


def test_a_fading_shape_is_taken_off_screen_when_its_fade_ends(player):
    fig, paint = sliding_scene()
    tracker = Tracker()
    play(player, fig, paint, tracker=tracker)
    before = len(player.on_screen())

    def bare(t):
        fig.axes[0].clear()
        fig.axes[0].set_axis_off()

    play(player, fig, bare, frames=12, motion=1, tracker=tracker)
    assert len(player.on_screen()) < before and not player._fading and not player._running
