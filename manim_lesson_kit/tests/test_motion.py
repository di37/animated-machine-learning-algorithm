"""Lives become tracks: reveals, transforms, cuts and fades on the right frames, inside the chapter."""

import numpy as np
import pytest

from manimkit.leaves import PathLeaf
from manimkit.motion import (
    APPEAR,
    CREATE,
    CUT,
    FADE_IN,
    FADE_OUT,
    FADE_SECONDS,
    MORPH,
    REMOVE,
    REVEAL_SECONDS,
    WRITE,
    PlanError,
    Planner,
    Track,
    grows_in,
    reveal_style,
)
from manimkit.paths import LINETO, MOVETO
from manimkit.timeline import CARRIED, Chapter, Life
from tests.test_leaves import square, text

FRAMES = 60
FPS = 30
FADE, REVEAL = round(FADE_SECONDS * FPS), round(REVEAL_SECONDS * FPS)  # 6 and 12 frames


def chapter(*lives: Life, frames: int = FRAMES) -> Chapter:
    return Chapter(frames, list(lives), {})


def life(start, changes=(), end=None, *, replaced=False, live=False, ident=0) -> Life:
    states = [(start, square())] + [(frame, square(frame)) for frame in changes]
    return Life(ident, ("k", ident), start, states, end, replaced, live)


def plan(*lives, **options):
    return Planner(chapter(*lives), **options).tracks()


def test_a_thing_that_pops_up_mid_chapter_is_revealed_over_reveal_frames():
    (track,) = plan(life(10))
    assert (track.kind, track.start, track.end) == (APPEAR, 10, 10 + REVEAL - 1)


def fading(start, frames) -> Life:
    """A square matplotlib fades in: its opacity climbs frame by frame."""
    states = [(frame, square(fill=(0.2, 0.4, 0.8, (frame - start + 1) / len(frames)))) for frame in frames]
    return Life(0, ("k", 0), start, states)


def test_a_thing_matplotlib_grows_in_is_revealed_over_those_frames_ending_on_its_last_state():
    tracks = plan(fading(10, range(10, 31)))
    assert len(tracks) == 1
    assert (tracks[0].kind, tracks[0].start, tracks[0].end) == (APPEAR, 10, 30)
    assert tracks[0].leaf.fill[3] == pytest.approx(1.0)


def test_a_thing_that_moves_from_the_moment_it_appears_is_drawn_in_then_moves_on():
    appear, move = plan(life(10, changes=range(11, 41)))
    assert (appear.kind, appear.start, appear.end) == (APPEAR, 10, 10 + REVEAL - 1)
    assert appear.leaf.anchor()[0] == pytest.approx(10 + REVEAL - 1 + 5)  # where it has got to by then
    assert (move.kind, move.start, move.end) == (MORPH, 10 + REVEAL, 40)


def test_growing_means_much_smaller_or_fainter_than_later():
    tiny = PathLeaf(("k", 0), np.array([[0, 0], [1, 0]], np.float32), np.array([MOVETO, LINETO], np.uint8),
                    fill=None, stroke=(0, 0, 0, 1), linewidth=1.0, dashes=None, cap="butt", join="miter")
    long = PathLeaf(("k", 0), np.array([[0, 0], [50, 0]], np.float32), np.array([MOVETO, LINETO], np.uint8),
                    fill=None, stroke=(0, 0, 0, 1), linewidth=1.0, dashes=None, cap="butt", join="miter")
    assert grows_in(tiny, long) and not grows_in(long, tiny)
    assert grows_in(text(color=(0, 0, 0, 0.1)), text()) and not grows_in(square(), square(40))


def test_at_the_first_frame_the_new_picture_waits_for_the_old_to_fade():
    fade, appear = plan(life(0, ident=0), life(CARRIED, end=0, ident=1))  # in order of their first frame
    assert (fade.kind, fade.start, fade.end) == (FADE_OUT, 0, FADE - 1)
    assert (appear.kind, appear.start) == (APPEAR, FADE)
    assert plan(life(0), transition=False)[0].start == 0


def test_a_carried_thing_that_moves_is_transformed_over_its_frames():
    (track,) = plan(life(CARRIED, changes=range(20, 41)))
    assert (track.kind, track.start, track.end) == (MORPH, 20, 40)


def test_a_single_jump_is_smoothed_and_cut_short_by_the_next_change():
    first, second = plan(life(CARRIED, changes=[20, 25]))
    assert (first.kind, first.start, first.end) == (MORPH, 20, 24)
    assert (second.start, second.end) == (25, 25 + REVEAL - 1)


def test_a_thing_that_goes_fades_unless_something_took_its_place():
    gone, cut = plan(life(CARRIED, end=30, ident=0)), plan(life(CARRIED, end=30, replaced=True, ident=1))
    assert (gone[0].kind, gone[0].start, gone[0].end) == (FADE_OUT, 30, 30 + FADE - 1)
    assert (cut[0].kind, cut[0].start) == (REMOVE, 30)


def test_a_reveal_finishes_before_the_thing_goes():
    appear, fade = plan(life(10, end=14))
    assert appear.end == 13 and fade.start == 14


def test_something_gone_before_the_old_picture_faded_is_never_drawn():
    assert plan(life(0, end=3)) == []  # nothing to draw in, and nothing to fade away


def test_live_things_and_held_states_are_cuts():
    live = plan(life(5, changes=[6, 7], end=8, live=True))
    assert [(t.kind, t.start) for t in live] == [(CUT, 5), (CUT, 6), (CUT, 7), (REMOVE, 8)]
    held = plan(life(CARRIED, changes=[0, 12, 30], end=40), cuts_after_start=True)
    assert [(t.kind, t.start) for t in held] == [(MORPH, 0), (CUT, 12), (CUT, 30), (REMOVE, 40)]


def sampled(start, frames, *, ident=0) -> Life:
    """A square sampled every second frame, moving on each sample."""
    states = [(frame, square(frame)) for frame in frames]
    return Life(ident, ("k", ident), start, states)


def test_sampled_motion_is_one_move_from_the_frame_after_the_last_still_sample():
    (track,) = Planner(chapter(sampled(CARRIED, [CARRIED, 20, 22, 24, 26])), step=2).tracks()
    assert (track.kind, track.start, track.end) == (MORPH, 19, 26)


def test_a_sampled_mover_is_drawn_in_on_a_sampled_state_then_moves_on():
    appear, move = Planner(chapter(sampled(10, range(10, 42, 2))), step=2).tracks()
    assert appear.end == 10 + REVEAL - 1 and appear.leaf.anchor()[0] == pytest.approx(20 + 5)  # frame 20's state
    assert (move.kind, move.end) == (MORPH, 40) and move.start > appear.end


def test_a_short_chapter_shortens_its_fades_and_reveals_to_fit():
    short = Planner(chapter(life(0, ident=0), life(CARRIED, end=0, ident=1), frames=6)).tracks()
    fade, appear = short
    assert (fade.start, fade.end) == (0, 1) and appear.start == 2 and appear.end <= 5


def test_every_track_ends_by_the_chapters_last_frame():
    tracks = plan(life(FRAMES - 2), life(CARRIED, changes=[FRAMES - 1], ident=1), life(CARRIED, end=FRAMES - 1, ident=2))
    assert all(track.end <= FRAMES - 1 for track in tracks)


def test_reveal_styles_follow_the_kind_of_thing():
    assert reveal_style(square(fill=None, stroke=(0, 0, 0, 1), width=1.0)) == CREATE
    assert reveal_style(square()) == FADE_IN
    assert reveal_style(text(size=24)) == WRITE and reveal_style(text(size=12)) == FADE_IN


def test_held_states_are_never_blended_even_on_the_first_frames():
    carried = plan(life(CARRIED, changes=[0, 1, 2, 3]), cuts_after_start=True)
    assert [(t.kind, t.start) for t in carried] == [(MORPH, 0), (CUT, 1), (CUT, 2), (CUT, 3)]
    assert carried[0].end == 0  # the first frame's change is smoothed only up to the next real state
    new = plan(life(0, changes=[1, 2, 3, 20]), cuts_after_start=True)
    assert new[0].kind == APPEAR and new[0].leaf.anchor()[0] == pytest.approx(3 + 5)  # as it is when drawn in
    assert [(t.kind, t.start) for t in new[1:]] == [(CUT, 20)]


def test_a_thing_that_changed_while_waiting_is_drawn_in_as_it_is_by_then():
    (appear, cut) = plan(life(0, changes=[3, 30]), cuts_after_start=True)
    assert appear.start == FADE and appear.leaf.anchor()[0] == pytest.approx(3 + 5) and cut.start == 30
    (appear,) = plan(life(0, changes=[3]))
    assert appear.leaf.anchor()[0] == pytest.approx(3 + 5)


def test_things_that_appear_while_the_old_picture_fades_wait_for_it():
    (track,) = plan(life(2))
    assert track.start == FADE


def test_reveals_and_fades_last_as_long_in_seconds_at_any_frame_rate():
    (track,) = Planner(chapter(life(10), frames=120), fps=10).tracks()
    assert track.end - track.start + 1 == round(REVEAL_SECONDS * 10)


def test_a_plan_that_would_not_end_on_the_last_picture_is_refused():
    planner = Planner(chapter(life(CARRIED, changes=[10])))
    bad = [Track(0, MORPH, 5, 70, square())]
    with pytest.raises(PlanError):
        planner._check(planner.chapter.lives[0], bad)
    with pytest.raises(PlanError):
        planner._check(planner.chapter.lives[0], [Track(0, MORPH, 10, 20, square())])  # not its last state


@pytest.mark.parametrize("seed", range(40))
def test_random_lives_get_ordered_tracks_inside_the_chapter_ending_on_their_last_state(seed):
    rng = np.random.default_rng(seed)
    lives = []
    for ident in range(12):
        start = int(rng.choice([CARRIED, 0, *rng.integers(0, FRAMES - 1, 3)]))
        first = max(start, 0)
        frames = sorted(set(int(f) for f in rng.integers(first + 1, FRAMES, rng.integers(0, 8)) if f > first))
        if rng.random() < 0.5 and frames:  # a run of moves on consecutive frames
            frames = sorted(set(frames) | set(range(frames[0], min(frames[0] + 6, FRAMES))))
        end = int(rng.integers(frames[-1] + 1 if frames else first + 1, FRAMES + 1)) if rng.random() < 0.4 else None
        end = None if end is not None and end >= FRAMES else end
        states = [(start, square(0))] + [(frame, square(frame % 40, fill=(0.1, 0.2, 0.3, 1.0))) for frame in frames]
        lives.append(Life(ident, ("k", ident), start, states, end, bool(rng.random() < 0.3), bool(rng.random() < 0.1)))
    for options in ({}, {"cuts_after_start": True}, {"step": 2}):
        tracks = Planner(chapter(*lives), **options).tracks()  # refuses any bad plan itself
        assert all(0 <= t.start <= t.end < FRAMES for t in tracks)
