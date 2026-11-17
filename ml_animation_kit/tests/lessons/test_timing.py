import pytest

from animkit.lessons.config import CHAPTER_SECONDS
from animkit.lessons.timing import CHAPTERS, MIN_CHAPTER_FRAMES, Timing, allocate_frames, ease, phase


def test_allocate_frames_sums_to_total_and_follows_weights():
    frames = allocate_frames((3.0, 1.0, 1.0, 1.0), 60)
    assert frames == (30, 10, 10, 10)


def test_allocate_frames_rounds_by_largest_remainder():
    frames = allocate_frames((1.0, 1.0, 1.0), 10)
    assert sum(frames) == 10 and sorted(frames) == [3, 3, 4]


def test_allocate_frames_gives_every_snapshot_at_least_one_frame():
    assert allocate_frames((100.0, 1.0, 1.0), 3) == (1, 1, 1)
    assert allocate_frames((1.0,) * 5, 2) == (1, 1, 1, 1, 1)
    assert min(allocate_frames((1000.0, 1.0), 20)) >= 1


@pytest.mark.parametrize("weights", [(), (1.0, -1.0), (0.0, 0.0), (1.0, float("nan"))])
def test_allocate_frames_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        allocate_frames(weights, 10)


def test_timing_turns_chapter_seconds_into_frames():
    timing = Timing(fps=30)
    assert timing.frames("training") == 30 * CHAPTER_SECONDS["training"]
    assert timing.total_frames == sum(timing.frames(chapter) for chapter in CHAPTERS)
    assert set(CHAPTERS) == set(CHAPTER_SECONDS)


def test_scaled_timing_keeps_a_floor_per_chapter():
    timing = Timing(fps=6, scale=0.01)
    assert all(timing.frames(chapter) == MIN_CHAPTER_FRAMES for chapter in CHAPTERS)


@pytest.mark.parametrize("fps, scale", [(0, 1.0), (30, 0.0), (30, -2.0)])
def test_timing_rejects_bad_values(fps, scale):
    with pytest.raises(ValueError):
        Timing(fps=fps, scale=scale)


def test_timing_rejects_unknown_chapters():
    with pytest.raises(KeyError):
        Timing().frames("credits")


def test_ease_is_clamped_smoothstep():
    assert (ease(-1.0), ease(0.0), ease(0.5), ease(1.0), ease(3.0)) == (0.0, 0.0, 0.5, 1.0, 1.0)
    assert ease(0.25) < 0.25 < 0.75 < ease(0.75)


def test_phase_maps_a_window_onto_zero_to_one():
    assert phase(0.1, 0.25, 0.75) == 0.0
    assert phase(0.5, 0.25, 0.75) == 0.5
    assert phase(0.9, 0.25, 0.75) == 1.0
    with pytest.raises(ValueError):
        phase(0.5, 0.6, 0.6)
