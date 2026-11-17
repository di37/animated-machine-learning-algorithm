"""Every chapter of every house-price Manim lesson ends on animkit's picture, frame for frame as long.

The comparison is ``manimkit.parity``'s: animkit's own players are the reference. After each chapter
nothing on screen may still be moving or differ from its last recorded state, and the last frames must
match in every tile; the module docstring there gives the limits and why.
"""

import shutil

import pytest

from animkit.lessons.encoder import probe
from animkit.lessons.stage import Stage, lesson_style
from animkit.lessons.timing import CHAPTERS, Timing
from house_anim.lessons.regression_scene import REGRESSION
from house_anim.lessons.stories import build_lesson, lesson_keys
from manimkit import parity
from manimkit.director import lesson_frames, plans, prepare_extras

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

TIMING = Timing(fps=10, scale=0.12)
DPI = 50.0  # 1280 x 720


@pytest.fixture(scope="module", params=lesson_keys())
def lesson(request, income_data):
    return build_lesson(request.param, income_data)


@pytest.fixture(scope="module")
def extras(lesson):
    return prepare_extras(lesson, REGRESSION)


@pytest.fixture(scope="module")
def animkit_run(lesson, extras):
    return parity.animkit_run(lesson, REGRESSION, extras, TIMING, DPI)


@pytest.fixture(scope="module")
def manim_run(lesson, extras, tmp_path_factory):
    path = tmp_path_factory.mktemp("manim") / f"{lesson.story.key}_lesson.mp4"
    return parity.manim_run(lesson, REGRESSION, extras, TIMING, DPI, path), path


def test_the_video_is_frame_for_frame_as_long_as_animkits(manim_run, animkit_run, lesson):
    run, path = manim_run
    info = probe(path)
    assert info.frames == run.frames == animkit_run.frames == lesson_frames(lesson, TIMING)
    assert (info.width, info.height) == (1280, 720)


def test_moving_chapters_are_sampled_on_the_frames_animkit_paints_them(lesson, extras, animkit_run):
    with lesson_style():
        ours = {plan.name: [number for number, _ in plan.samples]
                for plan in plans(Stage(dpi=10), lesson, TIMING, REGRESSION, extras)}
    animated = [name for name in CHAPTERS if animkit_run.painted[name]]  # training and the final model are held
    assert len(animated) == 8
    for name in animated:
        assert ours[name] == animkit_run.painted[name], name


def test_every_chapter_ends_settled_on_its_last_recorded_picture(manim_run):
    assert {name: problems for name, problems in manim_run[0].unsettled.items() if problems} == {}


def test_every_chapter_ends_on_animkits_picture(manim_run, animkit_run):
    failures = [f"{chapter}, {what}: {difference:.2f} >= {limit}" for chapter in CHAPTERS
                for what, difference, limit in parity.chapter_differences(chapter, manim_run[0].ends[chapter],
                                                                          animkit_run.ends[chapter])
                if difference >= limit]
    assert failures == []


def test_a_wrong_picture_is_far_above_the_limits(manim_run, animkit_run):
    ends, expected = manim_run[0].ends, animkit_run.ends
    assert parity.worst_tile(ends["training"], expected["inference"]) > 3 * parity.MOST_TILE_DIFFERENCE
    assert parity.worst_tile(ends["takeaway"], expected["title"]) > 3 * parity.CARD_TILE_DIFFERENCE
    rows, columns = parity.box_3d(ends["surface"])
    assert parity.worst_tile(ends["surface"][rows, columns], expected["exam"][rows, columns]) > \
        3 * parity.SHEET_TILE_DIFFERENCE
