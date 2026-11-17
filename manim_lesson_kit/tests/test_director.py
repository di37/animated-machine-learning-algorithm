"""The chapters are animkit's, sampled on the frames animkit paints them, with the 3-D view live."""

from types import SimpleNamespace

import pytest

from animkit.lessons.director import ANIMATION_STEP, Extras
from animkit.lessons.stage import Stage, lesson_style
from animkit.lessons.timing import CHAPTERS, Timing, allocate_frames
from manimkit import director
from manimkit.recorder import record

TIMING = Timing(fps=10, scale=0.2)


def fake_lesson(snapshots: int = 5):
    story = SimpleNamespace(training="training words", initial="first guess", final="final words")
    return SimpleNamespace(story=story, snapshots=[object()] * snapshots, final_index=snapshots - 1,
                           frame_weights=tuple(1.0 / (1 + i) for i in range(snapshots)))


@pytest.fixture
def chapter_plans():
    with lesson_style():
        stage = Stage(dpi=20)
        yield director.plans(stage, fake_lesson(), TIMING, scene=None, extras=Extras(None, None, None))


def test_the_ten_chapters_come_in_animkits_order_at_animkits_lengths(chapter_plans):
    assert [plan.name for plan in chapter_plans] == list(CHAPTERS)
    lengths = [TIMING.frames(name) if name != "training" else sum(director.training_holds(fake_lesson(), TIMING))
               for name in CHAPTERS]
    assert [plan.frames for plan in chapter_plans] == lengths
    assert sum(lengths) == director.lesson_frames(fake_lesson(), TIMING)


def test_a_moving_chapter_is_sampled_where_animkit_renders_it_and_where_its_hold_begins(chapter_plans):
    data = chapter_plans[1]
    motion = max(1, round(data.frames * 0.7))
    numbers = [number for number, _ in data.samples]
    assert numbers == [*range(0, motion, ANIMATION_STEP), motion] and data.step == ANIMATION_STEP


def test_animated_chapters_paint_the_t_animkit_paints_on_each_sampled_frame():
    seen = []
    for _, paint in director.animated("x", 20, 0.5, seen.append).samples:  # ten moving frames, then a hold
        paint()
    assert seen == [frame / 9 for frame in (0, 2, 4, 6, 8)] + [1.0]  # animkit: 0, 2, 4, 6, 8, then t = 1
    seen.clear()
    for _, paint in director.animated("y", 6, 0.2, seen.append).samples:  # one moving frame: t = 0, then t = 1
        paint()
    assert seen == [0.0, 1.0]
    seen.clear()
    for _, paint in director.animated("z", 10, 1.0, seen.append).samples:  # all moving: no closing t = 1
        paint()
    assert seen == [frame / 9 for frame in (0, 2, 4, 6, 8)]
    assert [n for n, _ in director.animated("w", 20, 0.5, seen.append, step=1).samples] == [*range(10), 10]


def test_training_paints_each_snapshot_on_the_frame_its_hold_starts(chapter_plans):
    training = next(plan for plan in chapter_plans if plan.name == "training")
    holds = allocate_frames(fake_lesson().frame_weights, training.frames)
    starts = [sum(holds[:i]) for i in range(len(holds))]
    assert [number for number, _ in training.samples] == starts and training.cuts_after_start


def test_only_the_three_d_chapter_is_live(chapter_plans):
    assert [plan.name for plan in chapter_plans if plan.live_3d] == ["surface"]
    assert director._live_3d((("figure", 0), ("axes", "main3d"), ("patch", 0), 0))
    assert not director._live_3d((("figure", 0), ("axes", "main"), 0))


def test_the_3d_boxs_own_groups_are_live_too():
    """mplot3d draws the box's panes, grid, ticks and labels outside the 3-D axes' group."""
    with lesson_style():
        stage = Stage(dpi=20)
        ax = stage.main_3d()
        ax.plot([0, 1], [0, 1], [0, 1])
        director.name_regions(stage)
        leaves = record(stage.fig)[0]
    box = [leaf for leaf in leaves if any(group[0] in director.LIVE_3D_GROUPS for group in leaf.key[:-1])]
    assert box and all(director._live_3d(leaf.key) for leaf in box)


def test_unknown_chapters_are_refused_before_anything_is_drawn(tmp_path):
    with pytest.raises(ValueError, match="unknown chapters"):
        director.render_lesson(fake_lesson(), tmp_path / "x.mp4", scene=None, chapters=("intro",))


def test_regions_are_named_so_their_keys_survive_switching_views():
    with lesson_style():
        stage = Stage(dpi=20)
        stage.main_3d()
        director.name_regions(stage)
        assert stage.main.get_gid() == "main" and stage._main_3d.get_gid() == "main3d"
        assert all(text.get_gid() for text in stage.fig.texts)
        stage.show_card()
        keys = {leaf.key[1] for leaf in record(stage.fig)[0] if len(leaf.key) > 2}
        assert ("axes", "card") in keys
