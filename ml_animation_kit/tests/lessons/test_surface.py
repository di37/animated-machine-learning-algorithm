"""The 3-D chapter: a fitted model's output drawn as height over two inputs, seen by a swinging camera."""

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection

from animkit.lessons import surface as view
from animkit.lessons import theme
from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.director import paint_surface
from animkit.lessons.fit import colliding_texts, overflowing_texts
from animkit.lessons.lesson import LessonError, Story
from animkit.lessons.stage import MAIN_RECT, TRACKER, Stage, lesson_style

TINY_DPI = RESOLUTIONS["preview"]
X, Y = np.linspace(0.0, 10.0, 6), np.linspace(0.0, 5.0, 4)


def sheet(offset: float = 0.0, label: str = "") -> view.Sheet:
    heights = np.add.outer(Y, X) / 2.0 + offset  # (4, 6), from 0 to 7.5 + offset
    return view.Sheet(heights, np.ones((*heights.shape, 4)), label)


def surface(**changes) -> view.Surface:
    values = dict(
        x=X, y=Y,
        x_axis=view.Axis("first input", (0.0, 10.0), (0.0, 5.0, 10.0), ("0", "5", "10")),
        y_axis=view.Axis("second input", (0.0, 5.0), (0.0, 5.0), ("0", "5")),
        z_axis=view.Axis("output", (0.0, 10.0), (0.0, 10.0), ("0", "10")),
        sheets=(sheet(),),
        dots=view.Dots(np.array([1.0, 2.0]), np.array([1.0, 3.0]), np.array([3.0, 1.0]), (theme.NAVY, theme.NAVY),
                       fitted=np.array([1.0, 2.5])),
        query=view.Marker(5.0, 2.0, 3.5, "new example: 3.5"),
        title="Two inputs, one output", caption="The same algorithm, given two inputs.",
        reading=("Height is the model's output.", "Dots are training rows."),
        scores=("Test error with one input: 4", "Test error with two inputs: 3"))
    return view.Surface(**{**values, **changes})


class FakeLesson:
    story = Story(key="demo", title="Demo", method="m", objective="o", initial="i", training="t", final="f",
                  inference="n", takeaway="k", limitation="l", progress_label="p")


class FakeScene:
    subtitle = "A demo task"


def painted(t: float = 0.5, shown: view.Surface | None = None) -> Stage:
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_surface(stage, FakeLesson(), shown or surface(), t, FakeScene())
        stage.render()
    return stage


def test_the_camera_swings_out_and_comes_back_to_where_it_started():
    angles = [view.camera(t) for t in np.linspace(0.0, 1.0, 41)]
    assert view.camera(0.0) == pytest.approx((view.ELEVATION, view.AZIMUTH))
    assert view.camera(1.0) == pytest.approx(view.camera(0.0))
    assert {elevation for elevation, _ in angles} == {view.ELEVATION}
    swing = max(abs(azimuth - view.AZIMUTH) for _, azimuth in angles)
    assert swing == pytest.approx(view.SWING_DEGREES)


def test_sheets_and_the_plane_are_one_depth_sorted_set_of_tiles():
    """One collection, so matplotlib orders every tile by depth: the plane shows where a sheet crosses it,
    and neither sheet is always painted over the other."""
    stage = painted(shown=surface(sheets=(sheet(0.0, "first"), sheet(2.0, "second")), plane=5.0,
                                  plane_label="threshold"))
    ax = stage.main_3d()
    assert isinstance(ax, Axes3D)
    tiles = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    assert len(tiles) == 1 and len(tiles[0].get_paths()) == 3 * (len(Y) - 1) * (len(X) - 1)
    labels = {text.get_text().strip() for text in ax.texts}
    assert {"first", "second", "threshold", "new example: 3.5"} <= labels


def test_nothing_in_the_box_is_clipped_to_the_square_3d_frame():
    stage = painted()
    ax = stage.main_3d()
    assert all(not artist.get_clip_on() for artist in (*ax.collections, *ax.get_lines()))


def test_the_sheet_spreads_past_the_square_3d_frame():
    red = np.zeros((len(Y), len(X), 4))
    red[..., 0] = red[..., 3] = 1.0
    flat = view.Sheet(np.full((len(Y), len(X)), 0.5), red)
    with lesson_style():
        stage = Stage(dpi=50.0)
        paint_surface(stage, FakeLesson(), surface(sheets=(flat,)), 0.0, FakeScene())
        stage.render()
    pixels = np.asarray(stage.fig.canvas.buffer_rgba())[..., :3].astype(int)
    left = int(stage.main_3d().bbox.x0)
    reddish = (pixels[..., 0] > 200) & (pixels[..., 1] < 140) & (pixels[..., 2] < 140)
    assert reddish[:, :left - 2].any()


def test_dots_under_the_sheet_are_painted_before_it_and_dots_above_after():
    ax = painted().main_3d()
    tiles = next(c for c in ax.collections if isinstance(c, Poly3DCollection))
    groups = {c.zorder < tiles.zorder: c for c in ax.collections
              if isinstance(c, Path3DCollection) and len(c._offsets3d[0]) and c.get_sizes()[0] == view.DOT_SIZE}  # noqa: SLF001
    under, over = (float(np.ravel(groups[side]._offsets3d[2])[0]) for side in (True, False))  # noqa: SLF001
    assert (under, over) == (1.0, 3.0)  # the dot at (2, 3) is below the sheet there, the one at (1, 1) above


def test_the_unseen_example_sits_at_its_own_height_and_dots_get_error_sticks():
    ax = painted().main_3d()
    points = [c for c in ax.collections if isinstance(c, Path3DCollection)]
    query = next(c for c in points if tuple(c.get_facecolor()[0]) == to_rgba(theme.UNSEEN))
    assert [float(np.ravel(axis)[0]) for axis in query._offsets3d] == [5.0, 2.0, 3.5]  # noqa: SLF001
    sticks = [line for line in ax.get_lines() if line.get_color() == theme.ERROR]
    assert len(sticks) == 2


def test_only_the_first_dots_get_sticks_when_fewer_fitted_heights_are_given():
    dots = view.Dots(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]),
                     (theme.NAVY,) * 3, fitted=np.array([2.0]))
    ax = painted(shown=surface(dots=dots)).main_3d()
    assert len([line for line in ax.get_lines() if line.get_color() == theme.ERROR]) == 1


def test_the_view_turns_with_time():
    ax = painted(0.25).main_3d()
    assert (ax.elev, ax.azim) == pytest.approx(view.camera(0.25))


def test_the_chapter_has_its_own_tracker_label_and_header():
    stage = painted()
    assert dict(TRACKER)["surface"] == "3-D view"
    texts = {text.get_text(): text for text in stage.fig.texts}
    assert texts["3-D view"].get_color() == theme.HEADING and texts["Two inputs, one output"].get_visible()
    assert "The same algorithm, given two inputs." in "\n".join(t.get_text() for t in stage.fig.findobj(Text))


def test_side_notes_fit_their_boxes_even_when_long():
    long = tuple("A long explanation that needs several lines to say what the picture shows. " * 2
                 for _ in range(5))
    stage = painted(1.0, surface(reading=long, scores=long[:3]))
    for ax in (stage.panel, stage.metric):
        assert overflowing_texts(ax) == () and colliding_texts(ax) == ()


def test_side_notes_fit_even_after_a_chapter_drew_a_map_with_equal_aspect_in_the_panel():
    """The exam chapter draws a map with equal aspect in the side panel; the notes must not inherit it."""
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        stage.panel.set_aspect("equal", adjustable="datalim")
        paint_surface(stage, FakeLesson(), surface(), 1.0, FakeScene())
        stage.render()
    assert stage.panel.get_xlim() == pytest.approx((0.0, 1.0))
    assert overflowing_texts(stage.panel) == ()


def test_notes_appear_one_after_another():
    early = [text.get_alpha() for text in painted(0.0).panel.texts[1:]]
    assert early[0] < 1.0 and early[-1] == pytest.approx(0.0)
    assert all(text.get_alpha() == pytest.approx(1.0) for text in painted(1.0).panel.texts[1:])


def test_the_3d_axes_fills_the_main_plot_and_hides_for_every_other_chapter():
    stage = painted()
    ax = stage.main_3d()
    assert ax.get_position(original=True).bounds == pytest.approx(MAIN_RECT)  # before the box aspect squeezes it
    assert ax.get_visible() and not stage.main.get_visible() and not stage.diagram.get_visible()
    stage.show_workspace()
    stage.main_facets(2)
    assert not ax.get_visible()
    stage.main_3d()
    stage.show_card()
    assert not ax.get_visible()


@pytest.mark.parametrize("t", [0.25, 0.75])  # the two ends of the swing
def test_the_box_and_its_labels_stay_on_the_main_panel_at_every_angle(t):
    from animkit.lessons import stage as layout

    dpi = 50.0
    long_ticks = view.Axis("first input", (0.0, 10.0), (0.0, 5.0, 10.0), ("$0k", "$50k", "$100k"))
    with lesson_style():
        stage = Stage(dpi=dpi)
        paint_surface(stage, FakeLesson(), surface(x_axis=long_ticks), t, FakeScene())
        stage.render()
    pixels = np.asarray(stage.fig.canvas.buffer_rgba())[..., :3].astype(int)
    top = round((layout.HEIGHT_IN - layout.WORK_BOTTOM_IN) * dpi) + 2  # just below the panel's bottom edge
    bottom = round((layout.HEIGHT_IN - layout.CAPTION_BOTTOM_IN - layout.CAPTION_HEIGHT_IN) * dpi) - 2
    left, right = (round(edge * dpi) for edge in (layout.MAIN_LEFT_IN, layout.MAIN_LEFT_IN + layout.MAIN_WIDTH_IN))
    canvas = np.array([int(theme.CANVAS[i:i + 2], 16) for i in (1, 3, 5)])
    assert not (np.abs(pixels[top:bottom, left:right] - canvas).sum(axis=2) > 60).any()


def test_the_same_3d_axes_is_reused_from_frame_to_frame():
    stage = painted()
    assert stage.main_3d() is stage.main_3d()


@pytest.mark.parametrize("change", [
    dict(sheets=()),
    dict(sheets=(view.Sheet(np.zeros((3, 6)), np.ones((3, 6, 4))),)),  # wrong number of rows
    dict(sheets=(sheet(5.0),)),  # pokes out of the top of the box
    dict(sheets=(view.Sheet(np.full((4, 6), np.nan), np.ones((4, 6, 4))),)),
    dict(query=view.Marker(5.0, 2.0, 11.0, "too high")),
    dict(dots=view.Dots(np.array([11.0]), np.array([1.0]), np.array([1.0]), (theme.NAVY,))),
    dict(plane=-1.0),
])
def test_anything_that_would_draw_outside_the_box_is_refused(change):
    with pytest.raises(ValueError):
        surface(**change)


def test_a_missing_value_is_refused_as_a_lesson_error():
    with pytest.raises(LessonError, match="finite"):
        surface(sheets=(view.Sheet(np.full((4, 6), np.nan), np.ones((4, 6, 4))),))


@pytest.mark.parametrize("part", ["reading", "scores"])
def test_too_many_notes_are_refused_when_the_surface_is_built_before_any_frame_is_encoded(part):
    with pytest.raises(ValueError, match="notes"):
        surface(**{part: ("a note",) * 40})


def test_a_dot_below_any_sheet_is_painted_under_the_tiles():
    upper = sheet(2.0, "upper")
    dots = view.Dots(np.array([2.0]), np.array([0.0]), np.array([1.0]), (theme.NAVY,))  # on the lower sheet
    ax = painted(shown=surface(sheets=(sheet(0.0, "lower"), upper), dots=dots)).main_3d()
    tiles = next(c for c in ax.collections if isinstance(c, Poly3DCollection))
    dot = next(c for c in ax.collections if isinstance(c, Path3DCollection) and c.get_sizes()[0] == view.DOT_SIZE)
    assert dot.zorder < tiles.zorder


def test_the_threshold_label_has_a_background_box():
    ax = painted(shown=surface(plane=5.0, plane_label="threshold")).main_3d()
    label = next(text for text in ax.texts if text.get_text().strip() == "threshold")
    assert label.get_bbox_patch() is not None


def test_parts_must_agree_in_length():
    with pytest.raises(ValueError):
        view.Sheet(np.zeros((4, 6)), np.ones((4, 5, 4)))
    with pytest.raises(ValueError):
        view.Dots(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 1.0]), ("a", "b"))
    with pytest.raises(ValueError):
        view.Dots(np.array([1.0]), np.array([1.0]), np.array([1.0]), ("a",), fitted=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        view.Axis("x", (0.0, 1.0), (0.0, 1.0), ("0",))
    with pytest.raises(ValueError):
        view.Axis("x", (1.0, 0.0), (), ())


def test_arrays_are_read_only_copies():
    heights = np.zeros((4, 6))
    shown = view.Sheet(heights, np.ones((4, 6, 4)))
    heights[0, 0] = 9.0
    assert shown.heights[0, 0] == 0.0 and not shown.heights.flags.writeable
