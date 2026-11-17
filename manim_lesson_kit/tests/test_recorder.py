"""What a figure draws, recorded as leaves, and drawn by Manim as Agg draws it."""

import numpy as np
import pytest
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageFilter

from manimkit.leaves import ImageLeaf, PathLeaf, TextLeaf
from manimkit.recorder import record
from tests.conftest import agg_pixels, figure

BLUR = 2
# Mean grey levels per pixel the Manim frame may differ from Agg's, after a blur: the plotted figure comes
# to 1.01; without its one note it is 1.39, without its 60 scatter dots 2.19.
TOLERANCE = 1.2


def manim_pixels(player, leaves, frame) -> np.ndarray:
    player.scene.mobjects = [leaf.build(frame) for leaf in leaves]
    player.scene.renderer.update_frame(player.scene)
    return player.scene.renderer.get_frame()[..., :3].astype(float)


def blurred(pixels: np.ndarray) -> np.ndarray:
    image = Image.fromarray(pixels.astype(np.uint8)).filter(ImageFilter.GaussianBlur(BLUR))
    return np.asarray(image, dtype=float)


def plotted():
    """A figure with every kind of thing a lesson draws: a plot, scatter, text, patches, dashes, an image."""
    fig = figure()
    ax = fig.add_axes((0.15, 0.2, 0.55, 0.7))
    rng = np.random.default_rng(0)
    ax.scatter(rng.uniform(0, 10, 60), rng.uniform(0, 10, 60), s=12, color="#2a78d6", alpha=0.6, linewidths=0)
    ax.scatter([2, 5, 8], [8, 2, 5], s=[30, 60, 90], c=["#e34948", "#0b8f9c", "#eb6834"])
    ax.plot([0, 12], [1, 14], color="#0f2541", linewidth=2.0)  # runs past the axes: clipped
    ax.axhline(5, color="#6b7f95", linestyle=(0, (3, 3)), linewidth=1.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("income")
    ax.set_ylabel("price")
    ax.text(1, 9, "note", bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    side = fig.add_axes((0.75, 0.2, 0.2, 0.7))
    side.set_axis_off()
    side.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.3, boxstyle="round,pad=0.02", facecolor="#cde2fb"))
    side.add_patch(Rectangle((0.1, 0.6), 0.6, 0.1, facecolor="#2a78d6"))
    side.imshow(rng.uniform(0, 1, (6, 6)), extent=(0.1, 0.9, 0.75, 0.95), cmap="Blues")
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    return fig


def test_every_kind_of_leaf_is_recorded():
    leaves, frame = record(plotted())
    kinds = {type(leaf) for leaf in leaves}
    assert kinds == {PathLeaf, TextLeaf, ImageLeaf}
    assert frame.size_px == (160, 90)
    dashed = [leaf for leaf in leaves if isinstance(leaf, PathLeaf) and leaf.dashes]
    assert dashed and dashed[0].dashes == (3.0, 3.0)


def test_manim_draws_the_recorded_frame_as_agg_draws_it(player):
    fig = plotted()
    expected = agg_pixels(fig)
    leaves, frame = record(fig)
    drawn = manim_pixels(player, leaves, frame)
    assert np.abs(blurred(drawn) - blurred(expected)).mean() < TOLERANCE


def test_a_second_draw_records_the_same_leaves_under_the_same_keys():
    fig = plotted()
    first, _ = record(fig)
    second, _ = record(fig)
    assert [leaf.key for leaf in first] == [leaf.key for leaf in second]
    assert [leaf.signature for leaf in first] == [leaf.signature for leaf in second]


def test_keys_follow_gids_so_a_hidden_axes_does_not_renumber_the_rest():
    fig = figure()
    a, b = fig.add_axes((0.1, 0.1, 0.3, 0.8)), fig.add_axes((0.6, 0.1, 0.3, 0.8))
    a.set_gid("left")
    b.set_gid("right")
    b.plot([0, 1], [0, 1])
    shown, _ = record(fig)
    a.set_visible(False)
    hidden, _ = record(fig)
    def line(leaves):
        return [leaf.key for leaf in leaves if ("axes", "right") in leaf.key and ("line2d", 0) in leaf.key]

    assert line(shown) == line(hidden) and line(shown)


def test_scatter_items_outside_the_axes_are_dropped_and_straddling_ones_keep_their_clip_box():
    fig = figure()
    ax = fig.add_axes((0.2, 0.2, 0.6, 0.6))
    ax.scatter([0.5, 5.0, 1.0], [0.5, 0.5, 0.99], s=[20, 20, 400], c=["r", "g", "b"])  # middle one off the axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    dots = [leaf for leaf in record(fig)[0] if isinstance(leaf, PathLeaf) and ("PathCollection", 0) in leaf.key]
    assert len(dots) == 2
    assert dots[0].clip is None and dots[1].clip is not None  # the big corner dot is clipped at the axes edge


def test_uniform_markers_are_one_leaf_each_with_their_own_keys():
    fig = figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.plot([0.2, 0.5, 0.8], [0.5, 0.5, 0.5], "o", color="#0f2541")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    markers = [leaf for leaf in record(fig)[0] if isinstance(leaf, PathLeaf) and leaf.key[2] == ("line2d", 0)
               and len(leaf.key) == 4]  # the plotted line itself, not a tick's
    assert len(markers) == 3 and len({leaf.key for leaf in markers}) == 3
    centres = sorted(leaf.anchor()[0] for leaf in markers)
    assert centres[0] < centres[1] < centres[2]


def test_a_line_along_the_axes_edge_is_clipped_line_width_and_all(player):
    fig = figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.6))
    ax.set_axis_off()
    ax.plot([0, 1], [0, 0], color="#2a78d6", linewidth=6)  # half of it lies below the axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    expected = agg_pixels(fig)
    leaves, frame = record(fig)
    line = next(leaf for leaf in leaves if isinstance(leaf, PathLeaf) and leaf.stroke)
    assert line.clip is not None
    drawn = manim_pixels(player, leaves, frame)
    assert np.abs(drawn - expected).mean() < 1.0
    assert drawn[int(90 * 0.8) + 3, 80].mean() > 250  # nothing below the axes' bottom edge


def centroid(pixels: np.ndarray) -> tuple[float, float]:
    ink = 255 - pixels.mean(axis=2)
    rows, columns = np.indices(ink.shape)
    return float((columns * ink).sum() / ink.sum()), float((rows * ink).sum() / ink.sum())


@pytest.mark.parametrize("offset", [0.0, 0.3, 0.5, 0.7])
def test_markers_and_thin_lines_land_on_the_pixels_agg_puts_them_on(player, offset):
    fig = figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.plot([40 + offset], [45 + offset], "o", markersize=5, color="black", markeredgewidth=0)
    ax.axhline(20 + offset, color="black", linewidth=1.44)  # one pixel at 50 dpi, snapped by Agg
    expected = agg_pixels(fig)
    leaves, frame = record(fig)
    drawn = manim_pixels(player, leaves, frame)
    dot = (slice(30, 60), slice(25, 55))
    assert np.allclose(centroid(drawn[dot]), centroid(expected[dot]), atol=0.15)
    line = (slice(55, 80), slice(0, 160))
    assert np.argmin(drawn[line].mean(axis=(1, 2))) == np.argmin(expected[line].mean(axis=(1, 2)))


def test_a_line_and_its_markers_never_share_a_key():
    fig = figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.plot([0.2, 0.5, 0.8], [0.5, 0.6, 0.5], "o-")
    keys = [leaf.key for leaf in record(fig)[0]]
    assert len(keys) == len(set(keys))


def test_a_collection_redrawn_on_new_axis_limits_is_a_new_drawing():
    from matplotlib.collections import PolyCollection
    fig = figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.add_collection(PolyCollection([[(0, 0), (1, 0), (1, 1)]], facecolors="b"))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    before = [leaf.signature for leaf in record(fig)[0] if ("PolyCollection", 0) in leaf.key]
    ax.set_xlim(0, 4)  # the first corner stays put on screen; the others move
    after = [leaf.signature for leaf in record(fig)[0] if ("PolyCollection", 0) in leaf.key]
    assert before and before != after


def test_the_frame_is_the_size_matplotlib_gives_the_canvas():
    fig = figure(25.6, 14.4, 96)
    assert record(fig)[1].size_px == fig.canvas.get_width_height() == (2457, 1382)


def test_a_clip_path_that_is_not_a_box_is_drawn_unclipped_with_a_warning():
    from matplotlib.patches import Circle
    from manimkit import recorder
    recorder._WARNED.clear()
    fig = figure()
    ax = fig.add_axes((0, 0, 1, 1))
    line, = ax.plot([0, 1], [0, 1])
    line.set_clip_path(Circle((0.5, 0.5), 0.2, transform=ax.transAxes))
    with pytest.warns(RuntimeWarning, match="clip path"):
        record(fig)


@pytest.mark.parametrize("alpha", [None, 0.5])
@pytest.mark.parametrize("colours", ["mapped", "rgba"])
def test_a_translucent_image_is_as_see_through_as_in_agg(player, alpha, colours):
    fig = figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    data = np.ones((4, 8)) if colours == "mapped" else np.tile([0.0, 0.0, 1.0, 0.6], (4, 8, 1))
    ax.imshow(data, cmap="Blues", vmin=0, vmax=1, alpha=alpha, aspect="auto", extent=(0, 1, 0, 1))
    expected = agg_pixels(fig)
    leaves, frame = record(fig)
    assert np.abs(manim_pixels(player, leaves, frame)[45, 80] - expected[45, 80]).max() <= 2


@pytest.mark.parametrize("origin", ["upper", "lower"])
def test_an_image_is_the_right_way_up(player, origin):
    fig = figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.imshow(np.linspace(0, 1, 8).reshape(8, 1) * np.ones((1, 4)), cmap="gray", origin=origin, aspect="auto",
              extent=(0, 1, 0, 1))
    expected = agg_pixels(fig)[:, 80, 0]
    leaves, frame = record(fig)
    drawn = manim_pixels(player, leaves, frame)[:, 80, 0]
    assert np.abs(drawn - expected).max() <= 2  # dark where Agg is dark, light where it is light


def test_hard_edged_collections_are_recorded_as_such():
    from matplotlib.collections import PolyCollection
    fig = figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.add_collection(PolyCollection([[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]], antialiased=False, facecolors="b"))
    tiles = [leaf for leaf in record(fig)[0] if isinstance(leaf, PathLeaf) and ("PolyCollection", 0) in leaf.key]
    assert tiles and not tiles[0].antialiased


def test_text_lands_on_the_pixel_agg_rounds_it_to():
    fig = figure(6, 2, 100)
    fig.text(0.1013, 0.5011, "Regression pg", fontsize=34, weight="bold")
    fig.text(0.1013, 0.2, "turned", rotation=90)
    straight, turned = [leaf for leaf in record(fig)[0] if isinstance(leaf, TextLeaf)]
    x, y = straight.origin
    assert x == round(x) and abs(y - 100.21) < 1.0 and (y * 2) == round(y * 2)  # whole or half pixels, as Agg
    assert turned.angle == 90 and turned.origin[0] != round(turned.origin[0])  # rotated text: left as laid out


@pytest.mark.parametrize("alpha", [0.0, 0.001])
def test_invisible_things_are_not_recorded(alpha):
    fig = figure()
    fig.text(0.5, 0.5, "gone", alpha=alpha)
    assert not [leaf for leaf in record(fig)[0] if isinstance(leaf, TextLeaf)]
