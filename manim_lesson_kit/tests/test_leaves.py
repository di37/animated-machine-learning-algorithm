"""Leaves: equal when they draw the same, matching when one can turn into the other, and their shapes."""

import numpy as np
import pytest
from manim import DashedVMobject, ImageMobject, VMobject
from matplotlib.font_manager import FontProperties

from manimkit.camera import ALIASED, CLIP
from manimkit.leaves import ImageLeaf, PathLeaf, TextLeaf, displacement_px
from manimkit.paths import CLOSEPOLY, LINETO, MOVETO, STOP

BLUE = (0.16, 0.47, 0.84, 1.0)
RED = (0.89, 0.29, 0.28, 0.8)


def square(x: float = 0.0, y: float = 0.0, *, fill=BLUE, stroke=None, width=0.0, dashes=None, antialiased=True,
           key=("k", 0)) -> PathLeaf:
    vertices = np.array([[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10], [x, y], [0, 0]], dtype=np.float32)
    codes = np.array([MOVETO, LINETO, LINETO, LINETO, CLOSEPOLY, STOP], dtype=np.uint8)
    return PathLeaf(key, vertices, codes, fill=fill, stroke=stroke, linewidth=width, dashes=dashes, cap="butt",
                    join="miter", antialiased=antialiased)


def text(string: str = "price", x: float = 10.0, color=BLUE, size: float = 13) -> TextLeaf:
    return TextLeaf(("t", 0), string, FontProperties(family=["Arial", "DejaVu Sans"], size=size), (x, 20.0), 0.0,
                    color, 40.0)


def test_equal_drawing_means_equal_signatures():
    assert square().signature == square().signature
    assert square().signature != square(1).signature
    assert square().signature != square(fill=RED).signature
    assert text().signature == text().signature != text("value").signature


def test_a_moved_or_recoloured_shape_matches_but_a_different_one_does_not():
    assert square().matches(square(5, 5, fill=RED))
    assert not square().matches(square(stroke=BLUE, width=1.0))  # gains a stroke
    assert not square().matches(text())
    assert text().matches(text(x=40.0, color=RED)) and not text().matches(text("other"))
    assert not text().matches(text(size=20))


def test_displacement_is_the_furthest_any_vertex_moved():
    assert displacement_px(square(), square(3, 4)) == pytest.approx(4.0)
    assert displacement_px(text(), text(x=25.0)) == pytest.approx(15.0)


def test_anchor_and_area_ignore_the_dummy_vertices():
    leaf = square(10, 20)
    np.testing.assert_allclose(leaf.anchor(), [15.0, 25.0])  # the four corners, not the close's repeat or the stop
    assert leaf.area_px() == pytest.approx(100.0)


def test_a_path_builds_a_filled_and_stroked_shape_in_units(small_frame):
    shape = square(80, 45, stroke=RED, width=2.0).build(small_frame)
    assert isinstance(shape, VMobject)
    np.testing.assert_allclose(shape.fill_rgbas[0], BLUE)
    np.testing.assert_allclose(shape.stroke_rgbas[0], RED)
    assert shape.stroke_width == pytest.approx(small_frame.stroke_width(2.0))
    np.testing.assert_allclose(shape.get_corner(np.array([-1, -1, 0]))[:2], [0.0, 0.0], atol=1e-6)  # pixel (80, 45)


def test_dashes_and_hard_edges_carry_over_to_the_shape(small_frame):
    dashed = square(stroke=RED, width=1.0, fill=None, dashes=(3.0, 3.0)).build(small_frame)
    assert isinstance(dashed, DashedVMobject) and len(dashed.submobjects) > 1
    assert getattr(square(antialiased=False).build(small_frame), ALIASED, False)


def test_a_clipped_path_carries_its_box_to_every_part_of_its_shape(small_frame):
    leaf = square(stroke=RED, width=1.0, dashes=(3.0, 3.0), key=("k", 0))
    clipped = PathLeaf(("k", 0), leaf.vertices, leaf.codes, fill=None, stroke=RED, linewidth=1.0, dashes=(3.0, 3.0),
                       cap="butt", join="miter", clip=(0.0, 0.0, 80.0, 45.0))
    shape = clipped.build(small_frame)
    assert all(getattr(part, CLIP) == pytest.approx((-1.6, -0.9, 0.0, 0.0)) for part in shape.get_family())
    assert clipped.signature != PathLeaf(("k", 0), leaf.vertices, leaf.codes, fill=None, stroke=RED, linewidth=1.0,
                                         dashes=(3.0, 3.0), cap="butt", join="miter").signature
    assert not clipped.matches(leaf)


def test_text_builds_one_filled_shape(small_frame):
    shape = text().build(small_frame)
    assert len(shape.points) > 0 and shape.get_stroke_width() == 0
    np.testing.assert_allclose(shape.fill_rgbas[0], BLUE)


def test_built_shapes_are_independent(small_frame):
    a, b = square().build(small_frame), square().build(small_frame)
    a.fill_rgbas[0, 3] = 0.1
    assert b.fill_rgbas[0, 3] == 1.0 and a.points is not b.points


def test_an_image_is_placed_pixel_for_pixel(small_frame):
    pixels = np.zeros((9, 16, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    leaf = ImageLeaf(("i", 0), pixels, (0.0, 0.0))
    image = leaf.build(small_frame)
    assert isinstance(image, ImageMobject)
    assert image.width == pytest.approx(16 / small_frame.dpi) and image.height == pytest.approx(9 / small_frame.dpi)
    np.testing.assert_allclose(image.get_corner(np.array([-1, -1, 0]))[:2], [-1.6, -0.9], atol=1e-6)
    assert leaf.matches(ImageLeaf(("i", 0), pixels.copy(), (5.0, 5.0)))
    assert leaf.area_px() == 144 and np.array_equal(leaf.anchor(), [0.0, 0.0])
