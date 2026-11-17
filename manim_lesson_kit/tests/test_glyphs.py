"""Text lines as glyph outlines: placed, spread and rotated as Agg draws them."""

import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties

from manimkit import glyphs

DPI = 100.0


def prop(size: float = 15.0, weight: str = "normal") -> FontProperties:
    return FontProperties(family=["Arial", "DejaVu Sans"], size=size, weight=weight)


def hinted_width(string: str, font: FontProperties) -> float:
    return RendererAgg(800, 200, DPI).get_text_width_height_descent(string, font, ismath=False)[0]


def test_a_line_spans_the_width_agg_gives_it():
    font = prop()
    string = "Gradient descent θ₀ → ✓"
    width = hinted_width(string, font)
    points = glyphs.line_points(string, font, (100.0, 50.0), 0.0, dpi=DPI, hinted_width_px=width)
    ink = (points[:, 0].max() - points[:, 0].min()) * DPI
    assert abs(ink - width) < 0.06 * width  # the ink sits inside the advance, a bearing or two short


def test_spreading_moves_glyphs_apart_but_keeps_their_shapes():
    font = prop()
    tight = glyphs.outline("ii", font, spread=1.0)
    wide = glyphs.outline("ii", font, spread=1.5)
    assert tight.shape == wide.shape
    half = len(tight) // 2
    np.testing.assert_allclose(np.ptp(tight[:half], axis=0), np.ptp(wide[:half], axis=0))  # first glyph unchanged
    assert wide[half:, 0].min() > tight[half:, 0].min()


def test_fallback_fonts_draw_characters_the_first_font_lacks():
    assert len(glyphs.outline("✓", prop())) > 0  # Arial has no check mark; DejaVu Sans draws it


def test_spaces_draw_nothing_and_a_line_starts_at_its_origin():
    font = prop()
    assert glyphs.outline("   ", font).shape == (0, 2)
    a = glyphs.line_points("A", font, (100.5, 50.0), 0.0, dpi=DPI, hinted_width_px=hinted_width("A", font))
    b = glyphs.line_points("A", font, (100.0, 50.0), 0.0, dpi=DPI, hinted_width_px=hinted_width("A", font))
    np.testing.assert_allclose(a - b, np.tile([0.5 / DPI, 0.0], (len(a), 1)), atol=1e-12)


def test_rotated_text_turns_about_its_baseline_origin():
    font = prop()
    flat = glyphs.place(glyphs.outline("Median", font), (1.0, 1.0), 0.0)
    up = glyphs.place(glyphs.outline("Median", font), (1.0, 1.0), 90.0)
    assert abs(np.ptp(up[:, 1]) - np.ptp(flat[:, 0])) < 1e-9
    assert up[:, 0].max() <= 1.0 + 0.01  # turned left of its origin (round letters overshoot the baseline a hair)
    assert up[:, 1].min() >= 1.0 - 0.01  # and rising from it


def test_bold_and_size_change_the_font_key():
    assert glyphs.font_key(prop()) != glyphs.font_key(prop(weight="bold"))
    assert glyphs.font_key(prop(12)) != glyphs.font_key(prop(13))


def test_mathtext_lines_are_drawn_from_their_whole_path():
    points = glyphs.line_points(r"$x^2$", prop(), (10.0, 10.0), 0.0, dpi=DPI, hinted_width_px=0.0, ismath=True)
    assert len(points) > 0 and points.shape[1] == 2
