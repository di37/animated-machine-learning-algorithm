r"""One line of matplotlib text as one Manim shape: the font's own glyph outlines, where Agg puts them.

matplotlib lays a text out and hands its renderer each line with its baseline origin, angle and font
(with fallback fonts for the characters the first one lacks). The outlines here come from the same
fonts through matplotlib's ``TextToPath``. Agg draws *hinted* glyphs (nudged onto the pixel grid,
which makes text a little wider), while outlines are *unhinted*; so each line's glyphs are spaced out
to the width Agg's hinted text takes. The recorder rounds the origin as Agg does, so a line starts
and ends where it does on the matplotlib frame.
"""

import math

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import text_to_path

from manimkit.paths import cubic_points

POINTS_PER_INCH = 72.0
_GLYPH_MAP: dict[str, tuple] = {}  # glyph id: matplotlib's (vertices, codes), so each glyph is loaded once
_OUTLINES: dict[str, np.ndarray] = {}  # glyph id: its outline as (4k, 2) points at FONT_SCALE


def font_key(prop: FontProperties) -> tuple:
    """What decides a line's glyphs, apart from its text.

    Args:
        prop: The line's font properties.

    Returns:
        A hashable tuple: two lines with equal keys and equal text draw the same glyphs.
    """
    return (tuple(prop.get_family()), prop.get_size_in_points(), prop.get_weight(), prop.get_style(),
            prop.get_stretch(), prop.get_variant(), prop.get_math_fontfamily())


def unhinted_width_pt(string: str, prop: FontProperties, ismath: bool | str = False) -> float:
    """The width matplotlib's unhinted outlines give a line.

    Args:
        string: The line's text.
        prop: Its font properties.
        ismath: Whether the text is mathtext, as matplotlib passes it.

    Returns:
        The width in points.
    """
    return float(text_to_path.get_text_width_height_descent(string, prop, ismath=ismath)[0])


def outline(string: str, prop: FontProperties, *, ismath: bool | str = False, spread: float = 1.0) -> np.ndarray:
    """A line's glyph outlines as Manim points, its baseline starting at (0, 0), unrotated.

    Each glyph's outline is loaded once and reused; spaces have no outline and add nothing.

    Args:
        string: The line's text.
        prop: Its font properties (with fallback fonts, as Agg resolves them).
        ismath: Whether the text is mathtext; it is then drawn as one path, without spreading.
        spread: How much to stretch the glyphs' positions (not their shapes) along the line.

    Returns:
        Manim points in inches, shape (4k, 2).
    """
    scale = prop.get_size_in_points() / text_to_path.FONT_SCALE / POINTS_PER_INCH
    if ismath:
        vertices, codes = text_to_path.get_text_path(prop, string, ismath=ismath)
        points = cubic_points(np.asarray(vertices), np.asarray(codes)) if len(codes) else np.empty((0, 2))
        return points * scale
    font = text_to_path._get_font(prop)  # the font with its fallbacks, as Agg resolves it
    glyph_info, new_glyphs, rects = text_to_path.get_glyphs_with_font(font, string, glyph_map=_GLYPH_MAP,
                                                                      return_new_glyphs_only=True)
    pieces = []
    for glyph_id, x, y, size in glyph_info:
        if glyph_id not in _OUTLINES:
            vertices, codes = new_glyphs[glyph_id]
            _GLYPH_MAP[glyph_id] = (vertices, codes)
            _OUTLINES[glyph_id] = cubic_points(np.asarray(vertices), np.asarray(codes))
        shape = _OUTLINES[glyph_id]
        if len(shape):  # a space has no outline
            pieces.append(shape * size + [x * spread, y])
    for vertices, codes in rects:
        pieces.append(cubic_points(np.asarray(vertices), np.asarray(codes)))
    return np.concatenate(pieces) * scale if pieces else np.empty((0, 2))


def place(points: np.ndarray, origin_in: tuple[float, float], angle_deg: float) -> np.ndarray:
    """Turns outline points about their baseline origin, then moves them to their place.

    Args:
        points: Outline points in inches, baseline origin at (0, 0), shape (n, 2).
        origin_in: Where the baseline starts, in inches from the figure's bottom left.
        angle_deg: The text's rotation, counter-clockwise, in degrees.

    Returns:
        The placed points, shape (n, 2).
    """
    if angle_deg:
        theta = math.radians(angle_deg)
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        points = points @ rotation.T
    return points + np.asarray(origin_in, dtype=float)


def line_points(string: str, prop: FontProperties, origin_px: tuple[float, float], angle_deg: float, *,
                dpi: float, hinted_width_px: float, ismath: bool | str = False) -> np.ndarray:
    """A line of text as Agg draws it: its outlines, spread to Agg's width and placed at its origin.

    Args:
        string: The line's text.
        prop: Its font properties.
        origin_px: Where its baseline starts, in display pixels (rounded by the recorder as Agg rounds it).
        angle_deg: Its rotation, in degrees.
        dpi: The figure's resolution.
        hinted_width_px: The width Agg's hinted text takes, in pixels.
        ismath: Whether the text is mathtext.

    Returns:
        The outline points in inches from the figure's bottom left, shape (4k, 2).
    """
    spread = 1.0
    if not ismath:
        unhinted_px = unhinted_width_pt(string, prop) * dpi / POINTS_PER_INCH
        spread = hinted_width_px / unhinted_px if unhinted_px > 0 and hinted_width_px > 0 else 1.0
    x, y = origin_px
    return place(outline(string, prop, ismath=ismath, spread=spread), (x / dpi, y / dpi), angle_deg)
