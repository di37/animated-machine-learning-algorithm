"""The shared animkit palette, plus the single-hue ramp the house-price views colour values with.

Every token is re-exported, so house modules keep reading colours as ``theme.X``.
"""

from matplotlib.colors import LinearSegmentedColormap

from animkit.theme import (
    BASELINE,
    CATEGORICAL,
    FONT_AXIS_TITLE,
    FONT_LABEL,
    FONT_SUBTITLE,
    FONT_TITLE,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    NEGATIVE,
    NOT_YET,
    POSITIVE,
    RC_PARAMS,
    SEQUENTIAL_BLUE,
    SERIES_1,
    SERIES_2,
    SURFACE,
    series_color,
    style_axes,
)

__all__ = [
    "BASELINE", "CATEGORICAL", "FONT_AXIS_TITLE", "FONT_LABEL", "FONT_SUBTITLE", "FONT_TITLE", "GRIDLINE",
    "INK_MUTED", "INK_PRIMARY", "INK_SECONDARY", "NEGATIVE", "NOT_YET", "POSITIVE", "RC_PARAMS",
    "SEQUENTIAL_BLUE", "SERIES_1", "SERIES_2", "SURFACE", "price_colormap", "series_color", "style_axes",
]


def price_colormap() -> LinearSegmentedColormap:
    """Single-hue light-to-dark ramp for house prices (magnitude, not identity).

    One hue that darkens as the price rises reads as "more", where a rainbow would suggest
    unrelated categories.

    Returns:
        A matplotlib colormap over animkit's ``SEQUENTIAL_BLUE`` steps, cheapest first.
    """
    return LinearSegmentedColormap.from_list("price_blue", SEQUENTIAL_BLUE)
