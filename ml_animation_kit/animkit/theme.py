"""Colour tokens and axis styling shared by every dashboard (validated reference palette).

Ink runs from primary (titles) to muted (ticks). The categorical palette's order was checked so
neighbouring colours stay distinct under colour-vision deficiency, and every race bar and panel is
also labelled by name, so no reading rests on colour alone. Chrome (grid, spines, ticks) is
recessive, so it sits behind the data. ``RC_PARAMS`` applies the same choices to matplotlib's defaults.
"""

from matplotlib.axes import Axes

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots 1 and 2: the adjacent pair passes colour-vision-deficiency checks.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
# Diverging pair for signed coefficients: blue pole positive, red pole negative.
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
NOT_YET = "#e1e0d9"

# Reference categorical order (validated adjacent-pair CVD separation). Every race bar and
# panel is also labelled by name, so identity never rests on colour alone.
CATEGORICAL = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)

SEQUENTIAL_BLUE = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)

FONT_TITLE = 17
FONT_SUBTITLE = 11
FONT_AXIS_TITLE = 11
FONT_LABEL = 9

RC_PARAMS = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK_PRIMARY,
    "axes.titlesize": FONT_AXIS_TITLE,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.labelsize": FONT_LABEL,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": FONT_LABEL,
    "ytick.labelsize": FONT_LABEL,
    "legend.fontsize": FONT_LABEL,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",  # equations in the classic LaTeX (Computer Modern) look
}


def series_color(slot: int) -> str:
    """The categorical colour for a palette slot, wrapping round after the eighth.

    Args:
        slot: The model's palette slot (see ``animkit.race.colour_slots``).

    Returns:
        The colour, as a hex string.
    """
    return CATEGORICAL[slot % len(CATEGORICAL)]


def style_axes(ax: Axes, grid_axis: str = "y") -> None:
    """Recessive chrome: hairline grid, no top/right spines, no tick marks.

    Args:
        ax: The axes to style.
        grid_axis: Which axis gets grid lines: ``"x"``, ``"y"`` or ``"both"``.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
