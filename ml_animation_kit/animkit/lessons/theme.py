"""Lesson palette: off-white/pale-cyan canvas, navy text, electric-blue headings.

The four mark colours were run through the dataviz palette validator on the canvas colour
(lightness, chroma, CVD and normal-vision separation all pass). Orange sits just under 3:1
contrast, so every orange mark carries a text label.

Each colour keeps one role in every chapter: blue is the model, orange is unseen data, teal is what
the algorithm is looking at right now, red is error, and training rows are a quiet grey-blue.
"""

from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

CANVAS = "#f2f8fb"
PANEL = "#ffffff"
PANEL_EDGE = "#d3e4ec"
GRID = "#e3eef3"

NAVY = "#0f2541"
INK_SECONDARY = "#3d5470"
INK_MUTED = "#6b7f95"
HEADING = "#1557d6"  # electric blue; headings only, never data

MODEL = "#2a78d6"  # the fitted model and anything "training"
UNSEEN = "#eb6834"  # validation and test rows, including the example predicted at inference
HIGHLIGHT = "#0b8f9c"  # teal: what the algorithm is looking at right now
ERROR = "#e34948"  # residuals and negative weights
DOTS = "#8ea3b8"  # training rows: data, not a series
INACTIVE = "#c9d6e0"

SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

FONT_FAMILY = ["Arial", "DejaVu Sans"]  # Arial for text; DejaVu fills in arrows, ✓ and maths
TITLE_SIZE = 34
SUBTITLE_SIZE = 17
CAPTION_SIZE = 21
AXIS_TITLE_SIZE = 15
TICK_SIZE = 13
PANEL_TITLE_SIZE = 15
NOTE_SIZE = 13
FOOTER_SIZE = 12
HEADER_SIZE = 13

RC_PARAMS = {
    "font.family": FONT_FAMILY,
    "figure.facecolor": CANVAS,
    "axes.facecolor": PANEL,
    "axes.edgecolor": PANEL_EDGE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": AXIS_TITLE_SIZE,
    "axes.titlecolor": NAVY,
    "axes.titlesize": PANEL_TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "grid.color": GRID,  # 3-D axes take their grid from here
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": NOTE_SIZE,
    "legend.frameon": False,
    "text.parse_math": False,  # captions hold dollar signs and Unicode maths, never LaTeX
}


def value_colormap() -> LinearSegmentedColormap:
    """Single-hue light-to-dark ramp for magnitudes (leaf values, activations).

    Returns:
        A colormap running from the palest to the darkest of ``SEQUENTIAL``.
    """
    return LinearSegmentedColormap.from_list("lesson_blue", SEQUENTIAL)


def style_axes(ax: Axes, grid_axis: str | None = "both") -> None:
    """Recessive chrome: hairline grid (None for no grid), no top/right spines, no tick marks.

    Args:
        ax: The axes to style.
        grid_axis: Which axis gets grid lines: ``"x"``, ``"y"``, ``"both"``, or ``None`` for none.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PANEL_EDGE)
    if grid_axis is not None:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
