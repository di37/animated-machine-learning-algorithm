"""Drawing every lesson shares: the teaching callout, plain tick labels and the worked-prediction panel.

``PLAIN_FORMAT`` is a tick formatter that prints numbers in their shortest form (``:g``). The
worked-prediction panel sizes its lines from the panel's real height, so it fits whether it has the
side column to itself or shares it with an algorithm diagram.
"""

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from animkit.lessons import theme
from animkit.lessons.lesson import Explanation
from animkit.lessons.stage import reset_axes

PLAIN_FORMAT = FuncFormatter(lambda value, _: f"{value:g}")
CALLOUT_BOX = dict(boxstyle="round,pad=0.55,rounding_size=0.4", facecolor=theme.PANEL,
                   edgecolor=theme.PANEL_EDGE, linewidth=1.2)
EXPLANATION_TOP = 0.72
EXPLANATION_BOTTOM = 0.04
EXPLANATION_PITCH = 0.13
EXPLANATION_SIZE = 13.5
LINE_SPACING = 1.3  # line pitch / font size, in points


def callout(ax: Axes, text: str) -> None:
    """Writes the teaching note in the main plot's top-left corner, on a rounded card.

    Args:
        ax: The main plot.
        text: The note; it may hold line breaks.
    """
    ax.text(0.018, 0.965, text, transform=ax.transAxes, ha="left", va="top", fontsize=15,
            color=theme.NAVY, bbox=CALLOUT_BOX, zorder=12, linespacing=1.4)


def draw_explanation(ax: Axes, explanation: Explanation, shown: int, title: str) -> None:
    """Draws the worked-prediction panel: a title and the first ``shown`` lines of the explanation.

    Pitch and font follow the panel's real height, so the lines still fit when the panel shares its space
    with an algorithm diagram. The last line, the model check, is set in bold.

    Args:
        ax: The panel's axes; it is cleared first.
        explanation: The worked prediction.
        shown: How many of its lines to show, so they can appear one by one.
        title: The panel's title.
    """
    reset_axes(ax)
    ax.text(0.04, 0.88, title, fontsize=theme.PANEL_TITLE_SIZE, weight="bold", color=theme.NAVY,
            va="center")
    height_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72.0
    step = min(EXPLANATION_PITCH, (EXPLANATION_TOP - EXPLANATION_BOTTOM) / max(len(explanation.lines), 1))
    size = min(EXPLANATION_SIZE, step * height_pt / LINE_SPACING)
    for row, line in enumerate(explanation.lines[:shown]):
        last = row == len(explanation.lines) - 1
        ax.text(0.04, EXPLANATION_TOP - row * step, line, fontsize=size, va="center",
                color=theme.NAVY if last else theme.INK_SECONDARY, weight="bold" if last else "normal")
