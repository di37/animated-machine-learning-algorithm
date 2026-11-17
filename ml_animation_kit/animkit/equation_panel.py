"""Draws an EquationCard as a column of styled lines inside its own axes.

Space is measured in points, not axes fractions, so a line keeps its height whatever the panel's
size. Each line kind has a style and a height; notes wrap to the panel's width, and mathtext with
stacked limits or fractions gets a taller line. A card longer than the panel is cut short with an
ellipsis rather than drawn past the edge. The panel makes its text objects once and reuses them for
every frame.
"""

import textwrap

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

from animkit import theme
from animkit.equations import HEADING, MATH, MONO, NOTE, EquationCard

DEFAULT_MAX_LINES = 24
TOP_MARGIN_PT = 33.0
BOTTOM_MARGIN_PT = 10.0
# kind -> (text style, vertical space the line takes, in points)
STYLES = {
    HEADING: (dict(fontsize=9.5, weight="bold", color=theme.INK_MUTED, family="sans-serif"), 23.8),
    MATH: (dict(fontsize=14, weight="normal", color=theme.INK_PRIMARY, family="sans-serif"), 27.6),
    MONO: (dict(fontsize=10.5, weight="normal", color=theme.INK_PRIMARY, family="monospace"), 19.1),
    NOTE: (dict(fontsize=10, weight="normal", color=theme.INK_SECONDARY, family="sans-serif"), 21.0),
}
HEADING_GAP_PT = 8.6  # extra air above each section heading
# Mathtext that stacks limits or fractions vertically needs about one more line of room.
TALL_MATH_MARKERS = ("\\sum", "\\frac", "\\min")
TALL_MATH_PT = 43.8
CHAR_WIDTH_EM = 0.58  # average glyph width of the note font, for wrapping


class EquationPanel:
    """A rounded card that shows one equation card at a time, redrawn in place each frame.

    Attributes:
        ax: The panel's axes, with its frame switched off.
        title: The text object holding the card's title.
        texts: The reusable text objects, one per line the panel can show.
    """

    def __init__(self, ax: Axes, max_lines: int = DEFAULT_MAX_LINES):
        """Draws the empty card and makes the text objects the panel reuses.

        Args:
            ax: The axes to draw in; its size sets how much fits.
            max_lines: The most lines the panel can show, the ellipsis included.
        """
        self.ax = ax
        box = ax.get_position()
        width_in, height_in = ax.figure.get_size_inches()
        self._per_point = 1.0 / (72.0 * box.height * height_in)  # points -> axes fraction
        note_size = STYLES[NOTE][0]["fontsize"]
        self._wrap_chars = max(20, int(box.width * width_in * 72.0 / (note_size * CHAR_WIDTH_EM)))
        ax.set_axis_off()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.02",
                                    transform=ax.transAxes, facecolor=theme.SURFACE,
                                    edgecolor=theme.BASELINE, linewidth=1.0, clip_on=False))
        self.title = ax.text(0.05, 0.975, "", va="top", fontsize=12, weight="semibold",
                             color=theme.INK_PRIMARY, transform=ax.transAxes)
        self.texts = [ax.text(0.05, 0.0, "", va="top", transform=ax.transAxes, visible=False)
                      for _ in range(max_lines)]

    def update(self, card: EquationCard) -> None:
        """Shows ``card``: its title, then its lines from the top until the panel is full.

        Headings get extra space above them. When the lines do not all fit, the last line shown is an
        ellipsis. Text objects left over are hidden.

        Args:
            card: The card to show.
        """
        self.title.set_text(card.title)
        lines = list(card.lines)
        if len(lines) > len(self.texts):
            lines = lines[:len(self.texts) - 1]
            truncated = True
        else:
            truncated = False
        y = 1.0 - TOP_MARGIN_PT * self._per_point
        bottom = BOTTOM_MARGIN_PT * self._per_point
        used = 0
        for line in lines:
            style = STYLES[line.kind][0]
            content, height = self._layout(line.kind, line.text)
            if line.kind == HEADING and used:
                y -= HEADING_GAP_PT * self._per_point
            if y - height < bottom:
                truncated = True
                break
            self._place(self.texts[used], content, style, y)
            y -= height
            used += 1
        if truncated:
            self._place(self.texts[used], "…", STYLES[NOTE][0], y)
            used += 1
        for text in self.texts[used:]:
            text.set_visible(False)

    def _layout(self, kind: str, text: str) -> tuple[str, float]:
        """The text as drawn (notes wrapped) and the vertical space it needs, in axes fractions."""
        points = STYLES[kind][1]
        if kind == MATH and any(marker in text for marker in TALL_MATH_MARKERS):
            points = TALL_MATH_PT
        elif kind == NOTE:
            text = textwrap.fill(text, self._wrap_chars)
            points *= text.count("\n") + 1
        return text, points * self._per_point

    @staticmethod
    def _place(text, content: str, style: dict, y: float) -> None:
        """Restyles one reusable text object as a line and moves it to height ``y``."""
        text.set_text(content)
        text.set_y(y)
        text.set_fontsize(style["fontsize"])
        text.set_fontweight(style["weight"])
        text.set_color(style["color"])
        text.set_fontfamily(style["family"])
        text.set_visible(True)
