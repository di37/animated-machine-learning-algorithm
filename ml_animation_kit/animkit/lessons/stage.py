"""The 16:9 canvas: header, main plot, side panel, metric strip, caption bar, footer, full-screen card.

The canvas is 25.6 × 14.4 inches, and every position is set in inches because text is set in points:
the header, caption and footer keep their distance from the edges at every resolution, and only the
dpi changes. ``_rect`` turns inches into the figure fractions matplotlib places axes by.

The ``Stage`` owns the figure and its axes. On the left is the main plot (or side-by-side plots, or
a 3-D plot); on the right a panel above a metric strip, with room for an algorithm diagram on top;
below them the caption bar and the footer; over everything, a full-screen card for the title and
takeaway. Painters draw into those axes, and ``render`` turns the figure into one frame.
"""

import textwrap
from contextlib import AbstractContextManager

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D

from animkit.lessons import theme
from animkit.lessons.config import DEFAULT_RESOLUTION, FIGSIZE, FOOTER, HEADER, RESOLUTIONS

# The layout is set in inches, since text is in points: the header, caption and footer keep fixed distances
# from the edges, and the workspace between them takes the rest of the canvas. The right-hand column is
# wide, so an algorithm diagram has room to be its biggest box.
WIDTH_IN, HEIGHT_IN = FIGSIZE
MARGIN_IN = 0.67  # left edge of the header, caption and footer
RIGHT_IN = WIDTH_IN - 0.58  # right edge of the side column, tracker, stage label and source note
TITLE_Y_IN, SUBTITLE_Y_IN, FOOTER_Y_IN = HEIGHT_IN - 0.7, HEIGHT_IN - 1.21, 0.3
HEADER_Y_IN = HEIGHT_IN - FOOTER_Y_IN  # the header line sits as far from the top as the footer from the bottom
HEADER_ZORDER = 11  # above the full-screen card (10), so the title and takeaway cards carry it too
WORK_TOP_IN, WORK_BOTTOM_IN = HEIGHT_IN - 1.67, 2.65  # main plot and side column, between header and caption
MAIN_LEFT_IN, MAIN_WIDTH_IN = 1.31, 14.0  # room on the left for the y-axis label
SIDE_LEFT_IN = MAIN_LEFT_IN + MAIN_WIDTH_IN + 1.13  # past the side panel's tick labels and y-axis label
TITLE_ROOM_IN = 0.42  # above a panel that sits under the diagram: its title
AXIS_ROOM_IN = 0.95  # between the panel and the metric strip: the panel's x axis and the strip's title
PANEL_HEIGHT_IN = 5.5
# When a lesson has an algorithm diagram it takes the top of the side column, about twice the height of the
# panel and the metric strip, which shrink below it.
DIAGRAM_HEIGHT_IN, SHARED_PANEL_HEIGHT_IN = 4.4, 2.4
CAPTION_BOTTOM_IN, CAPTION_HEIGHT_IN = 0.76, 1.08
CARD_BOTTOM_IN = 0.65  # the full-screen card stops above the footer
FACET_GAP_IN = 0.58  # between side-by-side main plots
FACET_TITLE_ROOM_IN = 0.38  # side-by-side plots carry titles, so they start a little lower


def _x(inches: float) -> float:
    """Inches across the canvas, as a figure fraction."""
    return inches / WIDTH_IN


def _y(inches: float) -> float:
    """Inches up the canvas, as a figure fraction."""
    return inches / HEIGHT_IN


def _rect(left: float, bottom: float, width: float, height: float) -> tuple[float, float, float, float]:
    """Inches to the figure-fraction (left, bottom, width, height) that matplotlib places axes by."""
    return _x(left), _y(bottom), _x(width), _y(height)


SIDE_WIDTH_IN = RIGHT_IN - SIDE_LEFT_IN
SHARED_PANEL_TOP_IN = WORK_TOP_IN - DIAGRAM_HEIGHT_IN - TITLE_ROOM_IN
MAIN_RECT = _rect(MAIN_LEFT_IN, WORK_BOTTOM_IN, MAIN_WIDTH_IN, WORK_TOP_IN - WORK_BOTTOM_IN)
PANEL_RECT = _rect(SIDE_LEFT_IN, WORK_TOP_IN - PANEL_HEIGHT_IN, SIDE_WIDTH_IN, PANEL_HEIGHT_IN)
METRIC_RECT = _rect(SIDE_LEFT_IN, WORK_BOTTOM_IN, SIDE_WIDTH_IN,
                    WORK_TOP_IN - PANEL_HEIGHT_IN - AXIS_ROOM_IN - WORK_BOTTOM_IN)
DIAGRAM_RECT = _rect(SIDE_LEFT_IN, WORK_TOP_IN - DIAGRAM_HEIGHT_IN, SIDE_WIDTH_IN, DIAGRAM_HEIGHT_IN)
SHARED_PANEL_RECT = _rect(SIDE_LEFT_IN, SHARED_PANEL_TOP_IN - SHARED_PANEL_HEIGHT_IN, SIDE_WIDTH_IN,
                          SHARED_PANEL_HEIGHT_IN)
SHARED_METRIC_RECT = _rect(SIDE_LEFT_IN, WORK_BOTTOM_IN, SIDE_WIDTH_IN,
                           SHARED_PANEL_TOP_IN - SHARED_PANEL_HEIGHT_IN - AXIS_ROOM_IN - WORK_BOTTOM_IN)
CAPTION_RECT = _rect(MARGIN_IN, CAPTION_BOTTOM_IN, RIGHT_IN - MARGIN_IN, CAPTION_HEIGHT_IN)
FACET_GAP = _x(FACET_GAP_IN)
FACET_TITLE_ROOM = _y(FACET_TITLE_ROOM_IN)
# Chapter tracker in the header: (chapter key, label) for the chapters after the title card.
TRACKER = (("data", "Data"), ("initial", "First guess"), ("training", "Training"),
           ("final", "Final model"), ("inference", "Prediction"), ("exam", "Final exam"),
           ("surface", "3-D view"), ("features", "More features"), ("takeaway", "Takeaway"))
TRACKER_SIZE = 14
TRACKER_CHAR_IN = 0.1075  # width per character of the 14 pt tracker labels
TRACKER_GAP_IN = 0.4
CAPTION_CHARS = 150  # wrap width for the caption at CAPTION_SIZE
CAPTION_MAX_LINES = 2


def tracker_positions() -> tuple[float, ...]:
    """Left edge of each tracker label, spaced by label length, the row ending at the right margin.

    Returns:
        One x position per ``TRACKER`` label, in figure fractions.
    """
    widths = [len(label) * TRACKER_CHAR_IN for _, label in TRACKER]
    x = RIGHT_IN - sum(widths) - TRACKER_GAP_IN * (len(widths) - 1)
    xs = []
    for width in widths:
        xs.append(_x(x))
        x += width + TRACKER_GAP_IN
    return tuple(xs)


def lesson_style() -> AbstractContextManager:
    """The lesson look for a ``with`` block: matplotlib's defaults set to the lesson theme.

    Returns:
        A context manager that applies ``theme.RC_PARAMS`` and puts the old settings back on exit.
    """
    return matplotlib.rc_context(theme.RC_PARAMS)


def rounded_card(ax: Axes, face: str = theme.PANEL, edge: str = theme.PANEL_EDGE) -> None:
    """A rounded panel filling ``ax`` (used where the axes frame itself is switched off).

    Args:
        ax: The axes to fill.
        face: The card's fill colour.
        edge: Its edge colour.
    """
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.025",
                                transform=ax.transAxes, facecolor=face, edgecolor=edge,
                                linewidth=1.2, zorder=0, clip_on=False))


def reset_axes(ax: Axes, card: bool = True) -> None:
    """Blank an axes for a hand-drawn panel: no frame, unit coordinates, optional rounded card.

    The aspect is reset too: clear() keeps it, so a map drawn with equal aspect in an earlier chapter
    would otherwise stretch the unit square and push text out of the card.

    Args:
        ax: The axes to blank.
        card: Whether to draw the rounded card behind it.
    """
    ax.clear()
    ax.set_aspect("auto", adjustable="box")
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if card:
        rounded_card(ax)


class Stage:
    """Owns the figure. Painters draw into its axes; ``render`` returns one RGBA frame.

    ``source_note`` is the dataset credit at the bottom right; each project's scene supplies its own.

    Attributes:
        fig: The figure, drawn off screen by matplotlib's Agg canvas.
        main: The main plot.
        panel: The side panel.
        metric: The metric strip under the panel.
        diagram: The algorithm diagram above the panel, hidden unless a lesson has one.
        card: The full-screen card for the title and takeaway, hidden until shown.
    """

    def __init__(self, dpi: float = RESOLUTIONS[DEFAULT_RESOLUTION], source_note: str = ""):
        """Builds the canvas: header, chapter tracker, plots, caption bar, footer and the hidden card.

        Args:
            dpi: The resolution; the canvas size is fixed, so this alone sets the frame's pixel size.
            source_note: The data credit in the footer's bottom-right corner.
        """
        self.fig = Figure(figsize=FIGSIZE, dpi=dpi, facecolor=theme.CANVAS)
        FigureCanvasAgg(self.fig)
        self._title = self.fig.text(_x(MARGIN_IN), _y(TITLE_Y_IN), "", fontsize=theme.TITLE_SIZE,
                                    weight="bold", color=theme.HEADING, va="center")
        self._subtitle = self.fig.text(_x(MARGIN_IN), _y(SUBTITLE_Y_IN), "", fontsize=theme.SUBTITLE_SIZE,
                                       color=theme.INK_SECONDARY, va="center")
        self._tracker = tuple(self.fig.text(x, _y(TITLE_Y_IN), label, fontsize=TRACKER_SIZE, va="center",
                                            ha="left") for x, (_, label) in zip(tracker_positions(), TRACKER))
        self._stage_label = self.fig.text(_x(RIGHT_IN), _y(SUBTITLE_Y_IN), "", fontsize=theme.SUBTITLE_SIZE,
                                          weight="bold", color=theme.NAVY, va="center", ha="right")
        self.main = self.fig.add_axes(MAIN_RECT)
        self.panel = self.fig.add_axes(PANEL_RECT)
        self.metric = self.fig.add_axes(METRIC_RECT)
        self.diagram = self.fig.add_axes(DIAGRAM_RECT)
        self.diagram.set_visible(False)
        self._caption_ax = self.fig.add_axes(CAPTION_RECT)
        reset_axes(self._caption_ax)
        self._caption = self._caption_ax.text(0.02, 0.5, "", fontsize=theme.CAPTION_SIZE,
                                              color=theme.NAVY, va="center", ha="left",
                                              linespacing=1.35)
        self.fig.text(_x(RIGHT_IN), _y(HEADER_Y_IN), HEADER, fontsize=theme.HEADER_SIZE, weight="bold",
                      color=theme.NAVY, va="center", ha="right", zorder=HEADER_ZORDER)
        self.fig.text(_x(MARGIN_IN), _y(FOOTER_Y_IN), FOOTER, fontsize=theme.FOOTER_SIZE,
                      color=theme.INK_MUTED, va="center")
        self.fig.text(_x(RIGHT_IN), _y(FOOTER_Y_IN), source_note, fontsize=theme.FOOTER_SIZE,
                      color=theme.INK_MUTED, va="center", ha="right")
        self.card = self.fig.add_axes((0.0, 0.0, 1.0, 1.0), zorder=10)
        self.card.set_visible(False)
        self._workspace = (self.main, self.panel, self.metric, self._caption_ax, self._stage_label,
                           *self._tracker)
        self._facets: tuple[Axes, ...] = ()
        self._main_3d: Axes3D | None = None
        self._backdrop_3d: Axes | None = None

    @property
    def size_px(self) -> tuple[int, int]:
        """The frame's size in pixels.

        Returns:
            ``(width, height)``, as the canvas renders it.
        """
        return self.fig.canvas.get_width_height()

    def render(self) -> bytes:
        """Draws the figure and returns it as one frame.

        Returns:
            The frame's raw RGBA bytes, row by row from the top.
        """
        self.fig.canvas.draw()
        return bytes(self.fig.canvas.buffer_rgba())

    def set_header(self, title: str, subtitle: str, chapter: str, stage_label: str = "") -> None:
        """Sets the title, the subtitle, the stage label and the chapter the tracker highlights.

        Chapters before the current one are drawn in navy, the current one in bold heading blue, and the
        ones still to come muted.

        Args:
            title: The lesson's title.
            subtitle: The line under it.
            chapter: The current chapter's key; a key not in ``TRACKER`` (the title card's) highlights none.
            stage_label: The bold label at the right-hand end of the subtitle line.
        """
        self._title.set_text(title)
        self._subtitle.set_text(subtitle)
        self._stage_label.set_text(stage_label)
        keys = [key for key, _ in TRACKER]
        current = keys.index(chapter) if chapter in keys else -1
        for index, text in enumerate(self._tracker):
            text.set_color(theme.HEADING if index == current else
                           theme.NAVY if index < current else theme.INK_MUTED)
            text.set_fontweight("bold" if index == current else "normal")

    def set_caption(self, text: str) -> None:
        """Sets the caption bar's text, wrapped to two lines; longer text is set smaller and wrapped wider.

        Args:
            text: The caption.
        """
        lines = textwrap.wrap(text, CAPTION_CHARS)
        size = theme.CAPTION_SIZE if len(lines) <= CAPTION_MAX_LINES else theme.CAPTION_SIZE * 0.82
        if len(lines) > CAPTION_MAX_LINES:
            lines = textwrap.wrap(text, int(CAPTION_CHARS / 0.82))
        self._caption.set_text("\n".join(lines))
        self._caption.set_fontsize(size)

    def main_facets(self, count: int) -> tuple[Axes, ...]:
        """The main plot, or ``count`` side-by-side plots in its place (made once, then reused).

        Args:
            count: How many plots.

        Returns:
            The plots, left to right.
        """
        self._hide_3d()
        if count == 1:
            return (self.main,)
        if len(self._facets) != count:
            for ax in self._facets:
                ax.remove()
            left, bottom, width, height = MAIN_RECT
            each = (width - FACET_GAP * (count - 1)) / count
            self._facets = tuple(self.fig.add_axes((left + i * (each + FACET_GAP), bottom, each,
                                                    height - FACET_TITLE_ROOM))
                                 for i in range(count))
        self.main.set_visible(False)
        for ax in self._facets:
            ax.set_visible(True)
        return self._facets

    def main_3d(self) -> Axes3D:
        """A 3-D plot in the main plot's place (made once, then reused); hides the 2-D main plots.

        A 3-D axes keeps a square frame, so a blank panel behind it fills the whole main plot and the
        3-D box, drawn with a transparent background, is free to use that width.

        Returns:
            The 3-D axes.
        """
        if self._main_3d is None:
            self._backdrop_3d = self.fig.add_axes(MAIN_RECT)
            self._backdrop_3d.set_xticks([])
            self._backdrop_3d.set_yticks([])
            self._main_3d = self.fig.add_axes(MAIN_RECT, projection="3d", computed_zorder=False)
            self._main_3d.set_facecolor((1.0, 1.0, 1.0, 0.0))  # kept through clear(), unlike the patch
        for ax in (self.main, *self._facets):
            ax.set_visible(False)
        self._backdrop_3d.set_visible(True)
        self._main_3d.set_visible(True)
        return self._main_3d

    def _hide_3d(self) -> None:
        """Hides the 3-D plot and its backdrop, if they have been made."""
        for ax in (self._main_3d, self._backdrop_3d):
            if ax is not None:
                ax.set_visible(False)

    def show_card(self) -> Axes:
        """Clear and show the full-screen card (title and takeaway); hides the workspace.

        Returns:
            The card's axes, blank, in unit coordinates.
        """
        self._hide_3d()
        for artist in (*self._workspace, *self._facets, self.diagram):
            artist.set_visible(False)
        reset_axes(self.card, card=False)
        bottom = _y(CARD_BOTTOM_IN)
        self.card.add_patch(FancyBboxPatch((0.0, bottom), 1.0, 1.0 - bottom, boxstyle="square,pad=0",
                                           transform=self.card.transAxes, facecolor=theme.CANVAS,
                                           edgecolor="none", zorder=0))
        self.card.set_visible(True)
        return self.card

    def share_panel(self, shared: bool) -> None:
        """Diagram on top and a smaller panel below it, or the panel alone at full size.

        Args:
            shared: True to make room for an algorithm diagram (which is cleared and shown), False to hide it.
        """
        self.panel.set_position(SHARED_PANEL_RECT if shared else PANEL_RECT)
        self.metric.set_position(SHARED_METRIC_RECT if shared else METRIC_RECT)
        self.diagram.clear()
        self.diagram.set_visible(shared)

    def show_workspace(self) -> None:
        """Hides the full-screen card and shows the workspace: plots, panel, caption, tracker and stage label."""
        self.card.set_visible(False)
        for artist in self._workspace:
            artist.set_visible(True)
