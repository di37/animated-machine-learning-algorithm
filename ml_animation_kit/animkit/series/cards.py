"""The series' text cards, drawn on the lessons' full-screen card in the lessons' theme.

Every card has a kicker (a small teal label) and a headline, then its own content, revealed in turn
while the chapter moves:

- ``paint_banner``: a welcome or thank-you slide: one large centred headline over a line or two.
- ``paint_text``: lines of text, each with an optional marker (✓, ✗, 1, 2 …) and a smaller detail below.
- ``paint_lineup``: up to nine numbered algorithm tiles, in playlist order.
- ``paint_roadmap``: a lesson's chapters as pills that light up one after another.
- ``paint_swatches``: the colour key, each colour beside what it always means, and one rule below.

Sizes are given on the cards' design canvas (``CARD_DESIGN_HEIGHT_IN`` tall) and scaled to the real
one, as the lessons' title and takeaway cards are, so a series card looks the same at every resolution.
"""

import textwrap
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle

from animkit.lessons import theme
from animkit.lessons.director import CARD_DESIGN_HEIGHT_IN, CARD_SCALE
from animkit.lessons.stage import Stage
from animkit.lessons.timing import phase

LEFT = 0.08  # the cards' left text edge, in card units
KICKER_Y, HEADLINE_Y = 0.85, 0.81
KICKER_PT, HEADLINE_PT, SUBHEAD_PT, LINE_PT, DETAIL_PT = 18, 44, 25, 25, 19
HEADLINE_CHARS, SUBHEAD_CHARS, LINE_CHARS, DETAIL_CHARS = 48, 88, 80, 105
POINTS_PER_INCH = 72
HEADING_SHARE = 0.25  # the kicker and headline fade in over this first share of the chapter's motion
LINE_SPACING = 1.32
GAP = 0.025  # between blocks, in card units
MARKER_INDENT = 0.035
MAX_TILES = 9
TILE_COLUMNS = 3
TILE_X = (0.08, 0.375, 0.67)
TILE_WIDTH, TILE_HEIGHT, TILE_TOP, TILE_STEP = 0.275, 0.15, 0.64, 0.18
PILLS_PER_ROW = 5
PILL_WIDTH, PILL_HEIGHT, PILL_GAP = 0.158, 0.075, 0.012
PILL_ROWS_Y = (0.6, 0.38)
LIT_FACE = theme.SEQUENTIAL[0]
SWATCH_WIDTH, SWATCH_HEIGHT = 0.03, 0.053  # about square on the 16:9 card
SWATCH_TOP, SWATCH_STEP = 0.6, 0.085
MEANING_X = 0.24
RULE_Y = 0.17
CENTRE = 0.5  # a banner's text is centred on the card
BANNER_KICKER_Y, BANNER_HEADLINE_Y, BANNER_RULE_Y, BANNER_SUBHEAD_Y = 0.71, 0.59, 0.495, 0.44
BANNER_PT, BANNER_SUBHEAD_PT, BANNER_NOTE_PT = 112, 36, 22
BANNER_RULE = (0.08, 0.006)  # the short teal rule under the headline: width, height
BANNER_SUBHEAD_CHARS, BANNER_NOTE_CHARS = 60, 90


@dataclass(frozen=True)
class Line:
    """One line of a text card.

    Attributes:
        text: The line.
        marker: A short mark before it, for example "✓"; none if empty.
        detail: A smaller line below it; none if empty.
        colour: The marker's colour.
    """

    text: str
    marker: str = ""
    detail: str = ""
    colour: str = theme.HIGHLIGHT


@dataclass(frozen=True)
class Algorithm:
    """One tile of the line-up: the lesson's title and how its algorithm learns."""

    title: str
    how: str


@dataclass(frozen=True)
class Step:
    """One pill of the road map: a chapter's name and what happens in it."""

    label: str
    note: str = ""


@dataclass(frozen=True)
class Swatch:
    """One row of the colour key: the colour, its name and what it always means."""

    colour: str
    name: str
    meaning: str


def card_pt(size: float) -> float:
    """A font size from the cards' design canvas, scaled to the real one."""
    return size * CARD_SCALE


def line_height(size: float, spacing: float = LINE_SPACING) -> float:
    """The height of one line of ``size``-point text, in card units."""
    return size * spacing / POINTS_PER_INCH / CARD_DESIGN_HEIGHT_IN


def wrapped(text: str, width: int) -> str:
    """``text`` wrapped at ``width`` characters."""
    return "\n".join(textwrap.wrap(text, width)) or text


def turn(t: float, index: int, count: int, start: float = HEADING_SHARE, end: float = 1.0) -> float:
    """How far the ``index``-th of ``count`` items has faded in at progress ``t``: each in its own turn."""
    window = (end - start) / max(count, 1)
    return phase(t, start + index * window, start + (index + 0.7) * window)


def begin(stage: Stage) -> Axes:
    """Shows a blank card with the lessons' accent bar beside its text."""
    card = stage.show_card()
    card.add_patch(Rectangle((0.055, 0.12), 0.006, 0.75, facecolor=theme.HEADING, edgecolor="none"))
    return card


def heading(card: Axes, kicker: str, headline: str, alpha: float) -> float:
    """Draws the kicker and the headline; returns where the content below them starts."""
    card.text(LEFT, KICKER_Y, kicker, fontsize=card_pt(KICKER_PT), weight="bold", color=theme.HIGHLIGHT,
              alpha=alpha)
    text = wrapped(headline, HEADLINE_CHARS)
    card.text(LEFT, HEADLINE_Y, text, fontsize=card_pt(HEADLINE_PT), weight="bold", color=theme.NAVY,
              alpha=alpha, va="top", linespacing=1.15)
    return HEADLINE_Y - (text.count("\n") + 1) * line_height(HEADLINE_PT, 1.15) - GAP


def _draw_line(card: Axes, line: Line, y: float, alpha: float) -> float:
    """Draws one line (marker, text, detail) from ``y`` down; returns where the next one starts."""
    x = LEFT + (MARKER_INDENT if line.marker else 0.0)
    if line.marker:
        card.text(LEFT, y, line.marker, fontsize=card_pt(LINE_PT), weight="bold", color=line.colour,
                  alpha=alpha, va="top")
    text = wrapped(line.text, LINE_CHARS)
    card.text(x, y, text, fontsize=card_pt(LINE_PT), color=theme.NAVY, alpha=alpha, va="top",
              linespacing=LINE_SPACING)
    y -= (text.count("\n") + 1) * line_height(LINE_PT)
    if line.detail:
        detail = wrapped(line.detail, DETAIL_CHARS)
        card.text(x, y - 0.005, detail, fontsize=card_pt(DETAIL_PT), color=theme.INK_SECONDARY, alpha=alpha,
                  va="top", linespacing=LINE_SPACING)
        y -= (detail.count("\n") + 1) * line_height(DETAIL_PT) + 0.005
    return y - GAP


def paint_text(stage: Stage, t: float, *, kicker: str, headline: str, lines: tuple[Line, ...] = (),
               subhead: str = "") -> None:
    """A headline, an optional subhead, then lines that appear one after another.

    Args:
        stage: The canvas.
        t: The chapter's progress, 0 to 1.
        kicker: The small label above the headline.
        headline: The card's main words.
        lines: The lines below, in order.
        subhead: A sentence under the headline; none if empty.
    """
    card = begin(stage)
    y = heading(card, kicker, headline, phase(t, 0.0, HEADING_SHARE))
    if subhead:
        text = wrapped(subhead, SUBHEAD_CHARS)
        card.text(LEFT, y, text, fontsize=card_pt(SUBHEAD_PT), color=theme.INK_SECONDARY,
                  alpha=phase(t, 0.0, HEADING_SHARE), va="top", linespacing=LINE_SPACING)
        y -= (text.count("\n") + 1) * line_height(SUBHEAD_PT) + GAP
    y -= GAP
    for index, line in enumerate(lines):
        y = _draw_line(card, line, y, turn(t, index, len(lines)))


def _tile(card: Axes, number: int, algorithm: Algorithm, x: float, top: float, alpha: float) -> None:
    """One numbered algorithm tile with its top-left corner at (x, top)."""
    card.add_patch(FancyBboxPatch((x, top - TILE_HEIGHT), TILE_WIDTH, TILE_HEIGHT,
                                  boxstyle="round,pad=0,rounding_size=0.012", facecolor=theme.PANEL,
                                  edgecolor=theme.PANEL_EDGE, linewidth=1.5, alpha=alpha))
    card.text(x + 0.015, top - 0.03, str(number), fontsize=card_pt(26), weight="bold", color=theme.HEADING,
              alpha=alpha, va="top")
    card.text(x + 0.05, top - 0.03, algorithm.title, fontsize=card_pt(22), weight="bold", color=theme.NAVY,
              alpha=alpha, va="top")
    card.text(x + 0.05, top - 0.075, wrapped(algorithm.how, 38), fontsize=card_pt(16),
              color=theme.INK_SECONDARY, alpha=alpha, va="top", linespacing=1.25)


def paint_lineup(stage: Stage, t: float, *, kicker: str, headline: str,
                 algorithms: tuple[Algorithm, ...]) -> None:
    """The playlist's algorithms as numbered tiles, three to a row, appearing in playlist order.

    Args:
        stage: The canvas.
        t: The chapter's progress, 0 to 1.
        kicker: The small label above the headline.
        headline: The card's main words.
        algorithms: The lessons' algorithms, in playlist order (at most ``MAX_TILES``).

    Raises:
        ValueError: If there are more algorithms than tiles.
    """
    if len(algorithms) > MAX_TILES:
        raise ValueError(f"the line-up holds at most {MAX_TILES} algorithms, got {len(algorithms)}")
    card = begin(stage)
    heading(card, kicker, headline, phase(t, 0.0, HEADING_SHARE))
    for index, algorithm in enumerate(algorithms):
        row, column = divmod(index, TILE_COLUMNS)
        _tile(card, index + 1, algorithm, TILE_X[column], TILE_TOP - row * TILE_STEP,
              turn(t, index, len(algorithms)))


def _pill(card: Axes, step: Step, x: float, y: float, lit: bool) -> None:
    """One road-map pill centred vertically on y: lit (pale blue, blue edge) or not yet (white, grey edge)."""
    card.add_patch(FancyBboxPatch((x, y - PILL_HEIGHT / 2), PILL_WIDTH, PILL_HEIGHT,
                                  boxstyle="round,pad=0,rounding_size=0.02",
                                  facecolor=LIT_FACE if lit else theme.PANEL,
                                  edgecolor=theme.MODEL if lit else theme.INACTIVE, linewidth=2,
                                  gid="lit" if lit else "unlit"))
    card.text(x + PILL_WIDTH / 2, y, step.label, fontsize=card_pt(19), weight="bold", ha="center", va="center",
              color=theme.NAVY if lit else theme.INK_MUTED)
    if step.note:
        card.text(x + PILL_WIDTH / 2, y - PILL_HEIGHT / 2 - 0.015, wrapped(step.note, 24), fontsize=card_pt(14),
                  ha="center", va="top", color=theme.INK_SECONDARY if lit else theme.INACTIVE, linespacing=1.2)


def paint_roadmap(stage: Stage, t: float, *, kicker: str, headline: str, steps: tuple[Step, ...],
                  note: str = "") -> None:
    """A lesson's chapters as pills, five to a row, each lighting up in its turn; a note below.

    Args:
        stage: The canvas.
        t: The chapter's progress, 0 to 1.
        kicker: The small label above the headline.
        headline: The card's main words.
        steps: The chapters, in order.
        note: A line under the pills, shown once every pill is lit; none if empty.
    """
    card = begin(stage)
    heading(card, kicker, headline, phase(t, 0.0, HEADING_SHARE))
    lit = sum(1 for index in range(len(steps)) if t >= index / len(steps) * 0.9)
    for index, step in enumerate(steps):
        row, column = divmod(index, PILLS_PER_ROW)
        _pill(card, step, LEFT + column * (PILL_WIDTH + PILL_GAP), PILL_ROWS_Y[min(row, len(PILL_ROWS_Y) - 1)],
              index < lit)
    if note:
        card.text(LEFT, RULE_Y, wrapped(note, 100), fontsize=card_pt(21), color=theme.INK_SECONDARY, va="top",
                  alpha=phase(t, 0.9, 1.0), linespacing=LINE_SPACING)


def paint_swatches(stage: Stage, t: float, *, kicker: str, headline: str, swatches: tuple[Swatch, ...],
                   rule: str = "") -> None:
    """The colour key: each colour's swatch, name and meaning in turn, then one rule in bold.

    Args:
        stage: The canvas.
        t: The chapter's progress, 0 to 1.
        kicker: The small label above the headline.
        headline: The card's main words.
        swatches: The colours, in order.
        rule: A closing line in bold; none if empty.
    """
    card = begin(stage)
    heading(card, kicker, headline, phase(t, 0.0, HEADING_SHARE))
    for index, swatch in enumerate(swatches):
        alpha = turn(t, index, len(swatches) + (1 if rule else 0))
        y = SWATCH_TOP - index * SWATCH_STEP
        card.add_patch(FancyBboxPatch((LEFT, y - SWATCH_HEIGHT / 2), SWATCH_WIDTH, SWATCH_HEIGHT,
                                      boxstyle="round,pad=0,rounding_size=0.006", facecolor=swatch.colour,
                                      edgecolor="none", alpha=alpha))
        card.text(LEFT + 0.045, y, swatch.name, fontsize=card_pt(24), weight="bold", color=theme.NAVY,
                  va="center", alpha=alpha)
        card.text(MEANING_X, y, swatch.meaning, fontsize=card_pt(23), color=theme.INK_SECONDARY, va="center",
                  alpha=alpha)
    if rule:
        card.text(LEFT, RULE_Y, wrapped(rule, 85), fontsize=card_pt(24), weight="bold", color=theme.NAVY,
                  va="top", alpha=turn(t, len(swatches), len(swatches) + 1), linespacing=LINE_SPACING)


def paint_banner(stage: Stage, t: float, *, kicker: str, headline: str, subhead: str = "", note: str = "") -> None:
    """A welcome or thank-you slide: one large headline in the centre, then a line or two below it.

    The kicker and headline come first; the subhead and note follow once they are up.

    Args:
        stage: The canvas.
        t: The chapter's progress, 0 to 1.
        kicker: The small label above the headline.
        headline: A word or two, set very large ("Welcome", "Thank you").
        subhead: A line under the headline; none if empty.
        note: A smaller line under that; none if empty.
    """
    card = stage.show_card()
    first, then = phase(t, 0.0, 0.45), phase(t, 0.45, 1.0)
    width, height = BANNER_RULE
    card.text(CENTRE, BANNER_KICKER_Y, kicker, fontsize=card_pt(KICKER_PT), weight="bold", color=theme.HIGHLIGHT,
              ha="center", va="center", alpha=first)
    card.text(CENTRE, BANNER_HEADLINE_Y, headline, fontsize=card_pt(BANNER_PT), weight="bold", color=theme.HEADING,
              ha="center", va="center", alpha=first)
    card.add_patch(Rectangle((CENTRE - width / 2, BANNER_RULE_Y), width, height, facecolor=theme.HIGHLIGHT,
                             edgecolor="none", alpha=first))
    y = BANNER_SUBHEAD_Y
    if subhead:
        text = wrapped(subhead, BANNER_SUBHEAD_CHARS)
        card.text(CENTRE, y, text, fontsize=card_pt(BANNER_SUBHEAD_PT), weight="bold", color=theme.NAVY, ha="center",
                  va="top", alpha=then, linespacing=LINE_SPACING)
        y -= (text.count("\n") + 1) * line_height(BANNER_SUBHEAD_PT) + GAP
    if note:
        card.text(CENTRE, y, wrapped(note, BANNER_NOTE_CHARS), fontsize=card_pt(BANNER_NOTE_PT),
                  color=theme.INK_SECONDARY, ha="center", va="top", alpha=then, linespacing=LINE_SPACING)
