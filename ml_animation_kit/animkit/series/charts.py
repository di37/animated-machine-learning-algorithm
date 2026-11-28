"""The series' data cards: real rows as a scatter, a scoreboard of bars, and each model before and after.

They follow the lessons' truthfulness rules. A bar appears at its real length (bars are revealed in
turn, never grown through values no model scored), and the "every input" score appears beside the
one-input score instead of sliding there, since nothing was measured in between.

- ``paint_scatter``: the dataset's rows, revealed progressively in a plot box, with facts beside it.
- ``paint_bars``: one bar per model (and a grey baseline), each the length of its real score.
- ``paint_shifts``: per model, its score with one input and then with every input, on one scale.
"""

import math
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle

from animkit.lessons import theme
from animkit.lessons.stage import Stage
from animkit.lessons.timing import phase
from animkit.series.cards import HEADING_SHARE, LEFT, begin, card_pt, heading, turn, wrapped

BOX = (0.08, 0.13, 0.5, 0.52)  # the scatter's panel: left, bottom, width, height, in card units
PLOT_MARGINS = (0.065, 0.075, 0.02, 0.06)  # inside the panel: left, bottom, right, top
POINT_PT = 6.0
FACTS_X, FACTS_TOP, FACTS_CHARS = 0.63, 0.6, 34
ROWS_TOP, ROWS_SPACE, MAX_ROW_STEP = 0.6, 0.44, 0.06  # the bar and shift rows share this band
BAR_X, BAR_SPAN, BAR_HEIGHT = 0.4, 0.4, 0.032
SHIFT_X, SHIFT_SPAN, SHIFT_PAD = 0.42, 0.38, 0.06
SCORE_GAP = 0.012  # between a dot and its score
VERDICT_Y, VERDICT_CHARS = 0.13, 110
BAR_COLOURS = {"model": theme.MODEL, "baseline": theme.DOTS}
SHIFT_START = 0.6  # "every input" scores start appearing at this share of the motion


@dataclass(frozen=True)
class Cloud:
    """Rows to scatter, already placed in the plot box (0 to 1 on each axis).

    Attributes:
        x: Each row's position across, 0 to 1.
        y: Each row's position up, 0 to 1.
        groups: Each row's group, an index into ``colours`` and ``legend``.
        colours: One colour per group.
        legend: One name per group.
        x_label: What across means.
        y_label: What up means.
        x_ticks: (position, label) pairs along the bottom.
        y_ticks: (position, label) pairs up the side.
        markers: One marker per group ("o", "X" …); dots for every group if empty.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    groups: tuple[int, ...]
    colours: tuple[str, ...]
    legend: tuple[str, ...]
    x_label: str
    y_label: str
    x_ticks: tuple[tuple[float, str], ...] = ()
    y_ticks: tuple[tuple[float, str], ...] = ()
    markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuses rows that are not all placed inside the box, or groups without a colour and name.

        Raises:
            ValueError: If the lengths differ, a position is outside 0 to 1, or a group has no colour.
        """
        if not len(self.x) == len(self.y) == len(self.groups):
            raise ValueError("x, y and groups need one value per row")
        if any(not 0.0 <= value <= 1.0 for value in (*self.x, *self.y)):
            raise ValueError("every row must be placed inside the box, 0 to 1 on each axis")
        if len(self.colours) != len(self.legend) or any(not 0 <= g < len(self.colours) for g in self.groups):
            raise ValueError("every group needs a colour and a name")
        if self.markers and len(self.markers) != len(self.colours):
            raise ValueError("give every group a marker, or none")

    def marker(self, group: int) -> str:
        """The group's marker: its own, or a dot."""
        return self.markers[group] if self.markers else "o"


@dataclass(frozen=True)
class Bar:
    """One scoreboard bar: whose score, its value (sets the length), how it is printed, and its role."""

    label: str
    value: float
    text: str
    role: str = "model"  # "model" or "baseline"


@dataclass(frozen=True)
class Shift:
    """One model's score with one input (``before``) and with every input (``after``), and their texts."""

    label: str
    before: float
    after: float
    before_text: str
    after_text: str


def _plot_area() -> tuple[float, float, float, float]:
    """The scatter's plotting area inside its panel: left, bottom, width, height."""
    left, bottom, width, height = BOX
    m_left, m_bottom, m_right, m_top = PLOT_MARGINS
    return left + m_left, bottom + m_bottom, width - m_left - m_right, height - m_bottom - m_top


def _scatter_frame(card: Axes, cloud: Cloud, alpha: float) -> None:
    """The scatter's panel, axis labels, ticks and legend."""
    left, bottom, width, height = BOX
    card.add_patch(FancyBboxPatch((left, bottom), width, height, boxstyle="round,pad=0,rounding_size=0.012",
                                  facecolor=theme.PANEL, edgecolor=theme.PANEL_EDGE, linewidth=1.5))
    x0, y0, w, h = _plot_area()
    for position, label in cloud.x_ticks:
        card.text(x0 + position * w, y0 - 0.02, label, fontsize=card_pt(13), color=theme.INK_MUTED, ha="center",
                  va="top", alpha=alpha)
    for position, label in cloud.y_ticks:
        card.text(x0 - 0.008, y0 + position * h, label, fontsize=card_pt(13), color=theme.INK_MUTED, ha="right",
                  va="center", alpha=alpha)
    card.text(x0 + w / 2, bottom + 0.012, cloud.x_label, fontsize=card_pt(15), color=theme.NAVY, ha="center",
              va="bottom", alpha=alpha)
    card.text(left + 0.012, y0 + h / 2, cloud.y_label, fontsize=card_pt(15), color=theme.NAVY, rotation=90,
              ha="left", va="center", alpha=alpha)
    for index, (colour, name) in enumerate(zip(cloud.colours, cloud.legend)):
        x = x0 + index * 0.13
        card.scatter([x], [bottom + height - 0.03], s=card_pt(9) ** 2, color=colour, marker=cloud.marker(index),
                     gid="legend", alpha=alpha)
        card.text(x + 0.01, bottom + height - 0.03, name, fontsize=card_pt(14), color=theme.NAVY, va="center",
                  alpha=alpha)


def paint_scatter(stage: Stage, t: float, *, kicker: str, headline: str, cloud: Cloud,
                  facts: tuple[str, ...]) -> None:
    """The dataset's rows, revealed progressively in a plot box, with facts appearing beside it.

    Args:
        stage: The canvas.
        t: The chapter's progress, 0 to 1.
        kicker: The small label above the headline.
        headline: The card's main words.
        cloud: The rows, placed in the box.
        facts: Lines beside the box, in order.
    """
    card = begin(stage)
    alpha = phase(t, 0.0, HEADING_SHARE)
    heading(card, kicker, headline, alpha)
    _scatter_frame(card, cloud, alpha)
    x0, y0, w, h = _plot_area()
    count = math.ceil(len(cloud.x) * phase(t, HEADING_SHARE, 0.85))
    for group, colour in enumerate(cloud.colours):
        rows = [i for i in range(count) if cloud.groups[i] == group]
        card.scatter([x0 + cloud.x[i] * w for i in rows], [y0 + cloud.y[i] * h for i in rows],
                     s=card_pt(POINT_PT) ** 2 / 4, color=colour, marker=cloud.marker(group), alpha=0.75,
                     linewidths=0, gid="rows")
    y = FACTS_TOP
    for index, fact in enumerate(facts):
        text = wrapped(fact, FACTS_CHARS)
        card.text(FACTS_X, y, text, fontsize=card_pt(22), color=theme.NAVY, va="top", linespacing=1.3,
                  alpha=turn(t, index, len(facts), start=0.4))
        y -= (text.count("\n") + 1) * 0.042 + 0.035


def _rows_step(count: int) -> float:
    """The distance between two rows, so ``count`` rows fit the band."""
    return min(MAX_ROW_STEP, ROWS_SPACE / max(count, 1))


def _metric_and_verdict(card: Axes, metric: str, verdict: str, t: float) -> None:
    """The metric's name under the headline, and the verdict at the bottom once every row is shown."""
    card.text(LEFT, ROWS_TOP + 0.05, metric, fontsize=card_pt(17), color=theme.INK_SECONDARY,
              alpha=phase(t, 0.0, HEADING_SHARE))
    card.text(LEFT, VERDICT_Y, wrapped(verdict, VERDICT_CHARS), fontsize=card_pt(21), weight="bold",
              color=theme.NAVY, va="top", linespacing=1.3, alpha=phase(t, 0.9, 1.0))


def paint_bars(stage: Stage, t: float, *, kicker: str, headline: str, metric: str, bars: tuple[Bar, ...],
               verdict: str) -> None:
    """A scoreboard: one bar per model, revealed in turn at its real length, then the verdict.

    Args:
        stage: The canvas.
        t: The chapter's progress, 0 to 1.
        kicker: The small label above the headline.
        headline: The card's main words.
        metric: What the bars measure, saying which way is better.
        bars: The bars, top to bottom.
        verdict: What the scoreboard shows, in one or two sentences.
    """
    card = begin(stage)
    heading(card, kicker, headline, phase(t, 0.0, HEADING_SHARE))
    _metric_and_verdict(card, metric, verdict, t)
    top, step = max(bar.value for bar in bars) or 1.0, _rows_step(len(bars))
    shown = sum(1 for index in range(len(bars)) if turn(t, index, len(bars), end=0.9) > 0)
    for index, bar in enumerate(bars[:shown]):
        y = ROWS_TOP - index * step
        width = BAR_SPAN * bar.value / top
        card.text(LEFT, y, bar.label, fontsize=card_pt(18), weight="bold", color=theme.NAVY, va="center")
        card.add_patch(Rectangle((BAR_X, y - BAR_HEIGHT / 2), width, BAR_HEIGHT, facecolor=BAR_COLOURS[bar.role],
                                 edgecolor="none", gid="bar"))
        card.text(BAR_X + width + 0.008, y, bar.text, fontsize=card_pt(18), weight="bold", color=theme.NAVY,
                  va="center")


def _shift_scale(shifts: tuple[Shift, ...]) -> tuple[float, float]:
    """The value range the shift rows share, padded so no dot sits on an edge."""
    values = [value for shift in shifts for value in (shift.before, shift.after)]
    low, high = min(values), max(values)
    pad = (high - low) * SHIFT_PAD or 1.0
    return low - pad, high + pad


def _place(value: float, scale: tuple[float, float]) -> float:
    """Where a score sits across the card, on the rows' shared scale."""
    low, high = scale
    return SHIFT_X + SHIFT_SPAN * (value - low) / (high - low)


def _score(card: Axes, x: float, y: float, text: str, *, outward: bool, strong: bool) -> None:
    """A score beside its dot, on the side away from the row's other dot, so the two never collide."""
    card.text(x + SCORE_GAP if outward else x - SCORE_GAP, y, text, va="center", ha="left" if outward else "right",
              fontsize=card_pt(16 if strong else 14), weight="bold" if strong else "normal",
              color=theme.NAVY if strong else theme.INK_MUTED)


def paint_shifts(stage: Stage, t: float, *, kicker: str, headline: str, metric: str,
                 shifts: tuple[Shift, ...], verdict: str) -> None:
    """Each model's one-input score (grey) in turn, then its every-input score (teal) beside it.

    Args:
        stage: The canvas.
        t: The chapter's progress, 0 to 1.
        kicker: The small label above the headline.
        headline: The card's main words.
        metric: What the scores measure, saying which way is better.
        shifts: One row per model, top to bottom.
        verdict: What the rows show, in one or two sentences.
    """
    card = begin(stage)
    heading(card, kicker, headline, phase(t, 0.0, HEADING_SHARE))
    _metric_and_verdict(card, metric, verdict, t)
    scale = _shift_scale(shifts)
    step = _rows_step(len(shifts))
    for index, shift in enumerate(shifts):
        y = ROWS_TOP - index * step
        if turn(t, index, len(shifts), end=SHIFT_START) <= 0:
            continue
        before, after = _place(shift.before, scale), _place(shift.after, scale)
        card.text(LEFT, y, shift.label, fontsize=card_pt(18), weight="bold", color=theme.NAVY, va="center")
        card.scatter([before], [y], s=card_pt(11) ** 2, color=theme.DOTS, gid="before", zorder=3)
        _score(card, before, y, shift.before_text, outward=before >= after, strong=False)
        if turn(t, index, len(shifts), start=SHIFT_START, end=0.9) <= 0:
            continue
        card.plot([before, after], [y, y], color=theme.HIGHLIGHT, linewidth=3, zorder=2)
        card.scatter([after], [y], s=card_pt(11) ** 2, color=theme.HIGHLIGHT, gid="after", zorder=3)
        _score(card, after, y, shift.after_text, outward=after > before, strong=True)
