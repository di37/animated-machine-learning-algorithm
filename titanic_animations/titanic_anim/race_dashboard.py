"""The classifier race figure: one error curve per model beside a sliding leaderboard.

On the left, a small chart per model plots its validation error against the share of its schedule
done, so every model's x-axis runs 0 … 100% and the curves can be compared. On the right, the
leaderboard ranks the models by validation error. Between ticks its bars glide to their new rank
and length, eased by animkit's smoothstep ``ease``. At the end, a card shows the top three in the
test exam.
"""

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from animkit import theme as base
from animkit.config import RACE_FIGSIZE
from animkit.panels import ProgressBar, add_header
from animkit.race_dashboard import ORDINALS, TEXTURE_HATCH, TEXTURE_LINESTYLE, ease
from titanic_anim.data import TitanicSplits
from titanic_anim.models import ClassifierSpec
from titanic_anim.panels import percent
from titanic_anim.race import ExamResult, RaceEntry, RaceFrame, colour_slots

BAR_HEIGHT = 0.68
HEADROOM = 1.12


def axis_percent(value: float, _position) -> str:
    """A matplotlib tick formatter that shows a share as a whole percentage.

    Args:
        value: The tick's value, a share.
        _position: The tick's position, which matplotlib passes and the label ignores.

    Returns:
        For example ``"25%"``.
    """
    return f"{value * 100:.0f}%"


class LaneGrid:
    """Small multiples: each model's validation error against the share of its schedule done.

    Every panel shares one y-axis top, which only grows (to at most 100%), so a taller curve means a
    worse model.

    Attributes:
        axes: One axes per model.
        lines: Each model's error curve.
        markers: A dot on each curve's latest point.
        values: Each panel's latest error, as text.
        steps: Each panel's place in the schedule, as text.
    """

    def __init__(self, figure, specs: Sequence[ClassifierSpec], area: tuple[float, float, float, float]):
        """Lays out one panel per model in a near-square grid.

        Args:
            figure: The figure to draw in.
            specs: The racing classifiers, one panel each.
            area: ``(left, right, bottom, top)`` of the grid, in figure fractions.
        """
        self._columns = math.ceil(math.sqrt(len(specs)))
        rows = math.ceil(len(specs) / self._columns)
        left, right, bottom, top = area
        grid = figure.add_gridspec(rows, self._columns, left=left, right=right, bottom=bottom, top=top,
                                   hspace=0.6, wspace=0.22)
        self.axes = [figure.add_subplot(grid[i // self._columns, i % self._columns]) for i in range(len(specs))]
        self.lines, self.markers, self.values, self.steps = [], [], [], []
        for index, (ax, spec, style) in enumerate(zip(self.axes, specs, colour_slots(specs))):
            self._style(ax, spec, index, style)
        self._y_top = 0.1

    def _style(self, ax: Axes, spec: ClassifierSpec, index: int, style: tuple[int, bool]) -> None:
        """Styles one panel and adds its line, dot and labels; only the edge panels keep tick labels."""
        base.style_axes(ax)
        color = base.series_color(style[0])
        self.lines.append(ax.plot([], [], color=color, linewidth=2.0,
                                  linestyle=TEXTURE_LINESTYLE if style[1] else "-")[0])
        self.markers.append(ax.plot([], [], "o", color=color, markersize=8, markeredgecolor=base.SURFACE,
                                    markeredgewidth=2.0)[0])
        self.values.append(ax.text(0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top",
                                   fontsize=12, weight="bold", color=base.INK_PRIMARY))
        self.steps.append(ax.text(0.98, 0.76, "", transform=ax.transAxes, ha="right", va="top",
                                  fontsize=base.FONT_LABEL, color=base.INK_SECONDARY))
        ax.set_title(spec.short_title)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0], ["0%", "50%", "100%"])
        ax.yaxis.set_major_formatter(axis_percent)
        if index + self._columns < len(self.axes):  # another panel sits below this one
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("training schedule done")
        if index % self._columns:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("val error")

    def update(self, entries: Sequence[RaceEntry]) -> None:
        """Redraws every model's curve and labels.

        Args:
            entries: Every model's standing, in the order of the panels.
        """
        self._y_top = min(max(self._y_top, max(e.history_error.max() for e in entries) * HEADROOM), 1.0)
        for i, entry in enumerate(entries):
            self.lines[i].set_data(entry.history_progress, entry.history_error)
            self.markers[i].set_data([entry.history_progress[-1]], [entry.history_error[-1]])
            self.values[i].set_text(f"{percent(entry.val_error)} wrong")
            self.steps[i].set_text(entry.step_text + ("  ·  done" if entry.finished else ""))
            self.axes[i].set_ylim(0.0, self._y_top)


class Leaderboard:
    """Horizontal bars of validation error, sorted best-first; positions and lengths ease between ticks.

    Each bar keeps its model's colour; only its row changes with its rank.

    Attributes:
        ax: The axes the bars are drawn on.
        bars: One bar per model, in the order the models were given.
        value_texts: Each bar's error label.
        positions: Each bar's current row, between ranks while it moves.
    """

    def __init__(self, ax: Axes, specs: Sequence[ClassifierSpec]):
        """Draws empty bars in the models' colours, with their names and the rank numbers.

        Args:
            ax: The axes to draw on.
            specs: The racing classifiers.
        """
        self.ax = ax
        n = len(specs)
        base.style_axes(ax, grid_axis="x")
        styles = colour_slots(specs)
        self.bars = ax.barh(np.arange(n), np.zeros(n), height=BAR_HEIGHT,
                            color=[base.series_color(slot) for slot, _ in styles])
        for bar, (_, textured) in zip(self.bars, styles):
            if textured:
                bar.set_hatch(TEXTURE_HATCH)
                bar.set_edgecolor(base.SURFACE)  # hatch lines take the edge colour
        names_at = blended_transform_factory(ax.transAxes, ax.transData)
        self._names = [ax.text(-0.03, i, spec.short_title, transform=names_at, ha="right", va="center",
                               fontsize=11, color=base.INK_PRIMARY) for i, spec in enumerate(specs)]
        for rank in range(n):
            ax.text(-0.62, rank, f"{rank + 1}", transform=names_at, ha="right", va="center",
                    fontsize=11, weight="bold", color=base.INK_MUTED)
        self.value_texts = [ax.text(0.0, i, "", va="center", fontsize=10, color=base.INK_SECONDARY)
                            for i in range(n)]
        ax.set_ylim(n - 0.5, -0.5)  # rank 1 on top
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(axis_percent)
        ax.set_xlabel("validation error (passengers misclassified) · shorter is better")
        self.positions = np.arange(n, dtype=float)
        self._widths = np.zeros(n)
        self._x_max = 0.1
        self._start = (self.positions, self._widths, self._x_max)
        self._tick = None

    def update(self, frame: RaceFrame, blend: float) -> None:
        """Moves the bars part of the way towards this tick's ranks and errors.

        Args:
            frame: The race at this tick.
            blend: How far from the previous tick's picture to this one, from 0 to 1 (eased).
        """
        if frame.tick != self._tick:  # a new tick: ease from wherever the bars are now
            self._start = (self.positions, self._widths, self._x_max)
            self._tick = frame.tick
        rank_of = {entry.key: rank for rank, entry in enumerate(frame.standings)}
        target_pos = np.array([rank_of[e.key] for e in frame.entries], dtype=float)
        target_widths = np.array([e.val_error for e in frame.entries])
        target_max = max(float(target_widths.max()), 0.05) * 1.3
        t = ease(blend)
        start_pos, start_widths, start_max = self._start
        self.positions = start_pos + (target_pos - start_pos) * t
        self._widths = start_widths + (target_widths - start_widths) * t
        self._x_max = start_max + (target_max - start_max) * t
        self._draw(frame.entries)

    def _draw(self, entries: Sequence[RaceEntry]) -> None:
        """Moves every bar, name and value label to its current eased row and width."""
        pad = self._x_max * 0.015
        for bar, name, value, entry, y, width in zip(
            self.bars, self._names, self.value_texts, entries, self.positions, self._widths
        ):
            bar.set_y(y - BAR_HEIGHT / 2)
            bar.set_width(width)
            name.set_y(y)
            value.set_position((width + pad, y))
            value.set_text(percent(width) + ("  ✓" if entry.finished else ""))
        self.ax.set_xlim(0.0, self._x_max)


class RaceDashboard:
    """The race figure: header, progress bar, status line, lane grid, leaderboard and results card.

    Attributes:
        figure: The matplotlib figure, which animkit's sinks show and record.
        progress: The bar showing how far through the race it is.
        status_text: The line naming the leader.
        lanes: Each model's error curve.
        leaderboard: The ranked bars.
        results_title: The results card's title, hidden until the end.
        results_text: The results card's podium, hidden until the end.
    """

    def __init__(self, specs: Sequence[ClassifierSpec], splits: TitanicSplits, figsize=RACE_FIGSIZE):
        """Builds the figure and every panel, empty until the first ``render``.

        Args:
            specs: The racing classifiers.
            splits: The splits; the header gives the number of training passengers.
            figsize: The figure's size in inches.
        """
        plt.rcParams.update(base.RC_PARAMS)
        self.figure = plt.figure(figsize=figsize)
        add_header(self.figure, "The Algorithm Race · who predicts Titanic survival best?",
                   f"{len(specs)} scikit-learn classifiers train side by side on the same "
                   f"{splits.n_train:,} passengers. Every frame shows each model equally far "
                   "through its own training schedule.")
        self.progress = ProgressBar(self.figure)
        self.status_text = self.figure.text(0.04, 0.855, "", fontsize=base.FONT_SUBTITLE, color=base.INK_PRIMARY)
        self.lanes = LaneGrid(self.figure, specs, area=(0.05, 0.60, 0.09, 0.76))
        self.figure.text(0.655, 0.795, "Leaderboard · validation error", fontsize=base.FONT_AXIS_TITLE,
                         weight="semibold", color=base.INK_PRIMARY)
        self.leaderboard = Leaderboard(self.figure.add_axes((0.79, 0.09, 0.18, 0.67)), specs)
        self.results_title, self.results_text = self._build_results_card()

    def _build_results_card(self):
        """The hidden title and text box that show the final exam's podium."""
        box = dict(boxstyle="round,pad=1.1", facecolor=base.SURFACE, edgecolor=base.BASELINE)
        title = self.figure.text(0.325, 0.495, "", ha="center", va="center", fontsize=16, weight="bold",
                                 color=base.INK_PRIMARY, visible=False, zorder=10)
        body = self.figure.text(0.325, 0.42, "", ha="center", va="center", fontsize=14, linespacing=1.9,
                                color=base.INK_PRIMARY, bbox=box, visible=False, zorder=9)
        return title, body

    @property
    def bar_positions(self) -> tuple[float, ...]:
        """Each leaderboard bar's current row.

        Returns:
            The rows, in the order the models were given.
        """
        return tuple(self.leaderboard.positions)

    def render(self, frame: RaceFrame, blend: float = 1.0) -> None:
        """Draws the race at one tick, the leaderboard ``blend`` of the way to its new order.

        Args:
            frame: The race at this tick.
            blend: How far the leaderboard has moved from the previous tick, from 0 to 1.
        """
        self.progress.update(frame.tick / frame.ticks)
        self.lanes.update(frame.entries)
        self.leaderboard.update(frame, blend)
        leader = frame.leader
        self.status_text.set_text(
            f"Race {frame.tick / frame.ticks:.0%} complete   ·   leader: {leader.short_title} at "
            f"{percent(leader.val_accuracy)} validation accuracy ({percent(leader.val_error)} misclassified)"
        )

    def show_results(self, exam: Sequence[ExamResult], n_test: int) -> None:
        """Shows the top three of the final exam.

        Args:
            exam: Every model's exam result, best first.
            n_test: How many test passengers sat the exam.
        """
        lines = [f"{place}   {r.short_title}   ·   test accuracy {percent(r.test_accuracy)}   ·   "
                 f"AUC {r.test_auc:.3f}" for place, r in zip(ORDINALS, exam)]
        self.results_title.set_text(f"Final exam on {n_test:,} unseen test passengers · top 3")
        self.results_text.set_text("\n\n" + "\n".join(lines))
        self.results_title.set_visible(True)
        self.results_text.set_visible(True)
