"""The Algorithm Race figure: one learning curve per model beside a sliding leaderboard.

The left half is small multiples (``LaneGrid``): one small chart per model of validation error
against the share of its schedule done, so models with long and short schedules line up. The right
half is a leaderboard (``Leaderboard``) whose bars re-sort as models overtake each other; between
two ticks they glide to their new rank instead of jumping. A model's colour never changes with its
rank. The sibling textures, podium ordinals and easing come from ``animkit.race_dashboard``.
"""

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory

from animkit.race_dashboard import ORDINALS, TEXTURE_HATCH, TEXTURE_LINESTYLE, ease
from house_anim import theme
from house_anim.config import RACE_FIGSIZE
from house_anim.data import HousingSplits
from house_anim.models import ModelSpec
from house_anim.panels import ProgressBar, add_header, money_k
from house_anim.race import ExamResult, RaceEntry, RaceFrame, colour_slots

BAR_HEIGHT = 0.68
HEADROOM = 1.12
# Tiny-data fits (e.g. linear regression on 50 houses) can miss by $700k+; cap both axes so
# one runaway model doesn't flatten everyone else. The true value is still printed.
ERROR_CAP_K = 250.0


class LaneGrid:
    """Small multiples: each model's validation error against the share of its schedule done.

    Every panel shares one y range, which only grows (capped at ``ERROR_CAP_K``), so the panels
    can be compared at a glance.

    Attributes:
        axes: One axes per model, in a near-square grid.
        lines: Each model's learning curve.
        markers: The dot at each curve's latest point.
        values: Each panel's current validation RMSE readout.
        steps: Each panel's "how far trained" readout.
    """

    def __init__(self, figure, specs: Sequence[ModelSpec], area: tuple[float, float, float, float]):
        """Lays out one styled, empty panel per model.

        Args:
            figure: The figure to draw on.
            specs: The models, in their colour order.
            area: The grid's (left, right, bottom, top), as fractions of the figure.
        """
        self._columns = math.ceil(math.sqrt(len(specs)))
        self._rows = math.ceil(len(specs) / self._columns)
        left, right, bottom, top = area
        grid = figure.add_gridspec(self._rows, self._columns, left=left, right=right,
                                   bottom=bottom, top=top, hspace=0.6, wspace=0.22)
        self.axes = [figure.add_subplot(grid[i // self._columns, i % self._columns])
                     for i in range(len(specs))]
        self.lines, self.markers, self.values, self.steps = [], [], [], []
        for index, (ax, spec, style) in enumerate(zip(self.axes, specs, colour_slots(specs))):
            self._style(ax, spec, index, style)
        self._y_top = 1.0

    def _style(self, ax: Axes, spec: ModelSpec, index: int, style: tuple[int, bool]) -> None:
        """Styles one panel and adds its curve, marker and readouts; only outer panels keep tick labels."""
        theme.style_axes(ax)
        color = theme.series_color(style[0])
        linestyle = TEXTURE_LINESTYLE if style[1] else "-"
        self.lines.append(ax.plot([], [], color=color, linewidth=2.0, linestyle=linestyle)[0])
        self.markers.append(ax.plot([], [], "o", color=color, markersize=8,
                                    markeredgecolor=theme.SURFACE, markeredgewidth=2.0)[0])
        self.values.append(ax.text(0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top",
                                   fontsize=12, weight="bold", color=theme.INK_PRIMARY))
        self.steps.append(ax.text(0.98, 0.76, "", transform=ax.transAxes, ha="right", va="top",
                                  fontsize=theme.FONT_LABEL, color=theme.INK_SECONDARY))
        ax.set_title(spec.short_title)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0], ["0%", "50%", "100%"])
        if index + self._columns < len(self.axes):  # another panel sits below this one
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("training schedule done")
        if index % self._columns:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("val RMSE (\\$k)")

    def update(self, entries: Sequence[RaceEntry]) -> None:
        """Redraws every panel's curve and readouts.

        Args:
            entries: One entry per model, in the order the panels were made.
        """
        tallest = max(e.history_val_k.max() for e in entries) * HEADROOM
        self._y_top = min(max(self._y_top, tallest), ERROR_CAP_K)
        for i, entry in enumerate(entries):
            self.lines[i].set_data(entry.history_progress, entry.history_val_k)
            self.markers[i].set_data([entry.history_progress[-1]], [entry.history_val_k[-1]])
            self.values[i].set_text(money_k(entry.val_rmse_k))
            self.steps[i].set_text(entry.step_text + ("  ·  done" if entry.finished else ""))
            self.axes[i].set_ylim(0.0, self._y_top)


class Leaderboard:
    """Horizontal bars sorted best-first; positions and lengths ease between ticks.

    Each bar belongs to one model and keeps its colour; only its y position (its rank) and length
    (its validation RMSE) change. A bar longer than the axis stops short of the edge, but its label
    still prints the true value.

    Attributes:
        ax: The leaderboard's axes.
        bars: One bar per model, in the order the models were given.
        value_texts: Each bar's value label.
        positions: Each bar's current y position (0 is the top rank), mid-glide between ticks.
    """

    def __init__(self, ax: Axes, specs: Sequence[ModelSpec]):
        """Draws empty bars, the model names and the rank numbers.

        Args:
            ax: The axes to draw in.
            specs: The models, in their colour order.
        """
        self.ax = ax
        n = len(specs)
        theme.style_axes(ax, grid_axis="x")
        styles = colour_slots(specs)
        self.bars = ax.barh(np.arange(n), np.zeros(n), height=BAR_HEIGHT,
                            color=[theme.series_color(slot) for slot, _ in styles])
        for bar, (_, textured) in zip(self.bars, styles):
            if textured:
                bar.set_hatch(TEXTURE_HATCH)
                bar.set_edgecolor(theme.SURFACE)  # hatch lines take the edge colour
        names_at = blended_transform_factory(ax.transAxes, ax.transData)
        self._names = [ax.text(-0.03, i, spec.short_title, transform=names_at, ha="right",
                               va="center", fontsize=11, color=theme.INK_PRIMARY)
                       for i, spec in enumerate(specs)]
        for rank in range(n):
            ax.text(-0.5, rank, f"{rank + 1}", transform=names_at, ha="right", va="center",
                    fontsize=11, weight="bold", color=theme.INK_MUTED)
        self.value_texts = [ax.text(0.0, i, "", va="center", fontsize=10,
                                    color=theme.INK_SECONDARY) for i in range(n)]
        ax.set_ylim(n - 0.5, -0.5)  # rank 1 on top
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("validation RMSE (\\$k) · shorter is better")
        self.positions = np.arange(n, dtype=float)
        self._widths = np.zeros(n)
        self._x_max = 1.0
        self._start = (self.positions, self._widths, self._x_max)
        self._tick = None

    def update(self, frame: RaceFrame, blend: float) -> None:
        """Moves the bars ``blend`` of the way from where they were when this tick began to its standings.

        Args:
            frame: The race at this tick.
            blend: How far through the glide, from 0 (where the bars were) to 1 (this tick's
                standings); eased so the bars start and stop gently.
        """
        if frame.tick != self._tick:  # a new tick: ease from wherever the bars are now
            self._start = (self.positions, self._widths, self._x_max)
            self._tick = frame.tick
        rank_of = {entry.key: rank for rank, entry in enumerate(frame.standings)}
        target_pos = np.array([rank_of[e.key] for e in frame.entries], dtype=float)
        target_widths = np.array([e.val_rmse_k for e in frame.entries])
        target_max = min(target_widths.max(), ERROR_CAP_K) * 1.25
        t = ease(blend)
        start_pos, start_widths, start_max = self._start
        self.positions = start_pos + (target_pos - start_pos) * t
        self._widths = start_widths + (target_widths - start_widths) * t
        self._x_max = start_max + (target_max - start_max) * t
        self._draw(frame.entries)

    def _draw(self, entries: Sequence[RaceEntry]) -> None:
        """Places every bar, name and value label at the current positions and widths."""
        pad = self._x_max * 0.015
        longest = self._x_max * 0.8  # runaway bars stop short of the edge, label still true
        for bar, name, value, entry, y, width in zip(
            self.bars, self._names, self.value_texts, entries, self.positions, self._widths
        ):
            shown = min(width, longest)
            bar.set_y(y - BAR_HEIGHT / 2)
            bar.set_width(shown)
            name.set_y(y)
            value.set_position((shown + pad, y))
            value.set_text(money_k(width) + ("  ✓" if entry.finished else ""))
        self.ax.set_xlim(0.0, self._x_max)


class RaceDashboard:
    """The whole race figure: header, progress bar, status line, lane grid, leaderboard and podium card.

    Attributes:
        figure: The matplotlib figure the sinks record.
        progress: The race's progress bar.
        status_text: The line naming the current leader.
        lanes: The small-multiples learning curves.
        leaderboard: The sliding leaderboard.
        results_title: The podium card's title, hidden until the final exam.
        results_text: The podium card's body, hidden until the final exam.

    Example:
        >>> dashboard = RaceDashboard(specs, splits)  # doctest: +SKIP
        >>> dashboard.render(frame, blend=0.5)  # doctest: +SKIP
    """

    def __init__(self, specs: Sequence[ModelSpec], splits: HousingSplits, figsize=RACE_FIGSIZE):
        """Builds the figure with every panel empty.

        Args:
            specs: The models in the race.
            splits: The data; its training row count goes in the subtitle.
            figsize: The figure size, in inches.
        """
        plt.rcParams.update(theme.RC_PARAMS)
        self.figure = plt.figure(figsize=figsize)
        add_header(self.figure, "The Algorithm Race · who prices California houses best?",
                   f"{len(specs)} scikit-learn regressors train side by side on the same "
                   f"{splits.n_train:,} houses. Every frame shows each model equally far "
                   "through its own training schedule.")
        self.progress = ProgressBar(self.figure)
        self.status_text = self.figure.text(0.04, 0.855, "", fontsize=theme.FONT_SUBTITLE,
                                            color=theme.INK_PRIMARY)
        self.lanes = LaneGrid(self.figure, specs, area=(0.05, 0.60, 0.09, 0.76))
        self.figure.text(0.655, 0.795, "Leaderboard · validation error", fontsize=theme.FONT_AXIS_TITLE,
                         weight="semibold", color=theme.INK_PRIMARY)
        self.leaderboard = Leaderboard(self.figure.add_axes((0.755, 0.09, 0.215, 0.67)), specs)
        self.results_title, self.results_text = self._build_results_card()

    def _build_results_card(self):
        """Adds the hidden podium card's title and body texts; returns (title, body)."""
        box = dict(boxstyle="round,pad=1.1", facecolor=theme.SURFACE, edgecolor=theme.BASELINE)
        title = self.figure.text(0.325, 0.495, "", ha="center", va="center", fontsize=16, weight="bold",
                                 color=theme.INK_PRIMARY, visible=False, zorder=10)
        body = self.figure.text(0.325, 0.42, "", ha="center", va="center", fontsize=14,
                                linespacing=1.9, color=theme.INK_PRIMARY, bbox=box,
                                visible=False, zorder=9)
        return title, body

    @property
    def bar_positions(self) -> tuple[float, ...]:
        """Where the leaderboard's bars are now (the tests read it to check the glide).

        Returns:
            Each bar's y position, in the order the models were given; 0 is the top rank.
        """
        return tuple(self.leaderboard.positions)

    def render(self, frame: RaceFrame, blend: float = 1.0) -> None:
        """Draws one tick, with the leaderboard ``blend`` of the way through its glide.

        Args:
            frame: The race at this tick.
            blend: How far the leaderboard's bars have glided towards this tick's standings, 0 to 1.
        """
        self.progress.update(frame.tick / frame.ticks)
        self.lanes.update(frame.entries)
        self.leaderboard.update(frame, blend)
        leader = frame.leader
        self.status_text.set_text(
            f"Race {frame.tick / frame.ticks:.0%} complete   ·   "
            f"leader: {leader.short_title} at {money_k(leader.val_rmse_k)} validation RMSE"
        )

    def show_results(self, exam: Sequence[ExamResult], n_test: int) -> None:
        """Shows the podium card: the top three on the final exam.

        Args:
            exam: The final exam's results, best first (only the first three are shown).
            n_test: How many test houses the exam used, for the title.
        """
        lines = [f"{place}   {r.short_title}   ·   test RMSE \\${r.test_rmse_k:.1f}k   ·   R² {r.test_r2:.3f}"
                 for place, r in zip(ORDINALS, exam)]
        self.results_title.set_text(f"Final exam on {n_test:,} unseen test houses")
        self.results_text.set_text("\n\n" + "\n".join(lines))
        self.results_title.set_visible(True)
        self.results_text.set_visible(True)
