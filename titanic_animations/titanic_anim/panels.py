"""Classification chart panels. Each owns its matplotlib artists and redraws from plain arrays.

A panel makes its lines, bars and labels once, in ``__init__``, and each ``update`` only changes
their data. That is quicker than drawing the chart again for every video frame, and the axes stay
put, so the eye follows the data rather than the chart. The panels use animkit's theme for their
axes, ink and series colours.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from animkit import theme as base
from titanic_anim import theme
from titanic_anim.config import (
    CLASS_NAMES,
    MARGIN_COLOUR_LIMIT,
    MARGIN_THRESHOLD,
    PROBABILITY_THRESHOLD,
    SCORE_BIN_COUNT,
    SCORE_MARGIN,
)
from titanic_anim.metrics import ClassificationScores, roc_points

HEADROOM = 1.12
RATE_FLOOR = 0.5  # accuracy axes start at 50%, or lower if a model does worse than a coin flip
COUNT_CMAP = LinearSegmentedColormap.from_list("counts", (base.SURFACE, *base.SEQUENTIAL_BLUE))
DARK_CELL = 0.55  # cells darker than this share of the colour range get light text


def percent(share: float) -> str:
    """A share as a percentage with one decimal.

    Args:
        share: A share, from 0 to 1.

    Returns:
        For example ``"38.2%"``.
    """
    return f"{share * 100:.1f}%"


class RateLines:
    """Two rates (e.g. train and validation accuracy) on one percent axis, with end labels.

    The axis runs from ``RATE_FLOOR`` (50%) to 100%; it only goes lower, in whole 5% steps, when a
    rate does, and never comes back up, so the curves do not jump.

    Attributes:
        ax: The axes the rates are drawn on.
        labels: The two rates' names, used in the legend and the end labels.
        lines: The two rate lines.
        markers: A dot on each line's latest point.
        end_labels: Each line's name and latest value, beside its dot.
    """

    def __init__(self, ax: Axes, labels: tuple[str, str], colours=(base.SERIES_1, base.SERIES_2)):
        """Styles the axes and makes the two empty lines, their dots and their labels.

        Args:
            ax: The axes to draw on.
            labels: The two rates' names.
            colours: The two lines' colours.
        """
        self.ax = ax
        base.style_axes(ax)
        self.labels = labels
        self.lines = tuple(ax.plot([], [], color=c, linewidth=2.0, label=label)[0]
                           for c, label in zip(colours, labels))
        self.markers = tuple(ax.plot([], [], "o", color=c, markersize=8, markeredgecolor=base.SURFACE,
                                     markeredgewidth=2.0)[0] for c in colours)
        self.end_labels = tuple(ax.annotate("", (0, 0), xytext=(-8, 0), textcoords="offset points",
                                            ha="right", va="center", fontsize=base.FONT_LABEL,
                                            color=base.INK_SECONDARY, annotation_clip=False)
                                for _ in colours)
        ax.legend(loc="lower right")  # rates climb toward the top, so this corner stays empty
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
        self._y_low = RATE_FLOOR

    def update(self, x: np.ndarray, series: tuple[np.ndarray, np.ndarray], floor_from: float | None = None) -> None:
        """Redraws both rates up to their latest point.

        Args:
            x: The x value of every point.
            series: The two rates at those x values, as shares from 0 to 1.
            floor_from: Only points at x >= this may lower the axis floor, since early rates are noisy;
                None lets any point lower it.
        """
        upper = int(np.argmax([y[-1] for y in series]))
        for index, (line, marker, label, name, y) in enumerate(
            zip(self.lines, self.markers, self.end_labels, self.labels, series)
        ):
            line.set_data(x, y)
            marker.set_data([x[-1]], [y[-1]])
            label.xy = (x[-1], y[-1])
            label.xyann = (-8, 11 if index == upper else -11)  # keep the two labels apart
            label.set_text(f"{name} {percent(y[-1])}")
        settled = x >= floor_from if floor_from is not None else np.ones(len(x), dtype=bool)
        if settled.any():
            lowest = min(float(s[settled].min()) for s in series)
            self._y_low = min(self._y_low, np.floor(lowest * 20) / 20)  # whole 5% steps
        self.ax.set_ylim(self._y_low, 1.0)


class RocCurve:
    """The ROC curve: survivors found against false alarms, as the threshold slides.

    A model that ranks every survivor above every non-survivor hugs the top-left corner; the dashed
    diagonal is random guessing. The title adds the area under the curve (AUC).

    Attributes:
        ax: The axes the curve is drawn on.
        line: The curve.
    """

    def __init__(self, ax: Axes):
        """Styles the axes and draws the guessing diagonal, with an empty curve.

        Args:
            ax: The axes to draw on.
        """
        self.ax = ax
        base.style_axes(ax, grid_axis="both")
        ax.plot((0, 1), (0, 1), linestyle="--", color=base.INK_MUTED, linewidth=1.0)
        ax.text(0.97, 0.04, "dashed line = random guessing", transform=ax.transAxes, ha="right",
                fontsize=base.FONT_LABEL, color=base.INK_MUTED)
        (self.line,) = ax.plot([], [], color=base.SERIES_1, linewidth=2.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("false alarms: non-survivors called survivors")
        ax.set_ylabel("survivors found")

    def update(self, y_true: np.ndarray, y_score: np.ndarray, auc: float, title: str) -> None:
        """Redraws the curve for new scores.

        Args:
            y_true: What really happened, 0 or 1 per passenger.
            y_score: Each passenger's P(survive) or margin.
            auc: The area under the curve, added to the title unless it is NaN.
            title: The panel's title.
        """
        fpr, tpr = roc_points(y_true, y_score)
        self.line.set_data(fpr, tpr)
        self.ax.set_title(f"{title} · AUC {auc:.3f}" if np.isfinite(auc) else title)


class ConfusionMatrixPanel:
    """2x2 counts: rows = what happened, columns = what the model said.

    Each cell shows its count and its share of all passengers, shaded by count relative to the
    largest cell.

    Attributes:
        ax: The axes the matrix is drawn on.
        image: The shaded 2x2 image.
        cells: The four cells' texts, ``cells[row][column]``.
    """

    def __init__(self, ax: Axes):
        """Draws the empty matrix and its labels, sizing the cell text to the cells.

        Args:
            ax: The axes to draw on.
        """
        self.ax = ax
        self.image = ax.imshow(np.zeros((2, 2)), cmap=COUNT_CMAP, vmin=0.0, vmax=1.0)
        short = ("did not\nsurvive", "survived")
        ax.set_xticks((0, 1), short)
        ax.set_yticks((0, 1), short)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        box = ax.get_position()
        cell_pt = min(box.width * ax.figure.get_figwidth(), box.height * ax.figure.get_figheight()) * 72 / 2
        size = min(12.0, max(7.0, cell_pt / 6.5))
        self.cells = [[ax.text(c, r, "", ha="center", va="center", fontsize=size, linespacing=1.4)
                       for c in range(2)] for r in range(2)]

    def update(self, scores: ClassificationScores, title: str) -> None:
        """Redraws the counts; darker cells get light text so it stays readable.

        Args:
            scores: The scores whose confusion counts to show.
            title: The panel's title; the accuracy is added to it.
        """
        counts = scores.confusion
        top = max(int(counts.max()), 1)
        self.image.set_data(counts / top)
        total = int(counts.sum())
        for r in range(2):
            for c in range(2):
                share = counts[r, c] / total if total else 0.0
                text = self.cells[r][c]
                text.set_text(f"{counts[r, c]:,}\n{share:.0%}")
                text.set_color(base.SURFACE if counts[r, c] / top > DARK_CELL else base.INK_PRIMARY)
        self.ax.set_title(f"{title} · {percent(scores.accuracy)} correct")


def _score_range(score_kind: str) -> tuple[float, float, float]:
    """The histogram's (low, high, threshold): 0 … 1 and 0.5, or ±1.5 × the margin colour limit and 0."""
    if score_kind == SCORE_MARGIN:
        return -MARGIN_COLOUR_LIMIT * 1.5, MARGIN_COLOUR_LIMIT * 1.5, MARGIN_THRESHOLD
    return 0.0, 1.0, PROBABILITY_THRESHOLD


class ScoreHistogram:
    """How confident the model is, split by what really happened; the line is the decision threshold.

    A good model piles the survivors (blue) to the right of the line and the others (red) to the left.
    The y-axis only grows, so the bars are not rescaled from frame to frame.

    Attributes:
        ax: The axes the histogram is drawn on.
    """

    def __init__(self, ax: Axes, score_kind: str):
        """Draws two empty step outlines, one per outcome, and the dashed threshold.

        Args:
            ax: The axes to draw on.
            score_kind: ``SCORE_PROBABILITY`` (bins over 0 … 1) or ``SCORE_MARGIN`` (bins around 0).
        """
        self.ax = ax
        base.style_axes(ax)
        low, high, threshold = _score_range(score_kind)
        self._edges = np.linspace(low, high, SCORE_BIN_COUNT + 1)
        self._steps = [ax.stairs(np.zeros(SCORE_BIN_COUNT), self._edges, color=colour, linewidth=2.0,
                                 label=f"actually {name}")
                       for colour, name in zip(theme.OUTCOME_COLOURS, CLASS_NAMES)]
        ax.axvline(threshold, color=base.INK_PRIMARY, linewidth=1.2, linestyle="--")
        ax.set_xlim(low, high)
        ax.set_xlabel("decision value f(x)   (> 0 → survived)" if score_kind == SCORE_MARGIN
                      else "predicted P(survive)   (> 0.5 → survived)")
        ax.set_ylabel("passengers")
        ax.legend(loc="upper center", ncols=2)
        self._y_top = 1.0

    def update(self, y_true: np.ndarray, y_score: np.ndarray, title: str) -> None:
        """Recounts both outcomes' scores; a score outside the bins is counted in the end bin.

        Args:
            y_true: What really happened, 0 or 1 per passenger.
            y_score: Each passenger's P(survive) or margin.
            title: The panel's title.
        """
        clipped = np.clip(y_score, self._edges[0], self._edges[-1] - 1e-9)
        tallest = 1
        for label, stairs in enumerate(self._steps):
            counts, _ = np.histogram(clipped[y_true == label], bins=self._edges)
            stairs.set_data(counts)
            tallest = max(tallest, int(counts.max(initial=0)))
        self._y_top = max(self._y_top, tallest * HEADROOM * 1.25)  # room for the legend
        self.ax.set_ylim(0.0, self._y_top)
        self.ax.set_title(title)
