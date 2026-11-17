"""Reusable house-price dashboard panels. Each owns its matplotlib artists and redraws from plain arrays.

The header, progress bar and step labels every dataset shares come from animkit.panels.

A panel is made once, with its empty artists, and then only updated: each ``update`` sets new data
on the same artists, which is much faster than clearing and redrawing the axes every frame. Values
arrive in $k. Labels put a backslash before every dollar sign, because matplotlib reads a bare
``$`` as the start of a maths expression.
"""

import numpy as np
from matplotlib.axes import Axes

from animkit.panels import ProgressBar, add_header, format_step
from house_anim import theme
from house_anim.config import GOOD_ENOUGH_K, PRICE_AXIS_MAX_K, RESIDUAL_BIN_COUNT, RESIDUAL_RANGE_K
from house_anim.metrics import share_within

__all__ = ["HEADROOM", "PredictedVsActual", "ProgressBar", "ResidualHistogram", "TwoLineChart", "add_header",
           "format_step", "money_k", "signed_money_k"]

HEADROOM = 1.12


def money_k(value_k: float) -> str:
    """A price in $k as a matplotlib label, such as $245k.

    Args:
        value_k: The price, in $k.

    Returns:
        The label, rounded to the nearest $1k, with the dollar sign escaped for matplotlib.
    """
    return f"\\${value_k:,.0f}k"


def signed_money_k(value_k: float) -> str:
    """A price change in $k with its sign, such as +$12k or −$3k (with a typographic minus).

    Args:
        value_k: The change, in $k.

    Returns:
        The label, rounded to the nearest $1k, with the dollar sign escaped; zero gets a plus sign.
    """
    sign = "+" if value_k >= 0 else "−"
    return f"{sign}\\${abs(value_k):,.0f}k"


class PredictedVsActual:
    """A scatter of predicted against actual price, with the dashed diagonal of perfect predictions.

    The better the model, the tighter the cloud hugs the diagonal. Both axes run from $0 to
    ``PRICE_AXIS_MAX_K``; the census caps values at $500k, which shows as a column of points there.

    Attributes:
        ax: The panel's axes.
        scatter: The points, one per house.
    """

    def __init__(self, ax: Axes):
        """Draws the empty panel: the diagonal, its legend note and the axis labels.

        Args:
            ax: The axes to draw in.
        """
        self.ax = ax
        theme.style_axes(ax, grid_axis="both")
        limit = (0.0, PRICE_AXIS_MAX_K)
        ax.plot(limit, limit, linestyle="--", color=theme.INK_MUTED, linewidth=1.0)
        ax.text(0.98, 0.04, "dashed line = perfect prediction", transform=ax.transAxes,
                color=theme.INK_MUTED, fontsize=theme.FONT_LABEL, ha="right")
        self.scatter = ax.scatter([], [], s=7, color=theme.SERIES_1, alpha=0.28, linewidths=0)
        ax.set_xlim(limit)
        ax.set_ylim(limit)
        ax.set_xlabel("actual median house value (\\$k, census caps at \\$500k)")
        ax.set_ylabel("predicted (\\$k)")

    def update(self, actual_k: np.ndarray, predicted_k: np.ndarray, title: str) -> None:
        """Moves the points to new predictions.

        Args:
            actual_k: The houses' true prices, in $k.
            predicted_k: Their predicted prices, in $k, in the same order.
            title: The panel's new title.
        """
        self.scatter.set_offsets(np.column_stack([actual_k, predicted_k]))
        self.ax.set_title(title)


class ResidualHistogram:
    """A histogram of residuals (predicted minus actual price), titled with the share within ±$50k.

    The bars sit on fixed bins from -``RESIDUAL_RANGE_K`` to +``RESIDUAL_RANGE_K``; a residual
    beyond either end is counted in the end bin. The y axis only grows, so a shrinking bar really
    is fewer houses.

    Attributes:
        ax: The panel's axes.
    """

    def __init__(self, ax: Axes):
        """Draws the empty bars, the zero line and the axis labels.

        Args:
            ax: The axes to draw in.
        """
        self.ax = ax
        theme.style_axes(ax)
        self._edges = np.linspace(-RESIDUAL_RANGE_K, RESIDUAL_RANGE_K, RESIDUAL_BIN_COUNT + 1)
        widths = np.diff(self._edges) * 0.86  # leaves a surface gap between adjacent bars
        self._bars = ax.bar(self._edges[:-1], np.zeros(RESIDUAL_BIN_COUNT), width=widths,
                            align="edge", color=theme.SERIES_1)
        ax.axvline(0.0, color=theme.INK_MUTED, linewidth=1.0)
        ax.set_xlim(self._edges[0], self._edges[-1])
        ax.set_xlabel("predicted − actual (\\$k)   ← under-priced · over-priced →")
        ax.set_ylabel("houses")
        self._y_top = 1.0

    def update(self, residuals_k: np.ndarray) -> None:
        """Recounts the bins and retitles the panel.

        Args:
            residuals_k: Predicted minus actual price for each house, in $k.
        """
        clipped = np.clip(residuals_k, self._edges[0], self._edges[-1] - 1e-9)
        counts, _ = np.histogram(clipped, bins=self._edges)
        for bar, count in zip(self._bars, counts):
            bar.set_height(count)
        self._y_top = max(self._y_top, counts.max() * HEADROOM)
        self.ax.set_ylim(0.0, self._y_top)
        share = share_within(residuals_k, GOOD_ENOUGH_K)
        self.ax.set_title(f"Residuals · {share:.0%} within ±{money_k(GOOD_ENOUGH_K)}")


class TwoLineChart:
    """Two series sharing one y-axis, with a legend, end markers and direct end labels.

    The learning curve uses it for train and validation RMSE. Each line ends in a dot labelled
    with its latest value, so the reader need not look up the legend; the upper label sits above
    its point and the lower one below, so they never overlap.

    Attributes:
        ax: The panel's axes.
        labels: The two series' names, as the legend and end labels show them.
        lines: The two lines.
        markers: The dot at each line's latest point.
        end_labels: The value label beside each dot.
    """

    def __init__(self, ax: Axes, labels: tuple[str, str]):
        """Draws two empty lines with their markers, labels and legend.

        Args:
            ax: The axes to draw in.
            labels: The two series' names (the first is drawn in ``SERIES_1``, the second in
                ``SERIES_2``).
        """
        self.ax = ax
        theme.style_axes(ax)
        self.labels = labels
        colors = (theme.SERIES_1, theme.SERIES_2)
        self.lines = tuple(ax.plot([], [], color=c, linewidth=2.0, label=label)[0]
                           for c, label in zip(colors, labels))
        self.markers = tuple(ax.plot([], [], "o", color=c, markersize=8,
                                     markeredgecolor=theme.SURFACE, markeredgewidth=2.0)[0]
                             for c in colors)
        self.end_labels = tuple(ax.annotate("", (0, 0), xytext=(-8, 0), textcoords="offset points",
                                            ha="right", va="center", fontsize=theme.FONT_LABEL,
                                            color=theme.INK_SECONDARY, annotation_clip=False)
                                for _ in colors)
        ax.legend(loc="lower left")  # error curves sit high on the left, so this corner stays empty
        self._y_top = 1.0

    def update(self, x: np.ndarray, series: tuple[np.ndarray, np.ndarray]) -> None:
        """Redraws both lines; the y axis only grows, with some headroom above the highest point.

        Args:
            x: The x values both series share.
            series: The two series' y values, in $k, each as long as ``x``.
        """
        upper = int(np.argmax([y[-1] for y in series]))
        for index, (line, marker, label, name, y) in enumerate(
            zip(self.lines, self.markers, self.end_labels, self.labels, series)
        ):
            line.set_data(x, y)
            marker.set_data([x[-1]], [y[-1]])
            label.xy = (x[-1], y[-1])
            label.xyann = (-8, 11 if index == upper else -11)  # keep the two labels apart
            label.set_text(f"{name} {money_k(y[-1])}")
        self._y_top = max(self._y_top, max(s.max() for s in series) * HEADROOM)
        self.ax.set_ylim(0.0, self._y_top)
