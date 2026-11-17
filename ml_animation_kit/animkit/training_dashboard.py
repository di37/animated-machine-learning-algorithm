"""Learned-weight bars for the training dashboards, and the fit-time label beside them.

The bars are made once and updated in place every frame, so a long training video never rebuilds its
axes.
"""

import numpy as np
from matplotlib.axes import Axes

from animkit import theme
from animkit.frames import KIND_COEFFICIENT, KIND_IMPORTANCE, KIND_SENSITIVITY, FeatureWeights

WEIGHT_TITLES = {
    KIND_COEFFICIENT: "Learned coefficients (standardised)",
    KIND_IMPORTANCE: "Feature importances",
    KIND_SENSITIVITY: "Input sensitivity (mean |weight|)",
}
WEIGHT_HEADROOM = 1.3


def format_seconds(seconds: float) -> str:
    """A fit time as a short label: two decimals under ten seconds, one above.

    Args:
        seconds: The time, in seconds.

    Returns:
        The label, such as ``"0.42s"`` or ``"12.3s"``.
    """
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.1f}s"


class WeightBars:
    """Horizontal bars for coefficients (diverging blue/red by sign) or importances (one hue).

    Attributes:
        ax: The axes the bars are drawn in.
        note: The text shown instead of bars when the model has no per-feature weights.
    """

    def __init__(self, ax: Axes, names: tuple[str, ...], empty_note: str):
        """Draws one empty bar per feature, the first feature on top, with a value label beside each.

        Args:
            ax: The axes to draw in.
            names: The feature names, one bar each.
            empty_note: What to say when the model has no per-feature weights.
        """
        self.ax = ax
        theme.style_axes(ax, grid_axis="x")
        positions = np.arange(len(names))[::-1]  # first feature on top
        self._bars = ax.barh(positions, np.zeros(len(names)), height=0.7, color=theme.POSITIVE)
        ax.set_yticks(positions, names)
        self._labels = [ax.text(0.0, p, "", va="center", fontsize=theme.FONT_LABEL,
                                color=theme.INK_SECONDARY) for p in positions]
        self._zero_line = ax.axvline(0.0, color=theme.INK_MUTED, linewidth=1.0)
        self.note = ax.text(0.5, 0.5, empty_note, transform=ax.transAxes, ha="center", va="center",
                            fontsize=theme.FONT_SUBTITLE, color=theme.INK_SECONDARY, visible=False)
        self._limit = 1e-6

    def update(self, weights: FeatureWeights | None) -> None:
        """Sets every bar to the model's current weights, or shows the note when there are none.

        Importances re-normalise every step, so their axis is fitted to each frame. Coefficients and
        sensitivities keep an axis that only grows, so their growth over training stays visible.

        Args:
            weights: The current weights, or ``None`` for a model without them.
        """
        if weights is None:
            self._show_note()
            return
        signed = weights.kind == KIND_COEFFICIENT
        needed = max(float(np.abs(weights.values).max()) * WEIGHT_HEADROOM, 1e-6)
        # Importances are shares that re-normalise every step, so fit them per frame;
        # coefficients and sensitivities keep an expanding axis so their growth stays visible.
        self._limit = needed if weights.kind == KIND_IMPORTANCE else max(self._limit, needed)
        pad = self._limit * 0.02
        for bar, label, value in zip(self._bars, self._labels, weights.values):
            bar.set_width(value)
            bar.set_color(theme.NEGATIVE if signed and value < 0 else theme.POSITIVE)
            label.set_x(value - pad if value < 0 else value + pad)
            label.set_ha("right" if value < 0 else "left")
            label.set_text(f"{value:+.2f}" if signed else f"{value:.2f}")
        self.ax.set_xlim(-self._limit if signed else 0.0, self._limit)
        self.ax.set_title(WEIGHT_TITLES[weights.kind])

    def _show_note(self) -> None:
        """Hides the bars, labels and zero line, and shows the empty note instead."""
        for artist in (*self._bars, *self._labels, self._zero_line):
            artist.set_visible(False)
        self.ax.set_axis_off()
        self.ax.set_title("Learned weights")
        self.note.set_visible(True)
