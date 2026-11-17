"""Gaussian Naive Bayes as 8 small bell-curve pairs: survivors vs others, one panel per feature.

Gaussian Naive Bayes learns, for every feature and outcome, a mean and a variance: one bell curve
for the survivors and one for the others. "Naive" means it treats the features as independent once
the outcome is known, so a passenger's evidence is a product of one density per feature. In logs
it is a sum: at the passenger's value, each feature adds log N(survivors) − log N(others), which is
positive where the survivors' curve is higher. The prior log-odds plus that sum is the log-odds of
surviving.
"""

import numpy as np
from matplotlib.axes import Axes

from animkit import theme as base
from titanic_anim import theme
from titanic_anim.algorithms.common import AlgorithmView, sigmoid
from titanic_anim.data import TitanicSplits
from titanic_anim.equations.kernels import log_likelihood_ratios

GRID_ROWS, GRID_COLUMNS = 4, 2
X_LIMIT = 3.5  # standardised units shown on every panel
CURVE_POINTS = 160
PANELS_TOP, PANELS_BOTTOM = 0.97, 0.1  # share of the view's height the panel grid spans


def gaussian(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    """The normal (bell-curve) density.

    Args:
        x: Where to evaluate it.
        mean: The curve's centre.
        variance: Its spread, the standard deviation squared.

    Returns:
        The density at every ``x``.
    """
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2.0 * np.pi * variance)


class NaiveBayesView(AlgorithmView):
    """Naive Bayes' diagram: a small panel per feature, with the two outcomes' bell curves.

    Attributes:
        panels: One axes per feature, in feature order, laid out in a 4x2 grid.
    """

    def __init__(self, ax: Axes, splits: TitanicSplits):
        """Lays out the grid of feature panels inside the view's area.

        Args:
            ax: The view's axes.
            splits: The splits, for the feature names.
        """
        super().__init__(ax, splits)
        pos = ax.get_position()
        top = pos.y0 + pos.height * PANELS_TOP
        cell_w, cell_h = pos.width / GRID_COLUMNS, pos.height * (PANELS_TOP - PANELS_BOTTOM) / GRID_ROWS
        self.panels = [
            ax.figure.add_axes([pos.x0 + c * cell_w + cell_w * 0.08, top - (r + 1) * cell_h + cell_h * 0.22,
                                cell_w * 0.84, cell_h * 0.6])
            for r in range(GRID_ROWS) for c in range(GRID_COLUMNS)
        ]
        self._grid = np.linspace(-X_LIMIT, X_LIMIT, CURVE_POINTS)

    def _curves(self, model) -> None:
        """Redraws every feature's panel with the survivors' (blue) and the others' (red) bell curve."""
        for index, (panel, name) in enumerate(zip(self.panels, self.splits.feature_names)):
            panel.clear()
            base.style_axes(panel)
            panel.set_yticks([])
            panel.set_xlim(-X_LIMIT, X_LIMIT)
            panel.tick_params(labelsize=8)
            panel.set_title(name, fontsize=10)
            for label in (0, 1):
                density = gaussian(self._grid, model.theta_[label, index], model.var_[label, index])
                panel.plot(self._grid, density, color=theme.OUTCOME_COLOURS[label], linewidth=1.8)
            panel.set_ylim(0.0, None)

    def show_training(self, model, frame) -> None:
        """Draws every feature's pair of bell curves and how many passengers were counted.

        Args:
            model: The fitted ``GaussianNB``.
            frame: The training frame; unused, since everything drawn comes from the model.
        """
        self._blank("The algorithm: two bell curves per feature")
        self._curves(model)
        counts = model.class_count_.astype(int)
        self.caption(f"counted {counts.sum():,} passengers: {counts[1]} survived (blue), {counts[0]} did not "
                     "(red)  ·  x = standardised feature value", y=0.0)

    def show_prediction(self, model, x: np.ndarray, passenger_number: int) -> None:
        """Marks the passenger's value on every feature's curves and adds up the evidence into P(survive).

        Each panel shows the feature's log-likelihood ratio, blue when it favours survival. A value beyond
        the panels' range is drawn at the edge.

        Args:
            model: The fitted ``GaussianNB``.
            x: The passenger's standardised features.
            passenger_number: The passenger's number, for the title.
        """
        self._blank(f"Passenger #{passenger_number:,}: which curve fits better?")
        self._curves(model)
        ratios = log_likelihood_ratios(model, x)
        for panel, value, ratio in zip(self.panels, x, ratios):
            shown = float(np.clip(value, -X_LIMIT, X_LIMIT))
            panel.axvline(shown, color=base.INK_PRIMARY, linewidth=1.4)
            panel.text(0.98, 0.92, f"{ratio:+.2f}", transform=panel.transAxes, ha="right", va="top",
                       fontsize=10, weight="bold", color=theme.OUTCOME_COLOURS[int(ratio > 0)])
        log_odds = float(np.log(model.class_prior_[1] / model.class_prior_[0]) + ratios.sum())
        self.caption(f"number = log N(survived) − log N(others) at this value  ·  prior + sum = "
                     f"{log_odds:+.2f}  →  P(survive) = {sigmoid(log_odds):.0%}", y=0.0)
