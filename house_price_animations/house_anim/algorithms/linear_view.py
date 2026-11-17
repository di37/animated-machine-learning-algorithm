"""Linear models as a weighted-sum diagram: 8 inputs -> weighted edges -> sum -> price.

A linear model prices a house as b + Σ_j w_j x_j. The diagram draws that sum: one node per
feature, an edge from each into the Σ node, then the output ŷ. While training, an edge's width is
the size of its weight and its colour the sign (blue +, red −); a weight of exactly 0 (Lasso)
turns its edge dashed. At inference each edge carries w_j × x_j, that feature's push on this
house's price, and the input nodes are coloured by the house's standardised values.

The view builds its artists once and only updates them each frame, rather than clearing the axes
and drawing again as the tree and map views do.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

from house_anim import theme
from house_anim.algorithms.common import AlgorithmView, price_k, signed_rgba, wrap_to_axes
from house_anim.data import HousingSplits

INPUT_X, SUM_X, OUTPUT_X = 0.26, 0.66, 0.9
SUM_Y = 0.5
MAX_EDGE_WIDTH = 9.0
X_SCALE = 2.5  # |z-score| that shows as a fully saturated input node
LABEL_OFFSET = 0.035


class WeightedSumView(AlgorithmView):
    """The weighted-sum diagram for Linear Regression, Ridge, Lasso and SGD.

    Attributes:
        edges: One edge per feature, from its input node to Σ.
        edge_labels: Each edge's weight (training) or push (inference).
        inputs: The input nodes.
        input_values: Each input node's label: the feature name, and the house's value at inference.
        bias_text: The intercept b under Σ (and the sum, at inference).
        output_text: The output: ŷ while training, the price at inference.
    """

    def __init__(self, ax: Axes, splits: HousingSplits):
        """Draws the inputs, edges, Σ node and output arrow, with nothing learned on them yet.

        Args:
            ax: The axes to draw in.
            splits: The data, for the feature names.
        """
        super().__init__(ax, splits)
        n = len(splits.feature_names)
        self._ys = np.linspace(0.88, 0.12, n)
        self.edges = [ax.plot([INPUT_X, SUM_X], [y, SUM_Y], solid_capstyle="round", zorder=1)[0]
                      for y in self._ys]
        # Labels sit just right of each input node, where the edges are still far apart.
        self.edge_labels = [ax.text(INPUT_X + LABEL_OFFSET, y, "", fontsize=10, ha="left", va="center",
                                    color=theme.INK_PRIMARY, zorder=4,
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor=theme.SURFACE,
                                              edgecolor="none", alpha=0.85))
                            for y in self._ys]
        self.inputs = ax.scatter(np.full(n, INPUT_X), self._ys, s=520, zorder=3,
                                 edgecolors=theme.BASELINE, linewidths=1.2)
        self.input_values = [ax.text(INPUT_X - 0.045, y, name, ha="right", va="center", fontsize=10.5,
                                     color=theme.INK_PRIMARY) for y, name in zip(self._ys, splits.feature_names)]
        ax.scatter([SUM_X], [SUM_Y], s=2600, color=theme.SURFACE, edgecolors=theme.INK_PRIMARY,
                   linewidths=1.6, zorder=3)
        ax.text(SUM_X, SUM_Y, "Σ", ha="center", va="center", fontsize=22, color=theme.INK_PRIMARY, zorder=4)
        self.bias_text = ax.text(SUM_X, SUM_Y - 0.075, "", ha="center", va="top", fontsize=10,
                                 color=theme.INK_SECONDARY)
        ax.annotate("", (OUTPUT_X - 0.035, SUM_Y), (SUM_X + 0.05, SUM_Y),
                    arrowprops=dict(arrowstyle="-|>", color=theme.INK_PRIMARY, linewidth=1.6))
        self.output_text = ax.text(OUTPUT_X, SUM_Y, "ŷ", ha="left", va="center", fontsize=16,
                                   weight="bold", color=theme.INK_PRIMARY)
        self._caption_text = ax.text(0.0, 0.0, "", transform=ax.transAxes,
                                     fontsize=theme.FONT_SUBTITLE, color=theme.INK_SECONDARY)

    def _draw_edges(self, strengths: np.ndarray, labels: list[str], zero_dashed: bool) -> None:
        """Sets each edge's width and colour from its strength (relative to the largest) and its label."""
        scale = float(np.abs(strengths).max()) or 1.0
        for edge, label, strength, text in zip(self.edges, self.edge_labels, strengths, labels):
            edge.set_linewidth(0.8 + MAX_EDGE_WIDTH * abs(strength) / scale)
            edge.set_color(signed_rgba(strength, scale, floor=0.35))
            edge.set_linestyle("--" if zero_dashed and strength == 0.0 else "-")
            label.set_text(text)

    def show_training(self, model, frame) -> None:
        """Draws the current weights on the edges and the intercept under Σ.

        Args:
            model: The fitted linear model.
            frame: The step's ``TrainingFrame`` (unused: the weights are the picture).
        """
        b, w = float(np.ravel(model.intercept_)[0]), np.ravel(model.coef_)
        self.ax.set_title("The algorithm: a weighted sum of 8 features", fontsize=13)
        self._draw_edges(w, [f"w = {v:+.2f}" for v in w], zero_dashed=True)
        self.inputs.set_facecolors([to_rgba(theme.NOT_YET)] * len(w))
        for text, name in zip(self.input_values, self.splits.feature_names):
            text.set_text(name)
        self.bias_text.set_text(f"+ b = {b:.2f}")
        self.output_text.set_text("ŷ")
        dashed = int(np.sum(w == 0.0))
        self._caption("edge width = |weight|  ·  blue +, red −"
                      + (f"  ·  {dashed} dashed = switched off (weight 0)" if dashed else ""))

    def show_prediction(self, model, x: np.ndarray, house_number: int) -> None:
        """Runs the house through the sum: each edge shows its push w_j × x_j, the output the price.

        Args:
            model: The fitted linear model.
            x: The house's scaled features.
            house_number: The house's position among the test houses, from 1, for the title.
        """
        b, w = float(np.ravel(model.intercept_)[0]), np.ravel(model.coef_)
        contributions = w * x
        value = b + float(contributions.sum())
        self.ax.set_title(f"House #{house_number:,} through the weighted sum", fontsize=13)
        self._draw_edges(contributions, [f"{c:+.2f}" for c in contributions], zero_dashed=False)
        self.inputs.set_facecolors([signed_rgba(v, X_SCALE, floor=0.1) for v in x])
        for text, name, v in zip(self.input_values, self.splits.feature_names, x):
            text.set_text(f"{name}  {v:+.2f}")
        self.bias_text.set_text(f"+ b = {b:.2f}  →  {value:.2f}")
        self.output_text.set_text(price_k(value))
        self._caption("node colour = this house's standardised value  ·  edge width = |w·x|, its push")

    def _caption(self, text: str) -> None:
        """Replaces the caption's text, wrapped to the view's width (the caption is one reused artist)."""
        self._caption_text.set_text(wrap_to_axes(self.ax, text, theme.FONT_SUBTITLE))
