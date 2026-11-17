"""Logistic models as a diagram: 8 inputs -> weighted edges -> Σ -> σ -> P(survive), above the S-curve.

Logistic regression adds up the features, each times its weight, plus a bias: z = b + Σ w·x, the
log-odds of surviving. The sigmoid σ squashes z into a probability. During training the edges show
the weights (width for size; blue raises the chance, red lowers it), and the strip below places
every validation passenger on the S-curve by its z, survivors at 1 and the others at 0. For one
passenger the edges show each feature's push on z, w·x, and the strip follows that z up to its
probability. SGDClassifier with log loss learns the same model, so it shares the view.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

from animkit import theme as base
from titanic_anim import theme
from titanic_anim.algorithms.common import AlgorithmView, sigmoid, signed_rgba, split_area, style_strip, wrap_to_axes
from titanic_anim.data import TitanicSplits
from titanic_anim.passengers import feature_text

INPUT_X, SUM_X, SIGMOID_X, OUTPUT_X = 0.34, 0.64, 0.8, 0.9
NODE_Y = 0.5
MAX_EDGE_WIDTH = 9.0
X_SCALE = 2.5  # |z-score| that shows as a fully saturated input node
LABEL_OFFSET = 0.035
Z_LIMIT = 6.0
CURVE_SHARE = 0.3


def _z(model, X: np.ndarray) -> np.ndarray:
    """The model's log-odds z for each passenger: its ``decision_function``."""
    return np.ravel(model.decision_function(X))


class LogisticView(AlgorithmView):
    """The logistic models' diagram: inputs, weighted edges, Σ and σ, above the S-curve strip.

    Attributes:
        curve: The strip below the diagram, for the S-curve.
        edges: One line per feature, from its input node to Σ.
        edge_labels: Each edge's number: a weight, or a passenger's w·x.
        inputs: The input nodes: grey in training, coloured by a passenger's standardised value.
        input_names: Each input's label: the feature's name, or its value in real units.
        nodes: The Σ and σ nodes.
        bias_text: The bias b (and, for a passenger, z) under Σ.
        output_text: The output: "P", or a passenger's probability.
    """

    def __init__(self, ax: Axes, splits: TitanicSplits):
        """Draws the diagram's fixed parts; the bottom of the area becomes the S-curve strip.

        Args:
            ax: The view's axes.
            splits: The splits, for the feature names, the scaling and the validation passengers.
        """
        self.curve = split_area(ax, CURVE_SHARE)
        super().__init__(ax, splits)
        n = len(splits.feature_names)
        self._ys = np.linspace(0.9, 0.1, n)
        self.edges = [ax.plot([INPUT_X, SUM_X], [y, NODE_Y], solid_capstyle="round", zorder=1)[0]
                      for y in self._ys]
        self.edge_labels = [ax.text(INPUT_X + LABEL_OFFSET, y, "", fontsize=10, ha="left", va="center",
                                    color=base.INK_PRIMARY, zorder=4,
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor=base.SURFACE,
                                              edgecolor="none", alpha=0.85))
                            for y in self._ys]
        self.inputs = ax.scatter(np.full(n, INPUT_X), self._ys, s=420, zorder=3,
                                 edgecolors=base.BASELINE, linewidths=1.2)
        self.input_names = [ax.text(INPUT_X - 0.04, y, name, ha="right", va="center", fontsize=10.5,
                                    color=base.INK_PRIMARY) for y, name in zip(self._ys, splits.feature_names)]
        self.nodes = [ax.scatter([x], [NODE_Y], s=size, color=base.SURFACE, edgecolors=base.INK_PRIMARY,
                                 linewidths=1.6, zorder=3) for x, size in ((SUM_X, 2400), (SIGMOID_X, 1500))]
        for x, symbol in ((SUM_X, "Σ"), (SIGMOID_X, "σ")):
            ax.text(x, NODE_Y, symbol, ha="center", va="center", fontsize=20, color=base.INK_PRIMARY, zorder=4)
        ax.annotate("", (SIGMOID_X - 0.04, NODE_Y), (SUM_X + 0.05, NODE_Y),
                    arrowprops=dict(arrowstyle="-|>", color=base.INK_PRIMARY, linewidth=1.4))
        self.bias_text = ax.text(SUM_X, NODE_Y - 0.075, "", ha="center", va="top", fontsize=10,
                                 color=base.INK_SECONDARY)
        self.output_text = ax.text(OUTPUT_X, NODE_Y + 0.07, "", ha="center", va="bottom", fontsize=15,
                                   weight="bold", color=base.INK_PRIMARY)
        self._caption_text = ax.text(0.0, 0.0, "", transform=ax.transAxes, fontsize=base.FONT_SUBTITLE,
                                     color=base.INK_SECONDARY)

    def _draw_edges(self, strengths: np.ndarray, labels: list[str]) -> None:
        """Sets each edge's width and colour from its strength (blue positive, red negative) and its label."""
        scale = float(np.abs(strengths).max()) or 1.0
        for edge, label, strength, text in zip(self.edges, self.edge_labels, strengths, labels):
            edge.set_linewidth(0.8 + MAX_EDGE_WIDTH * abs(strength) / scale)
            edge.set_color(signed_rgba(strength, scale, floor=0.35))
            label.set_text(text)

    def _curve(self, title: str) -> None:
        """Redraws the strip under ``title``: the S-curve σ(z) and the 0.5 threshold."""
        style_strip(self.curve, title, "z = b + Σ w·x")
        z = np.linspace(-Z_LIMIT, Z_LIMIT, 200)
        self.curve.plot(z, sigmoid(z), color=base.INK_PRIMARY, linewidth=1.6)
        self.curve.axhline(0.5, color=base.INK_MUTED, linestyle="--", linewidth=0.8)
        self.curve.set_xlim(-Z_LIMIT, Z_LIMIT)
        self.curve.set_ylim(-0.1, 1.1)
        self.curve.set_yticks((0.0, 0.5, 1.0), ("0", "0.5", "1"))
        self.curve.set_ylabel("P(survive)")

    def show_training(self, model, frame) -> None:
        """Shows the current weights on the edges and the validation passengers on the S-curve.

        Args:
            model: The fitted ``LogisticRegression`` or ``SGDClassifier``.
            frame: The training frame; unused, since everything drawn comes from the model.
        """
        b, w = float(np.ravel(model.intercept_)[0]), np.ravel(model.coef_)
        self.ax.set_title("The algorithm: a weighted sum, squashed by σ", fontsize=13)
        self._draw_edges(w, [f"w = {v:+.2f}" for v in w])
        self.inputs.set_facecolors([to_rgba(base.NOT_YET)] * len(w))
        for text, name in zip(self.input_names, self.splits.feature_names):
            text.set_text(name)
        self.bias_text.set_text(f"+ b = {b:+.2f}")
        self.output_text.set_text("P")
        self.nodes[1].set_facecolors([base.SURFACE])
        self._caption("edge width = |weight|  ·  blue raises, red lowers the chance of surviving")
        self._curve("validation passengers by z: survivors at 1, the others at 0")
        z = np.clip(_z(model, self.splits.X_val), -Z_LIMIT, Z_LIMIT)
        jitter = np.random.default_rng(0).uniform(-0.04, 0.04, len(z))
        self.curve.scatter(z, self.splits.y_val + jitter, s=14, linewidths=0, zorder=3,
                           c=[theme.outcome_rgba(y, 0.45) for y in self.splits.y_val])

    def show_prediction(self, model, x: np.ndarray, passenger_number: int) -> None:
        """Works one passenger through the diagram: each feature's push w·x, the sum z, and σ(z).

        Args:
            model: The fitted ``LogisticRegression`` or ``SGDClassifier``.
            x: The passenger's standardised features.
            passenger_number: The passenger's number, for the title.
        """
        b, w = float(np.ravel(model.intercept_)[0]), np.ravel(model.coef_)
        contributions = w * x
        z = b + float(contributions.sum())
        p = float(sigmoid(z))
        raw = x * self.splits.feature_scale + self.splits.feature_mean
        self.ax.set_title(f"Passenger #{passenger_number:,} through the equation", fontsize=13)
        self._draw_edges(contributions, [f"{c:+.2f}" for c in contributions])
        self.inputs.set_facecolors([signed_rgba(v, X_SCALE, floor=0.1) for v in x])
        for text, name, value in zip(self.input_names, self.splits.feature_names, raw):
            text.set_text(feature_text(name, value))
        self.bias_text.set_text(f"+ b = {b:+.2f}  →  z = {z:+.2f}")
        self.output_text.set_text(f"{p:.0%}")
        self.nodes[1].set_facecolors([theme.survival_rgba(p)])
        self._caption("node colour = this passenger's standardised value  ·  edge = w·x, its push on z")
        self._curve("σ turns this passenger's z into a probability")
        shown = float(np.clip(z, -Z_LIMIT, Z_LIMIT))
        self.curve.plot([shown, shown, -Z_LIMIT], [-0.1, p, p], color=base.INK_SECONDARY,
                        linestyle=":", linewidth=1.0)
        self.curve.scatter([shown], [p], s=110, color=theme.survival_rgba(p), edgecolors=base.INK_PRIMARY,
                           linewidths=1.0, zorder=4)

    def _caption(self, text: str) -> None:
        """Sets the caption under the diagram, wrapped to the axes' width."""
        self._caption_text.set_text(wrap_to_axes(self.ax, text, base.FONT_SUBTITLE))
