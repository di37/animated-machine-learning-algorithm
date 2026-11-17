"""The MLP as a live network diagram: every weight is an edge; activations light up per prediction.

Subclasses say what the single output unit means: ``output_label``, ``subject`` and ``_output``.

The network has two hidden ReLU layers and one output unit, drawn as four columns of nodes joined by
every weight. During training the edges show the weights themselves. For one prediction they show the
signal each edge carries, the source unit's activation times the weight, so units that stay silent
(ReLU at zero) carry nothing.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from animkit import theme
from animkit.algorithms.common import AlgorithmView, signed_rgba, wrap_to_axes
from animkit.data import ScaledSplits

LAYER_X = (0.2, 0.47, 0.74, 0.93)
EDGE_FLOOR = 0.02  # keeps 2,500+ faint edges from turning the diagram into a grey wall
X_SCALE = 2.5


def _layer_positions(sizes: tuple[int, ...]) -> list[np.ndarray]:
    """Node positions, one (n, 2) array per layer: a column at its ``LAYER_X``, nodes spread top to bottom."""
    return [np.column_stack([np.full(n, x), np.linspace(0.9, 0.1, n) if n > 1 else [0.5]])
            for x, n in zip(LAYER_X, sizes)]


class NetworkView(AlgorithmView):
    """A scikit-learn MLP drawn as a network, redrawn from its weights at every frame.

    Attributes:
        edges: Every weight as one line, once the network is built (``None`` before).
        output_label: What the last layer's single unit means, as its column's label.
        subject: What one prediction is about, for the title.
        nodes: Every unit, as one scatter of points.
        output_text: The text under the output node.
        caption_text: The caption under the diagram.
    """

    edges = None
    output_label = "output"  # what the last layer's single unit means
    subject = "Example"  # what one prediction is about, for the title

    def _output(self, z: float) -> tuple[str, object]:
        """Text and colour of the output node for the network's raw output ``z``."""
        raise NotImplementedError

    def __init__(self, ax: Axes, splits: ScaledSplits):
        """Starts with a blank diagram; the network is built from the first model it is shown.

        Args:
            ax: The panel's axes.
            splits: The project's splits, for the input features' names.
        """
        super().__init__(ax, splits)
        self._built_for: tuple[int, ...] | None = None

    def _build(self, model) -> None:
        """Lays out the nodes, edges and labels for the model's layer sizes, unless they are built already."""
        sizes = (model.coefs_[0].shape[0], *(w.shape[1] for w in model.coefs_))
        if sizes == self._built_for:
            return
        self._blank("")
        self._layers = _layer_positions(sizes)
        segments = [(tuple(a), tuple(b)) for left, right in zip(self._layers, self._layers[1:])
                    for a in left for b in right]
        self.edges = LineCollection(segments, linewidths=0.4, zorder=1)
        self.ax.add_collection(self.edges)
        points = np.vstack(self._layers)
        hidden = len(points) - sizes[0] - 1
        self.nodes = self.ax.scatter(points[:, 0], points[:, 1], s=[90] * sizes[0] + [28] * hidden + [260],
                                     edgecolors=theme.BASELINE, linewidths=0.6, zorder=3)
        for (x, y), name in zip(self._layers[0], self.splits.feature_names):
            self.ax.text(x - 0.03, y, name, ha="right", va="center", fontsize=10, color=theme.INK_PRIMARY)
        for x, label in zip(LAYER_X, ("inputs", f"{sizes[1]} hidden", f"{sizes[2]} hidden", self.output_label)):
            self.ax.text(x, 0.955, label, ha="center", fontsize=10, color=theme.INK_SECONDARY)
        self.output_text = self.ax.text(LAYER_X[-1], 0.43, "", ha="center", va="top", fontsize=13,
                                        weight="bold", color=theme.INK_PRIMARY)
        self.caption_text = self.ax.text(0.0, 0.0, "", transform=self.ax.transAxes,
                                         fontsize=theme.FONT_SUBTITLE, color=theme.INK_SECONDARY)
        self._built_for = sizes

    def _edge_style(self, strengths: list[np.ndarray]) -> None:
        """Colours every edge by its value's sign and widens it with its size, each layer on its own scale."""
        colours, widths = [], []
        for layer in strengths:
            scale = float(np.abs(layer).max()) or 1.0
            for value in layer.ravel():
                colours.append(signed_rgba(value, scale, floor=EDGE_FLOOR))
                widths.append(0.2 + 1.8 * abs(value) / scale)
        self.edges.set_colors(colours)
        self.edges.set_linewidths(widths)

    def show_training(self, model, frame) -> None:
        """Shows the network's current weights: edge colour is the sign, edge width the size.

        Args:
            model: The MLP as it stands after this training step.
            frame: The project's training frame (not used here).
        """
        self._build(model)
        hidden = "-".join(str(n) for n in model.hidden_layer_sizes)
        self.ax.set_title(f"The algorithm: a {hidden} neural network learning its weights", fontsize=13)
        self._edge_style(list(model.coefs_))
        neutral = to_rgba(theme.NOT_YET)
        self.nodes.set_facecolors([neutral] * len(self.nodes.get_offsets()))
        self.output_text.set_text("ŷ")
        self.caption_text.set_text(wrap_to_axes(
            self.ax, f"{sum(w.size for w in model.coefs_):,} weights  ·  edge colour = sign, "
            "strength = size  ·  each frame is one more epoch of Adam", theme.FONT_SUBTITLE))

    def show_prediction(self, model, x: np.ndarray, number: int) -> None:
        """Shows one example flowing through the network.

        The hidden activations are recomputed from the weights (ReLU), so each edge can show the signal it
        carries, activation × weight, and each hidden unit how strongly it fires.

        Args:
            model: The trained MLP.
            x: The example's scaled feature row.
            number: The example's number, for the title.
        """
        self._build(model)
        (w1, w2, w3), (b1, b2, b3) = model.coefs_, model.intercepts_
        h1 = np.maximum(x @ w1 + b1, 0.0)
        h2 = np.maximum(h1 @ w2 + b2, 0.0)
        text, colour = self._output(float(h2 @ w3[:, 0] + b3[0]))
        # Signal on each edge = source activation x weight, so silent units carry nothing.
        self._edge_style([x[:, None] * w1, h1[:, None] * w2, h2[:, None] * w3])
        peak = max(float(h1.max()), float(h2.max()), 1e-9)
        hidden = [signed_rgba(a, peak, floor=0.06) if a > 0 else to_rgba(theme.NOT_YET) for a in (*h1, *h2)]
        inputs = [signed_rgba(v, X_SCALE, floor=0.1) for v in x]
        self.nodes.set_facecolors(inputs + hidden + [colour])
        self.ax.set_title(f"{self.subject} #{number:,} flowing through the network", fontsize=13)
        self.output_text.set_text(text)
        self.caption_text.set_text(wrap_to_axes(
            self.ax, f"{np.count_nonzero(h1)}/{h1.size} and {np.count_nonzero(h2)}/{h2.size} hidden units "
            "fire (ReLU > 0)  ·  edges show activation × weight", theme.FONT_SUBTITLE))
