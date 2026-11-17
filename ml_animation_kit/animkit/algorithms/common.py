"""Shared pieces for the animated algorithm diagrams: base class, signed colours, tree drawing.

Every diagram is an ``AlgorithmView``: it owns one axes in unit coordinates and redraws itself from
the live model, once per training step and once per example at inference. Signed values follow one
rule in every diagram: blue for positive, red for negative, more opaque the bigger they are.
``draw_tree`` draws the top levels of a fitted scikit-learn tree, with the path one example takes
through it highlighted.
"""

import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from animkit import theme
from animkit.data import ScaledSplits

CAPTION_STYLE = dict(fontsize=theme.FONT_SUBTITLE, color=theme.INK_SECONDARY)
SIGNED_NODE_FLOOR = 0.3  # small corrections stay visible instead of fading into the background


CHAR_WIDTH_EM = 0.55  # average glyph width of the UI font, for wrapping


def wrap_to_axes(ax: Axes, text: str, fontsize: float) -> str:
    """Wrap ``text`` so it never runs past the right edge of ``ax``.

    The width in characters is estimated from the axes' width and an average glyph width.

    Args:
        ax: The axes the text is drawn in.
        text: The text; its own line breaks are kept.
        fontsize: The font size it is drawn at, in points.

    Returns:
        The text with line breaks added.
    """
    width_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    chars = max(20, int(width_in * 72.0 / (fontsize * CHAR_WIDTH_EM)))
    return "\n".join(textwrap.fill(part, chars) for part in text.split("\n"))


def signed_rgba(value: float, scale: float, floor: float = 0.12):
    """Blue for positive, red for negative; opacity grows with magnitude relative to ``scale``.

    Args:
        value: The signed value to colour.
        scale: The magnitude drawn fully opaque; bigger values are capped there.
        floor: The opacity of a zero value, so small values stay visible.

    Returns:
        An RGBA colour.
    """
    strength = min(abs(value) / scale, 1.0) if scale > 0 else 0.0
    base = theme.POSITIVE if value >= 0 else theme.NEGATIVE
    return to_rgba(base, floor + (1.0 - floor) * strength)


class AlgorithmView:
    """One diagram that redraws itself from a live model during training and per example at inference.

    Subclasses implement ``show_training`` and ``show_prediction``.

    Attributes:
        ax: The diagram's axes, in unit coordinates with the frame switched off.
        splits: The project's splits: feature names, and the means and scales that turn scaled values
            back into real units.
    """

    def __init__(self, ax: Axes, splits: ScaledSplits):
        """Takes over ``ax`` and blanks it.

        Args:
            ax: The axes to draw in.
            splits: The project's splits.
        """
        self.ax = ax
        self.splits = splits
        self._blank("")

    def _blank(self, title: str) -> None:
        """Clears the axes to an empty unit square with no frame, under the given title."""
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_title(title, fontsize=13)

    def caption(self, text: str, y: float = 0.015) -> None:
        """Writes a caption at the bottom of the diagram, wrapped to its width.

        Args:
            text: The caption.
            y: Its height in axes fractions; a negative value hangs it below the axes.
        """
        wrapped = wrap_to_axes(self.ax, text, CAPTION_STYLE["fontsize"])
        self.ax.text(0.0, y, wrapped, transform=self.ax.transAxes, va="top" if y < 0 else "bottom",
                     **CAPTION_STYLE)

    def show_training(self, model, frame) -> None:
        """Redraws the diagram from the model after a training step.

        Args:
            model: The model as it stands after this step.
            frame: The project's training frame for this step.

        Raises:
            NotImplementedError: Always; each subclass draws its own diagram.
        """
        raise NotImplementedError

    def show_prediction(self, model, x: np.ndarray, number: int) -> None:
        """Redraws the diagram to show how the model handles one example.

        Args:
            model: The trained model.
            x: The example's scaled feature row.
            number: The example's number, for the title.

        Raises:
            NotImplementedError: Always; each subclass draws its own diagram.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class DrawnTree:
    """What ``draw_tree`` put on screen, so callers and tests can check it.

    Attributes:
        nodes: How many nodes are on screen.
        path_edges: How many highlighted path edges are on screen.
        path_end: Where the visible path stops, in axes coordinates; ``None`` without a path.
    """

    nodes: int  # how many nodes are on screen
    path_edges: int  # how many highlighted path edges are on screen
    path_end: tuple[float, float] | None  # where the visible path stops


def _layout(t, max_depth: int) -> dict[int, tuple[float, int]]:
    """Node id -> (x in 0..1, depth) for the top ``max_depth`` levels; leaves spread evenly."""
    order: list[int] = []
    depth_of: dict[int, int] = {}

    def visit(node: int, depth: int) -> None:
        """Records each node's depth, and collects the leaves (and nodes at ``max_depth``) left to right.

        Args:
            node: The node's id in the tree.
            depth: Its depth, the root being 0.
        """
        depth_of[node] = depth
        left, right = t.children_left[node], t.children_right[node]
        if left == -1 or depth == max_depth:
            order.append(node)
            return
        visit(left, depth + 1)
        visit(right, depth + 1)

    visit(0, 0)
    x_of = {node: (i + 0.5) / len(order) for i, node in enumerate(order)}

    def centre(node: int) -> float:
        """A node's x: its own if it is a leaf, otherwise midway between its children's.

        Args:
            node: The node's id in the tree.

        Returns:
            Its x, from 0 to 1.
        """
        if node not in x_of:
            x_of[node] = (centre(t.children_left[node]) + centre(t.children_right[node])) / 2
        return x_of[node]

    centre(0)
    return {node: (x_of[node], depth_of[node]) for node in depth_of}


def default_split_label(name: str, threshold: float) -> str:
    """Words a split as "name ≤ threshold", the threshold to two decimals.

    Args:
        name: The feature's name.
        threshold: The split threshold, in the feature's real units.

    Returns:
        The label.
    """
    return f"{name} ≤ {threshold:.2f}"


def draw_tree(ax: Axes, t, box: tuple[float, float, float, float], max_depth: int,
              path=None, node_size: float = 160.0, label_depth: int = 0,
              splits: ScaledSplits | None = None, signed_scale: float | None = None,
              node_colour: Callable[[np.ndarray], object] | None = None,
              split_label: Callable[[str, float], str] = default_split_label) -> DrawnTree:
    """Draw a fitted tree's top levels inside ``box`` (x0, y0, w, h in axes units).

    ``node_colour`` maps a node's ``tree_.value`` row to a colour; without it a node's value is
    coloured by sign against ``signed_scale``, so one of the two is required.
    ``split_label(name, threshold)`` words a split in real units.

    A node's marker area is ``node_size`` times the square root of the share of training rows that reach
    it, so busy nodes look bigger.

    Args:
        ax: The axes to draw in.
        t: A fitted scikit-learn tree structure (an estimator's ``tree_``).
        box: Where the tree goes, as ``(x0, y0, width, height)`` in axes units.
        max_depth: How many levels below the root to draw.
        path: The node ids one example passes through, root first, to highlight; ``None`` for none.
        node_size: The root's marker area, in points².
        label_depth: Splits shallower than this get a label; 0 labels none.
        splits: The project's splits, which turn thresholds back into real units; labels need them.
        signed_scale: The value drawn at full strength when nodes are coloured by sign.
        node_colour: Maps a node's value row to its colour.
        split_label: Words a split from the feature's name and the threshold in real units.

    Returns:
        What was drawn: the node count, the highlighted edges and where the path ends.

    Raises:
        ValueError: If neither ``node_colour`` nor a positive ``signed_scale`` is given.
    """
    if node_colour is None and not signed_scale:
        raise ValueError("draw_tree needs node_colour, or a positive signed_scale to colour nodes by sign")
    x0, y0, w, h = box
    layout = _layout(t, max_depth)
    deepest = max(depth for _, depth in layout.values()) or 1
    point = {n: (x0 + x * w, y0 + h - depth / deepest * h) for n, (x, depth) in layout.items()}
    edges = [(point[n], point[c]) for n in layout for c in (t.children_left[n], t.children_right[n])
             if c != -1 and c in point]
    ax.add_collection(LineCollection(edges, colors=theme.BASELINE, linewidths=1.0, zorder=1))

    nodes = list(layout)
    colours = [node_colour(t.value[n].ravel()) if node_colour else
               signed_rgba(float(t.value[n].ravel()[0]), signed_scale, floor=SIGNED_NODE_FLOOR) for n in nodes]
    sizes = [max(14.0, node_size * np.sqrt(t.n_node_samples[n] / t.n_node_samples[0])) for n in nodes]
    xs, ys = zip(*(point[n] for n in nodes))
    ax.scatter(xs, ys, s=sizes, c=colours, edgecolors=theme.SURFACE, linewidths=1.0, zorder=3)
    if splits is not None:
        _label_splits(ax, t, layout, point, label_depth, splits, split_label)
    return _draw_path(ax, path, point, len(nodes))


def _label_splits(ax, t, layout, point, label_depth: int, splits: ScaledSplits,
                  split_label: Callable[[str, float], str]) -> None:
    """Labels each split shallower than ``label_depth`` with its feature and its threshold in real units."""
    for node, (_, depth) in layout.items():
        if depth >= label_depth or t.children_left[node] == -1:
            continue
        feature = int(t.feature[node])
        threshold = t.threshold[node] * splits.feature_scale[feature] + splits.feature_mean[feature]
        x, y = point[node]
        ax.text(x, y + 0.022, split_label(splits.feature_names[feature], float(threshold)), ha="center",
                va="bottom", fontsize=8.5, color=theme.INK_PRIMARY, zorder=4)


def _draw_path(ax, path, point, nodes_drawn: int) -> DrawnTree:
    """Highlights the visible part of ``path`` and reports what was drawn."""
    if path is None:
        return DrawnTree(nodes_drawn, 0, None)
    visible = [n for n in path if n in point]
    segments = [(point[a], point[b]) for a, b in zip(visible, visible[1:])]
    ax.add_collection(LineCollection(segments, colors=theme.INK_PRIMARY, linewidths=3.0, zorder=2))
    xs, ys = zip(*(point[n] for n in visible))
    ax.scatter(xs, ys, s=60, facecolors="none", edgecolors=theme.INK_PRIMARY, linewidths=1.6, zorder=5)
    return DrawnTree(nodes_drawn, len(segments), point[visible[-1]])
