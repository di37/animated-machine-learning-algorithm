"""House-price colours for the algorithm diagrams, on top of animkit's base view and tree drawing.

Prices are coloured with the single-hue price ramp on a fixed scale, $0 to $500k (``PRICE_NORM``,
in the model's $100k units), the same scale as the price maps, so a colour means the same price in
every view. ``draw_tree`` is animkit's tree drawing with these price colours as the default.
"""

from collections.abc import Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from animkit.algorithms.common import AlgorithmView, DrawnTree, default_split_label, signed_rgba, wrap_to_axes
from animkit.algorithms.common import draw_tree as draw_any_tree
from house_anim import theme
from house_anim.config import TARGET_TO_THOUSANDS
from house_anim.data import HousingSplits

__all__ = ["PRICE_CMAP", "PRICE_NORM", "AlgorithmView", "DrawnTree", "draw_tree", "price_k", "price_rgba",
           "signed_rgba", "wrap_to_axes"]

PRICE_NORM = Normalize(0.0, 5.0)  # target units ($100k), matching the price maps
PRICE_CMAP = theme.price_colormap()


def price_k(value: float) -> str:
    """A model output as a dollar label for matplotlib, such as $245k (the dollar sign escaped).

    Args:
        value: A price in the model's units, $100k.

    Returns:
        The label, rounded to the nearest $1k.
    """
    return f"\\${value * TARGET_TO_THOUSANDS:,.0f}k"


def price_rgba(value: float):
    """The price ramp's colour for a price.

    Args:
        value: A price in $100k; values beyond $0 to $500k take the ramp's end colours.

    Returns:
        An (r, g, b, a) colour.
    """
    return PRICE_CMAP(PRICE_NORM(value))


def _price_node_colour(value: np.ndarray):
    """A regression node's value row holds its mean price."""
    return price_rgba(float(value[0]))


def draw_tree(ax: Axes, t, box: tuple[float, float, float, float], max_depth: int,
              path=None, node_size: float = 160.0, label_depth: int = 0,
              splits: HousingSplits | None = None, signed_scale: float | None = None,
              node_colour: Callable[[np.ndarray], object] | None = None,
              split_label: Callable[[str, float], str] = default_split_label) -> DrawnTree:
    """Draws a tree with animkit's draw_tree, colouring nodes by price unless told otherwise.

    Nodes are coloured on the price ramp unless ``signed_scale`` or ``node_colour`` is given, and a
    node's size grows with the number of training houses in it.

    Args:
        ax: The axes to draw in.
        t: A fitted tree's ``tree_`` structure.
        box: The drawing's (x0, y0, width, height), in axes units.
        max_depth: How many levels below the root to draw.
        path: Node ids of one house's root-to-leaf path, to highlight; None highlights nothing.
        node_size: The root's marker size; other nodes shrink with their share of the houses.
        label_depth: How many levels of splits, from the root, are labelled with their rule in
            real units; 0 labels none.
        splits: The data, to read split thresholds in real units; None draws no labels.
        signed_scale: Colour nodes blue (positive) or red (negative) against this size instead,
            for boosting's corrections.
        node_colour: Maps a node's value row to a colour, instead of the price ramp.
        split_label: Words a split from its feature name and real-unit threshold.

    Returns:
        animkit's ``DrawnTree``: how many nodes and path edges were drawn, and where the path stops.
    """
    if node_colour is None and not signed_scale:
        node_colour = _price_node_colour
    return draw_any_tree(ax, t, box, max_depth, path=path, node_size=node_size, label_depth=label_depth,
                         splits=splits, signed_scale=signed_scale, node_colour=node_colour,
                         split_label=split_label)
