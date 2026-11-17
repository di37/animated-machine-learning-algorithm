"""Shared pieces for the classifier diagrams, on top of animkit's AlgorithmView and tree drawing.

animkit provides the base view, ``signed_rgba`` (blue for positive, red for negative, stronger for
larger values), ``wrap_to_axes`` and ``draw_tree``. This module adds what the Titanic views share:
a strip chart under a view (``split_area``, ``style_strip``), and trees coloured by survival share
with their splits asked as plain questions (``draw_survival_tree``).
"""

import numpy as np
from matplotlib.axes import Axes

from animkit import theme as base
from animkit.algorithms.common import AlgorithmView, draw_tree, signed_rgba, wrap_to_axes
from titanic_anim import theme
from titanic_anim.metrics import sigmoid
from titanic_anim.passengers import split_label

__all__ = ["AlgorithmView", "draw_survival_tree", "sigmoid", "signed_rgba", "split_area", "style_strip",
           "survival_node_colour", "wrap_to_axes"]


def split_area(ax: Axes, share: float, gap: float = 1.35) -> Axes:
    """Shrinks ``ax`` to the top of its area and returns a strip chart axes underneath it.

    Args:
        ax: The view's axes; it is moved up to make room.
        share: The strip's share of the area's height.
        gap: Where the main axes now starts, in strip heights from the bottom; above 1 leaves room for
            the strip's title.

    Returns:
        The new strip axes, at the bottom of the area.
    """
    pos = ax.get_position()
    strip_height = pos.height * share
    main_bottom = pos.y0 + strip_height * gap  # leaves room for the strip's title
    ax.set_position([pos.x0, main_bottom, pos.width, pos.y0 + pos.height - main_bottom])
    return ax.figure.add_axes([pos.x0 + pos.width * 0.1, pos.y0 + strip_height * 0.15,
                               pos.width * 0.86, strip_height * 0.62])


def style_strip(strip: Axes, title: str, xlabel: str) -> None:
    """Clears a strip axes and styles it again, with a title and an x label.

    Args:
        strip: The strip axes.
        title: Its title.
        xlabel: Its x-axis label.
    """
    strip.clear()
    base.style_axes(strip)
    strip.set_title(title, fontsize=10)
    strip.set_xlabel(xlabel)


def survival_node_colour(value: np.ndarray):
    """The colour of a classifier tree's node, from its share of survivors.

    Args:
        value: The node's value row: [share who died, share who survived].

    Returns:
        An RGBA tuple: red for a node of non-survivors, blue for one of survivors.
    """
    return theme.survival_rgba(float(value[1]))


def draw_survival_tree(ax: Axes, t, box, max_depth: int, splits=None, **kwargs):
    """Draws a tree with animkit's ``draw_tree``: nodes coloured by survival share, splits as questions.

    Args:
        ax: The axes to draw in.
        t: A fitted tree's ``tree_`` structure.
        box: ``(x0, y0, width, height)`` to draw inside, in axes units.
        max_depth: How many levels to draw; deeper nodes are left out.
        splits: The splits, to word the questions in real units; None draws no questions.
        **kwargs: Passed on to ``draw_tree``, such as ``path``, ``node_size`` and ``label_depth``.

    Returns:
        animkit's ``DrawnTree``: how many nodes and path edges are on screen, and where the path stops.
    """
    return draw_tree(ax, t, box, max_depth, splits=splits, node_colour=survival_node_colour,
                     split_label=split_label, **kwargs)

