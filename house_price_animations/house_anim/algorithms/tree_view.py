"""A single decision tree: the diagram grows during training; a house's path lights up at inference.

Each node is coloured by the mean price of the training houses in it and sized by how many there
are. A prediction is a walk from the root: at each node the house goes left if its feature is at
most the threshold, right if not, until it lands in a leaf, whose mean price is the prediction.
Only the top ``DRAW_DEPTH`` levels are drawn; the captions say how much is hidden.
"""

import numpy as np

from house_anim import theme
from house_anim.algorithms.common import AlgorithmView, draw_tree, price_k

DRAW_DEPTH = 5  # deeper levels exist but would be unreadable; captions say how much is hidden
TREE_BOX = (0.03, 0.12, 0.94, 0.78)
LABEL_DEPTH = 3


class TreeView(AlgorithmView):
    """The decision tree's diagram.

    Attributes:
        nodes_drawn: How many nodes the last drawing put on screen (the tests read it).
        path_drawn: How many edges of the house's path the last drawing highlighted.
    """

    nodes_drawn = 0
    path_drawn = 0

    def show_training(self, model, frame) -> None:
        """Draws the tree as it is now, with its depth and leaf count.

        Args:
            model: The fitted ``DecisionTreeRegressor``.
            frame: The step's ``TrainingFrame`` (unused: the tree itself is the picture).
        """
        self._blank("The algorithm: a decision tree, growing")
        drawn = draw_tree(self.ax, model.tree_, TREE_BOX, DRAW_DEPTH, node_size=420.0,
                          label_depth=LABEL_DEPTH, splits=self.splits)
        self.nodes_drawn, self.path_drawn = drawn.nodes, 0
        hidden = "" if model.get_depth() <= DRAW_DEPTH else f"  (top {DRAW_DEPTH} levels shown)"
        self.caption(f"depth {model.get_depth()}  ·  {model.get_n_leaves():,} leaves{hidden}\n"
                     "node colour = mean price of its houses  ·  size = how many houses", y=0.0)

    def show_prediction(self, model, x: np.ndarray, house_number: int) -> None:
        """Highlights the house's path through the splits, and names its leaf's size and mean price.

        Args:
            model: The fitted ``DecisionTreeRegressor``.
            x: The house's scaled features.
            house_number: The house's position among the test houses, from 1, for the title.
        """
        self._blank(f"House #{house_number:,} follows the splits")
        t = model.tree_
        path = model.decision_path(x[None, :]).indices
        drawn = draw_tree(self.ax, t, TREE_BOX, DRAW_DEPTH, path=path, node_size=420.0,
                          label_depth=LABEL_DEPTH, splits=self.splits)
        self.nodes_drawn, self.path_drawn = drawn.nodes, drawn.path_edges
        below = len(path) - 1 - drawn.path_edges
        if below > 0 and drawn.path_end is not None:
            self.ax.text(drawn.path_end[0], drawn.path_end[1] - 0.04, f"↓ {below} more splits",
                         ha="center", va="top", fontsize=10, color=theme.INK_PRIMARY)
        leaf = path[-1]
        houses = int(t.n_node_samples[leaf])
        self.caption(f"lands in a leaf of {houses:,} training house{'s' if houses != 1 else ''}  →  "
                     f"prediction = their mean price, {price_k(float(t.value[leaf].ravel()[0]))}", y=0.0)
