"""Tree classifiers: one tree's splits, a forest of voting trees, boosting's log-odds corrections.

- A decision tree asks yes/no questions, one feature at a time. A leaf's answer is the share of
  its training passengers who survived. Training grows the tree a level deeper each step, and a
  prediction lights up one passenger's path from the root to a leaf.
- A random forest grows many deep trees, each on its own bootstrap sample (passengers drawn with
  replacement), and averages their P(survive), so each tree's noise cancels out.
- Gradient boosting starts from the base rate's log-odds and adds small trees, each fit to the
  errors left so far. Their corrections, times the learning rate ν, add up to the log-odds F(x),
  and σ(F) is P(survive).
"""

import numpy as np
from matplotlib.axes import Axes

from animkit import theme as base
from animkit.algorithms.common import draw_tree
from titanic_anim import theme
from titanic_anim.algorithms.common import (
    AlgorithmView,
    draw_survival_tree,
    sigmoid,
    signed_rgba,
    split_area,
    style_strip,
)
from titanic_anim.data import TitanicSplits
from titanic_anim.equations.trees import leaf_note
from titanic_anim.passengers import split_label

DRAW_DEPTH = 5  # deeper levels exist but would be unreadable; captions say how much is hidden
TREE_BOX = (0.03, 0.12, 0.94, 0.78)
LABEL_DEPTH = 3
TILE_ROWS, TILE_COLUMNS = 3, 4
TILE_DEPTH = 3
STRIP_SHARE = 0.24
NODE_COLOUR_NOTE = "node colour = share who survived (blue) or did not (red)  ·  size = passengers"


class TreeView(AlgorithmView):
    """The decision tree's diagram: the tree growing, then one passenger's path through it.

    Attributes:
        nodes_drawn: How many nodes the last drawing put on screen.
        path_drawn: How many edges of the passenger's path are on screen (0 in training).
    """

    nodes_drawn = 0
    path_drawn = 0

    def show_training(self, model, frame) -> None:
        """Draws the tree's top levels, nodes coloured by survival share and sized by passengers.

        Args:
            model: The fitted ``DecisionTreeClassifier``.
            frame: The training frame; unused, since everything drawn comes from the model.
        """
        self._blank("The algorithm: a decision tree, growing")
        drawn = draw_survival_tree(self.ax, model.tree_, TREE_BOX, DRAW_DEPTH, splits=self.splits,
                                   node_size=420.0, label_depth=LABEL_DEPTH)
        self.nodes_drawn, self.path_drawn = drawn.nodes, 0
        hidden = "" if model.get_depth() <= DRAW_DEPTH else f"  (top {DRAW_DEPTH} levels shown)"
        self.caption(f"depth {model.get_depth()}  ·  {model.get_n_leaves():,} leaves{hidden}\n"
                     f"{NODE_COLOUR_NOTE}  ·  left = yes", y=0.0)

    def show_prediction(self, model, x: np.ndarray, passenger_number: int) -> None:
        """Highlights the passenger's path and says how many of the leaf's training passengers survived.

        A path deeper than the drawn levels ends in a note of how many splits are hidden.

        Args:
            model: The fitted ``DecisionTreeClassifier``.
            x: The passenger's standardised features.
            passenger_number: The passenger's number, for the title.
        """
        self._blank(f"Passenger #{passenger_number:,} answers the questions")
        t = model.tree_
        path = model.decision_path(x[None, :]).indices
        drawn = draw_survival_tree(self.ax, t, TREE_BOX, DRAW_DEPTH, splits=self.splits, path=path,
                                   node_size=420.0, label_depth=LABEL_DEPTH)
        self.nodes_drawn, self.path_drawn = drawn.nodes, drawn.path_edges
        below = len(path) - 1 - drawn.path_edges
        if below > 0 and drawn.path_end is not None:
            self.ax.text(drawn.path_end[0], drawn.path_end[1] - 0.04, f"↓ {below} more splits",
                         ha="center", va="top", fontsize=10, color=base.INK_PRIMARY)
        leaf = path[-1]
        passengers = int(t.n_node_samples[leaf])
        p = float(t.value[leaf].ravel()[1])
        self.caption(f"{leaf_note(round(p * passengers), passengers)}  →  P(survive) = {p:.0%}", y=0.0)


class ForestView(AlgorithmView):
    """The random forest's diagram: the first trees as tiles, and every tree's vote for a passenger.

    Attributes:
        strip: The strip below the tiles, for every tree's P(survive); hidden during training.
        tiles_drawn: How many tree tiles the last drawing showed.
        votes_drawn: How many trees' votes the last prediction plotted.
    """

    tiles_drawn = 0
    votes_drawn = 0

    def __init__(self, ax: Axes, splits: TitanicSplits):
        """Makes room for the vote strip under the tiles.

        Args:
            ax: The view's axes.
            splits: The splits.
        """
        self.strip = split_area(ax, STRIP_SHARE)
        super().__init__(ax, splits)

    def _tiles(self, model, x: np.ndarray | None) -> None:
        """Draws the first trees as small tiles; with a passenger, each shows its path and P(survive)."""
        trees = model.estimators_[:TILE_ROWS * TILE_COLUMNS]
        width, height = 1.0 / TILE_COLUMNS, 0.92 / TILE_ROWS
        for index, tree in enumerate(trees):
            row, column = divmod(index, TILE_COLUMNS)
            box = (column * width + 0.02, 0.92 - (row + 1) * height + 0.05, width - 0.04, height - 0.1)
            path = tree.decision_path(x[None, :]).indices if x is not None else None
            draw_survival_tree(self.ax, tree.tree_, box, TILE_DEPTH, path=path, node_size=45.0)
            if x is not None:
                vote = float(tree.predict_proba(x[None, :])[0, 1])
                self.ax.text(box[0] + box[2] / 2, box[1] - 0.012, f"{vote:.0%}", ha="center",
                             va="top", fontsize=9, color=base.INK_PRIMARY)
        self.tiles_drawn = len(trees)

    def show_training(self, model, frame) -> None:
        """Draws the first trees' top levels and says how many trees the forest has.

        Args:
            model: The fitted ``RandomForestClassifier``.
            frame: The training frame; unused, since everything drawn comes from the model.
        """
        self._blank("The algorithm: a forest of trees, each on its own bootstrap sample")
        self._tiles(model, None)
        self.strip.set_visible(False)
        shown = min(len(model.estimators_), TILE_ROWS * TILE_COLUMNS)
        self.caption(f"{len(model.estimators_)} trees so far (first {shown} shown, top {TILE_DEPTH} "
                     "levels each)  ·  the forest averages their P(survive)", y=-0.02)

    def show_prediction(self, model, x: np.ndarray, passenger_number: int) -> None:
        """Shows every tree's P(survive) for the passenger and their average, the forest's answer.

        Args:
            model: The fitted ``RandomForestClassifier``.
            x: The passenger's standardised features.
            passenger_number: The passenger's number, for the title (and the strip's jitter seed).
        """
        self._blank(f"Passenger #{passenger_number:,}: every tree gives its P(survive)")
        self._tiles(model, x)
        votes = np.array([tree.predict_proba(x[None, :])[0, 1] for tree in model.estimators_])
        self.votes_drawn = len(votes)
        self.strip.set_visible(True)
        style_strip(self.strip, f"all {len(votes)} trees  ·  the forest's answer = their average "
                    f"= {votes.mean():.0%}", "one tree's P(survive)")
        jitter = np.random.default_rng(passenger_number).uniform(-0.3, 0.3, len(votes))
        self.strip.scatter(votes, jitter, s=18, c=[theme.survival_rgba(v) for v in votes], linewidths=0)
        self.strip.axvline(0.5, color=base.INK_MUTED, linestyle="--", linewidth=0.8)
        self.strip.axvline(votes.mean(), color=base.INK_PRIMARY, linewidth=2.0)
        self.strip.set_xlim(-0.02, 1.02)
        self.strip.set_ylim(-0.6, 0.6)
        self.strip.set_yticks([])


class BoostingView(AlgorithmView):
    """Gradient boosting's diagram: the newest tree's corrections, then a passenger's chance built tree by tree.

    Attributes:
        strip: The bar chart below: each tree's correction.
        bars_drawn: How many bars the last training drawing showed.
        trajectory: The passenger's P(survive) line, after each tree (None before a prediction).
    """

    bars_drawn = 0
    trajectory = None

    def __init__(self, ax: Axes, splits: TitanicSplits):
        """Makes room for the bar strip under the diagram.

        Args:
            ax: The view's axes.
            splits: The splits, to word the newest tree's questions in real units.
        """
        self.strip = split_area(ax, STRIP_SHARE)
        super().__init__(ax, splits)

    @staticmethod
    def _mean_correction(tree) -> float:
        """A tree's mean |correction| over the training passengers, from its leaves' values and sizes."""
        t = tree.tree_
        leaves = t.children_left == -1
        return float(np.sum(t.n_node_samples[leaves] * np.abs(t.value[leaves].ravel())) / t.n_node_samples[0])

    def show_training(self, model, frame) -> None:
        """Draws the newest tree, leaves coloured by the correction they add, and every tree's mean correction.

        Args:
            model: The fitted ``GradientBoostingClassifier``.
            frame: The training frame; unused, since everything drawn comes from the model.
        """
        self._blank(f"The algorithm: tree #{model.n_estimators_} corrects the log-odds so far")
        latest = model.estimators_[-1, 0]
        scale = float(np.abs(latest.tree_.value).max()) or 1.0
        draw_tree(self.ax, latest.tree_, (0.05, 0.12, 0.9, 0.76), TILE_DEPTH, node_size=520.0,
                  label_depth=2, splits=self.splits, signed_scale=scale, split_label=split_label)
        self.caption("leaf colour = the log-odds correction it adds (blue raises, red lowers the chance)",
                     y=0.0)
        sizes = [self._mean_correction(tree) for tree in model.estimators_[:, 0]]
        style_strip(self.strip, "average size of each tree's correction (shrinks as the errors shrink)",
                    "tree number")
        self.strip.bar(np.arange(1, len(sizes) + 1), sizes, width=0.8, color=base.SERIES_1)
        self.bars_drawn = len(sizes)

    def show_prediction(self, model, x: np.ndarray, passenger_number: int) -> None:
        """Plots the passenger's P(survive) after each tree, from the base rate, with each tree's correction.

        Args:
            model: The fitted ``GradientBoostingClassifier``.
            x: The passenger's standardised features.
            passenger_number: The passenger's number, for the title.
        """
        self._blank("")
        self.ax.set_axis_on()
        base.style_axes(self.ax)
        rate = float(model.learning_rate)
        prior = float(model.init_.class_prior_[1])
        start = float(np.log(prior / (1.0 - prior)))
        steps = np.array([tree.predict(x[None, :])[0] for tree in model.estimators_[:, 0]]) * rate
        chance = sigmoid(np.concatenate([[start], start + np.cumsum(steps)]))
        (self.trajectory,) = self.ax.plot(np.arange(len(chance)), chance, color=base.SERIES_1, linewidth=2.2)
        self.ax.axhline(0.5, color=base.INK_MUTED, linestyle="--", linewidth=1.0)
        self.ax.plot([len(chance) - 1], [chance[-1]], "o", color=theme.survival_rgba(chance[-1]),
                     markersize=10, markeredgecolor=base.INK_PRIMARY, markeredgewidth=1.0)
        self.ax.set_xlim(0, len(chance) - 1)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_title(f"Passenger #{passenger_number:,}: the chance is built tree by tree", fontsize=13)
        self.ax.set_xlabel("trees added so far (m)")
        self.ax.set_ylabel("σ(F_m(x)) = P(survive)")
        self.ax.annotate(f"base rate {prior:.0%}", (0, chance[0]), xytext=(6, 6), textcoords="offset points",
                         fontsize=10, color=base.INK_SECONDARY)
        self.ax.annotate(f"P = {chance[-1]:.0%}", (len(chance) - 1, chance[-1]), xytext=(-10, 10),
                         textcoords="offset points", ha="right", fontsize=11, color=base.INK_PRIMARY)
        style_strip(self.strip, "each tree's log-odds correction ν·h_m(x) for this passenger", "tree number")
        scale = float(np.abs(steps).max()) or 1.0
        self.strip.bar(np.arange(1, len(steps) + 1), steps, width=0.8,
                       color=[signed_rgba(s, scale, floor=0.5) for s in steps])
        self.strip.axhline(0.0, color=base.INK_MUTED, linewidth=0.8)
