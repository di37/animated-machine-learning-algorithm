"""Ensembles: a random forest as a grid of voting trees; gradient boosting as a chain of corrections.

- A random forest is many trees, each grown on its own bootstrap sample (training houses drawn
  with replacement). The view tiles the first dozen; at inference each tile highlights the house's
  path and prints its tree's vote, and a strip chart shows every vote and their average, which is
  the forest's prediction.
- Gradient boosting starts from the mean price F₀ and adds trees one at a time, each fitted to
  what the trees before it got wrong. The view draws the newest tree, its leaves coloured by the
  correction they add, over a strip of every tree's average correction size, which shrinks as the
  residuals do. At inference it plots F_m(x), the price after m trees, climbing from F₀ to ŷ.

Both views give the bottom of their area to the strip chart (``STRIP_SHARE``).
"""

import numpy as np
from matplotlib.axes import Axes

from house_anim import theme
from house_anim.algorithms.common import AlgorithmView, draw_tree, price_k, signed_rgba
from house_anim.config import TARGET_TO_THOUSANDS
from house_anim.data import HousingSplits

TILE_ROWS, TILE_COLUMNS = 3, 4
TILE_DEPTH = 3
STRIP_SHARE = 0.24  # bottom share of the view's area given to the strip chart


def _split_area(ax: Axes) -> Axes:
    """Shrink ``ax`` to the top of its area and return a strip chart axes underneath it."""
    pos = ax.get_position()
    strip_height = pos.height * STRIP_SHARE
    main_bottom = pos.y0 + strip_height * 1.35  # leaves room for the main chart's x-label
    ax.set_position([pos.x0, main_bottom, pos.width, pos.y0 + pos.height - main_bottom])
    strip = ax.figure.add_axes([pos.x0 + pos.width * 0.1, pos.y0 + strip_height * 0.15,
                                pos.width * 0.86, strip_height * 0.62])
    return strip


def _style_strip(strip: Axes, title: str, xlabel: str) -> None:
    """Clears the strip chart and gives it the house style, a title and an x label."""
    strip.clear()
    theme.style_axes(strip)
    strip.set_title(title, fontsize=10)
    strip.set_xlabel(xlabel)


class ForestView(AlgorithmView):
    """The random forest's diagram: a grid of small trees, and at inference a strip of every vote.

    Attributes:
        strip: The strip chart's axes, under the grid (hidden while training).
        tiles_drawn: How many trees the last drawing tiled (the tests read it).
        votes_drawn: How many votes the last prediction drew on the strip.
    """

    tiles_drawn = 0
    votes_drawn = 0

    def __init__(self, ax: Axes, splits: HousingSplits):
        """Splits the view's area into the tree grid and the strip chart under it.

        Args:
            ax: The view's axes; it is shrunk to make room for the strip.
            splits: The data.
        """
        self.strip = _split_area(ax)
        super().__init__(ax, splits)

    def _tiles(self, model, x: np.ndarray | None) -> list[float]:
        """Draws the first trees as tiles; with a house ``x``, highlights its paths and returns the tiles' votes."""
        votes = []
        trees = model.estimators_[:TILE_ROWS * TILE_COLUMNS]
        width, height = 1.0 / TILE_COLUMNS, 0.92 / TILE_ROWS
        for index, tree in enumerate(trees):
            row, column = divmod(index, TILE_COLUMNS)
            box = (column * width + 0.02, 0.92 - (row + 1) * height + 0.05, width - 0.04, height - 0.1)
            path = tree.decision_path(x[None, :]).indices if x is not None else None
            draw_tree(self.ax, tree.tree_, box, TILE_DEPTH, path=path, node_size=45.0)
            if x is not None:
                vote = float(tree.predict(x[None, :])[0])
                votes.append(vote)
                self.ax.text(box[0] + box[2] / 2, box[1] - 0.012, price_k(vote), ha="center",
                             va="top", fontsize=9, color=theme.INK_PRIMARY)
        self.tiles_drawn = len(trees)
        return votes

    def show_training(self, model, frame) -> None:
        """Tiles the forest's first trees and says how many it has so far.

        Args:
            model: The fitted ``RandomForestRegressor``.
            frame: The step's ``TrainingFrame`` (unused: the trees are the picture).
        """
        self._blank("The algorithm: a forest of trees, each on its own bootstrap sample")
        self._tiles(model, None)
        self.strip.set_visible(False)
        shown = min(len(model.estimators_), TILE_ROWS * TILE_COLUMNS)
        self.caption(f"{len(model.estimators_)} trees so far (first {shown} shown, top {TILE_DEPTH} "
                     "levels each)  ·  the forest predicts their average", y=-0.02)

    def show_prediction(self, model, x: np.ndarray, house_number: int) -> None:
        """Shows every tree's vote for the house and their average.

        The strip spreads the votes vertically at random so they do not hide each other; the
        spread is seeded by ``house_number``, so a house always looks the same.

        Args:
            model: The fitted ``RandomForestRegressor``.
            x: The house's scaled features.
            house_number: The house's position among the test houses, from 1.
        """
        self._blank(f"House #{house_number:,}: every tree votes")
        self._tiles(model, x)
        votes = np.array([tree.predict(x[None, :])[0] for tree in model.estimators_])
        self.votes_drawn = len(votes)
        self.strip.set_visible(True)
        _style_strip(self.strip, f"all {len(votes)} votes  ·  prediction = their average",
                     "tree vote (\\$k)")
        jitter = np.random.default_rng(house_number).uniform(-0.3, 0.3, len(votes))
        self.strip.scatter(votes * TARGET_TO_THOUSANDS, jitter, s=18, color=theme.SERIES_1, alpha=0.6)
        self.strip.axvline(votes.mean() * TARGET_TO_THOUSANDS, color=theme.INK_PRIMARY, linewidth=2.0)
        self.strip.set_ylim(-0.6, 0.6)
        self.strip.set_yticks([])


class BoostingView(AlgorithmView):
    """Gradient boosting's diagram: the newest correcting tree, and at inference the price built tree by tree.

    Attributes:
        strip: The strip chart's axes, under the main chart.
        bars_drawn: How many trees' bars the last training drawing put on the strip (the tests read it).
        trajectory: The F_m(x) line of the last prediction; None before the first.
    """

    bars_drawn = 0
    trajectory = None

    def __init__(self, ax: Axes, splits: HousingSplits):
        """Splits the view's area into the main chart and the strip chart under it.

        Args:
            ax: The view's axes; it is shrunk to make room for the strip.
            splits: The data, to label the newest tree's splits in real units.
        """
        self.strip = _split_area(ax)
        super().__init__(ax, splits)

    @staticmethod
    def _mean_correction(tree) -> float:
        """How much a tree changes a training house's price on average: |leaf value|, weighted by leaf size."""
        t = tree.tree_
        leaves = t.children_left == -1
        return float(np.sum(t.n_node_samples[leaves] * np.abs(t.value[leaves].ravel())) / t.n_node_samples[0])

    def show_training(self, model, frame) -> None:
        """Draws the newest tree, coloured by its corrections, over every tree's average correction size.

        Args:
            model: The fitted ``GradientBoostingRegressor``.
            frame: The step's ``TrainingFrame`` (unused: the trees are the picture).
        """
        self._blank(f"The algorithm: tree #{model.n_estimators_} corrects what the others got wrong")
        latest = model.estimators_[-1, 0]
        scale = float(np.abs(latest.tree_.value).max()) or 1.0
        draw_tree(self.ax, latest.tree_, (0.05, 0.12, 0.9, 0.76), TILE_DEPTH, node_size=520.0,
                  label_depth=2, splits=self.splits, signed_scale=scale)
        self.caption("leaf colour = the correction it adds (blue raises, red lowers the price)", y=0.0)
        sizes = [self._mean_correction(tree) for tree in model.estimators_[:, 0]]
        _style_strip(self.strip, "average size of each tree's correction (shrinks as residuals shrink)",
                     "tree number")
        self.strip.bar(np.arange(1, len(sizes) + 1), sizes, width=0.8, color=theme.SERIES_1)
        self.bars_drawn = len(sizes)

    def show_prediction(self, model, x: np.ndarray, house_number: int) -> None:
        """Plots the house's price after each tree, F₀ to ŷ, over each tree's correction ν·h_m(x).

        Args:
            model: The fitted ``GradientBoostingRegressor``.
            x: The house's scaled features.
            house_number: The house's position among the test houses, from 1, for the title.
        """
        self._blank("")
        self.ax.set_axis_on()
        theme.style_axes(self.ax)
        rate = float(model.learning_rate)
        start = float(np.ravel(model.init_.constant_)[0])
        steps = np.array([tree.predict(x[None, :])[0] for tree in model.estimators_[:, 0]]) * rate
        path = np.concatenate([[start], start + np.cumsum(steps)]) * TARGET_TO_THOUSANDS
        (self.trajectory,) = self.ax.plot(np.arange(len(path)), path, color=theme.SERIES_1, linewidth=2.2)
        self.ax.axhline(start * TARGET_TO_THOUSANDS, color=theme.INK_MUTED, linestyle="--", linewidth=1.0)
        self.ax.plot([len(path) - 1], [path[-1]], "o", color=theme.SERIES_1, markersize=9,
                     markeredgecolor=theme.SURFACE, markeredgewidth=2.0)
        self.ax.set_xlim(0, len(path) - 1)
        spread = max(float(np.ptp(path)), 1.0)
        self.ax.set_ylim(path.min() - 0.15 * spread, path.max() + 0.25 * spread)
        self.ax.set_title(f"House #{house_number:,}: the price is built tree by tree", fontsize=13)
        self.ax.set_xlabel("trees added so far (m)")
        self.ax.set_ylabel("F_m(x) in \\$k")
        self.ax.annotate(f"F0 = {price_k(start)}", (0, start * TARGET_TO_THOUSANDS), xytext=(6, 6),
                         textcoords="offset points", fontsize=10, color=theme.INK_SECONDARY)
        self.ax.annotate(f"ŷ = {price_k(path[-1] / TARGET_TO_THOUSANDS)}", (len(path) - 1, path[-1]),
                         xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=11,
                         color=theme.INK_PRIMARY)
        _style_strip(self.strip, "each tree's correction ν·h_m(x) for this house", "tree number")
        scale = float(np.abs(steps).max()) or 1.0
        self.strip.bar(np.arange(1, len(steps) + 1), steps * TARGET_TO_THOUSANDS, width=0.8,
                       color=[signed_rgba(s, scale, floor=0.5) for s in steps])
        self.strip.axhline(0.0, color=theme.INK_MUTED, linewidth=0.8)
