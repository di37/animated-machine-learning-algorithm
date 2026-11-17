"""A random forest built tree by tree, each on its own bootstrap sample, its answer their plain average.

Each new tree is fitted on its own bootstrap sample, and the forest's answer is the plain average of
every tree so far (aggregation, not gradient updates).

A bootstrap sample is as many districts as the training set, drawn with replacement, so each tree
sees a slightly different data set and over-fits it in its own way. One deep tree is jagged: its
small leaves chase local noise. The trees disagree mostly on that noise, so averaging them cancels
much of it (lower variance) while keeping what they agree on. Unlike boosting, no tree learns from
another: adding a tree never changes the ones already there. The forest is fitted with scikit-learn's
``warm_start``, so growing it to the next count in ``TREE_COUNTS`` fits only the new trees, and every
snapshot is the real forest at that size.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from sklearn.ensemble import RandomForestRegressor

from animkit.lessons import theme
from animkit.lessons.timing import ease
from house_anim.config import RANDOM_STATE
from house_anim.data import read_only
from house_anim.lessons.config import INCOME_TO_K, PRICE_TO_K, X_RANGE, Y_RANGE_K
from house_anim.lessons.data import IncomeData, Query
from house_anim.lessons.lesson import (
    Explanation,
    Lesson,
    Panel,
    Snapshot,
    Story,
    make_snapshot,
    money_k,
    plural,
    predict_one,
    verified,
)
from house_anim.lessons.lesson_diagrams import forest_diagram
from house_anim.lessons.plots import K_FORMAT

TREE_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50)
OVERLAY_TREES = 12  # faint single-tree curves drawn on the main plot (the newest ones)


def min_leaf(n_rows: int) -> int:
    """The smallest leaf allowed: small (one in 600 of the rows), so each tree is deep and jagged.

    Args:
        n_rows: How many rows the trees train on.

    Returns:
        The minimum number of training rows per leaf, at least 1.
    """
    return max(1, n_rows // 600)


@dataclass(frozen=True)
class ForestState:
    """The forest at one size, for the panel and the main plot's faint single-tree curves.

    Attributes:
        n_trees: How many trees the forest has.
        tree_curves: Every tree's own prediction on the grid ($100k), shape (n_trees, grid), oldest first.
    """

    n_trees: int
    tree_curves: np.ndarray  # (n_trees, grid) — every tree's own prediction curve


def _average_of(trees: list) -> Callable[[np.ndarray], np.ndarray]:
    """A predictor that averages these trees' predictions: the forest at this size."""
    return lambda x: np.mean([t.predict(np.asarray(x, dtype=float).reshape(-1, 1)) for t in trees], axis=0)


class ForestPanel(Panel):
    """The newest tree against the forest's average while training; at inference, every tree's vote."""

    def __init__(self, data: IncomeData, forest: RandomForestRegressor, final_curves: np.ndarray):
        """Keeps the data, the fitted forest and the full forest's single-tree curves.

        Args:
            data: The lesson's data.
            forest: The fully grown forest.
            final_curves: Every tree's prediction on the grid ($100k), for the inference overlay.
        """
        self._data = data
        self._forest = forest
        self._final_curves = final_curves

    def _income_axes(self, ax: Axes, title: str) -> None:
        """Sets up the panel with the main plot's income and price axes."""
        theme.style_axes(ax)
        ax.set_xlim(X_RANGE[0] * INCOME_TO_K, X_RANGE[1] * INCOME_TO_K)
        ax.set_ylim(*Y_RANGE_K)
        ax.xaxis.set_major_formatter(K_FORMAT)
        ax.yaxis.set_major_formatter(K_FORMAT)
        ax.set_title(title)

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws the newest tree's jagged curve beside the smoother average of all the trees so far.

        Args:
            ax: The side panel.
            snapshot: The forest being shown; its ``detail`` is a ``ForestState``.
        """
        state: ForestState = snapshot.detail
        self._income_axes(ax, f"Newest tree (#{state.n_trees}) vs the forest average")
        grid = self._data.grid * INCOME_TO_K
        ax.plot(grid, state.tree_curves[-1] * PRICE_TO_K, color=theme.HIGHLIGHT, linewidth=1.8,
                label=f"tree #{state.n_trees} alone")
        ax.plot(grid, snapshot.curve * PRICE_TO_K, color=theme.MODEL, linewidth=3.0,
                label=f"average of {plural(state.n_trees, 'tree')}")
        ax.legend(loc="upper left")

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Draws every tree's vote for the unseen district, sorted, then their average once all are in.

        Args:
            ax: The side panel.
            query: The unseen district.
            t: How far through the votes, from 0 to 1.
        """
        x = np.array([[query.income]])
        votes = np.sort([tree.predict(x)[0] for tree in self._forest.estimators_]) * PRICE_TO_K
        shown = int(np.ceil(len(votes) * ease(t)))
        theme.style_axes(ax, grid_axis="x")
        ax.set_xlim(votes.min() - 40, votes.max() + 40)
        ax.set_ylim(-1, len(votes))
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(K_FORMAT)
        ax.set_title(f"Each of the {len(votes)} trees votes a price")
        ax.scatter(votes[:shown], np.arange(shown), s=34, color=theme.HIGHLIGHT, zorder=3)
        if shown == len(votes):
            mean = float(votes.mean())
            ax.axvline(mean, color=theme.MODEL, linewidth=3.0, zorder=4)
            ax.text(mean, len(votes) * 0.98, f" average ${mean:,.0f}k", fontsize=13, weight="bold",
                    color=theme.NAVY, va="top")

    def overlay_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws the newest single trees as faint curves under the forest's average on the main plot.

        Args:
            ax: The main plot.
            snapshot: The forest being shown; its ``detail`` is a ``ForestState``.
        """
        for curve in snapshot.detail.tree_curves[-OVERLAY_TREES:]:
            ax.plot(self._data.grid * INCOME_TO_K, curve * PRICE_TO_K, color=theme.HIGHLIGHT,
                    linewidth=1.0, alpha=0.18, zorder=3)

    def overlay_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Draws the full forest's newest single trees, fainter still, behind the final curve.

        Args:
            ax: The main plot.
            query: The unseen district (unused: the overlay is the same for any district).
            t: How far through the animation (unused: the overlay does not move).
        """
        for curve in self._final_curves[-OVERLAY_TREES:]:
            ax.plot(self._data.grid * INCOME_TO_K, curve * PRICE_TO_K, color=theme.HIGHLIGHT,
                    linewidth=1.0, alpha=0.1, zorder=3)


def _story(first: Snapshot, last: Snapshot) -> Story:
    """Every word of the random-forest video, with one tree's errors against the full forest's."""
    return Story(
        key="random_forest", title="Random Forest",
        method=f"{TREE_COUNTS[-1]} bootstrapped trees, averaged",
        objective=("A single deep tree is jagged. A random forest grows many trees, each on a "
                   "different bootstrap sample of districts, and averages them. Watch the jagged "
                   "single tree turn into a smooth, stable curve as trees are added."),
        initial=(f"One tree on its own bootstrap sample: jagged, because small leaves chase local noise. Validation "
                 f"RMSE ${first.val_rmse_k:,.0f}k."),
        training=("Each frame adds already-fitted trees and re-averages. Nothing is optimised "
                  "here: faint teal lines are the newest single trees, the blue line is the average of all."),
        final=(f"{TREE_COUNTS[-1]} trees: train RMSE ${last.train_rmse_k:,.0f}k, validation "
               f"${last.val_rmse_k:,.0f}k (one tree alone: ${first.val_rmse_k:,.0f}k). "
               "Averaging cancels out what the trees disagree on."),
        inference=("For an unseen district every tree casts a vote (its own prediction); the "
                   "forest reports the average of all the votes."),
        takeaway=("Averaging many over-fitted, slightly different trees cancels their individual "
                  "noise: lower variance, better predictions."),
        limitation=("the forest is slower and far less readable than one tree, and like every tree "
                    "model it cannot predict beyond the prices it has seen."),
        progress_label="number of trees")


def build(data: IncomeData) -> Lesson:
    """Builds the random-forest lesson: one snapshot per forest size in ``TREE_COUNTS``.

    The final model is the full forest.

    Args:
        data: The income view to learn from.

    Returns:
        The lesson, with the votes panel and animkit's forest diagram.
    """
    forest = RandomForestRegressor(n_estimators=TREE_COUNTS[0], min_samples_leaf=min_leaf(data.n_train),
                                   bootstrap=True, warm_start=True, n_jobs=1, random_state=RANDOM_STATE)
    X = data.x_train.reshape(-1, 1)
    built: list[Snapshot] = []
    for previous, count in zip((0, *TREE_COUNTS), TREE_COUNTS):
        forest.set_params(n_estimators=count).fit(X, data.y_train)  # warm start: only the new trees fit
        trees = list(forest.estimators_)
        curves = read_only(np.array([t.predict(data.grid.reshape(-1, 1)) for t in trees]))
        note = ("Tree 1, fitted on a bootstrap sample\non its own it is jagged: small leaves chase local noise"
                if count == 1 else f"{count} trees: {plural(count - previous, 'new tree')} added\n"
                                   "the average smooths out each tree's quirks")
        built.append(make_snapshot(_average_of(trees), data, label=plural(count, "tree"), progress=count,
                                   note=note, detail=ForestState(count, curves)))
    snapshots = tuple(built)

    def predict(x: np.ndarray) -> np.ndarray:
        """The full forest's prices.

        Args:
            x: Raw incomes ($10k).

        Returns:
            The predicted prices ($100k).
        """
        return forest.predict(np.asarray(x, dtype=float).reshape(-1, 1))

    def explain(income: float) -> Explanation:
        """Works the forest's price for one income: the spread of the trees' votes, then their average.

        Args:
            income: The income ($10k).

        Returns:
            The worked lines, checked against the model's own prediction.
        """
        votes = np.array([tree.predict(np.array([[income]]))[0] for tree in forest.estimators_])
        worked = float(votes.mean())
        low, high = np.percentile(votes, [25, 75])
        lines = [f"{len(votes)} trees, each trained on its own bootstrap sample",
                 f"lowest vote {money_k(votes.min())}, highest {money_k(votes.max())}",
                 f"middle half of the votes: {money_k(low)} to {money_k(high)}",
                 f"ŷ = average of the {len(votes)} votes = {money_k(worked)}"]
        return verified(lines, worked, predict_one(predict, income))

    story = _story(snapshots[0], snapshots[-1])
    return Lesson(
        story=story, data=data, snapshots=snapshots, final_index=len(snapshots) - 1, predict=predict,
        explain=explain, panel=ForestPanel(data, forest, snapshots[-1].detail.tree_curves),
        diagram=forest_diagram(forest),
        config={"model": "RandomForestRegressor", "n_estimators": str(TREE_COUNTS[-1]),
                "min_samples_leaf": str(min_leaf(data.n_train)), "bootstrap": "True",
                "random_state": str(RANDOM_STATE)})
