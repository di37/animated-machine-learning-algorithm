"""A regression tree grown one split at a time (best-first), then an unseen house walking root to leaf.

Each snapshot is a separate fit with ``max_leaf_nodes`` one higher. scikit-learn grows such trees
best-first, so every fit is the previous tree plus the single split that cuts squared error most;
the tests check that prefix property, which is what lets the diagram grow node by node.
Nodes are keyed by their path from the root ("", "L", "LR", …) so the layout never jumps.

A regression tree asks yes/no questions about the input ("is income ≤ $52k?") and sends a district
left on yes, right on no, until it reaches a leaf; the leaf's prediction is the mean price of the
training districts that landed there. Growing it is greedy: each new split is the one income cut
that lowers the total squared error the most. The curve it draws over income is therefore a
staircase, one flat step per leaf. The side panel draws the tree itself (laid out by animkit's
``tree_layout``), and the main plot marks every split as a vertical line.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from sklearn.tree import DecisionTreeRegressor

from animkit.lessons import theme
from animkit.lessons.stage import reset_axes
from animkit.lessons.timing import ease
from animkit.lessons.tree_layout import layout
from house_anim.config import RANDOM_STATE
from house_anim.lessons.config import INCOME_TO_K, X_RANGE, Y_RANGE_K
from house_anim.lessons.data import IncomeData, Query
from house_anim.lessons.lesson import (
    Explanation,
    Lesson,
    Panel,
    Snapshot,
    Story,
    income_k,
    make_snapshot,
    money_k,
    plural,
    predict_one,
    verified,
)

MAX_LEAVES = 14
LABEL_DEPTH = 3  # thresholds are printed on the top levels only; deeper nodes stay as dots


def min_leaf(n_rows: int) -> int:
    """The smallest leaf allowed: one in 200 of the training rows.

    Args:
        n_rows: How many rows the tree trains on.

    Returns:
        The minimum number of training rows per leaf, at least 1.
    """
    return max(1, n_rows // 200)


@dataclass(frozen=True)
class TreeState:
    """One tree's structure, keyed by path, for the panel and the main plot's split lines.

    Attributes:
        splits: (path key, income threshold in $10k) of every internal node, depth first from the root.
        leaves: (path key, mean price in $100k, training districts) of every leaf, left to right.
        newest: The key of the split this snapshot added; None for the single-leaf root.
    """

    splits: tuple[tuple[str, float], ...]  # (path key, income threshold) of every internal node
    leaves: tuple[tuple[str, float, int], ...]  # (path key, mean price, training districts)
    newest: str | None  # the split this snapshot added


def tree_state(model: DecisionTreeRegressor, previous: TreeState | None) -> TreeState:
    """Reads a fitted tree's splits and leaves, keyed by path, and finds the split it added.

    Args:
        model: The fitted tree.
        previous: The tree with one leaf fewer; None if there is none.

    Returns:
        The tree's state; ``newest`` is the first split not in ``previous``.
    """
    t = model.tree_
    splits, leaves = [], []

    def visit(node: int, key: str) -> None:
        """Records ``node`` under ``key``, then its left and right subtrees, depth first.

        Args:
            node: The node's index in scikit-learn's tree arrays.
            key: Its path from the root.
        """
        if t.children_left[node] == -1:
            leaves.append((key, float(t.value[node].ravel()[0]), int(t.n_node_samples[node])))
            return
        splits.append((key, float(t.threshold[node])))
        visit(t.children_left[node], key + "L")
        visit(t.children_right[node], key + "R")

    visit(0, "")
    before = set() if previous is None else {key for key, _ in previous.splits}
    added = [key for key, _ in splits if key not in before]
    return TreeState(tuple(splits), tuple(leaves), added[0] if added else None)


def root_state(data: IncomeData) -> TreeState:
    """The tree before any split: one leaf holding every training district at their mean price.

    Args:
        data: The lesson's data.

    Returns:
        The single-leaf state.
    """
    return TreeState((), (("", float(data.y_train.mean()), data.n_train),), None)


def path_keys(state: TreeState, income: float) -> list[str]:
    """Root-to-leaf keys for one income, following ≤ threshold to the left.

    Args:
        state: The tree.
        income: The income to route ($10k).

    Returns:
        The keys of every node on its path, from the root ("") to its leaf.
    """
    thresholds = dict(state.splits)
    keys = [""]
    while keys[-1] in thresholds:
        keys.append(keys[-1] + ("L" if income <= thresholds[keys[-1]] else "R"))
    return keys


class TreePanel(Panel):
    """The tree itself, growing node by node; at inference the new house's path is lit to its leaf.

    Every node sits where the final tree puts it, so the layout does not jump as the tree grows. Leaf
    size follows how many training districts it holds; leaf colour follows its mean price.
    """

    def __init__(self, final: TreeState):
        """Lays out the final tree once and fixes the leaf colour scale to its prices.

        Args:
            final: The fully grown tree.
        """
        self._final = final
        self._points = layout([key for key, _, _ in final.leaves], [key for key, _ in final.splits])
        values = [v for _, v, _ in final.leaves]
        self._norm = (min(values), max(values))
        self._cmap = theme.value_colormap()

    def _colour(self, value: float):
        """A leaf's colour for its mean price, on the final tree's range (darker = higher)."""
        low, high = self._norm
        return self._cmap(0.15 + 0.85 * (value - low) / max(high - low, 1e-9))

    def _draw(self, ax: Axes, state: TreeState, title: str, path: list[str] | None = None) -> None:
        """Draws a tree's edges, splits (top levels and the newest labelled) and leaves, and lights ``path``."""
        reset_axes(ax)
        ax.text(0.04, 0.9, title, fontsize=theme.PANEL_TITLE_SIZE, weight="bold", color=theme.NAVY,
                va="center")
        p = self._points
        edges = [(p[key], p[key + side]) for key, _ in state.splits for side in "LR"]
        ax.add_collection(LineCollection(edges, colors=theme.INACTIVE, linewidths=1.4, zorder=1))
        if path:
            ax.add_collection(LineCollection([(p[a], p[b]) for a, b in zip(path, path[1:])],
                                             colors=theme.HIGHLIGHT, linewidths=4.0, zorder=2))
        for key, threshold in state.splits:
            newest = key == state.newest
            ax.scatter(*p[key], s=110 if newest else 60, color=theme.NAVY, zorder=3,
                       edgecolors=theme.HIGHLIGHT if newest else theme.PANEL, linewidths=3 if newest else 1.5)
            if len(key) < LABEL_DEPTH or newest:
                ax.text(p[key][0], p[key][1] + 0.035, f"≤ ${threshold * INCOME_TO_K:,.0f}k", ha="center",
                        va="bottom", fontsize=11, color=theme.NAVY, weight="bold" if newest else "normal",
                        zorder=5, bbox=dict(boxstyle="round,pad=0.15", facecolor=theme.PANEL,
                                            edgecolor="none", alpha=0.85))
        total = sum(n for _, _, n in state.leaves)
        for key, value, count in state.leaves:
            ax.scatter(*p[key], s=50 + 260 * np.sqrt(count / total), color=self._colour(value), zorder=3,
                       edgecolors=theme.PANEL, linewidths=1.5)
        ax.text(0.96, 0.03, "leaf colour: mean price (darker = higher) · left = income ≤ split",
                ha="right", fontsize=11, color=theme.INK_SECONDARY)

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws the tree as it stands at this snapshot, its newest split ringed.

        Args:
            ax: The side panel.
            snapshot: The tree being shown; its ``detail`` is a ``TreeState``.
        """
        state: TreeState = snapshot.detail
        self._draw(ax, state, f"The tree: {plural(len(state.leaves), 'leaf')}, {plural(len(state.splits), 'split')}")

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Lights the unseen district's path one node at a time, then writes its leaf's price.

        Args:
            ax: The side panel.
            query: The unseen district.
            t: How far down the path, from 0 (the root) to 1 (the leaf).
        """
        keys = path_keys(self._final, query.income)
        shown = keys[:max(1, int(np.ceil(len(keys) * ease(t))))]
        self._draw(ax, self._final, "The new house answers yes/no questions", path=shown)
        leaf = {key: value for key, value, _ in self._final.leaves}
        if shown[-1] in leaf:
            x, y = self._points[shown[-1]]
            ax.text(x, y - 0.06, money_k(leaf[shown[-1]]), ha="center", va="top", fontsize=13,
                    weight="bold", color=theme.NAVY, zorder=6)

    def overlay_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Marks every split's income on the main plot, the newest highlighted and labelled.

        Args:
            ax: The main plot.
            snapshot: The tree being shown; its ``detail`` is a ``TreeState``.
        """
        state: TreeState = snapshot.detail
        for key, threshold in state.splits:
            newest = key == state.newest
            ax.axvline(threshold * INCOME_TO_K, color=theme.HIGHLIGHT if newest else theme.INACTIVE,
                       linewidth=2.4 if newest else 1.2, zorder=1)
            if newest:
                ax.text(threshold * INCOME_TO_K + 1, Y_RANGE_K[1] * 0.78, f"new split\n${threshold * INCOME_TO_K:,.1f}k",
                        fontsize=13, color=theme.NAVY, weight="bold", va="top", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=theme.PANEL, edgecolor="none", alpha=0.9))

    def overlay_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Shades the income band the district's answers so far allow, narrowing to its leaf's band.

        Args:
            ax: The main plot.
            query: The unseen district.
            t: How far down the path, from 0 (the root) to 1 (the leaf).
        """
        keys = path_keys(self._final, query.income)
        steps = keys[:max(1, int(np.ceil(len(keys) * ease(t))))]
        thresholds = dict(self._final.splits)
        low, high = X_RANGE[0], X_RANGE[1]
        for parent, child in zip(steps, steps[1:]):
            if child.endswith("L"):
                high = min(high, thresholds[parent])
            else:
                low = max(low, thresholds[parent])
        ax.axvspan(low * INCOME_TO_K, high * INCOME_TO_K, color=theme.HIGHLIGHT, alpha=0.12, zorder=0)


def _note(state: TreeState) -> str:
    """The callout for one tree: its newest split's question, or the single leaf's average price."""
    if state.newest is None:
        return f"1 leaf, no splits yet\nevery district gets the average price, {money_k(state.leaves[0][1])}"
    threshold = dict(state.splits)[state.newest]
    return (f"Split {len(state.splits)}: is income ≤ {income_k(threshold)}?\n"
            "chosen because it cuts squared error the most")


def _story(snapshots: tuple[Snapshot, ...]) -> Story:
    """Every word of the decision-tree video, with the full tree's errors."""
    return Story(
        key="decision_tree", title="Decision Tree",
        method=f"grown best-first to {MAX_LEAVES} leaves",
        objective=("A tree prices a home by asking yes/no questions about income. Watch it grow one "
                   "split at a time, always choosing the cut that reduces squared error the most, "
                   "until the curve becomes a staircase."),
        initial=("Before any split the tree is a single leaf: it predicts the average price for "
                 "every district. The red lines show how wrong that is."),
        training=("Each frame adds one split: the income cut that lowers the total squared error the "
                  "most. Every leaf predicts the mean price of the training districts inside it."),
        final=(f"{MAX_LEAVES} leaves: train RMSE ${snapshots[-1].train_rmse_k:,.0f}k, validation "
               f"${snapshots[-1].val_rmse_k:,.0f}k. Each step of the staircase is one leaf."),
        inference=("The unseen district follows the yes/no questions from the root down to one leaf. "
                   "Its prediction is that leaf's average training price."),
        takeaway=("A tree splits the input into boxes and predicts each box's average: greedy, "
                  "readable, and blind to anything between its split points."),
        limitation=("predictions are flat steps, so two districts on either side of a split get "
                    "very different prices, and deep trees memorise noise."),
        progress_label="number of leaves")


def build(data: IncomeData) -> Lesson:
    """Builds the decision-tree lesson: one fit per leaf count, from the single leaf up to ``MAX_LEAVES``.

    The final model is the full tree. The lesson has no separate diagram: its panel already is the tree.

    Args:
        data: The income view to learn from.

    Returns:
        The lesson.
    """
    states, models = [root_state(data)], [None]
    for leaves in range(2, MAX_LEAVES + 1):
        model = DecisionTreeRegressor(max_leaf_nodes=leaves, min_samples_leaf=min_leaf(data.n_train),
                                      random_state=RANDOM_STATE)
        model.fit(data.x_train.reshape(-1, 1), data.y_train)
        states.append(tree_state(model, states[-1]))
        models.append(model)

    def predictor(model):
        """A fitted tree as a predictor: raw incomes in, prices out.

        Args:
            model: The fitted tree; None stands for the single leaf, which predicts the mean price.

        Returns:
            The predictor.
        """
        if model is None:
            mean = float(data.y_train.mean())
            return lambda x: np.full(np.shape(x), mean)
        return lambda x: model.predict(np.asarray(x, dtype=float).reshape(-1, 1))

    snapshots = tuple(make_snapshot(predictor(model), data, label=plural(len(state.leaves), "leaf"),
                                    progress=len(state.leaves), note=_note(state), detail=state)
                      for model, state in zip(models, states))
    final_state, predict = states[-1], predictor(models[-1])
    leaf_of = {key: (value, count) for key, value, count in final_state.leaves}

    def explain(income: float) -> Explanation:
        """Works the tree's price for one income: its yes/no questions, then its leaf's mean price.

        More than four questions are cut to the first three and a count of the rest.

        Args:
            income: The income ($10k).

        Returns:
            The worked lines, checked against the model's own prediction.
        """
        keys = path_keys(final_state, income)
        thresholds = dict(final_state.splits)
        questions = [f"income {income_k(income)} ≤ {income_k(thresholds[a])}? "
                     f"{'yes → left' if b.endswith('L') else 'no → right'}" for a, b in zip(keys, keys[1:])]
        if len(questions) > 4:
            questions = [*questions[:3], f"… {len(questions) - 3} more questions …"]
        value, count = leaf_of[keys[-1]]
        lines = [*questions, f"leaf: {count:,} training districts, mean price {money_k(value)}",
                 f"ŷ = that mean = {money_k(value)}"]
        return verified(lines, value, predict_one(predict, income))

    story = _story(snapshots)
    return Lesson(
        story=story, data=data, snapshots=snapshots, final_index=len(snapshots) - 1, predict=predict,
        explain=explain, panel=TreePanel(final_state),
        config={"model": "DecisionTreeRegressor", "max_leaf_nodes": f"1-{MAX_LEAVES} (best-first)",
                "min_samples_leaf": str(min_leaf(data.n_train)), "random_state": str(RANDOM_STATE)})
