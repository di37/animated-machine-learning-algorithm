"""A classification tree grown one split at a time (best-first), each leaf predicting a survival share.

Each refit with one more leaf adds the split that cuts Gini impurity most, and every leaf predicts the
share of its passengers who survived. A group's Gini impurity is 1 − Σ p², the chance that two of its
passengers drawn at random had different fates: 0 for a group that all survived or all died.

As in the regression lesson, nodes are keyed by their path ("", "L", "LR", …) and the tests check
that every fit is the previous tree plus one split.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from sklearn.tree import DecisionTreeClassifier

from animkit.lessons import theme
from animkit.lessons.lesson import Story, plural
from animkit.lessons.stage import reset_axes
from animkit.lessons.timing import ease
from animkit.lessons.tree_layout import layout
from titanic_anim.config import RANDOM_STATE
from titanic_anim.lessons.config import AGE, FEMALE
from titanic_anim.lessons.data import Passenger, PassengerData
from titanic_anim.lessons.lesson import (
    PROBABILITY,
    ClassLesson,
    ClassSnapshot,
    make_snapshot,
    percent,
    verified_score,
)
from titanic_anim.lessons.maps import SCORE_CMAP

MAX_LEAVES = 12
LABEL_DEPTH = 3


def min_leaf(n_rows: int) -> int:
    """The fewest training passengers a leaf may hold, so a leaf never describes a handful of people.

    Args:
        n_rows: How many training passengers there are.

    Returns:
        One in 60 of them, and at least 1.
    """
    return max(1, n_rows // 60)


@dataclass(frozen=True)
class TreeState:
    """A fitted tree, as the panel and the explanation read it.

    A node's key is its path from the root: "" is the root, "L" its left child, "LR" that child's right.

    Attributes:
        splits: Each split as (path key, feature, threshold); a passenger goes left when feature ≤ threshold.
        leaves: Each leaf as (path key, P(survive), training passengers in it).
        newest: The key of the split this tree added to the previous one; None for the first state.
    """

    splits: tuple[tuple[str, int, float], ...]  # (path key, feature, threshold)
    leaves: tuple[tuple[str, float, int], ...]  # (path key, P(survive), training passengers)
    newest: str | None


def split_text(feature: int, value: float) -> str:
    """A split as a yes/no question.

    Args:
        feature: The column it splits on.
        value: Its threshold, in the column's units (log fare for fare).

    Returns:
        'man?' for sex (female ≤ 0.5), 'age ≤ 9.5', or 'fare ≤ $52' with the fare back in dollars.
    """
    if feature == FEMALE:
        return "man?"  # female ≤ 0.5
    if feature == AGE:
        return f"age ≤ {value:.1f}"
    return f"fare ≤ ${np.expm1(value):,.0f}"


def tree_state(model: DecisionTreeClassifier, previous: TreeState | None) -> TreeState:
    """Reads a fitted tree's splits and leaves, keyed by path, from scikit-learn's arrays.

    Args:
        model: The fitted tree.
        previous: The tree with one leaf fewer, to find the new split; None if there is none.

    Returns:
        The tree's state.
    """
    t = model.tree_
    splits, leaves = [], []

    def visit(node: int, key: str) -> None:
        """Records a node and, for a split, walks on into both children.

        Args:
            node: The node's index in scikit-learn's arrays.
            key: Its path key.
        """
        if t.children_left[node] == -1:
            counts = t.value[node].ravel()
            leaves.append((key, float(counts[1] / counts.sum()), int(t.n_node_samples[node])))
            return
        splits.append((key, int(t.feature[node]), float(t.threshold[node])))
        visit(t.children_left[node], key + "L")
        visit(t.children_right[node], key + "R")

    visit(0, "")
    before = set() if previous is None else {key for key, _, _ in previous.splits}
    added = [key for key, _, _ in splits if key not in before]
    return TreeState(tuple(splits), tuple(leaves), added[0] if added else None)


def path_keys(state: TreeState, features: np.ndarray) -> list[str]:
    """The nodes a passenger passes through, from the root to their leaf.

    Args:
        state: The tree.
        features: One passenger's (age, log fare, female) row.

    Returns:
        The path keys, root ("") first, leaf last.
    """
    rules = {key: (feature, value) for key, feature, value in state.splits}
    keys = [""]
    while keys[-1] in rules:
        feature, value = rules[keys[-1]]
        keys.append(keys[-1] + ("L" if features[feature] <= value else "R"))
    return keys


class TreePanel:
    """The tree lesson's side panel (a ``ClassPanel``): the tree itself, grown split by split.

    Splits are navy dots labelled with their question, the newest highlighted; leaves are coloured by
    their survival share and sized by how many passengers they hold. Every state is laid out on the
    final tree's positions, so a node never moves as the tree grows.
    """

    def __init__(self, final: TreeState):
        """Lays out the final tree.

        Args:
            final: The final tree, used for the layout and at inference.
        """
        self._final = final
        self._points = layout([key for key, _, _ in final.leaves], [key for key, _, _ in final.splits])

    def _draw(self, ax: Axes, state: TreeState, title: str, path: list[str] | None = None) -> None:
        """Draws one tree: edges, split questions (deep ones only when new), leaves, and a highlighted path."""
        reset_axes(ax)
        ax.text(0.04, 0.9, title, fontsize=theme.PANEL_TITLE_SIZE, weight="bold", color=theme.NAVY, va="center")
        p = self._points
        ax.add_collection(LineCollection([(p[key], p[key + side]) for key, _, _ in state.splits for side in "LR"],
                                         colors=theme.INACTIVE, linewidths=1.4, zorder=1))
        if path:
            ax.add_collection(LineCollection([(p[a], p[b]) for a, b in zip(path, path[1:])],
                                             colors=theme.HIGHLIGHT, linewidths=4.0, zorder=2))
        for key, feature, value in state.splits:
            newest = key == state.newest
            ax.scatter(*p[key], s=110 if newest else 60, color=theme.NAVY, zorder=3,
                       edgecolors=theme.HIGHLIGHT if newest else theme.PANEL, linewidths=3 if newest else 1.5)
            if len(key) < LABEL_DEPTH or newest:
                ax.text(p[key][0], p[key][1] + 0.035, split_text(feature, value), ha="center", va="bottom",
                        fontsize=11, color=theme.NAVY, weight="bold" if newest else "normal", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=theme.PANEL, edgecolor="none", alpha=0.85))
        total = sum(n for _, _, n in state.leaves)
        for key, share, count in state.leaves:
            ax.scatter(*p[key], s=50 + 260 * np.sqrt(count / total), color=SCORE_CMAP(share), zorder=3,
                       edgecolors=theme.NAVY, linewidths=0.8)
        ax.text(0.96, 0.03, "leaf colour: share who survived · left branch = yes", ha="right", fontsize=11,
                color=theme.INK_SECONDARY)

    def draw_training(self, ax: Axes, snapshot: ClassSnapshot) -> None:
        """Draws the tree at one moment of training.

        Args:
            ax: The side panel's axes.
            snapshot: The moment to draw.
        """
        state: TreeState = snapshot.detail
        self._draw(ax, state, f"The tree: {plural(len(state.leaves), 'leaf')}, "
                              f"{plural(len(state.splits), 'split')}")

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the unseen passenger's path down the final tree as ``t`` grows, then their leaf's share.

        Args:
            ax: The side panel's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        keys = path_keys(self._final, query.features)
        shown = keys[:max(1, int(np.ceil(len(keys) * ease(t))))]
        self._draw(ax, self._final, "The new passenger answers yes/no questions", path=shown)
        shares = {key: share for key, share, _ in self._final.leaves}
        if shown[-1] in shares:
            x, y = self._points[shown[-1]]
            ax.text(x, y - 0.06, f"{shares[shown[-1]]:.0%}", ha="center", va="top", fontsize=13,
                    weight="bold", color=theme.NAVY, zorder=6)

    def overlay_training(self, axes, snapshot) -> None:
        """The maps already show the boxes.

        Args:
            axes: The two maps, women's first.
            snapshot: The moment being drawn.
        """

    def overlay_inference(self, axes, query, t: float) -> None:
        """The panel shows the path.

        Args:
            axes: The two maps, women's first.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """


def _note(state: TreeState) -> str:
    """The training callout: the newest split's question, or the one-leaf start."""
    if state.newest is None:
        return "1 leaf, no splits yet\neveryone gets the training survival rate"
    feature, value = next((f, v) for k, f, v in state.splits if k == state.newest)
    return (f"Split {len(state.splits)}: {split_text(feature, value)}\n"
            "chosen because it cuts Gini impurity the most")


def _story(data: PassengerData, last: ClassSnapshot) -> Story:
    """The lesson's narration; ``last`` is the final snapshot, for its size and accuracies."""
    return Story(
        key="decision_tree", title="Decision Tree",
        method=f"grown best-first to {plural(len(last.detail.leaves), 'leaf')}",
        objective=("A tree sorts passengers with yes/no questions about sex, age and fare. Watch it add "
                   "one question at a time, always the one that separates survivors from the others best, "
                   "and carve both maps into boxes."),
        initial=("Before any split the tree is one leaf: everyone gets the training survival rate, so "
                 "nobody is predicted to survive. Circles mark the mistakes."),
        training=("Each frame adds the split that cuts Gini impurity the most. Every box on the maps is "
                  "a leaf, shaded by the share of its training passengers who survived."),
        final=(f"{plural(len(last.detail.leaves), 'leaf')}: train accuracy {last.train_accuracy:.0%}, validation "
               f"{last.val_accuracy:.0%}. The first question is the most telling: man or woman?"),
        inference=("The unseen passenger answers the questions from the root down to one leaf; the "
                   "prediction is that leaf's survival share."),
        takeaway=("A tree splits the inputs into boxes with readable questions, and predicts each box's "
                  "share of survivors."),
        limitation=("its boundaries are boxy steps, and grown deep it memorises small groups of "
                    "passengers instead of general patterns."),
        progress_label="number of leaves")


def build(data: PassengerData) -> ClassLesson:
    """Builds the tree lesson: a one-leaf start, then a refit with one more leaf per frame.

    Growth stops at ``MAX_LEAVES``, or earlier when no split is left that keeps ``min_leaf`` passengers on
    each side (small datasets).

    Args:
        data: The passengers.

    Returns:
        The lesson, with the largest tree as the final model.
    """
    rate = data.baseline_rate
    states, models = [TreeState((), (("", rate, data.n_train),), None)], [None]
    for leaves in range(2, MAX_LEAVES + 1):
        model = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=min_leaf(data.n_train),
                                       random_state=RANDOM_STATE).fit(data.x_train, data.y_train)
        state = tree_state(model, states[-1])
        if len(state.leaves) <= len(states[-1].leaves):
            break  # no split left that respects min_samples_leaf (small datasets)
        states.append(state)
        models.append(model)

    def scorer(model):
        """One state's model as a scorer.

        Args:
            model: A fitted tree, or None for the one-leaf start.

        Returns:
            A function from feature rows to P(survive): the training survival rate for everyone at the start.
        """
        if model is None:
            return lambda x: np.full(len(x), rate)
        return lambda x: model.predict_proba(np.asarray(x, dtype=float))[:, 1]

    snapshots = tuple(make_snapshot(scorer(m), data, PROBABILITY, label=plural(len(s.leaves), "leaf"),
                                    progress=len(s.leaves), note=_note(s), detail=s)
                      for m, s in zip(models, states))
    final, score = states[-1], scorer(models[-1])
    leaf_of = {key: (share, count) for key, share, count in final.leaves}

    def explain(features: np.ndarray):
        """Works one passenger's prediction through: the questions down the tree, then the leaf's share.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            The verified explanation (a long path is shortened to its first three questions).
        """
        keys = path_keys(final, features)
        rules = {key: (feature, value) for key, feature, value in final.splits}
        questions = [f"{split_text(*rules[a])} {'yes → left' if b.endswith('L') else 'no → right'}"
                     for a, b in zip(keys, keys[1:])]
        if len(questions) > 4:
            questions = [*questions[:3], f"… {len(questions) - 3} more questions …"]
        share, count = leaf_of[keys[-1]]
        lines = [*questions, f"leaf: {count} training passengers, {percent(share)} survived",
                 f"P(survive) = that share = {percent(share)}"]
        return verified_score(lines, share, float(score(features[None, :])[0]), PROBABILITY)

    return ClassLesson(
        story=_story(data, snapshots[-1]), data=data, snapshots=snapshots, final_index=len(snapshots) - 1,
        score=score, kind=PROBABILITY, explain=explain, panel=TreePanel(final),
        config={"model": "DecisionTreeClassifier",
                "max_leaf_nodes": f"1-{len(final.leaves)} (best-first, cap {MAX_LEAVES})",
                "criterion": "gini", "min_samples_leaf": str(min_leaf(data.n_train)),
                "random_state": str(RANDOM_STATE)})
