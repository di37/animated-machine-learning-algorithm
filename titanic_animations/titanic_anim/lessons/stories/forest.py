"""A random forest built tree by tree (warm start), its P(survive) the average of its trees'.

Each tree is fitted on its own bootstrap sample (passengers drawn at random with replacement); the
forest's P(survive) is the average of its trees' probabilities (aggregation, not optimisation). With
``warm_start``, raising ``n_estimators`` and fitting again adds new trees and keeps the old ones, so each
frame is the previous forest plus more trees.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from sklearn.ensemble import RandomForestClassifier

from animkit.data import read_only
from animkit.lessons import theme
from animkit.lessons.lesson import Story, plural
from titanic_anim.config import RANDOM_STATE
from titanic_anim.lessons import maps
from titanic_anim.lessons.data import Passenger, PassengerData
from titanic_anim.lessons.lesson import (
    PROBABILITY,
    ClassLesson,
    ClassSnapshot,
    make_snapshot,
    percent,
    verified_score,
)
from titanic_anim.lessons.lesson_diagrams import forest_diagram
from titanic_anim.lessons.timing_helpers import revealed

TREE_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50)


def min_leaf(n_rows: int) -> int:
    """The fewest training passengers a leaf of any tree may hold.

    Args:
        n_rows: How many training passengers there are.

    Returns:
        One in 200 of them, and at least 1: small leaves, so each tree on its own over-fits.
    """
    return max(1, n_rows // 200)


@dataclass(frozen=True)
class ForestState:
    """The forest at one frame.

    Attributes:
        n_trees: How many trees it has.
        tree_surfaces: Every tree's own P(survive) on the maps, shape (n_trees, 2, fares, ages).
    """

    n_trees: int
    tree_surfaces: np.ndarray  # (n_trees, 2, fares, ages): every tree's own P(survive) map


def _average_of(trees: list) -> Callable[[np.ndarray], np.ndarray]:
    """The forest of ``trees`` as a scorer: the mean of their P(survive)."""
    return lambda x: np.mean([t.predict_proba(np.asarray(x, dtype=float))[:, 1] for t in trees], axis=0)


class ForestPanel:
    """The forest's side panel (a ``ClassPanel``): the newest tree's own maps, then every tree's vote."""

    def __init__(self, data: PassengerData, forest: RandomForestClassifier):
        """Sets up the panel.

        Args:
            data: The passengers.
            forest: The final fitted forest, used at inference.
        """
        self._data = data
        self._forest = forest

    def draw_training(self, ax: Axes, snapshot: ClassSnapshot) -> None:
        """Draws the newest tree's two maps on their own, for contrast with the averaged main maps.

        Args:
            ax: The side panel's axes.
            snapshot: The moment to draw.
        """
        state: ForestState = snapshot.detail
        maps.mini_maps(ax, self._data, state.tree_surfaces[-1], f"Tree #{state.n_trees} on its own")

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws each tree's P(survive) for the unseen passenger, sorted, then their average.

        Args:
            ax: The side panel's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        votes = np.sort([tree.predict_proba(query.features[None, :])[0, 1] for tree in self._forest.estimators_])
        shown = revealed(len(votes), t)
        theme.style_axes(ax, grid_axis="x")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-1, len(votes))
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(maps.PERCENT)
        ax.set_title(f"Each of the {len(votes)} trees votes a P(survive)")
        ax.axvline(0.5, color=theme.INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        ax.scatter(votes[:shown], np.arange(shown), s=34, color=theme.HIGHLIGHT, zorder=3)
        if shown == len(votes):
            mean = float(votes.mean())
            ax.axvline(mean, color=theme.MODEL, linewidth=3.0, zorder=4)
            ax.text(mean, len(votes) * 0.98, f" average {mean:.0%}", fontsize=13, weight="bold",
                    color=theme.NAVY, va="top")

    def overlay_training(self, axes, snapshot) -> None:
        """The maps show the average; the panel shows the newest tree alone.

        Args:
            axes: The two maps, women's first.
            snapshot: The moment being drawn.
        """

    def overlay_inference(self, axes, query, t: float) -> None:
        """The panel shows the votes.

        Args:
            axes: The two maps, women's first.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """


def _story(first: ClassSnapshot, last: ClassSnapshot) -> Story:
    """The forest's narration; ``first`` is the one-tree snapshot, ``last`` the full forest."""
    return Story(
        key="random_forest", title="Random Forest",
        method=f"{TREE_COUNTS[-1]} bootstrapped trees, probabilities averaged",
        objective=("One deep tree draws a patchy, over-confident map. A random forest grows many trees, "
                   "each on a different bootstrap sample of passengers, and averages their probabilities. "
                   "Watch the patchwork smooth out as trees are added."),
        initial=(f"One tree on its own bootstrap sample: hard-edged boxes, many of them 0% or 100%. "
                 f"Validation accuracy {first.val_accuracy:.0%}."),
        training=("Each frame adds already-fitted trees and re-averages: nothing is optimised here. The "
                  "panel shows the newest tree alone; the maps show the average of all trees so far."),
        final=(f"{TREE_COUNTS[-1]} trees: train accuracy {last.train_accuracy:.0%}, validation "
               f"{last.val_accuracy:.0%} (one tree alone: {first.val_accuracy:.0%})."),
        inference=("For an unseen passenger each tree gives its own P(survive); the forest reports the "
                   "average of all the votes."),
        takeaway="Averaging many different over-fitted trees cancels much of their noise and softens their certainty.",
        limitation=("a forest is slow to inspect and far less readable than one tree, and its training "
                    "accuracy flatters it: judge it on held-out passengers."),
        progress_label="number of trees")


def build(data: PassengerData) -> ClassLesson:
    """Builds the forest lesson: one snapshot at each tree count in ``TREE_COUNTS``.

    Args:
        data: The passengers.

    Returns:
        The lesson, with the full forest as the final model.
    """
    forest = RandomForestClassifier(n_estimators=TREE_COUNTS[0], min_samples_leaf=min_leaf(data.n_train),
                                    warm_start=True, n_jobs=1, random_state=RANDOM_STATE)
    built: list[ClassSnapshot] = []
    for previous, count in zip((0, *TREE_COUNTS), TREE_COUNTS):
        forest.set_params(n_estimators=count).fit(data.x_train, data.y_train)  # warm start: new trees only
        trees = list(forest.estimators_)
        surfaces = read_only(np.array([t.predict_proba(data.grid)[:, 1].reshape(data.surface_shape)
                                       for t in trees]))
        note = ("Tree 1, fitted on a bootstrap sample\nalone it draws hard 0% / 100% boxes" if count == 1
                else f"{count} trees: {plural(count - previous, 'new tree')} added\nthe average softens each tree's boxes")
        built.append(make_snapshot(_average_of(trees), data, PROBABILITY, label=plural(count, "tree"),
                                   progress=count, note=note, detail=ForestState(count, surfaces)))
    snapshots = tuple(built)

    def score(x: np.ndarray) -> np.ndarray:
        """The full forest's P(survive), from scikit-learn.

        Args:
            x: (age, log fare, female) rows.

        Returns:
            One P(survive) per row.
        """
        return forest.predict_proba(np.asarray(x, dtype=float))[:, 1]

    def explain(features: np.ndarray):
        """Works one passenger's prediction through: the spread of the trees' votes, then their average.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            The verified explanation, checked against the forest's ``predict_proba``.
        """
        votes = np.array([tree.predict_proba(features[None, :])[0, 1] for tree in forest.estimators_])
        worked = float(votes.mean())
        lines = [f"{len(votes)} trees, each trained on its own bootstrap sample",
                 f"lowest vote {percent(votes.min())}, highest {percent(votes.max())}",
                 f"{int(np.sum(votes > 0.5))} of {len(votes)} trees say 'survived'",
                 f"P(survive) = average of the votes = {percent(worked)}"]
        return verified_score(lines, worked, float(score(features[None, :])[0]), PROBABILITY)

    return ClassLesson(
        story=_story(snapshots[0], snapshots[-1]), data=data, snapshots=snapshots,
        final_index=len(snapshots) - 1, score=score, kind=PROBABILITY, explain=explain,
        panel=ForestPanel(data, forest), diagram=forest_diagram(forest),
        config={"model": "RandomForestClassifier", "n_estimators": str(TREE_COUNTS[-1]),
                "min_samples_leaf": str(min_leaf(data.n_train)), "bootstrap": "True",
                "random_state": str(RANDOM_STATE)})
