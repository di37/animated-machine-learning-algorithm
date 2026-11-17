"""Shared algorithm diagrams: a forest's trees voting, boosting's chain of corrections and kNN's lookup.

A project supplies a ``Reading`` (what its query is, the model row it becomes, the model's
prediction and how that reads on screen) plus the few words and formats that differ between tasks.

Diagrams are drawn from the fitted models at every frame; nothing on them is precomputed or invented.

Each class is a lesson ``Diagram`` (see ``director``): ``draw_training`` draws the model at one
training snapshot, and ``draw_inference`` shows it working on the unseen example as ``t`` runs from
0 to 1. The drawing itself is done by ``animkit.lessons.diagrams``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from animkit.algorithms.common import signed_rgba
from animkit.lessons import diagrams
from animkit.lessons.lesson import plural

SHOWN_LEVELS = 3  # forest thumbnails draw this many levels of each tree

Format = Callable[[float], str]


@dataclass(frozen=True)
class Reading:
    """How one task reads on screen.

    Attributes:
        subject: What one query is called on screen, a single noun.
        row: Turns a query into the one-row matrix the models take.
        predict: Takes a fitted model (or one of its trees) and a row, and returns its prediction.
        show: Formats a prediction as shown, for example "81.1%".
    """

    subject: str  # what one query is called on screen, a single noun
    row: Callable[[Any], np.ndarray]  # query → the one-row matrix the models take
    predict: Callable[[Any, np.ndarray], float]  # a fitted model (or one of its trees), row → its prediction
    show: Format  # a prediction as shown, e.g. "$302k" or "81.1%"


class ForestDiagram:
    """The forest's first trees, their top levels drawn, feeding an average.

    At inference each drawn tree's path is highlighted under its vote, and the average of every tree's
    vote appears at the end.
    """

    def __init__(self, forest, reading: Reading, *, colour: Callable[[np.ndarray], Any], summary: str,
                 show_vote: Format | None = None):
        """Wraps a fitted forest of trees.

        Args:
            forest: The fitted forest; its ``estimators_`` are the trees.
            reading: How the task reads on screen.
            colour: Maps a node's value row to its colour.
            summary: The text in the average box during training.
            show_vote: Formats one tree's vote; ``reading.show`` when not given.
        """
        self._forest = forest
        self._reading = reading
        self._colour = colour
        self._summary = summary
        self._show_vote = show_vote or reading.show

    def drawn(self, count: int) -> list:
        """The trees in the forest when it had ``count`` of them (warm start only ever appends).

        Args:
            count: How many trees the forest had at that snapshot.

        Returns:
            The first ``count`` fitted trees.
        """
        return list(self._forest.estimators_[:count])

    def draw_training(self, ax: Axes, snapshot: Any) -> None:
        """Draws the trees the forest had at this snapshot.

        Args:
            ax: The diagram's axes.
            snapshot: The training snapshot; ``snapshot.detail.n_trees`` says how many trees it had.
        """
        count = snapshot.detail.n_trees
        diagrams.tree_grid(ax, [tree.tree_ for tree in self.drawn(count)], self._colour, depth=SHOWN_LEVELS,
                           title=f"The forest: {plural(count, 'tree')}, top {SHOWN_LEVELS} levels shown",
                           total=count, summary=self._summary)

    def draw_inference(self, ax: Axes, query: Any, t: float) -> None:
        """Shows every tree voting on the query: each drawn tree's path and vote, then the average.

        Args:
            ax: The diagram's axes.
            query: The unseen example, as the project holds it.
            t: Progress through the reveal, 0 to 1; the average is shown at 1.
        """
        reading, row = self._reading, self._reading.row(query)
        estimators = self._forest.estimators_
        votes = [reading.predict(tree, row) for tree in estimators]
        drawn = estimators[:diagrams.GRID_TREES]
        summary = f"average\n{reading.show(float(np.mean(votes)))}" if t >= 1.0 else "average"
        diagrams.tree_grid(ax, [tree.tree_ for tree in estimators], self._colour, depth=SHOWN_LEVELS,
                           title=f"Every tree routes the {reading.subject} to a leaf and votes",
                           total=len(estimators), summary=summary,
                           paths=[tree.decision_path(row).indices for tree in drawn],
                           votes=[self._show_vote(vote) for vote in votes[:len(drawn)]])


def leaf_colours(trees: Sequence) -> Callable[[np.ndarray], Any]:
    """Signed colours on the drawn trees' largest leaf value (leaves hold each tree's actual step).

    Args:
        trees: The tree structures (``tree_``) being drawn.

    Returns:
        A function from a node's value row to its colour, on one scale for all the trees.
    """
    scale = max((float(np.max(np.abs(tree.value[tree.children_left == -1]))) for tree in trees), default=1.0)
    return lambda value: signed_rgba(float(np.ravel(value)[0]), scale)


class BoostingDiagram:
    """F₀ plus a shrunken correction from each small tree, optionally passed through a ``link`` such as σ.

    The newest few trees are drawn with their corrections; the rest are summed into one note.

    ``booster`` needs ``.model`` (a fitted scikit-learn gradient-boosting model) and
    ``.trees(first, last, row)`` = Σ h_i(row) over trees first..last (1-based, unshrunk).
    """

    def __init__(self, booster, reading: Reading, *, f0: float, show_score: Format, show_step: Format,
                 link: str = "", step_noun: str = "correction"):
        """Wraps a fitted booster.

        Args:
            booster: The booster, with ``.model`` and ``.trees(first, last, row)`` as described above.
            reading: How the task reads on screen.
            f0: The starting score F₀, before any tree.
            show_score: Formats a raw score, such as F₀.
            show_step: Formats one tree's shrunken correction.
            link: The name of the function the final score passes through, such as "σ"; empty for none.
            step_noun: What one tree adds, in words.
        """
        self._booster = booster
        self._reading = reading
        self._f0 = f0
        self._show_score = show_score
        self._show_step = show_step
        self._link = link
        self._step_noun = step_noun

    @property
    def booster(self):
        """The booster the diagram reads.

        Returns:
            The booster it was made with.
        """
        return self._booster

    @property
    def f0(self) -> float:
        """The starting score F₀, before any tree.

        Returns:
            F₀, in raw score units.
        """
        return self._f0

    def _output(self, stage: str) -> str:
        """The output's name, wrapped in the link when there is one: "σ(F3)" or just "F3"."""
        return f"{self._link}({stage})" if self._link else stage

    def draw_training(self, ax: Axes, snapshot: Any) -> None:
        """Draws the chain as it stood after ``snapshot.detail.stage`` trees.

        Args:
            ax: The diagram's axes.
            snapshot: The training snapshot; ``snapshot.detail.stage`` is how many trees it had.
        """
        model, m = self._booster.model, snapshot.detail.stage
        trees = [tree.tree_ for tree in model.estimators_[:m, 0]]
        then = f", then {self._link}" if self._link else ""
        diagrams.chain(ax, trees, leaf_colours(trees[-diagrams.CHAIN_TREES:]), total=m,
                       title=f"Boosting: F₀ + {plural(m, 'small correction')}{then}",
                       start=f"F₀\n{self._show_score(self._f0)}", end=self._output(f"F{m}" if m else "F₀"),
                       step_label=f"{model.learning_rate} ×")

    def draw_inference(self, ax: Axes, query: Any, t: float) -> None:
        """Shows the query's score being built: F₀, the earlier trees' sum, and each newest tree's correction.

        Args:
            ax: The diagram's axes.
            query: The unseen example, as the project holds it.
            t: Progress through the reveal, 0 to 1; the prediction is shown at 1.
        """
        model, row = self._booster.model, self._reading.row(query)
        newest = model.estimators_[-diagrams.CHAIN_TREES:, 0]
        earlier = model.n_estimators - len(newest)
        notes = [self._show_step(model.learning_rate * float(tree.predict(row)[0])) for tree in newest]
        earlier_sum = model.learning_rate * float(self._booster.trees(1, earlier, row)[0]) if earlier else 0.0
        output = self._output("F") if self._link else "ŷ"
        end = f"{output}\n{self._reading.show(self._reading.predict(model, row))}" if t >= 1.0 else output
        drawn = [tree.tree_ for tree in newest]
        diagrams.chain(ax, [tree.tree_ for tree in model.estimators_[:, 0]], leaf_colours(drawn),
                       title=f"This {self._reading.subject}: F₀ plus every tree's {self._step_noun}",
                       total=model.n_estimators, start=f"F₀\n{self._show_score(self._f0)}", end=end,
                       step_label=f"{model.learning_rate} ×", earlier=f"\n{self._show_step(earlier_sum)}",
                       notes=notes, paths=[tree.decision_path(row).indices for tree in newest])


class NeighboursDiagram:
    """kNN has no weights: store, find the k nearest, combine their answers.

    ``steps(k)`` names the boxes and ``values(query)`` fills them in for one query, from the fitted model.
    """

    def __init__(self, reading: Reading, k: int, *, steps: Callable[[int], list[str]],
                 values: Callable[[Any], list[str]], verb: str = "predicts"):
        """Sets up the flow of steps.

        Args:
            reading: How the task reads on screen.
            k: How many neighbours the trained model uses.
            steps: Names the flow's boxes for a given k.
            values: The value under each box for one query, from the fitted model.
            verb: What kNN does, for the training title ("predicts", "classifies").
        """
        self._reading = reading
        self._k = k
        self._steps = steps
        self._values = values
        self._verb = verb

    def draw_training(self, ax: Axes, snapshot: Any) -> None:
        """Draws the flow of steps for this snapshot's k, with nothing filled in.

        Args:
            ax: The diagram's axes.
            snapshot: The training snapshot; ``snapshot.detail.k`` is its number of neighbours.
        """
        diagrams.flow(ax, self._steps(snapshot.detail.k), title=f"How kNN {self._verb} (no training beyond storing)")

    def draw_inference(self, ax: Axes, query: Any, t: float) -> None:
        """Draws the flow for the query, its values appearing box by box as ``t`` grows.

        Args:
            ax: The diagram's axes.
            query: The unseen example, as the project holds it.
            t: Progress through the reveal, 0 to 1.
        """
        diagrams.flow(ax, self._steps(self._k), title=f"This {self._reading.subject}'s neighbours, step by step",
                      values=self._values(query), reveal=t)
