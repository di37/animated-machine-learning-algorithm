"""Gradient boosting for survival: small trees added one at a time in log-odds, then P = σ(F).

Start from the log-odds of the base rate, F₀; then each small tree h_m fits what is still wrong and is
added in log-odds, shrunk by the learning rate: F_m = F_m−1 + learning rate × h_m. For the log loss,
"what is still wrong" is each passenger's y − σ(F): the loss's negative gradient with respect to F.

One scikit-learn model is fitted with all ``N_TREES`` trees. ``Booster`` reads F after any number of
them from its public API, so every frame is the real model part-way through, not a refit.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import GradientBoostingClassifier

from animkit.data import read_only
from animkit.lessons import theme
from animkit.lessons.lesson import Story
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
from titanic_anim.lessons.lesson_diagrams import boosting_diagram
from titanic_anim.lessons.stories.common import sigmoid
from titanic_anim.lessons.timing_helpers import revealed

N_TREES = 80
LEARNING_RATE = 0.1
TREE_DEPTH = 2
STAGES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80)
CONTRIBUTION_GROUPS = ((1, 10), (11, 30), (31, N_TREES))
CORRECTION_LIMIT = 1.0  # log-odds drawn fully saturated in the correction maps


def min_leaf(n_rows: int) -> int:
    """The fewest training passengers a leaf of any tree may hold.

    Args:
        n_rows: How many training passengers there are.

    Returns:
        One in 60 of them, and at least 1.
    """
    return max(1, n_rows // 60)


@dataclass(frozen=True)
class BoostState:
    """The model at one frame, and what that frame's trees added; each map has shape (2, fares, ages).

    Attributes:
        first: The first tree added in this frame (1-based); 0 at the start.
        stage: How many trees the model has after this frame.
        learning_rate: The shrinkage each tree's output is multiplied by.
        log_odds_before: The previous frame's F on both maps.
        correction: The sum of the outputs of the trees added in this frame (before shrinkage), on both maps.
        log_odds: This frame's F on both maps.
    """

    first: int  # first tree added in this frame (1-based); 0 at the start
    stage: int
    learning_rate: float
    log_odds_before: np.ndarray  # previous frame's F on both maps
    correction: np.ndarray  # Σ h_i of the trees added in this frame (unshrunk), both maps
    log_odds: np.ndarray  # this frame's F

    @property
    def added(self) -> str:
        """The trees this frame added, in words.

        Returns:
            'tree 3' for one tree, 'trees 11–13' for several.
        """
        return f"tree {self.stage}" if self.first == self.stage else f"trees {self.first}–{self.stage}"


class Booster:
    """F_m(x) for any m, from public scikit-learn API: decision_function and each tree's raw output.

    Attributes:
        model: The fitted ``GradientBoostingClassifier``, with all its trees.

    Example:
        >>> booster = Booster(GradientBoostingClassifier().fit(x, y))  # doctest: +SKIP
        >>> booster.log_odds(10, x)  # doctest: +SKIP
    """

    def __init__(self, model: GradientBoostingClassifier):
        """Wraps a fitted model.

        Args:
            model: The fitted ``GradientBoostingClassifier``.
        """
        self.model = model

    def trees(self, first: int, last: int, x: np.ndarray) -> np.ndarray:
        """The summed raw outputs of trees ``first`` to ``last`` (1-based, both included), unshrunk.

        Args:
            first: The first tree.
            last: The last tree; below ``first``, no trees are summed.
            x: (age, log fare, female) rows.

        Returns:
            One sum per row, in log-odds before the learning rate is applied.
        """
        x = np.asarray(x, dtype=float)
        return sum((self.model.estimators_[i - 1, 0].predict(x) for i in range(first, last + 1)), np.zeros(len(x)))

    def base(self, x: np.ndarray) -> np.ndarray:
        """F_0: the final log-odds minus every tree's shrunken contribution (the prior log-odds).

        Args:
            x: (age, log fare, female) rows.

        Returns:
            F₀ for each row: the same for every passenger, the log-odds of the training survival rate.
        """
        return self.model.decision_function(np.asarray(x, dtype=float)) - self.model.learning_rate * self.trees(
            1, self.model.n_estimators, x)

    def log_odds(self, m: int, x: np.ndarray) -> np.ndarray:
        """F_m: the log-odds after the first ``m`` trees.

        Args:
            m: How many trees to include; 0 gives F₀.
            x: (age, log fare, female) rows.

        Returns:
            F_m for each row; σ of it is the model's P(survive) at that stage.
        """
        return self.base(x) + self.model.learning_rate * self.trees(1, m, x)

    def staged_at(self, features: np.ndarray) -> np.ndarray:
        """One passenger's log-odds after every number of trees.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            F₀, F₁, ..., F_N: ``N_TREES + 1`` values.
        """
        x = features[None, :]
        steps = [self.model.learning_rate * self.trees(i, i, x)[0] for i in range(1, self.model.n_estimators + 1)]
        return self.base(x)[0] + np.concatenate([[0.0], np.cumsum(steps)])


class BoostPanel:
    """Boosting's side panel (a ``ClassPanel``): what each frame's trees add, then one passenger's climb.

    During training two small maps show the frame's shrunken correction in log-odds (blue raises the
    chance of survival, red lowers it). At inference a line traces the passenger's log-odds tree by tree.
    """

    def __init__(self, booster: Booster, data: PassengerData):
        """Sets up the panel.

        Args:
            booster: Reads the final model's log-odds after any number of trees.
            data: The passengers.
        """
        self._booster = booster
        self._data = data

    def draw_training(self, ax: Axes, snapshot: ClassSnapshot) -> None:
        """Draws F₀ at the start, then what the frame's trees add, in log-odds.

        Args:
            ax: The side panel's axes.
            snapshot: The moment to draw.
        """
        state: BoostState = snapshot.detail
        if state.stage == 0:
            maps.mini_maps(ax, self._data, state.log_odds, "Start: F₀, the log-odds of the base rate",
                           norm=TwoSlopeNorm(0.0, -CORRECTION_LIMIT, CORRECTION_LIMIT), boundary=False)
            return
        maps.mini_maps(ax, self._data, state.learning_rate * state.correction,
                       f"What {state.added} add (log-odds; blue = up)",
                       norm=TwoSlopeNorm(0.0, -CORRECTION_LIMIT, CORRECTION_LIMIT), boundary=False)

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the unseen passenger's log-odds as trees are added, revealed as ``t`` grows.

        Args:
            ax: The side panel's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        staged = self._booster.staged_at(query.features)
        shown = revealed(len(staged), t)
        theme.style_axes(ax)
        ax.set_xlim(-2, len(staged) + 1)
        pad = max(staged.max() - staged.min(), 0.5) * 0.25
        ax.set_ylim(staged.min() - pad, staged.max() + pad)
        ax.set_xlabel("trees added", fontsize=theme.NOTE_SIZE)
        ax.set_title("This passenger's log-odds, tree by tree")
        ax.axhline(0.0, color=theme.INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        ax.plot(np.arange(shown), staged[:shown], color=theme.MODEL, linewidth=2.4)
        ax.scatter([shown - 1], [staged[shown - 1]], s=70, color=theme.MODEL, zorder=4,
                   edgecolors=theme.PANEL, linewidths=2.0)
        ax.text(0, staged[0], f"  F₀ = {staged[0]:+.2f} ({sigmoid(staged[0]):.0%})", va="bottom", fontsize=12,
                color=theme.NAVY)
        ax.text(0.99, 0.04, "dashed: log-odds 0 = 50%", transform=ax.transAxes, ha="right", fontsize=11,
                color=theme.INK_SECONDARY)

    def overlay_training(self, axes, snapshot) -> None:
        """The maps show σ(F).

        Args:
            axes: The two maps, women's first.
            snapshot: The moment being drawn.
        """

    def overlay_inference(self, axes, query, t: float) -> None:
        """The panel shows the climb.

        Args:
            axes: The two maps, women's first.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """


def _note(state: BoostState) -> str:
    """The training callout: the start, one new tree, or a run of trees added in one frame."""
    if state.stage == 0:
        return "Start: F₀ = log-odds of the survival rate\nno trees yet"
    if state.first == state.stage:
        return (f"Tree {state.stage}: a depth-{TREE_DEPTH} tree fitted to what is\n"
                f"still wrong, added at {LEARNING_RATE:.0%} strength")
    return (f"Trees {state.first}–{state.stage} ({state.stage - state.first + 1} trees), each fitted to\n"
            f"the errors left before it, each at {LEARNING_RATE:.0%} strength")


def _states(booster: Booster, data: PassengerData) -> list[BoostState]:
    """Every frame's state on the map grid, at the tree counts in ``STAGES``."""
    shape = data.surface_shape
    states = []
    for previous, m in zip((0, *STAGES), STAGES):
        before = booster.log_odds(previous, data.grid)
        correction = booster.trees(previous + 1, m, data.grid) if m else np.zeros(len(data.grid))
        states.append(BoostState(first=previous + 1 if m else 0, stage=m, learning_rate=LEARNING_RATE,
                                 log_odds_before=read_only(before.reshape(shape)),
                                 correction=read_only(correction.reshape(shape)),
                                 log_odds=read_only(booster.log_odds(m, data.grid).reshape(shape))))
    return states


def _story(last: ClassSnapshot) -> Story:
    """The lesson's narration; ``last`` is the final snapshot, for its accuracies."""
    return Story(
        key="gradient_boosting", title="Gradient Boosting",
        method=f"{N_TREES} small trees, each correcting the log-odds",
        objective=("Boosting builds its answer as a sum of small trees. Each new tree looks at what the "
                   "model still gets wrong and nudges the log-odds of survival to fix a little of it. "
                   "Watch the maps sharpen one correction at a time."),
        initial=("Start: F₀ is the log-odds of the training survival rate, the same for everyone, so "
                 "nobody is predicted to survive yet."),
        training=(f"Each tree fits the current errors and is added at {LEARNING_RATE} strength in log-odds: "
                  f"F_m = F_m−1 + {LEARNING_RATE} × h_m, P = σ(F). Later frames skip ahead several trees."),
        final=(f"{N_TREES} trees: train accuracy {last.train_accuracy:.0%}, validation "
               f"{last.val_accuracy:.0%}. Small steps keep any one tree from over-reacting."),
        inference=("For an unseen passenger boosting starts from the base-rate log-odds, every tree adds "
                   "its correction, and the sigmoid turns the total into a probability."),
        takeaway="Boosting turns many weak trees into a strong model by fitting each one to the errors that remain.",
        limitation=("trees are added in sequence, and with too many trees or a high learning rate it "
                    "starts fitting noise in a dataset this small."),
        progress_label="number of trees")


def build(data: PassengerData) -> ClassLesson:
    """Builds the boosting lesson: one fit, then one snapshot at each tree count in ``STAGES``.

    Args:
        data: The passengers.

    Returns:
        The lesson, with all ``N_TREES`` trees as the final model.
    """
    model = GradientBoostingClassifier(n_estimators=N_TREES, learning_rate=LEARNING_RATE, max_depth=TREE_DEPTH,
                                       min_samples_leaf=min_leaf(data.n_train), random_state=RANDOM_STATE)
    model.fit(data.x_train, data.y_train)
    booster = Booster(model)
    snapshots = tuple(make_snapshot(lambda x, m=s.stage: sigmoid(booster.log_odds(m, x)), data, PROBABILITY,
                                    label=f"{s.stage} trees", progress=s.stage, note=_note(s), detail=s)
                      for s in _states(booster, data))

    def score(x: np.ndarray) -> np.ndarray:
        """The final model's P(survive), from scikit-learn.

        Args:
            x: (age, log fare, female) rows.

        Returns:
            One P(survive) per row.
        """
        return model.predict_proba(np.asarray(x, dtype=float))[:, 1]

    def explain(features: np.ndarray):
        """Works one passenger's prediction through: F₀, what each group of trees adds, then σ.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            The verified explanation, checked against the model's ``predict_proba``.
        """
        staged = booster.staged_at(features)
        lines = [f"start: F₀ = log-odds of the survival rate = {staged[0]:+.2f}"]
        lines += [f"trees {a}–{b} add {staged[b] - staged[a - 1]:+.2f}" for a, b in CONTRIBUTION_GROUPS]
        worked = float(sigmoid(staged[-1]))
        lines.append(f"F = {staged[-1]:+.2f} → P(survive) = σ(F) = {percent(worked)}")
        return verified_score(lines, worked, float(score(features[None, :])[0]), PROBABILITY)

    return ClassLesson(
        story=_story(snapshots[-1]), data=data, snapshots=snapshots, final_index=len(snapshots) - 1,
        score=score, kind=PROBABILITY, explain=explain, panel=BoostPanel(booster, data),
        diagram=boosting_diagram(booster, data),
        config={"model": "GradientBoostingClassifier", "n_estimators": str(N_TREES),
                "learning_rate": str(LEARNING_RATE), "max_depth": str(TREE_DEPTH),
                "min_samples_leaf": str(min_leaf(data.n_train)), "loss": "log_loss",
                "random_state": str(RANDOM_STATE)})
