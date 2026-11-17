"""Gradient boosting: start from the average, then each small tree fixes a little of what is still wrong.

Each small tree fits what is still wrong (the residuals) and is added, shrunk by the learning rate.
F_m(x) = F_0 + ν · Σ_{i≤m} h_i(x).

Frames skip ahead (1, 2, … 6, then 8, 10, …), so every frame says which trees it added and the
side panel shows those trees' combined correction against the previous frame's residuals.

F_0 is the average training price; h_i is the i-th tree, a small regression tree (``TREE_DEPTH``
levels) fitted to the residuals y − F_{i−1}(x) left by the trees before it; ν is the learning rate.
For squared error the residuals are, up to a constant factor, the loss's negative gradient, which is
where the name comes from: each tree is one small step of gradient descent in the space of functions.
Shrinking each step by ν makes the model need more trees but over-react less to any one. The model
is fitted once, by scikit-learn; ``Booster`` reads every intermediate F_m back out of its trees, so
each frame is a real stage.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from sklearn.ensemble import GradientBoostingRegressor

from animkit.lessons import theme
from animkit.lessons.timing import ease
from house_anim.config import RANDOM_STATE
from house_anim.data import read_only
from house_anim.lessons.config import INCOME_TO_K, PRICE_TO_K, X_RANGE
from house_anim.lessons.data import IncomeData, Query
from house_anim.lessons.lesson import (
    Explanation,
    Lesson,
    Panel,
    Snapshot,
    Story,
    make_snapshot,
    money_k,
    predict_one,
    verified,
)
from house_anim.lessons.lesson_diagrams import boosting_diagram
from house_anim.lessons.plots import K_FORMAT

N_TREES = 80
LEARNING_RATE = 0.15
TREE_DEPTH = 2
STAGES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80)
RESIDUAL_LIMIT_K = 320.0
CONTRIBUTION_GROUPS = ((1, 10), (11, 30), (31, N_TREES))


def min_leaf(n_rows: int) -> int:
    """Leaves need a few dozen districts, so no tree spends a split on one outlier.

    Args:
        n_rows: How many rows the trees train on.

    Returns:
        The minimum number of training rows per leaf: one in 250, at least 1.
    """
    return max(1, n_rows // 250)


@dataclass(frozen=True)
class BoostState:
    """One frame of boosting: which trees it added, and what they were fitted to.

    Attributes:
        first: The first tree added in this frame (1-based); 0 at the start.
        stage: How many trees the model has after this frame.
        learning_rate: The shrinkage ν each tree is added with.
        before: The previous frame's model on the grid ($100k).
        correction: Σ h_i of the trees added in this frame, unshrunk, on the grid ($100k).
        residuals: y − the previous frame's model, at the displayed training districts ($100k).
    """

    first: int  # first tree added in this frame (1-based); 0 at the start
    stage: int  # trees in the model after this frame
    learning_rate: float
    before: np.ndarray  # the previous frame's model on the grid
    correction: np.ndarray  # Σ h_i of the trees added in this frame, unshrunk, on the grid
    residuals: np.ndarray  # y − previous frame's model, at the displayed training districts

    @property
    def added(self) -> str:
        """The trees this frame added, as the panel names them.

        Returns:
            "tree 7" for one tree, "trees 9–10" for several.
        """
        return f"tree {self.stage}" if self.first == self.stage else f"trees {self.first}–{self.stage}"


class Booster:
    """Evaluates F_m(x) for any m from the fitted model's public init_ and estimators_.

    It is also what animkit's ``BoostingDiagram`` reads the trees through (``.model`` and ``.trees``).

    Attributes:
        model: The fitted gradient-boosting model.
    """

    def __init__(self, model: GradientBoostingRegressor):
        """Wraps a fitted model.

        Args:
            model: The fitted gradient-boosting model (one input, raw income).
        """
        self.model = model

    def base(self, x: np.ndarray) -> np.ndarray:
        """F_0, the model's starting guess, at some incomes.

        Args:
            x: Raw incomes ($10k).

        Returns:
            The initial prediction at each ($100k): the average training price.
        """
        return self.model.init_.predict(np.asarray(x, dtype=float).reshape(-1, 1))

    def trees(self, first: int, last: int, x: np.ndarray) -> np.ndarray:
        """Σ h_i(x) for trees first..last (1-based, inclusive), unshrunk.

        Args:
            first: The first tree to sum (1-based).
            last: The last tree to sum, inclusive; less than ``first`` sums nothing.
            x: Raw incomes ($10k).

        Returns:
            The trees' summed output at each income ($100k), before the learning rate.
        """
        column = np.asarray(x, dtype=float).reshape(-1, 1)
        return sum((self.model.estimators_[i - 1, 0].predict(column) for i in range(first, last + 1)),
                   np.zeros(len(column)))

    def stage(self, m: int) -> Callable[[np.ndarray], np.ndarray]:
        """The model after its first ``m`` trees, F_m = F_0 + ν · Σ_{i≤m} h_i.

        Args:
            m: How many trees to include; 0 is F_0 alone.

        Returns:
            F_m as a predictor: raw incomes in, prices out.
        """
        return lambda x: self.base(x) + self.model.learning_rate * self.trees(1, m, x)

    def staged_at(self, income: float) -> np.ndarray:
        """F_0 … F_M at one income.

        Args:
            income: The income ($10k).

        Returns:
            M + 1 predictions ($100k): F_0, then the running total after each tree.
        """
        x = np.array([income])
        steps = [self.model.learning_rate * self.trees(i, i, x)[0] for i in range(1, self.model.n_estimators + 1)]
        return self.base(x)[0] + np.concatenate([[0.0], np.cumsum(steps)])


class BoostPanel(Panel):
    """The residuals each frame's trees were fitted to, with their correction; at inference, F_m tree by tree."""

    def __init__(self, booster: Booster, data: IncomeData):
        """Keeps the model and the data the panel draws from.

        Args:
            booster: The fitted model's stage reader.
            data: The lesson's data.
        """
        self._booster = booster
        self._data = data

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws what the previous frame still got wrong, and the correction this frame's trees fitted to it.

        Args:
            ax: The side panel.
            snapshot: The frame being shown; its ``detail`` is a ``BoostState``.
        """
        state: BoostState = snapshot.detail
        theme.style_axes(ax)
        ax.set_xlim(X_RANGE[0] * INCOME_TO_K, X_RANGE[1] * INCOME_TO_K)
        ax.set_ylim(-RESIDUAL_LIMIT_K, RESIDUAL_LIMIT_K)
        ax.xaxis.set_major_formatter(K_FORMAT)
        ax.yaxis.set_major_formatter(K_FORMAT)
        ax.axhline(0.0, color=theme.INK_MUTED, linewidth=1.0)
        ax.scatter(self._data.display_x * INCOME_TO_K, state.residuals * PRICE_TO_K, s=10,
                   color=theme.DOTS, alpha=0.6, linewidths=0)
        if state.stage == 0:
            ax.set_title("What is still wrong: price − F₀ (no trees yet)")
            return
        ax.set_title(f"What is still wrong, and what {state.added} fit")
        ax.plot(self._data.grid * INCOME_TO_K, state.correction * PRICE_TO_K, color=theme.HIGHLIGHT,
                linewidth=3.0, label=f"{state.added}, added × {state.learning_rate}")
        ax.legend(loc="upper left")

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Draws the unseen district's prediction as each tree adds its step, from F₀ to the final price.

        Args:
            ax: The side panel.
            query: The unseen district.
            t: How far through the trees, from 0 (F₀ only) to 1 (all of them).
        """
        staged = self._booster.staged_at(query.income) * PRICE_TO_K
        shown = max(1, int(np.ceil(len(staged) * ease(t))))
        theme.style_axes(ax)
        ax.set_xlim(-2, len(staged) + 1)
        pad = max(staged.max() - staged.min(), 20.0) * 0.25
        ax.set_ylim(staged.min() - pad, staged.max() + pad)
        ax.yaxis.set_major_formatter(K_FORMAT)
        ax.set_xlabel("trees added", fontsize=theme.NOTE_SIZE)
        ax.set_title("This house's prediction, tree by tree")
        ax.plot(np.arange(shown), staged[:shown], color=theme.MODEL, linewidth=2.4)
        ax.scatter([shown - 1], [staged[shown - 1]], s=70, color=theme.MODEL, zorder=4,
                   edgecolors=theme.PANEL, linewidths=2.0)
        ax.text(0, staged[0], f"  F₀ ${staged[0]:,.0f}k", va="bottom", fontsize=12, color=theme.NAVY)


def _note(state: BoostState) -> str:
    """The callout for one frame: the start, one tree, or a run of trees, each fitted to the residuals."""
    if state.stage == 0:
        return "Start: F₀ = the average price for everyone\nno trees yet"
    if state.first == state.stage:
        return (f"Tree {state.stage}: a depth-{TREE_DEPTH} tree fitted to the residuals\n"
                f"added at {LEARNING_RATE:.0%} strength, so the curve moves a little")
    count = state.stage - state.first + 1
    return (f"Trees {state.first}–{state.stage} ({count} trees), each fitted to the\n"
            f"residuals the one before it left, each added at {LEARNING_RATE:.0%} strength")


def _states(booster: Booster, data: IncomeData) -> list[BoostState]:
    """One state per entry of ``STAGES``: the trees added since the previous stage, and its residuals."""
    states = []
    for previous, m in zip((0, *STAGES), STAGES):
        before = booster.stage(previous)
        correction = booster.trees(previous + 1, m, data.grid) if m else np.zeros_like(data.grid)
        states.append(BoostState(first=previous + 1 if m else 0, stage=m, learning_rate=LEARNING_RATE,
                                 before=read_only(before(data.grid)), correction=read_only(correction),
                                 residuals=read_only(data.display_y - before(data.display_x))))
    return states


def _story(snapshots: tuple[Snapshot, ...]) -> Story:
    """Every word of the gradient-boosting video, with the full model's errors."""
    last = snapshots[-1]
    return Story(
        key="gradient_boosting", title="Gradient Boosting",
        method=f"{N_TREES} small trees, each fixing the last one's mistakes",
        objective=("Boosting builds a model as a team of small trees. Each new tree studies what the "
                   "model still gets wrong and nudges the prediction to fix a little of it. Watch the "
                   "curve sharpen one correction at a time."),
        initial=("Start: F₀ predicts the average price for everyone. The residuals (price − F₀) are "
                 "what the first tree will try to explain."),
        training=(f"Each tree fits the residuals left so far and is added at {LEARNING_RATE} strength: "
                  f"F_m = F_m−1 + {LEARNING_RATE} × h_m. Later frames skip ahead several trees at once."),
        final=(f"{N_TREES} trees: train RMSE ${last.train_rmse_k:,.0f}k, validation "
               f"${last.val_rmse_k:,.0f}k. Small, shrunken steps keep any single tree from over-reacting."),
        inference=("For an unseen district boosting starts at the average price, then every tree "
                   "adds its small correction; the running total is the prediction."),
        takeaway=("Boosting turns many weak trees into a strong model by fitting each new tree to "
                  "the errors that remain."),
        limitation=("trees are added in sequence, so training cannot be parallelised across trees, "
                    "and too many trees or a high learning rate will fit noise."),
        progress_label="number of trees")


def build(data: IncomeData) -> Lesson:
    """Builds the gradient-boosting lesson: one fit of ``N_TREES`` trees, shown at each of ``STAGES``.

    The final model is all the trees.

    Args:
        data: The income view to learn from.

    Returns:
        The lesson, with the residuals panel and animkit's boosting diagram.
    """
    model = GradientBoostingRegressor(n_estimators=N_TREES, learning_rate=LEARNING_RATE,
                                      max_depth=TREE_DEPTH, min_samples_leaf=min_leaf(data.n_train),
                                      random_state=RANDOM_STATE)
    model.fit(data.x_train.reshape(-1, 1), data.y_train)
    booster = Booster(model)
    snapshots = tuple(make_snapshot(booster.stage(s.stage), data, label=f"{s.stage} trees",
                                    progress=s.stage, note=_note(s), detail=s)
                      for s in _states(booster, data))

    def predict(x: np.ndarray) -> np.ndarray:
        """The full model's prices.

        Args:
            x: Raw incomes ($10k).

        Returns:
            The predicted prices ($100k).
        """
        return model.predict(np.asarray(x, dtype=float).reshape(-1, 1))

    def explain(income: float) -> Explanation:
        """Works boosting's price for one income: F₀, then what each group of trees adds.

        Args:
            income: The income ($10k).

        Returns:
            The worked lines, checked against the model's own prediction.
        """
        staged = booster.staged_at(income)
        lines = [f"start: F₀ = average training price = {money_k(staged[0])}"]
        lines += [f"trees {first}–{last} add {money_k(staged[last] - staged[first - 1], signed=True)}"
                  for first, last in CONTRIBUTION_GROUPS]
        worked = float(staged[-1])
        lines.append(f"ŷ = F₀ + {LEARNING_RATE} × (sum of {N_TREES} trees) = {money_k(worked)}")
        return verified(lines, worked, predict_one(predict, income))

    return Lesson(
        story=_story(snapshots), data=data, snapshots=snapshots, final_index=len(snapshots) - 1,
        predict=predict, explain=explain, panel=BoostPanel(booster, data),
        diagram=boosting_diagram(booster, data),
        config={"model": "GradientBoostingRegressor", "n_estimators": str(N_TREES),
                "learning_rate": str(LEARNING_RATE), "max_depth": str(TREE_DEPTH),
                "min_samples_leaf": str(min_leaf(data.n_train)), "loss": "squared_error",
                "random_state": str(RANDOM_STATE)})
