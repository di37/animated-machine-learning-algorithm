"""The regression lesson contract: immutable training snapshots of a real model, and the final model.

Every snapshot is computed from a real model state. Nothing between two snapshots is invented:
the director holds each one on screen rather than interpolating a curve that never existed.
The words of the video (Story) and the verified worked prediction come from animkit.lessons.lesson.

A ``Lesson`` is what one algorithm's story module builds and the director plays: the ``Story`` (every
word the video says), the ``IncomeData``, a tuple of ``Snapshot`` states the model passed through while
it trained (each with its curve and its train and validation error), which of them is the final model,
the final model's ``predict``, an ``explain`` that works one prediction through by hand, and a ``Panel``
that draws the algorithm's own side panel. Prices are in the dataset's $100k units throughout; the
helpers here turn them into the "$305k" text the video shows, and errors into RMSE in $k.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from animkit.lessons.lesson import Explanation, LessonError, Story, plural
from animkit.lessons.lesson import verified as verified_with
from house_anim.data import read_only
from house_anim.lessons.config import PRICE_TO_K
from house_anim.lessons.data import IncomeData, Query

__all__ = ["Explanation", "Lesson", "LessonError", "Panel", "Predictor", "Snapshot", "Story", "baseline_rmse_k",
           "best_validation", "income_k", "make_snapshot", "money_k", "plural", "predict_one", "rmse_k", "verified"]

Predictor = Callable[[np.ndarray], np.ndarray]  # raw MedInc -> MedHouseVal ($100k)


def rmse_k(predicted: np.ndarray, actual: np.ndarray) -> float:
    """The root-mean-square error of some predictions: the typical size of a miss.

    Args:
        predicted: The predicted prices ($100k).
        actual: The true prices ($100k), in the same order.

    Returns:
        The RMSE, in $k.
    """
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2)) * PRICE_TO_K)


def money_k(value: float, signed: bool = False) -> str:
    """$100k units -> '$305k' (or '+$12k' / '−$12k' when ``signed``).

    Args:
        value: An amount in $100k units.
        signed: Whether to write a sign in front, for a correction or a difference.

    Returns:
        The amount in whole $k, as the video shows it.
    """
    amount = value * PRICE_TO_K
    if not signed:
        return f"${amount:,.0f}k"
    sign = "+" if amount >= 0 else "−"
    return f"{sign}${abs(amount):,.0f}k"


def income_k(value: float) -> str:
    """A median income as the video shows it: 6.0 -> '$60.0k'.

    Args:
        value: An income in ``MedInc`` units ($10k).

    Returns:
        The income in $k, to one decimal place.
    """
    return f"${value * 10.0:,.1f}k"


@dataclass(frozen=True)
class Snapshot:
    """One real state of the model while it trains, as the training chapter shows it.

    Attributes:
        label: The header text, e.g. "Step 12 of 60".
        progress: Its x position on the metric chart (a step, a depth, a tree count, ...).
        curve: The model's predictions on ``data.grid`` ($100k), drawn as its curve.
        fitted: Its predictions at ``data.residual_x``, where the residual lines end.
        train_rmse_k: Its RMSE on the training districts, in $k.
        val_rmse_k: Its RMSE on the validation districts, in $k.
        note: What changed at this snapshot, shown in the callout.
        detail: Algorithm-specific state for the side panel (weights, a tree, ...); None if there is none.
    """

    label: str  # header text, e.g. "Step 12 of 60"
    progress: float  # x position on the metric chart
    curve: np.ndarray  # predictions on data.grid
    fitted: np.ndarray  # predictions at data.residual_x, for residual lines
    train_rmse_k: float
    val_rmse_k: float
    note: str  # what changed at this snapshot, shown in the caption
    detail: Any = None  # algorithm-specific state for the side panel


def make_snapshot(predict: Predictor, data: IncomeData, *, label: str, progress: float, note: str,
                  detail: Any = None) -> Snapshot:
    """Records one model state: its curve, its residual ends and its train and validation errors.

    Args:
        predict: The model in this state, raw income ($10k) in, price ($100k) out.
        data: The lesson's data.
        label: The header text.
        progress: The x position on the metric chart.
        note: What changed at this snapshot.
        detail: Algorithm-specific state for the side panel.

    Returns:
        The snapshot, its arrays read-only.
    """
    return Snapshot(
        label=label,
        progress=float(progress),
        curve=read_only(predict(data.grid)),
        fitted=read_only(predict(data.residual_x)),
        train_rmse_k=rmse_k(predict(data.x_train), data.y_train),
        val_rmse_k=rmse_k(predict(data.x_val), data.y_val),
        note=note,
        detail=detail,
    )


def verified(lines: list[str], worked: float, model_value: float, *,
             show: Callable[[float], str] = money_k, call: str = "model.predict(x)") -> Explanation:
    """Wraps animkit's ``verified`` check line, showing the model's value as a house price by default.

    Args:
        lines: The worked prediction, one line per step.
        worked: The prediction the worked lines arrive at ($100k).
        model_value: The model's own prediction for the same input ($100k).
        show: Formats the model's value for the check line.
        call: The code shown as producing the model's value.

    Returns:
        The lines with a ✓ check line added, and the prediction.

    Raises:
        LessonError: If the worked prediction does not match the model's.
    """
    return verified_with(lines, worked, model_value, show=show, call=call)


class Panel:
    """Algorithm-specific drawing. Subclasses fill the side panel and may overlay the main plot."""

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Fills the side panel for one training snapshot.

        Args:
            ax: The side panel.
            snapshot: The model state being shown.

        Raises:
            NotImplementedError: Always; each subclass draws its own panel.
        """
        raise NotImplementedError

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Fills the side panel while the final model prices the unseen district.

        Args:
            ax: The side panel.
            query: The unseen district.
            t: How far through the panel's animation, from 0 to 1.

        Raises:
            NotImplementedError: Always; each subclass draws its own panel.
        """
        raise NotImplementedError

    def overlay_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Extras on the main plot during training (default: none).

        Args:
            ax: The main plot.
            snapshot: The model state being shown.
        """

    def overlay_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Extras on the main plot during inference (default: none).

        Args:
            ax: The main plot.
            query: The unseen district.
            t: How far through the overlay's animation, from 0 to 1.
        """


@dataclass(frozen=True)
class Lesson:
    """One algorithm's lesson, ready for the director: its words, its data and its real training history.

    Attributes:
        story: Every word the video says about the algorithm.
        data: The income view the model learned from.
        snapshots: The model's states while it trained, in the order shown.
        final_index: The snapshot kept for inference (the best validation score for sweeps).
        predict: The final model, raw income ($10k) in, price ($100k) out.
        explain: Works the final model's prediction for one income through by hand, checked.
        panel: Draws the algorithm's side panel, and any extras on the main plot.
        config: The model settings, written to the summary for reproducibility (a read-only copy).
        frame_weights: The relative screen time of each snapshot; None gives them equal time.
        diagram: The algorithm diagram shown above the panel; None when the panel already is one.
    """

    story: Story
    data: IncomeData
    snapshots: tuple[Snapshot, ...]
    final_index: int  # the model kept for inference (the best validation score for sweeps)
    predict: Predictor
    explain: Callable[[float], Explanation]
    panel: Panel
    config: Mapping[str, str]  # model settings, written to the summary for reproducibility
    frame_weights: tuple[float, ...] | None = None  # relative screen time per snapshot
    diagram: Any = None  # algorithm diagram shown above the panel; None when the panel already is one

    def __post_init__(self):
        """Freezes ``config`` as a read-only copy and checks the snapshots.

        Raises:
            ValueError: If there are no snapshots, or ``final_index`` points past them.
        """
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))  # read-only copy
        if not self.snapshots:
            raise ValueError("a lesson needs at least one snapshot")
        if not 0 <= self.final_index < len(self.snapshots):
            raise ValueError("final_index must point at a snapshot")

    @property
    def final(self) -> Snapshot:
        """The snapshot of the model kept for inference.

        Returns:
            ``snapshots[final_index]``.
        """
        return self.snapshots[self.final_index]


def best_validation(snapshots: tuple[Snapshot, ...]) -> int:
    """Which snapshot scores best on the validation districts: how a sweep picks its setting.

    Args:
        snapshots: The model states to choose from.

    Returns:
        The index of the one with the lowest validation RMSE (the first, on a tie).
    """
    return int(np.argmin([snap.val_rmse_k for snap in snapshots]))


def baseline_rmse_k(data: IncomeData, actual: np.ndarray | None = None) -> float:
    """Error of always predicting the full training average (validation by default): the bar to beat.

    Args:
        data: The lesson's data, which holds the training average.
        actual: The true prices to score against ($100k); None uses the validation prices.

    Returns:
        The baseline's RMSE, in $k.
    """
    target = data.y_val if actual is None else actual
    return rmse_k(np.full_like(target, data.baseline_mean), target)


def predict_one(predict: Predictor, income: float) -> float:
    """A model's price for one income.

    Args:
        predict: The model, raw income ($10k) in, price ($100k) out.
        income: The income ($10k).

    Returns:
        The predicted price ($100k).
    """
    return float(predict(np.array([income]))[0])
