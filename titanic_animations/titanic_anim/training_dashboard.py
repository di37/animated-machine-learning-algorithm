"""Live training dashboard: accuracy curve, ROC curve, learned weights, confusion matrix.

One figure per classifier, redrawn after every training step. The algorithm's diagram sits on the
left, the four charts in a 2x2 grid in the middle, and the equation card on the right, under a
header, a progress bar and a status line. The ROC curve and the confusion matrix use the
validation passengers, which the model never trains on; the learning curve sets their accuracy
beside the training accuracy.

The pieces every project shares come from animkit: the header and progress bar, the weight bars,
the equation panel and the panel positions.
"""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from animkit import theme as base
from animkit.config import ALGORITHM_PANEL_RECT, EQUATION_PANEL_RECT, TRAINING_FIGSIZE
from animkit.equation_panel import EquationPanel
from animkit.equations import EquationCard
from animkit.panels import ProgressBar, add_header, format_step
from animkit.training_dashboard import WeightBars, format_seconds
from titanic_anim.algorithms import algorithm_view
from titanic_anim.data import TitanicSplits
from titanic_anim.frames import (
    KIND_COEFFICIENT,
    KIND_IMPORTANCE,
    KIND_SENSITIVITY,
    KIND_SEPARATION,
    FeatureWeights,
    TrainingFrame,
)
from titanic_anim.models import ClassifierSpec
from titanic_anim.panels import ConfusionMatrixPanel, RateLines, RocCurve, percent

WEIGHT_TITLES = {
    KIND_COEFFICIENT: "Learned weights (log-odds per std. dev.)",
    KIND_IMPORTANCE: "Feature importances",
    KIND_SENSITIVITY: "Input sensitivity (mean |weight|)",
    KIND_SEPARATION: "Survivors' mean − others' mean (in σ)",
}


class ClassifierWeightBars(WeightBars):
    """animkit's weight bars with classification titles; NB separations are signed like weights."""

    def update(self, weights: FeatureWeights | None) -> None:
        """Redraws the bars, titled for the kind of weight; a model with no weights shows its note instead.

        Args:
            weights: The model's per-feature view, or None if it has none.
        """
        signed = weights is not None and weights.kind == KIND_SEPARATION
        super().update(replace(weights, kind=KIND_COEFFICIENT) if signed else weights)
        if weights is not None:
            self.ax.set_title(WEIGHT_TITLES[weights.kind])


class TrainingDashboard:
    """The training figure of one classifier.

    Attributes:
        figure: The matplotlib figure, which animkit's sinks show and record.
        progress: The bar showing how far through its schedule training is.
        status_text: The line under the header: the step, fit time and validation scores.
        algorithm: The classifier's animated diagram.
        curve: The learning curve: train and validation accuracy.
        roc: The validation ROC curve.
        weights: The learned weights, importances or class separations.
        confusion: The validation confusion matrix.
        equations: The equation card.

    Example:
        >>> dashboard = TrainingDashboard(get_spec("random_forest"), splits)  # doctest: +SKIP
        >>> dashboard.render(step.frame, card, model=step.model)  # doctest: +SKIP
    """

    def __init__(self, spec: ClassifierSpec, splits: TitanicSplits, figsize=TRAINING_FIGSIZE):
        """Builds the figure and every panel, empty until the first ``render``.

        Args:
            spec: The classifier being trained.
            splits: The splits; the diagram reads them, and the validation labels feed the ROC curve.
            figsize: The figure's size in inches.
        """
        plt.rcParams.update(base.RC_PARAMS)
        self._spec = spec
        self._y_val = splits.y_val
        self.figure = plt.figure(figsize=figsize)
        add_header(self.figure, f"{spec.title} · training live", spec.note)
        self.progress = ProgressBar(self.figure)
        self.status_text = self.figure.text(0.04, 0.855, "", fontsize=base.FONT_SUBTITLE,
                                            color=base.INK_PRIMARY)
        self.algorithm = algorithm_view(spec, self.figure.add_axes(ALGORITHM_PANEL_RECT), splits)
        grid = self.figure.add_gridspec(2, 2, left=0.355, right=0.705, top=0.78, bottom=0.08,
                                        hspace=0.38, wspace=0.32)
        self.curve = self._build_curve(self.figure.add_subplot(grid[0, 0]))
        self.roc = RocCurve(self.figure.add_subplot(grid[0, 1]))
        self.weights = ClassifierWeightBars(self.figure.add_subplot(grid[1, 0]), splits.feature_names,
                                            spec.no_weights_note)
        self.confusion = ConfusionMatrixPanel(self.figure.add_subplot(grid[1, 1]))
        self.equations = EquationPanel(self.figure.add_axes(EQUATION_PANEL_RECT))
        self._x_range_set = False

    def _build_curve(self, ax: Axes) -> RateLines:
        """The learning-curve panel, against the spec's step, on a log x-axis if the spec asks for one."""
        chart = RateLines(ax, ("train", "validation"))
        ax.set_title("Learning curve · accuracy")
        ax.set_xlabel(self._spec.step_label)
        ax.set_ylabel("passengers predicted correctly")
        if self._spec.log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
        return chart

    @property
    def train_line(self):
        """The learning curve's training-accuracy line.

        Returns:
            The matplotlib line.
        """
        return self.curve.lines[0]

    @property
    def weights_note(self):
        """The note the weights panel shows for a model without per-feature weights.

        Returns:
            The matplotlib text.
        """
        return self.weights.note

    def _set_x_range(self, x_range: tuple[float, float]) -> None:
        """Fixes the learning curve's x-axis to the whole schedule, widening a span of one step."""
        low, high = x_range
        if high <= low:
            low, high = low - 0.5, high + 0.5
        self.curve.ax.set_xlim(low, high)
        self._x_range_set = True

    def render(self, frame: TrainingFrame, card: EquationCard | None = None, model=None) -> None:
        """Draws one training frame.

        Args:
            frame: The training frame to draw.
            card: The equation card for this step; None leaves the card as it is.
            model: The live model, to redraw the algorithm's diagram; None leaves the diagram as it is.

        Raises:
            ValueError: If the frame's validation predictions do not match the validation set.
        """
        if frame.val_pred.shape != self._y_val.shape:
            raise ValueError("Frame validation predictions do not match the validation set")
        if card is not None:
            self.equations.update(card)
        if model is not None:
            self.algorithm.show_training(model, frame)
        if not self._x_range_set:
            self._set_x_range(frame.x_range)
        steps = np.array([p.step for p in frame.history])
        train = np.array([p.train_accuracy for p in frame.history])
        val = np.array([p.val_accuracy for p in frame.history])
        scores = frame.val_scores

        self.progress.update((frame.step_index + 1) / frame.total_steps)
        self.curve.update(steps, (train, val))
        self.roc.update(self._y_val, frame.val_score, scores.auc, "Validation ROC curve")
        self.weights.update(frame.weights)
        self.confusion.update(scores, "Validation")
        self.status_text.set_text(
            f"{self._spec.step_label}: {format_step(frame.step)}   ·   "
            f"step {frame.step_index + 1}/{frame.total_steps}   ·   "
            f"total fit time {format_seconds(frame.fit_seconds)}   ·   "
            f"validation accuracy {percent(scores.accuracy)}   ·   precision {percent(scores.precision)}"
            f"   ·   recall {percent(scores.recall)}"
        )
