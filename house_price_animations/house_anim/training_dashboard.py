"""Live training dashboard: learning curve, validation predictions, learned weights, residuals.

The figure has three columns. On the left, an animated diagram of the algorithm (from
``algorithms``); in the middle, four charts; on the right, the equation card for this step,
drawn by animkit's ``EquationPanel``. The four charts, all in $k:

- the learning curve: train and validation RMSE per step, where underfitting (both high),
  convergence (both flat) and overfitting (validation turning back up) show;
- validation predicted vs actual: the cloud tightens onto the diagonal as the model learns;
- the learned weights (animkit's ``WeightBars``): coefficients, importances or an MLP's
  first-layer weight sizes;
- the residual histogram, with the share of validation houses priced within ±$50k.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from animkit.equation_panel import EquationPanel
from animkit.training_dashboard import WeightBars, format_seconds
from house_anim import theme
from house_anim.algorithms import algorithm_view
from house_anim.config import (
    ALGORITHM_PANEL_RECT,
    EQUATION_PANEL_RECT,
    TARGET_TO_THOUSANDS,
    TRAINING_FIGSIZE,
)
from house_anim.data import HousingSplits
from house_anim.equations import EquationCard
from house_anim.frames import TrainingFrame
from house_anim.models import ModelSpec
from house_anim.panels import (
    PredictedVsActual,
    ProgressBar,
    ResidualHistogram,
    TwoLineChart,
    add_header,
    format_step,
    money_k,
)


class TrainingDashboard:
    """The training figure for one model, redrawn after every training step.

    Attributes:
        figure: The matplotlib figure the sinks record.
        progress: The progress bar through the schedule.
        status_text: The line with the step, total fit time and validation RMSE.
        algorithm: The model's animated algorithm diagram.
        curve: The learning curve.
        predictions: Validation predicted vs actual.
        weights: The learned-weights bars.
        residuals: The validation residual histogram.
        equations: The equation card's panel.

    Example:
        >>> dashboard = TrainingDashboard(spec, splits)  # doctest: +SKIP
        >>> for step in training_steps(spec, splits):  # doctest: +SKIP
        ...     dashboard.render(step.frame, card, model=step.model)
    """

    def __init__(self, spec: ModelSpec, splits: HousingSplits, figsize=TRAINING_FIGSIZE):
        """Builds the figure with every panel empty.

        Args:
            spec: The model being trained (its title, note, step label and x scale).
            splits: The data; the validation prices are what every frame is compared with.
            figsize: The figure size, in inches.
        """
        plt.rcParams.update(theme.RC_PARAMS)
        self._spec = spec
        self._actual_k = splits.y_val * TARGET_TO_THOUSANDS
        self.figure = plt.figure(figsize=figsize)
        add_header(self.figure, f"{spec.title} · training live", spec.note)
        self.progress = ProgressBar(self.figure)
        self.status_text = self.figure.text(0.04, 0.855, "", fontsize=theme.FONT_SUBTITLE,
                                            color=theme.INK_PRIMARY)
        self.algorithm = algorithm_view(spec, self.figure.add_axes(ALGORITHM_PANEL_RECT), splits)
        grid = self.figure.add_gridspec(2, 2, left=0.355, right=0.705, top=0.78, bottom=0.08,
                                        hspace=0.38, wspace=0.32)
        self.curve = self._build_curve(self.figure.add_subplot(grid[0, 0]))
        self.predictions = PredictedVsActual(self.figure.add_subplot(grid[0, 1]))
        self.weights = WeightBars(self.figure.add_subplot(grid[1, 0]), splits.feature_names,
                                  spec.no_weights_note)
        self.residuals = ResidualHistogram(self.figure.add_subplot(grid[1, 1]))
        self.equations = EquationPanel(self.figure.add_axes(EQUATION_PANEL_RECT))
        self._x_range_set = False

    def _build_curve(self, ax: Axes) -> TwoLineChart:
        """The learning curve, labelled with the spec's step label, on a log x axis if the spec asks."""
        chart = TwoLineChart(ax, ("train", "validation"))
        ax.set_title("Learning curve · RMSE (\\$k)")
        ax.set_xlabel(self._spec.step_label)
        ax.set_ylabel("RMSE (\\$k)")
        if self._spec.log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        return chart

    # Handles the test suite reads to check what is on screen.
    @property
    def scatter(self):
        """The predicted-vs-actual points, for the tests.

        Returns:
            The matplotlib scatter collection.
        """
        return self.predictions.scatter

    @property
    def train_line(self):
        """The learning curve's training-RMSE line, for the tests.

        Returns:
            The matplotlib line.
        """
        return self.curve.lines[0]

    @property
    def weights_note(self):
        """The note shown instead of bars when a model has no weights (KNN), for the tests.

        Returns:
            The matplotlib text.
        """
        return self.weights.note

    def _set_x_range(self, x_range: tuple[float, float]) -> None:
        """Fixes the learning curve's x range once, widening a one-step range so it is not empty."""
        low, high = x_range
        if high <= low:
            low, high = low - 0.5, high + 0.5
        self.curve.ax.set_xlim(low, high)
        self._x_range_set = True

    def render(self, frame: TrainingFrame, card: EquationCard | None = None, model=None) -> None:
        """Draw a frame; with ``model`` the algorithm diagram, with ``card`` the equations, update too.

        Args:
            frame: The snapshot after one training step.
            card: The equation card for this step; None leaves the panel as it is.
            model: The live model, for the algorithm diagram; None leaves the diagram as it is.

        Raises:
            ValueError: If the frame's validation predictions do not match the validation set.
        """
        if frame.val_pred.shape != self._actual_k.shape:
            raise ValueError("Frame validation predictions do not match the validation set")
        if card is not None:
            self.equations.update(card)
        if model is not None:
            self.algorithm.show_training(model, frame)
        if not self._x_range_set:
            self._set_x_range(frame.x_range)
        predicted_k = frame.val_pred * TARGET_TO_THOUSANDS
        steps = np.array([p.step for p in frame.history])
        train_k = np.array([p.train_rmse for p in frame.history]) * TARGET_TO_THOUSANDS
        val_k = np.array([p.val_rmse for p in frame.history]) * TARGET_TO_THOUSANDS

        self.progress.update((frame.step_index + 1) / frame.total_steps)
        self.curve.update(steps, (train_k, val_k))
        self.predictions.update(self._actual_k, predicted_k,
                                f"Validation · predicted vs actual · R² {frame.val_scores.r2:.3f}")
        self.weights.update(frame.weights)
        self.residuals.update(predicted_k - self._actual_k)
        self.status_text.set_text(
            f"{self._spec.step_label}: {format_step(frame.step)}   ·   "
            f"step {frame.step_index + 1}/{frame.total_steps}   ·   "
            f"total fit time {format_seconds(frame.fit_seconds)}   ·   "
            f"validation RMSE {money_k(frame.val_scores.rmse * TARGET_TO_THOUSANDS)}"
        )
