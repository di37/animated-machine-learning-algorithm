"""Immutable snapshots handed from the training/inference loops to the dashboards.

The loops do the work and the dashboards only draw: after each training step or prediction batch,
a loop yields a frozen snapshot of everything the next picture needs. A dashboard cannot change
the model or the data, and a snapshot never changes after it is drawn. ``FeatureWeights`` (the
weight bars' data) comes from animkit and is re-exported here.

Prices in the snapshots are in the dataset's units, $100k; the dashboards convert them to $k.
"""

from dataclasses import dataclass

import numpy as np

from animkit.frames import KIND_COEFFICIENT, KIND_IMPORTANCE, KIND_SENSITIVITY, FeatureWeights
from house_anim.metrics import RegressionScores

__all__ = ["KIND_COEFFICIENT", "KIND_IMPORTANCE", "KIND_SENSITIVITY", "FeatureWeights", "InferenceFrame",
           "MetricPoint", "TrainingFrame", "TrainingPoint"]


@dataclass(frozen=True)
class TrainingPoint:
    """One point of the learning curve.

    Attributes:
        step: The step's x-axis value (epochs, rows, trees, depth or -log10 alpha).
        train_rmse: RMSE on the training rows the model has seen (the first ``TRAIN_EVAL_SAMPLE`` at
            most), in $100k.
        val_rmse: RMSE on the validation set, in $100k.
    """

    step: float
    train_rmse: float
    val_rmse: float


@dataclass(frozen=True)
class TrainingFrame:
    """What the training dashboard draws after one training step.

    Attributes:
        step: This step's x-axis value.
        step_index: Its position in the schedule, from 0.
        total_steps: How many steps the schedule has.
        x_range: The learning curve's (first, last) x value, fixed before training starts.
        history: Every learning-curve point so far, this step's last.
        val_pred: The model's predictions for every validation house, in $100k.
        val_scores: Validation RMSE, MAE and R².
        weights: What the model has learned per feature, or None (KNN learns no weights).
        fit_seconds: All fitting so far in this run (a sweep sums every refit).
        model_fit_seconds: What it cost to produce the model shown in this frame.
    """

    step: float
    step_index: int
    total_steps: int
    x_range: tuple[float, float]
    history: tuple[TrainingPoint, ...]
    val_pred: np.ndarray
    val_scores: RegressionScores
    weights: FeatureWeights | None
    fit_seconds: float  # all fitting so far in this run (a sweep sums every refit)
    model_fit_seconds: float  # what it cost to produce the model shown in this frame

    @property
    def is_final(self) -> bool:
        """Whether this is the schedule's last step.

        Returns:
            True for the final frame of training.
        """
        return self.step_index == self.total_steps - 1


@dataclass(frozen=True)
class MetricPoint:
    """The running test scores after a prediction batch.

    Attributes:
        n_done: How many test houses have been predicted so far.
        rmse: RMSE over those houses, in $100k.
        mae: Mean absolute error over those houses, in $100k.
        r2: R² over those houses (NaN until two have been scored).
    """

    n_done: int
    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class InferenceFrame:
    """What the inference dashboard draws after one ``predict`` batch on the test houses.

    Attributes:
        n_done: Test houses predicted so far.
        n_total: Test houses in all.
        batch_start: The index of the latest batch's first house (the batch is
            ``batch_start:n_done``).
        y_true: True prices of the houses predicted so far, in $100k.
        y_pred: Their predicted prices, in $100k.
        coords: Their raw (longitude, latitude), for the map.
        history: The running scores after every batch so far, this one's last.
        batch_latency_ms: How long this batch's ``predict`` call took, in milliseconds.
        throughput_per_s: Houses predicted per second of predict time, over all batches so far.
    """

    n_done: int
    n_total: int
    batch_start: int
    y_true: np.ndarray
    y_pred: np.ndarray
    coords: np.ndarray
    history: tuple[MetricPoint, ...]
    batch_latency_ms: float
    throughput_per_s: float

    @property
    def scores(self) -> MetricPoint:
        """The running scores after this batch.

        Returns:
            The last point of ``history``.
        """
        return self.history[-1]

    @property
    def is_final(self) -> bool:
        """Whether every test house has been predicted.

        Returns:
            True for the final frame of inference.
        """
        return self.n_done == self.n_total
