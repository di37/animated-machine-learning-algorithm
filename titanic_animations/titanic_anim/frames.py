"""Immutable snapshots handed from the training/inference loops to the dashboards.

A loop does real work, then freezes what it measured into a frame: a ``TrainingFrame`` after each
training step, an ``InferenceFrame`` after each batch of test passengers. A dashboard only reads
frames, so it draws exactly what the loop measured, and a frame never changes once drawn.

``FeatureWeights`` (from animkit) is the per-feature view of a model; its ``kind`` says what the
numbers are. ``KIND_SEPARATION`` adds Naive Bayes' class separation to animkit's three kinds.
"""

from dataclasses import dataclass

import numpy as np

from animkit.frames import KIND_COEFFICIENT, KIND_IMPORTANCE, KIND_SENSITIVITY, FeatureWeights
from titanic_anim.metrics import ClassificationScores

__all__ = ["KIND_COEFFICIENT", "KIND_IMPORTANCE", "KIND_SENSITIVITY", "KIND_SEPARATION",
           "FeatureWeights", "InferenceFrame", "MetricPoint", "TrainingFrame", "TrainingPoint"]

# Naive Bayes: how far apart the survivors' and non-survivors' means are, in standard deviations.
KIND_SEPARATION = "separation"


@dataclass(frozen=True)
class TrainingPoint:
    """One point of the learning curve.

    Attributes:
        step: The step's x-axis value (epochs, row count, number of trees, log10 C ...).
        train_accuracy: The share of the training passengers seen so far predicted correctly.
        val_accuracy: The share of the validation passengers predicted correctly.
        train_loss: The loss the model minimises (log loss, or hinge loss for the SVM) on the training
            passengers seen so far.
    """

    step: float
    train_accuracy: float
    val_accuracy: float
    train_loss: float  # the loss the model minimises (log loss, or hinge for the SVM)


@dataclass(frozen=True)
class TrainingFrame:
    """Everything a training dashboard draws after one training step.

    Attributes:
        step: This step's x-axis value.
        step_index: This step's position in the schedule, from 0.
        total_steps: How many steps the schedule has.
        x_range: The learning curve's x-axis span, fixed for the whole run.
        history: Every learning-curve point so far, this step's last.
        val_pred: The predicted label of every validation passenger.
        val_score: Every validation passenger's P(survive), or the SVM's margin.
        val_scores: The validation scores (accuracy, AUC, confusion counts ...).
        weights: The per-feature view of the model, or None if it has none.
        fit_seconds: All fitting so far in this run (a sweep adds up every refit).
        model_fit_seconds: What it cost to produce the model shown in this frame.
    """

    step: float
    step_index: int
    total_steps: int
    x_range: tuple[float, float]
    history: tuple[TrainingPoint, ...]
    val_pred: np.ndarray  # predicted labels for the validation passengers
    val_score: np.ndarray  # P(survive), or the SVM's margin
    val_scores: ClassificationScores
    weights: FeatureWeights | None
    fit_seconds: float  # all fitting so far in this run (a sweep sums every refit)
    model_fit_seconds: float  # what it cost to produce the model shown in this frame

    @property
    def is_final(self) -> bool:
        """Whether this is the schedule's last step.

        Returns:
            True for the last step.
        """
        return self.step_index == self.total_steps - 1


@dataclass(frozen=True)
class MetricPoint:
    """The running test scores after some of the test passengers have been classified.

    Attributes:
        n_done: How many test passengers have been classified.
        accuracy: The share of them predicted correctly.
        precision: Of those predicted to survive, the share who did.
        recall: Of those who survived, the share the model found.
        f1: The harmonic mean of precision and recall.
    """

    n_done: int
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class InferenceFrame:
    """Everything an inference dashboard draws after one batch of test passengers.

    Attributes:
        n_done: How many test passengers have been classified, this batch included.
        n_total: How many test passengers there are.
        batch_start: The index of this batch's first passenger.
        y_true: What really happened to the first ``n_done`` test passengers.
        y_pred: The model's labels for them.
        y_score: The model's scores for them: P(survive), or the SVM's margin.
        history: The running scores after every batch so far.
        batch_latency_ms: How long this batch's ``predict`` call took, in milliseconds.
        throughput_per_s: Test passengers classified per second of ``predict`` time so far.
    """

    n_done: int
    n_total: int
    batch_start: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
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
        """Whether every test passenger has been classified.

        Returns:
            True for the last batch.
        """
        return self.n_done == self.n_total
