"""Classification scores as immutable snapshots.

"Positive" means survived (label 1) throughout. For a set of predictions:

- accuracy is the share of passengers predicted correctly;
- precision: of the passengers predicted to survive, the share who did;
- recall: of the passengers who survived, the share the model found;
- F1 is the harmonic mean of precision and recall, high only when both are;
- AUC, the area under the ROC curve, is the chance that a random survivor scores higher than a
  random non-survivor (0.5 is guessing). It needs scores, not only labels, and both outcomes;
- log loss is how far the predicted probabilities are from what happened (probability models
  only);
- the confusion matrix counts each (what happened, what was predicted) pair.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from titanic_anim.config import CLASSES

PROBABILITY_CLIP = 1e-15


def sigmoid(z):
    """σ(z) = 1 / (1 + e^-z), without overflow for large |z|; floats in, floats out.

    It turns log-odds z into a probability: 0 becomes 0.5, large positive z nearly 1.

    Args:
        z: A number or an array of numbers.

    Returns:
        A float for a single number, otherwise an array of the same shape.
    """
    value = expit(np.asarray(z, dtype=float))
    return float(value) if value.ndim == 0 else value


@dataclass(frozen=True)
class ClassificationScores:
    """Every classification score of one set of predictions.

    Attributes:
        accuracy: The share of passengers predicted correctly.
        precision: Of the passengers predicted to survive, the share who did (0 if none was).
        recall: Of the passengers who survived, the share the model found (0 if none did).
        f1: The harmonic mean of precision and recall.
        auc: The area under the ROC curve; NaN until both outcomes have been seen, or without scores.
        log_loss: The mean log loss; NaN for margin models, which give no probabilities.
        tn: True negatives: non-survivors predicted not to survive.
        fp: False positives: non-survivors predicted to survive.
        fn: False negatives: survivors predicted not to survive.
        tp: True positives: survivors predicted to survive.
    """

    accuracy: float
    precision: float  # of the passengers predicted to survive, the share who did
    recall: float  # of the passengers who survived, the share the model found
    f1: float
    auc: float  # NaN until both outcomes have been seen
    log_loss: float  # NaN for margin models, which give no probabilities
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def error_rate(self) -> float:
        """The share of passengers predicted wrongly.

        Returns:
            1 − accuracy.
        """
        return 1.0 - self.accuracy

    @property
    def confusion(self) -> np.ndarray:
        """The 2x2 confusion matrix: rows are what happened, columns what was predicted.

        Returns:
            ``[[tn, fp], [fn, tp]]``, as an int array.
        """
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])


def _validate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> None:
    """Refuses mismatched shapes, empty or non-1-D arrays, labels other than 0 and 1, and non-finite scores."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")
    if y_true.ndim != 1:
        raise ValueError("score expects 1-D arrays")
    if y_true.size == 0:
        raise ValueError("score needs at least one prediction")
    if not (np.isin(y_true, CLASSES).all() and np.isin(y_pred, CLASSES).all()):
        raise ValueError("labels must be 0 (did not survive) or 1 (survived)")
    if y_score is not None and (y_score.shape != y_true.shape or not np.isfinite(y_score).all()):
        raise ValueError("y_score must be finite and match y_true's shape")


def score(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None,
          probabilities: bool = True) -> ClassificationScores:
    """Scores a set of predicted labels; with ``y_score`` it adds AUC and log loss.

    Args:
        y_true: What really happened, 0 or 1 per passenger.
        y_pred: The predicted labels, 0 or 1 per passenger.
        y_score: Each passenger's P(survive) or margin, or None for labels only.
        probabilities: Whether ``y_score`` holds probabilities; log loss is computed only if it does.

    Returns:
        The scores, as an immutable ``ClassificationScores``.

    Raises:
        ValueError: If the arrays are empty, not 1-D or of different shapes, a label is not 0 or 1,
            or a score is not finite.
    """
    _validate(y_true, y_pred, y_score)
    both = len(np.unique(y_true)) == 2
    auc = float(roc_auc_score(y_true, y_score)) if both and y_score is not None else float("nan")
    loss = (float(log_loss(y_true, np.clip(y_score, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP), labels=CLASSES))
            if probabilities and y_score is not None else float("nan"))
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=CLASSES).ravel())
    return ClassificationScores(
        accuracy=float(np.mean(y_true == y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=auc,
        log_loss=loss,
        tn=tn, fp=fp, fn=fn, tp=tp,
    )


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ROC curve: false positive rate against true positive rate for every threshold.

    Sliding the threshold from strict to lenient trades missed survivors for false alarms; each
    threshold is one point.

    Args:
        y_true: What really happened, 0 or 1 per passenger.
        y_score: Each passenger's P(survive) or margin.

    Returns:
        ``(false positive rates, true positive rates)``; the guessing diagonal if only one outcome is
        present, since no curve can be drawn then.
    """
    if len(np.unique(y_true)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr
