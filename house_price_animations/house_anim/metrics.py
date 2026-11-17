"""Regression scores as immutable snapshots.

Three scores describe how far predictions are from the truth:

- RMSE (root mean squared error) squares each error before averaging, so big misses weigh most;
  it is in the target's units.
- MAE (mean absolute error) is the typical miss, in the target's units.
- R² is the share of the target's variance the model explains: 1 is perfect, 0 is no better than
  always guessing the mean, and below 0 is worse.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


@dataclass(frozen=True)
class RegressionScores:
    """RMSE, MAE and R² for one set of predictions.

    Attributes:
        rmse: Root mean squared error, in the target's units ($100k here).
        mae: Mean absolute error, in the target's units.
        r2: The coefficient of determination; NaN when fewer than two points have been scored.
    """

    rmse: float
    mae: float
    r2: float  # NaN when fewer than two points have been scored


def score(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionScores:
    """Scores predictions against the true values.

    Args:
        y_true: The true values, a 1-D array.
        y_pred: The predictions, the same shape.

    Returns:
        Their RMSE, MAE and R² (R² is NaN for a single point, where it is undefined).

    Raises:
        ValueError: If the shapes differ, the arrays are not 1-D or are empty, or a value is not
            finite.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")
    if y_true.ndim != 1:
        raise ValueError("score expects 1-D arrays")
    if y_true.size == 0:
        raise ValueError("score needs at least one prediction")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("score received values that are not finite")

    r2 = float(r2_score(y_true, y_pred)) if y_true.size >= 2 else float("nan")
    return RegressionScores(
        rmse=float(root_mean_squared_error(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=r2,
    )


def share_within(residuals: np.ndarray, tolerance: float) -> float:
    """Fraction of residuals whose magnitude is at most ``tolerance``.

    Args:
        residuals: Prediction errors (predicted minus true, or the reverse: only size matters).
        tolerance: The largest error that still counts as close, in the residuals' units.

    Returns:
        A share from 0 to 1; 0 when there are no residuals.
    """
    if residuals.size == 0:
        return 0.0
    return float(np.mean(np.abs(residuals) <= tolerance))
