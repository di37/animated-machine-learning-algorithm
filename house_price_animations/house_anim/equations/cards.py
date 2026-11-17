"""House-price equation lines on top of animkit's equation cards (EquationLine, heading, math_line …).

animkit's builders make one ``EquationLine`` each: ``heading``, ``math_line`` (matplotlib mathtext
between dollar signs), ``mono`` (aligned numbers) and ``note``. This module re-exports them and adds
the lines every house-price card needs: a price in both the model's units and $k, the check
against ``model.predict``, and the training MSE.
"""

import numpy as np

from animkit.equations.cards import (
    HEADING,
    KINDS,
    MATH,
    MONO,
    NOTE,
    PREDICT_TOLERANCE,
    EquationCard,
    EquationLine,
    heading,
    linear_terms,
    math_line,
    mono,
    note,
    power_of_ten,
    raw_value,
    subscript_name,
    validate_row,
)
from house_anim.config import TARGET_TO_THOUSANDS

__all__ = ["HEADING", "KINDS", "MATH", "MONO", "NOTE", "PREDICT_TOLERANCE", "EquationCard", "EquationLine",
           "check_line", "heading", "linear_terms", "math_line", "mono", "note", "power_of_ten", "price_line",
           "raw_value", "subscript_name", "training_mse", "validate_house"]


def price_line(value: float, symbol: str = "\\hat{y}") -> EquationLine:
    """A maths line giving a prediction in the model's units and in dollars, such as ŷ = 2.453 → $245k.

    Args:
        value: The prediction, in $100k.
        symbol: The mathtext symbol on the left of the equals sign.

    Returns:
        The line.
    """
    return math_line(f"${symbol} = {value:.3f}$   →   \\${value * TARGET_TO_THOUSANDS:,.0f}k")


def check_line(model, x: np.ndarray, value: float) -> EquationLine:
    """Proves the worked equation is the model's own computation.

    It calls ``model.predict`` on the same house and marks the line ✓ when the two agree within
    ``PREDICT_TOLERANCE``, or ✗ when they do not.

    Args:
        model: The fitted model.
        x: The house's scaled features, shape (8,).
        value: What the worked equation arrived at, in $100k.

    Returns:
        A note line: ``check: model.predict(x) = ...`` with the mark.
    """
    predicted = float(model.predict(x[None, :])[0])
    mark = "✓" if abs(predicted - value) <= PREDICT_TOLERANCE else "✗ (mismatch)"
    return note(f"check: model.predict(x) = {predicted:.3f}  {mark}")


def training_mse(frame) -> float:
    """The training mean squared error at a step: its training RMSE squared.

    Args:
        frame: A ``TrainingFrame``.

    Returns:
        The MSE, in ($100k)².
    """
    return frame.history[-1].train_rmse ** 2


def validate_house(x: np.ndarray, n_features: int) -> np.ndarray:
    """One house's features as floats, checked to be a single row.

    Args:
        x: The house's features.
        n_features: How many features a house has.

    Returns:
        The features as a float array of shape (n_features,).

    Raises:
        ValueError: If ``x`` is not of shape (n_features,).
    """
    return validate_row(x, n_features, noun="house")
