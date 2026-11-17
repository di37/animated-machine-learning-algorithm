"""Classification helpers on top of animkit's equation cards (EquationLine, heading, math_line …).

A card is a list of lines of four kinds, all from animkit: ``heading`` (a section title),
``math_line`` (matplotlib mathtext between dollar signs), ``mono`` (an aligned row of numbers) and
``note`` (plain text). This module adds how a classification card ends: the probability, the
verdict with the rule that produced it, and a check line that asks the model itself and marks the
worked answer ✓, or ✗ if the two differ by more than ``PREDICT_TOLERANCE``.
"""

import numpy as np

from animkit.equations.cards import (
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
    validate_row,
)
from titanic_anim.config import PROBABILITY_THRESHOLD
from titanic_anim.metrics import sigmoid
from titanic_anim.passengers import outcome

__all__ = ["P_SURVIVE", "EquationCard", "EquationLine", "accuracy_note", "heading", "linear_terms",
           "margin_check_line", "math_line", "mono", "note", "power_of_ten", "probability_check_line",
           "probability_line", "raw_value", "sigmoid", "validate_passenger", "verdict_line"]

P_SURVIVE = r"P(\mathrm{survive})"
SIGMOID = r"$\sigma(z) = \frac{1}{1 + e^{-z}}$  squashes any number into 0 … 1"


def validate_passenger(x: np.ndarray, n_features: int) -> np.ndarray:
    """One passenger's features as floats, checked for shape.

    Args:
        x: The passenger's standardised features.
        n_features: How many features a passenger has.

    Returns:
        The row as a float array.

    Raises:
        ValueError: If ``x`` is not a single row of ``n_features`` values.
    """
    return validate_row(x, n_features, noun="passenger")


def probability_line(p: float, symbol: str = P_SURVIVE) -> EquationLine:
    """A probability, as an equation and in words.

    Args:
        p: The probability, from 0 to 1.
        symbol: The mathtext for what the probability is of.

    Returns:
        A math line such as "P(survive) = 0.731   →   73% chance of surviving".
    """
    return math_line(f"${symbol} = {p:.3f}$   →   {p:.0%} chance of surviving")


def verdict_line(model, x: np.ndarray, rule: str) -> EquationLine:
    """The model's own label for x, with the rule that produced it.

    Args:
        model: The trained model; its ``predict`` gives the label.
        x: The passenger's standardised features.
        rule: The rule, worked, as text, such as "0.731 > 0.5".

    Returns:
        A note such as "0.731 > 0.5  →  predicted: survived".
    """
    label = int(model.predict(x[None, :])[0])
    return note(f"{rule}  →  predicted: {outcome(label)}")


def probability_rule(p: float) -> str:
    """The decision rule a probability meets, worked, as text.

    Args:
        p: The probability, from 0 to 1.

    Returns:
        For example "0.731 > 0.5". A probability within 0.0005 of the threshold is shown to six
        decimals, so the side it falls on can be seen.
    """
    sign = ">" if p > PROBABILITY_THRESHOLD else "≤"
    digits = 3 if abs(p - PROBABILITY_THRESHOLD) >= 5e-4 else 6  # show enough digits to see the side
    return f"{p:.{digits}f} {sign} {PROBABILITY_THRESHOLD}"


def probability_check_line(model, x: np.ndarray, value: float) -> EquationLine:
    """Proves the worked equation is the model's own computation.

    Args:
        model: The trained model; its ``predict_proba`` is asked for P(survive).
        x: The passenger's standardised features.
        value: The probability the worked equation arrived at.

    Returns:
        A note with the model's own probability, marked ✓ if it matches ``value``, else ✗.
    """
    predicted = float(model.predict_proba(x[None, :])[0, 1])
    mark = "✓" if abs(predicted - value) <= PREDICT_TOLERANCE else "✗ (mismatch)"
    return note(f"check: model.predict_proba(x)[survived] = {predicted:.3f}  {mark}")


def margin_check_line(model, x: np.ndarray, value: float) -> EquationLine:
    """Proves the worked equation is the model's own computation, for a model scored by a margin.

    Args:
        model: The trained model; its ``decision_function`` is asked for f(x).
        x: The passenger's standardised features.
        value: The margin the worked equation arrived at.

    Returns:
        A note with the model's own margin, marked ✓ if it matches ``value``, else ✗.
    """
    predicted = float(model.decision_function(x[None, :])[0])
    mark = "✓" if abs(predicted - value) <= PREDICT_TOLERANCE else "✗ (mismatch)"
    return note(f"check: model.decision_function(x) = {predicted:+.3f}  {mark}")


def probability_ending(model, x: np.ndarray, p: float) -> list[EquationLine]:
    """The last lines of a probability card: the probability, the verdict and the check.

    Args:
        model: The trained model.
        x: The passenger's standardised features.
        p: The probability the worked equation arrived at.

    Returns:
        The three lines.
    """
    return [probability_line(p), verdict_line(model, x, probability_rule(p)),
            probability_check_line(model, x, p)]


def accuracy_note(frame) -> EquationLine:
    """The note with this step's training and validation accuracy.

    Args:
        frame: The training frame.

    Returns:
        A note such as "training accuracy 81.2%  ·  validation 79.4%".
    """
    point = frame.history[-1]
    return note(f"training accuracy {point.train_accuracy:.1%}  ·  validation {frame.val_scores.accuracy:.1%}")
