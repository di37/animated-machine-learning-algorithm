"""The lesson contract every project shares: the words of the video and the verified worked prediction.

Every number a lesson shows comes from a real model state; ``verified`` refuses to print a ✓ for a
worked prediction that does not reproduce the model's own output.

A project's lesson carries a ``Story`` (every word the video says), a worked ``Explanation`` of one
prediction, and a ``Comparison`` against the same algorithm given more inputs. These check
themselves, so a lesson that does not add up fails loudly instead of showing a wrong number.
"""

from collections.abc import Callable
from dataclasses import dataclass

# A worked explanation must reproduce the model's own output to within this tolerance (in its units).
CHECK_TOLERANCE = 1e-9


class LessonError(ValueError):
    """A lesson cannot be shown truthfully (a solver did not converge, a check did not add up)."""


def plural(count: int, word: str) -> str:
    """A count and a noun, the noun made plural when the count is not one.

    Nouns ending in "f" take "ves"; others take "s".

    Args:
        count: How many.
        word: The noun, in the singular.

    Returns:
        For example '1 leaf', '14 leaves', '1 tree', '50 trees'.

    Example:
        >>> plural(14, "leaf")
        '14 leaves'
    """
    if count == 1:
        return f"1 {word}"
    return f"{count} {word[:-1] + 'ves' if word.endswith('f') else word + 's'}"


@dataclass(frozen=True)
class Explanation:
    """A worked prediction, checked against the model.

    Attributes:
        lines: The worked prediction, step by step; the last line is the model check.
        prediction: The worked value, in the model's own output units.
    """

    lines: tuple[str, ...]  # worked prediction, last line is the model check
    prediction: float  # in the model's own output units


def verified(lines: list[str], worked: float, model_value: float, *,
             show: Callable[[float], str], call: str = "model.predict(x)") -> Explanation:
    """Append the check line; refuse to print a ✓ the arithmetic did not earn.

    ``show`` formats the model's value for the check line (an amount, a probability, a score).

    Args:
        lines: The worked prediction's lines.
        worked: The value the worked lines arrive at.
        model_value: What the model itself predicts for the same row.
        show: Formats the model's value for the check line.
        call: The call written on the check line.

    Returns:
        The explanation, its last line "check: <call> = <value>  ✓".

    Raises:
        LessonError: If the worked value and the model's differ by more than ``CHECK_TOLERANCE``.
    """
    if abs(worked - model_value) > CHECK_TOLERANCE:
        raise LessonError(f"worked prediction {worked!r} disagrees with the model's {model_value!r}")
    return Explanation(lines=(*lines, f"check: {call} = {show(model_value)}  ✓"), prediction=float(worked))


@dataclass(frozen=True)
class Story:
    """Every word the video says about one algorithm.

    Attributes:
        key: The lesson's key.
        title: The algorithm's name, for example "Linear Regression".
        method: How it is trained, for the subtitle, for example "trained by batch gradient descent".
        objective: The learning goal on the title card.
        initial: The caption for the first guess.
        training: The caption during training.
        final: The caption for the final model.
        inference: The caption for the prediction on an unseen example.
        takeaway: The takeaway card's main line.
        limitation: What the algorithm does badly, for the takeaway card.
        progress_label: The metric chart's x-axis label.
        log_progress: Whether that axis is logarithmic.
    """

    key: str
    title: str  # "Linear Regression"
    method: str  # subtitle, e.g. "trained by batch gradient descent"
    objective: str  # title-card learning goal
    initial: str
    training: str
    final: str
    inference: str
    takeaway: str
    limitation: str
    progress_label: str  # metric chart x-axis
    log_progress: bool = False


ROLES = ("baseline", "lesson", "richer")


@dataclass(frozen=True)
class ComparisonRow:
    """One bar of the more-features comparison.

    Attributes:
        label: The model's name on the bar.
        value: The score, in the comparison's metric units; it sets the bar's length.
        text: The value as printed, for example "81%".
        detail: What this model saw.
        role: "baseline", "lesson" (the model the video trained) or "richer" (the same algorithm, more
            inputs).
    """

    label: str
    value: float  # in the comparison's metric units; sets the bar length
    text: str  # the value as printed, e.g. "$84k" or "81%"
    detail: str  # what this model saw
    role: str  # "baseline", "lesson" (the model the video trained) or "richer" (same algorithm, more inputs)

    def __post_init__(self):
        """Refuses a role that is not one of ``ROLES``.

        Raises:
            ValueError: If ``role`` is not "baseline", "lesson" or "richer".
        """
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")


@dataclass(frozen=True)
class Comparison:
    """The lesson's model against the same algorithm given every feature, scored on the same test rows.

    Attributes:
        metric: The metric's name, saying which way is better, for example "Test RMSE (lower is better)".
        higher_is_better: Whether a higher score is better.
        rows: The bars, top to bottom; exactly one has the role "lesson" and one "richer".
        headline: The card's headline.
        note: What was held equal, and what was not.
        takeaway: One short line for the takeaway card.
    """

    metric: str  # e.g. "Test RMSE (lower is better)"
    higher_is_better: bool
    rows: tuple[ComparisonRow, ...]
    headline: str
    note: str  # what was held equal, and what was not
    takeaway: str  # one short line for the takeaway card

    def __post_init__(self):
        """Refuses rows without exactly one lesson row and one richer row.

        Raises:
            ValueError: If there is not exactly one "lesson" row and one "richer" row.
        """
        roles = [row.role for row in self.rows]
        if roles.count("lesson") != 1 or roles.count("richer") != 1:
            raise ValueError("a comparison needs exactly one lesson row and one richer row")

    def _value(self, role: str) -> float:
        """The score of the row with this role."""
        return next(row.value for row in self.rows if row.role == role)

    def improvement(self) -> float:
        """How much better (positive) the richer model scores than the lesson's model.

        Returns:
            The gap in the metric's units, signed so that positive always means the richer model is better.
        """
        gap = self._value("richer") - self._value("lesson")
        return gap if self.higher_is_better else -gap
