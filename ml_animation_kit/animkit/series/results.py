"""Each lesson's final scores, as the conclusion reports them: a scoreboard and before-and-after rows.

A ``Result`` holds what one lesson's final model scored (train, validation, test, in the metric's own
units) and what its more-features chapter measured on the same test rows: the lesson's model and the
same algorithm given every input. The conclusion only reorders and draws these numbers; it computes
no new ones.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from animkit.lessons.lesson import Comparison
from animkit.series.charts import Bar, Shift


@dataclass(frozen=True)
class Result:
    """One lesson's final model, and the same algorithm given every input.

    Attributes:
        key: The lesson's key.
        title: The lesson's title.
        train: The final model's training score.
        validation: Its validation score.
        test: Its test score.
        training_rows: How many rows it learned from.
        one_input: The more-features chapter's score for the lesson's model.
        every_input: Its score for the same algorithm given every input.
        one_input_text: ``one_input`` as the chapter printed it.
        every_input_text: ``every_input`` as the chapter printed it.
        verdict: The chapter's headline: whether every input helped, on paired evidence.
    """

    key: str
    title: str
    train: float
    validation: float
    test: float
    training_rows: int
    one_input: float
    every_input: float
    one_input_text: str
    every_input_text: str
    verdict: str


def result_of(key: str, title: str, *, train: float, validation: float, test: float, training_rows: int,
              comparison: Comparison) -> Result:
    """A lesson's result, with the every-input numbers read from its more-features comparison.

    Args:
        key: The lesson's key.
        title: The lesson's title.
        train: The final model's training score.
        validation: Its validation score.
        test: Its test score.
        training_rows: How many rows it learned from.
        comparison: The lesson's more-features comparison.

    Returns:
        The result. A ``Comparison`` always has exactly one "lesson" and one "richer" bar.
    """
    rows = {row.role: row for row in comparison.rows}
    lesson, richer = rows["lesson"], rows["richer"]
    return Result(key, title, train, validation, test, training_rows, lesson.value, richer.value, lesson.text,
                  richer.text, comparison.headline)


def scoreboard(results: Sequence[Result], *, text: Callable[[float], str], higher_is_better: bool, baseline: Bar,
               full_rows: int | None = 0) -> tuple[Bar, ...]:
    """One bar per lesson's test score, best first, then the baseline.

    A lesson that learned from fewer rows than the others says so on its bar, so a viewer does not
    compare it with the rest unawares.

    Args:
        results: The lessons' results.
        text: Prints a score, for example "$84k" or "81%".
        higher_is_better: Whether a higher score is better.
        baseline: The baseline's bar, drawn last.
        full_rows: The usual training size; 0 takes the largest, None marks no bar.

    Returns:
        The bars, top to bottom.
    """
    usual = max(result.training_rows for result in results) if full_rows == 0 else full_rows
    ordered = sorted(results, key=lambda result: result.test, reverse=higher_is_better)
    return (*(Bar(_label(result, usual), result.test, text(result.test)) for result in ordered), baseline)


def _label(result: Result, usual: int | None) -> str:
    """The bar's label: the title, and the training size when it is smaller than the usual one."""
    if usual is None or result.training_rows >= usual:
        return result.title
    return f"{result.title} (trained on {result.training_rows:,} rows)"


def shifts(results: Sequence[Result], *, higher_is_better: bool) -> tuple[Shift, ...]:
    """Each lesson's one-input and every-input scores, the best every-input score first.

    Args:
        results: The lessons' results.
        higher_is_better: Whether a higher score is better.

    Returns:
        One row per lesson.
    """
    ordered = sorted(results, key=lambda result: result.every_input, reverse=higher_is_better)
    return tuple(Shift(result.title, result.one_input, result.every_input, result.one_input_text,
                       result.every_input_text) for result in ordered)


def spread(results: Sequence[Result], field: str) -> tuple[float, float]:
    """The lowest and highest of one score across the lessons, for example ``spread(results, "test")``."""
    values = [getattr(result, field) for result in results]
    return min(values), max(values)


def widest_gap(results: Sequence[Result], *, higher_is_better: bool) -> Result:
    """The lesson whose final model looks best on its training rows compared with validation.

    That gap is over-fitting made visible: a model scored on the rows it learned from looks better than it is.

    Args:
        results: The lessons' results.
        higher_is_better: Whether a higher score is better.

    Returns:
        The result with the widest gap in the training rows' favour.
    """
    def gap(result: Result) -> float:
        return result.train - result.validation if higher_is_better else result.validation - result.train
    return max(results, key=gap)
