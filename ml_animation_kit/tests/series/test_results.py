"""Each lesson's final scores, turned into the conclusion's scoreboard and before-and-after rows."""

import pytest

from animkit.lessons.lesson import Comparison, ComparisonRow
from animkit.series.charts import Bar
from animkit.series.results import Result, result_of, scoreboard, shifts, spread, widest_gap


def comparison(lesson: float, richer: float, unit: str = "$", headline: str = "Seven more inputs cut it.") -> Comparison:
    return Comparison(metric="Test RMSE (lower is better)", higher_is_better=False, rows=(
        ComparisonRow("Always the average", 114.0, f"{unit}114k", "no input", "baseline"),
        ComparisonRow("Lesson model", lesson, f"{unit}{lesson:.0f}k", "one input", "lesson"),
        ComparisonRow("Every input", richer, f"{unit}{richer:.0f}k", "eight inputs", "richer"),
    ), headline=headline, note="Same test rows.", takeaway="More inputs help.")


def result(key, test, every, rows=1000):
    return result_of(key, key.title(), train=test - 1, validation=test + 1, test=test, training_rows=rows,
                     comparison=comparison(test, every))


RESULTS = (result("linear", 84.2, 75.0), result("forest", 84.4, 55.0), result("tree", 83.5, 76.0, rows=30))


def test_a_result_reads_the_lesson_and_every_input_bars_of_its_comparison():
    linear = RESULTS[0]
    assert (linear.one_input, linear.every_input) == (84.2, 75.0)
    assert (linear.one_input_text, linear.every_input_text) == ("$84k", "$75k")
    assert linear.verdict == "Seven more inputs cut it."


def test_the_scoreboard_puts_the_best_first_and_the_baseline_last():
    baseline = Bar("Always the average", 114.5, "$114k", role="baseline")
    bars = scoreboard(RESULTS, text=lambda v: f"${v:.1f}k", higher_is_better=False, baseline=baseline)
    assert [bar.label for bar in bars][:3] == ["Tree (trained on 30 rows)", "Linear", "Forest"]
    assert bars[-1] == baseline and bars[0].text == "$83.5k"


def test_a_higher_is_better_scoreboard_puts_the_highest_first():
    bars = scoreboard(RESULTS, text=str, higher_is_better=True, baseline=Bar("b", 50, "50", "baseline"),
                      full_rows=None)
    assert [bar.label for bar in bars][:3] == ["Forest", "Linear", "Tree"]


def test_the_rows_put_the_best_every_input_score_first():
    rows = shifts(RESULTS, higher_is_better=False)
    assert [row.label for row in rows] == ["Forest", "Linear", "Tree"]
    assert (rows[0].before, rows[0].after, rows[0].after_text) == (84.4, 55.0, "$55k")


def test_the_spread_is_the_lowest_and_highest_value():
    assert spread(RESULTS, "test") == (83.5, 84.4)
    assert spread(RESULTS, "every_input") == (55.0, 76.0)


def test_a_result_is_frozen():
    with pytest.raises(AttributeError):
        RESULTS[0].test = 1.0  # type: ignore[misc]
    assert isinstance(RESULTS[0], Result)


def test_the_widest_gap_is_the_model_that_fits_its_training_rows_best_compared_with_validation():
    tight = result_of("tight", "Tight", train=80.0, validation=81.0, test=81.0, training_rows=10,
                      comparison=comparison(81.0, 70.0))
    loose = result_of("loose", "Loose", train=60.0, validation=84.0, test=84.0, training_rows=10,
                      comparison=comparison(84.0, 70.0))
    assert widest_gap((tight, loose), higher_is_better=False) == loose
    accurate = result_of("acc", "Acc", train=0.87, validation=0.75, test=0.78, training_rows=10,
                         comparison=comparison(0.78, 0.8))
    honest = result_of("hon", "Hon", train=0.77, validation=0.78, test=0.81, training_rows=10,
                       comparison=comparison(0.81, 0.8))
    assert widest_gap((honest, accurate), higher_is_better=True) == accurate
