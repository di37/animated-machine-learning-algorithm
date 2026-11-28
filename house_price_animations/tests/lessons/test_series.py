"""The house-price playlist's introduction and conclusion: what they say, built from the lessons' numbers."""

import pytest

from animkit.lessons import theme
from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.lesson import Comparison, ComparisonRow
from animkit.lessons.speech_text import spoken, unexpanded_acronyms
from animkit.lessons.stage import Stage
from animkit.series.cards import Algorithm
from animkit.series.charts import Cloud
from animkit.series.results import result_of
from house_anim.lessons.series import Playlist, conclusion, introduction

TITLES = ("Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression", "Decision Tree",
          "Random Forest", "Gradient Boosting", "k-Nearest Neighbours", "Neural Network (MLP)")
KEYS = ("linear_regression", "polynomial_regression", "ridge", "lasso", "decision_tree", "random_forest",
        "gradient_boosting", "knn", "mlp")
TESTS = (84.21, 84.06, 85.04, 84.68, 83.49, 84.44, 83.33, 83.34, 83.30)
EVERY = (75.0, 75.0, 75.0, 75.0, 76.0, 55.0, 57.0, 65.0, 59.0)
ROWS = (12384, 30, 30, 30, 12384, 12384, 12384, 12384, 12384)


def comparison(own: float, richer: float) -> Comparison:
    return Comparison(metric="Test RMSE (lower is better)", higher_is_better=False, headline=(
        f"Seven more inputs cut the typical error from ${own:.0f}k to ${richer:.0f}k."), note="n", takeaway="t",
        rows=(ComparisonRow("Lesson", own, f"${own:.0f}k", "income", "lesson"),
              ComparisonRow("Every input", richer, f"${richer:.0f}k", "all 8", "richer")))


@pytest.fixture(scope="module")
def playlist() -> Playlist:
    results = tuple(result_of(key, title, train=60.78 if key == "polynomial_regression" else test - 1,
                              validation=83.44 if key == "polynomial_regression" else test - 0.5, test=test,
                              training_rows=rows, comparison=comparison(test, every))
                    for key, title, test, every, rows in zip(KEYS, TITLES, TESTS, EVERY, ROWS))
    cloud = Cloud(x=(0.1, 0.5), y=(0.2, 1.0), groups=(0, 0), colours=(theme.DOTS,), legend=("district",),
                  x_label="median income", y_label="median house value")
    return Playlist(algorithms=tuple(Algorithm(title, "learns somehow") for title in TITLES), cloud=cloud,
                    train_rows=12384, val_rows=4128, test_rows=4128, shown_rows=900, results=results,
                    baseline=114.48)


def said(series) -> dict[str, str]:
    return {chapter.key: spoken(chapter.narration) for chapter in series.chapters}


def test_the_introduction_says_who_the_series_is_for_and_that_it_is_not_for_beginners(playlist):
    intro = introduction(playlist)
    audience = next(chapter for chapter in intro.chapters if chapter.key == "audience")
    assert "not a first course" in audience.narration.lower()
    assert "introductory course" in audience.narration
    assert [chapter.key for chapter in intro.chapters][:3] == ["welcome", "title", "audience"]


def test_the_introduction_counts_the_real_districts(playlist):
    data = next(chapter for chapter in introduction(playlist).chapters if chapter.key == "data")
    assert "20,640" in data.narration
    assert "4,128" in " ".join(str(value) for value in data.paint.keywords["facts"])


def test_the_introduction_hands_over_to_the_first_lesson(playlist):
    assert introduction(playlist).chapters[-1].narration == "Let's start with Linear Regression."


def test_the_conclusion_reports_the_one_input_tie_from_the_real_scores(playlist):
    board = next(chapter for chapter in conclusion(playlist).chapters if chapter.key == "one_input")
    assert "$83k and $85k" in board.narration and "$114k" in board.narration
    bars = board.paint.keywords["bars"]
    assert bars[0].label == "Neural Network (MLP)" and bars[-1].role == "baseline"
    assert any("trained on 30 rows" in bar.label for bar in bars)


def test_the_conclusion_names_the_best_models_given_every_input(playlist):
    shift = next(chapter for chapter in conclusion(playlist).chapters if chapter.key == "every_input")
    assert "Random Forest reaches $55k" in shift.narration and "Gradient Boosting $57k" in shift.narration
    assert "all nine" in shift.paint.keywords["verdict"]


def test_the_lessons_learned_quote_polynomial_regressions_gap(playlist):
    learned = next(chapter for chapter in conclusion(playlist).chapters if chapter.key == "learned")
    details = " ".join(line.detail for line in learned.paint.keywords["lines"])
    assert "$61k" in details and "$83k" in details


@pytest.mark.parametrize("make", [introduction, conclusion])
def test_every_word_can_be_said_in_full(playlist, make):
    assert all(unexpanded_acronyms(text) == () for text in said(make(playlist)).values())
    assert "multi-layer perceptron" in " ".join(said(make(playlist)).values()) or make is introduction


@pytest.mark.parametrize("make", [introduction, conclusion])
def test_every_card_paints_without_error(playlist, make):
    stage = Stage(dpi=RESOLUTIONS["preview"])
    for chapter in make(playlist).chapters:
        chapter.paint(stage, 1.0)
        assert stage.render()


def test_the_introduction_opens_with_a_welcome_and_the_conclusion_closes_with_a_thank_you(playlist):
    assert introduction(playlist).chapters[0].narration.startswith("Welcome to")
    last = conclusion(playlist).chapters[-1]
    assert last.key == "thank_you" and "California House Prices" in last.narration


def test_the_introduction_says_the_training_and_inference_are_real(playlist):
    keys = [item.key for item in introduction(playlist).chapters]
    assert keys[:4] == ["welcome", "title", "audience", "real"]
