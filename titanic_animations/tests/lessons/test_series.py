"""The Titanic playlist's introduction and conclusion: what they say, built from the lessons' numbers."""

import pytest

from animkit.lessons import theme
from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.lesson import Comparison, ComparisonRow
from animkit.lessons.speech_text import spoken, unexpanded_acronyms
from animkit.lessons.stage import Stage
from animkit.series.cards import Algorithm
from animkit.series.charts import Cloud
from animkit.series.results import result_of
from titanic_anim.lessons.scene import CLASSIFICATION
from titanic_anim.lessons.series import Playlist, conclusion, introduction

KEYS = ("logistic_regression", "perceptron", "linear_svm", "naive_bayes", "decision_tree", "random_forest",
        "gradient_boosting", "knn", "mlp")
TITLES = ("Logistic Regression", "Perceptron", "Linear SVM", "Naive Bayes", "Decision Tree", "Random Forest",
          "Gradient Boosting", "k-Nearest Neighbours", "Neural Network (MLP)")
TRAIN = (0.7671, 0.7592, 0.7352, 0.7656, 0.7895, 0.8660, 0.8166, 0.7656, 0.7847)
VAL = (0.7799, 0.7464, 0.7225, 0.7703, 0.7656, 0.7512, 0.7608, 0.7751, 0.7703)
TEST = (0.8095, 0.7905, 0.7619, 0.8000, 0.7952, 0.7810, 0.7952, 0.8238, 0.8190)
EVERY = (0.82, 0.80, 0.72, 0.77, 0.83, 0.81, 0.83, 0.84, 0.83)


def comparison(own: float, richer: float) -> Comparison:
    return Comparison(metric="Test accuracy (higher is better)", higher_is_better=True, headline=(
        f"About the same: {richer:.0%} with every input against {own:.0%} with age, fare and sex."), note="n",
        takeaway="t", rows=(ComparisonRow("Lesson", own, f"{own:.0%}", "3 inputs", "lesson"),
                            ComparisonRow("Every input", richer, f"{richer:.0%}", "all", "richer")))


@pytest.fixture(scope="module")
def playlist() -> Playlist:
    results = tuple(result_of(key, title, train=train, validation=val, test=test, training_rows=627,
                              comparison=comparison(test, every))
                    for key, title, train, val, test, every in zip(KEYS, TITLES, TRAIN, VAL, TEST, EVERY))
    cloud = Cloud(x=(0.1, 0.5), y=(0.2, 0.9), groups=(0, 1), colours=(theme.ERROR, theme.MODEL),
                  legend=("did not survive", "survived"), x_label="age", y_label="fare paid (log scale)",
                  markers=("X", "o"))
    return Playlist(algorithms=tuple(Algorithm(title, "learns somehow") for title in TITLES), cloud=cloud,
                    listed=1309, train_rows=627, val_rows=209, test_rows=210, shown_rows=627, results=results,
                    baseline=0.5905)


def chapter(series, key):
    return next(item for item in series.chapters if item.key == key)


def test_the_introduction_says_who_the_series_is_for_and_that_it_is_not_for_beginners(playlist):
    audience = chapter(introduction(playlist), "audience")
    assert "not a first course" in audience.narration


def test_the_introduction_counts_the_real_passengers(playlist):
    data = chapter(introduction(playlist), "data")
    assert "1,046" in data.narration
    assert any("263" in fact for fact in data.paint.keywords["facts"])


def test_the_conclusion_reports_the_range_and_how_few_passengers_separate_the_top(playlist):
    board = chapter(conclusion(playlist), "three_inputs")
    assert "76% and 82%" in board.narration and "59%" in board.narration
    assert "within 3 passengers" in board.narration  # 173, 172 and 170 of the 210 called right
    assert board.paint.keywords["bars"][0].label == "k-Nearest Neighbours"


def test_the_conclusion_says_every_input_made_no_difference_a_test_could_confirm(playlist):
    shift = chapter(conclusion(playlist), "every_input")
    assert "all nine" in shift.paint.keywords["verdict"] and "McNemar" in shift.paint.keywords["verdict"]


def test_the_lessons_learned_quote_the_widest_train_validation_gap(playlist):
    details = " ".join(line.detail for line in chapter(conclusion(playlist), "learned").paint.keywords["lines"])
    assert "Random Forest: 87% right on its training passengers, 75% on validation" in details


@pytest.mark.parametrize("make", [introduction, conclusion])
def test_every_word_can_be_said_in_full(playlist, make):
    said = [spoken(item.narration, CLASSIFICATION.spoken_phrases) for item in make(playlist).chapters]
    assert all(unexpanded_acronyms(text) == () for text in said)


@pytest.mark.parametrize("make", [introduction, conclusion])
def test_every_card_paints_without_error(playlist, make):
    stage = Stage(dpi=RESOLUTIONS["preview"])
    for item in make(playlist).chapters:
        item.paint(stage, 1.0)
        assert stage.render()


def test_the_introduction_opens_with_a_welcome_and_the_conclusion_closes_with_a_thank_you(playlist):
    assert [item.key for item in introduction(playlist).chapters][:3] == ["welcome", "title", "audience"]
    last = conclusion(playlist).chapters[-1]
    assert last.key == "thank_you" and "Titanic Survival" in last.narration


def test_the_introduction_says_the_training_and_inference_are_real(playlist):
    keys = [item.key for item in introduction(playlist).chapters]
    assert keys[:4] == ["welcome", "title", "audience", "real"]
