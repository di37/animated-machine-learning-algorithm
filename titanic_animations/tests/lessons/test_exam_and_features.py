"""The classifier final exam scores the real model on every test passenger; the comparison refits for real."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from animkit.lessons.director import paint_exam
from animkit.lessons.stage import Stage, lesson_style
from tests.lessons.conftest import TINY_DPI
from titanic_anim.lessons.config import AGE, FEMALE, LOG_FARE
from titanic_anim.lessons.exam import BLOCK, GRID_COLUMNS, draw_class_grid, prepare_exam, running_scores
from titanic_anim.lessons.lesson import accuracy
from titanic_anim.lessons.more_features import RICHER, compare, richer_model
from titanic_anim.lessons.scene import CLASSIFICATION
from titanic_anim.lessons.stories import build_lesson, lesson_keys


@pytest.fixture(scope="module")
def lessons(passenger_data):
    return {key: build_lesson(key, passenger_data) for key in lesson_keys()}


def test_every_lesson_has_an_all_features_counterpart():
    assert set(RICHER) == set(lesson_keys())


def test_full_features_and_classes_cover_the_same_passengers(passenger_data):
    full = passenger_data.full
    assert full.X_train.shape == (passenger_data.n_train, len(full.names)) and len(full.names) == 8
    np.testing.assert_array_equal(full.y_train, passenger_data.y_train)
    assert set(np.unique(passenger_data.test_class)) <= {1, 2, 3}
    assert len(passenger_data.test_class) == len(passenger_data.y_test)
    names = full.names
    for X, x in ((full.X_train, passenger_data.x_train), (full.X_test, passenger_data.x_test)):
        np.testing.assert_array_equal(X[:, names.index("female")] > 0, x[:, FEMALE] == 1.0)
        assert np.corrcoef(X[:, names.index("age")], x[:, AGE])[0, 1] > 0.999999
        assert np.corrcoef(X[:, names.index("log_fare")], x[:, LOG_FARE])[0, 1] > 0.999999


def test_class_grid_puts_every_passenger_in_its_own_class_and_sex(passenger_data):
    from collections import Counter

    from titanic_anim.lessons.exam import class_blocks

    expected = Counter(zip((passenger_data.test_class - 1).tolist(),
                           np.where(passenger_data.x_test[:, FEMALE] == 1.0, 0, 1).tolist()))
    assert Counter(class_blocks(passenger_data)) == expected


def test_exam_scores_are_the_lessons_model_on_every_test_passenger(lessons):
    lesson = lessons["perceptron"]
    exam = prepare_exam(lesson)
    np.testing.assert_allclose(exam.scores, lesson.score(lesson.data.x_test))
    assert sorted(exam.order.tolist()) == list(range(len(lesson.data.y_test)))
    assert exam.speed.single_us > 0 and exam.speed.batch_us > 0


def test_running_scores_match_scikit_learn_on_the_scored_subset():
    scores = np.array([0.9, 0.2, 0.7, 0.4, 0.1, 0.8])
    labels = np.array([1, 0, 0, 1, 0, 1])
    result = running_scores(scores, labels, "probability")
    predicted = (scores > 0.5).astype(int)
    assert result.accuracy == pytest.approx(accuracy_score(labels, predicted))
    assert result.recall == pytest.approx(recall_score(labels, predicted))
    assert result.auc == pytest.approx(roc_auc_score(labels, scores))
    assert (result.caught, result.missed, result.false_alarms, result.right_no) == (2, 1, 1, 2)
    assert np.isnan(running_scores(scores[:1], labels[:1], "probability").auc)


def test_grid_squares_grow_with_the_panel_and_never_touch(lessons):
    lesson = lessons["decision_tree"]
    exam = prepare_exam(lesson)
    areas = []
    for width_in, height_in in ((4.5, 2.75), (9.0, 5.5)):
        ax = Figure(figsize=(width_in, height_in)).add_axes((0.0, 0.0, 1.0, 1.0))
        draw_class_grid(ax, lesson, exam, len(exam.order))
        squares, crosses = ax.collections[:2]
        side = float(np.sqrt(squares.get_sizes()[0]))
        assert side < BLOCK["width"] / GRID_COLUMNS * width_in * 72.0  # a gap stays between neighbours
        assert crosses.get_sizes()[0] < squares.get_sizes()[0]  # a wrong call's ✕ sits inside its square
        areas.append(squares.get_sizes()[0])
    assert areas[1] == pytest.approx(4 * areas[0])  # twice the panel, twice the side


def test_exam_frame_shows_the_scoreboard_and_the_class_grid(lessons):
    lesson = lessons["decision_tree"]
    exam = prepare_exam(lesson)
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_exam(stage, lesson, exam, 1.0, CLASSIFICATION)
    texts = "\n".join(t.get_text() for t in stage.fig.findobj(match=Text))
    n = len(lesson.data.y_test)
    assert f"{n:,} of {n:,}" in texts and "accuracy" in texts and "AUC" in texts
    assert "class" in stage.panel.get_title(loc="left").lower()


@pytest.mark.parametrize("key", lesson_keys())
def test_comparison_rows_are_real_test_accuracies(lessons, key):
    lesson = lessons[key]
    data = lesson.data
    comparison = compare(lesson)
    by_role = {row.role: row for row in comparison.rows}
    assert by_role["lesson"].value == pytest.approx(accuracy(lesson.score(data.x_test), data.y_test, lesson.kind))
    model, _ = richer_model(lesson)
    model.fit(data.full.X_train, data.full.y_train)
    assert by_role["richer"].value == pytest.approx(np.mean(model.predict(data.full.X_test) == data.y_test))
    assert by_role["baseline"].value == pytest.approx(np.mean(data.y_test == 0))
    assert comparison.higher_is_better is True


@pytest.mark.parametrize("key", lesson_keys())
def test_comparison_headline_needs_a_mcnemar_test(lessons, key):
    comparison = compare(lessons[key])
    by_role = {row.role: row for row in comparison.rows}
    assert "McNemar" in comparison.note
    if "lift" in comparison.headline:
        assert comparison.improvement() > 0 and by_role["lesson"].text != by_role["richer"].text


def test_knn_k_is_re_chosen_on_validation(lessons):
    assert "re-chosen on validation" in richer_model(lessons["knn"])[1]
