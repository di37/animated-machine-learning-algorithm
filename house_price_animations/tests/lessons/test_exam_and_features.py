"""The final exam scores the real model on every test district; the comparison refits for real."""

import numpy as np
import pytest
from matplotlib.text import Text

from animkit.lessons.stage import Stage, lesson_style
from house_anim.lessons.exam import prepare_exam, running_scores
from house_anim.lessons.lesson import rmse_k
from house_anim.lessons.more_features import RICHER, compare, richer_model
from house_anim.lessons.regression_scene import REGRESSION
from house_anim.lessons.stories import build_lesson, lesson_keys
from tests.lessons.conftest import TINY_DPI


@pytest.fixture(scope="module")
def lessons(income_data):
    return {key: build_lesson(key, income_data) for key in lesson_keys()}


def test_every_lesson_has_an_all_features_counterpart():
    assert set(RICHER) == set(lesson_keys())


def test_full_features_cover_the_same_rows(income_data, splits):
    full = income_data.full
    assert full.X_train.shape == splits.X_train.shape and len(full.names) == full.X_train.shape[1]
    np.testing.assert_array_equal(full.y_train, splits.y_train)
    income = full.names.index("MedInc")
    for X, x in ((full.X_train, income_data.x_train), (full.X_test, income_data.x_test)):
        np.testing.assert_allclose(X[:, income] * splits.feature_scale[income] + splits.feature_mean[income], x)
    np.testing.assert_array_equal(income_data.test_coords, splits.coords_test)


def test_exam_predictions_are_the_lessons_model_on_every_test_district(lessons):
    lesson = lessons["linear_regression"]
    exam = prepare_exam(lesson)
    np.testing.assert_allclose(exam.predicted, lesson.predict(lesson.data.x_test))
    assert sorted(exam.order.tolist()) == list(range(len(lesson.data.y_test)))
    assert exam.speed.single_us > 0 and exam.speed.batch_us > 0 and exam.speed.batch_size == len(lesson.data.y_test)


def test_running_scores_match_numpy_on_the_scored_subset():
    predicted, actual = np.array([1.0, 2.0, 4.0, 3.0]), np.array([1.5, 2.0, 3.0, 5.0])
    scores = running_scores(predicted[:3], actual[:3])
    errors = predicted[:3] - actual[:3]
    assert scores.rmse_k == pytest.approx(np.sqrt(np.mean(errors ** 2)) * 100)
    assert scores.mae_k == pytest.approx(np.mean(np.abs(errors)) * 100)
    assert scores.r2 == pytest.approx(1 - np.sum(errors ** 2) / np.sum((actual[:3] - actual[:3].mean()) ** 2))
    assert scores.within_share == pytest.approx(2 / 3)  # within ±$50k = ±0.5 in $100k units
    assert np.isnan(running_scores(predicted[:1], actual[:1]).r2)


def test_exam_frame_shows_the_scoreboard_and_the_error_map(lessons):
    lesson = lessons["decision_tree"]
    exam = prepare_exam(lesson)
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        from animkit.lessons.director import paint_exam
        paint_exam(stage, lesson, exam, 1.0, REGRESSION)
    texts = "\n".join(t.get_text() for t in stage.fig.findobj(match=Text))
    n = len(lesson.data.y_test)
    assert f"{n:,} of {n:,}" in texts and "RMSE" in texts and "R²" in texts
    assert "actual − predicted" in stage.panel.get_title(loc="left")


@pytest.mark.parametrize("key", lesson_keys())
def test_comparison_rows_are_real_test_scores(lessons, key):
    lesson = lessons[key]
    data = lesson.data
    comparison = compare(lesson)
    by_role = {row.role: row for row in comparison.rows}
    assert by_role["lesson"].value == pytest.approx(rmse_k(lesson.predict(data.x_test), data.y_test))
    model, _ = richer_model(lesson)
    model.fit(data.full.X_train, data.full.y_train)
    assert by_role["richer"].value == pytest.approx(rmse_k(model.predict(data.full.X_test), data.y_test))
    baseline = rmse_k(np.full_like(data.y_test, data.baseline_mean), data.y_test)
    assert by_role["baseline"].value == pytest.approx(baseline)
    assert comparison.higher_is_better is False


@pytest.mark.parametrize("key", lesson_keys())
def test_comparison_headline_needs_paired_evidence(lessons, key):
    comparison = compare(lessons[key])
    by_role = {row.role: row for row in comparison.rows}
    assert "bootstrap interval" in comparison.note
    if "cut the typical error" in comparison.headline:
        assert comparison.improvement() > 0 and by_role["lesson"].text != by_role["richer"].text
    if "About the same" in comparison.headline:
        assert "cut" not in comparison.headline and "worse" not in comparison.headline


def test_knn_and_penalties_are_re_chosen_on_validation(lessons):
    assert "re-chosen on validation" in richer_model(lessons["knn"])[1]
    for key in ("ridge", "lasso"):
        assert "linear terms only" in richer_model(lessons[key])[1]


def test_ridge_comparison_picks_its_penalty_on_validation_only(lessons):
    _, settings = richer_model(lessons["ridge"])
    assert "validation" in settings
