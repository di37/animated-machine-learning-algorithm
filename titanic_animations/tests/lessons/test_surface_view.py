"""The 3-D chapter: the lesson's own final model, its score as height over age and fare, one sheet per sex."""

import numpy as np
import pytest

from animkit.lessons.director import paint_surface
from animkit.lessons.fit import colliding_texts, overflowing_texts
from animkit.lessons.stage import Stage, lesson_style
from animkit.lessons.surface import Surface
from tests.lessons.conftest import TINY_DPI
from titanic_anim.lessons import surface_view
from titanic_anim.lessons.config import AGE, LOG_FARE
from titanic_anim.lessons.lesson import PROBABILITY, accuracy, threshold
from titanic_anim.lessons.scene import CLASSIFICATION
from titanic_anim.lessons.stories import build_lesson, lesson_keys


@pytest.fixture(scope="module")
def lessons(passenger_data):
    return {key: build_lesson(key, passenger_data) for key in lesson_keys()}


@pytest.fixture(scope="module")
def views(lessons):
    return {key: surface_view.prepare_surface(lesson) for key, lesson in lessons.items()}


def test_every_lesson_has_a_shape_sentence():
    assert set(surface_view.SHAPES) == set(lesson_keys())


def test_grid_picks_keep_both_ends():
    picks = surface_view.grid_picks(120, 40)
    assert picks[0] == 0 and picks[-1] == 119 and len(picks) == 40 and np.all(np.diff(picks) > 0)
    assert surface_view.grid_picks(10, 40).tolist() == list(range(10))


@pytest.mark.parametrize("key", lesson_keys())
def test_the_sheets_are_the_lessons_own_final_maps(lessons, views, key):
    lesson, shown = lessons[key], views[key]
    ages = surface_view.grid_picks(len(lesson.data.ages), surface_view.AGE_POINTS)
    fares = surface_view.grid_picks(len(lesson.data.log_fares), surface_view.FARE_POINTS)
    np.testing.assert_allclose(shown.x, lesson.data.ages[ages])
    np.testing.assert_allclose(shown.y, lesson.data.log_fares[fares])
    assert [sheet.label for sheet in shown.sheets] == ["Women", "Men"]
    for sex, sheet in enumerate(shown.sheets):
        np.testing.assert_allclose(sheet.heights, lesson.final.surface[sex][np.ix_(fares, ages)])
    assert shown.plane == threshold(lesson.kind)


def test_the_new_passenger_sits_at_the_lessons_own_score(lessons, views):
    for key in ("logistic_regression", "linear_svm"):
        lesson, query = lessons[key], views[key].query
        features = lesson.data.query.features
        assert (query.x, query.y) == (features[AGE], features[LOG_FARE])
        assert query.z == pytest.approx(lesson.score_one(features))


def test_probabilities_are_labelled_as_probabilities_and_scores_are_not(views):
    probability, margin = views["logistic_regression"], views["perceptron"]
    assert probability.z_axis.limits == (0.0, 1.0) and "P(survive)" in probability.z_axis.label
    assert "not a probability" in margin.z_axis.label and "P(" not in margin.z_axis.label
    low, high = margin.z_axis.limits
    assert low == -high and high == int(high)
    assert "50%" in probability.plane_label and "score 0" in margin.plane_label


def test_passengers_sit_on_the_top_face_if_they_survived(lessons, views):
    lesson, shown = lessons["random_forest"], views["random_forest"]
    low, high = shown.z_axis.limits
    survived = lesson.data.y_train[lesson.data.display] == 1
    np.testing.assert_array_equal(shown.dots.z, np.where(survived, high, low))
    assert shown.dots.fitted is None and len(shown.dots.x) == lesson.data.n_train


def test_the_women_share_is_measured_on_the_full_grid_and_says_so(lessons, views):
    lesson = lessons["logistic_regression"]
    share = float(np.mean(lesson.final.surface[0] > lesson.final.surface[1]))
    reading = "\n".join(views["logistic_regression"].reading)
    ages, fares = len(lesson.data.ages), len(lesson.data.log_fares)
    assert f"{surface_view.share_text(share)} the maps' full {ages} × {fares} grid" in reading
    assert f"{surface_view.AGE_POINTS} × {surface_view.FARE_POINTS} of its points" in reading


@pytest.mark.parametrize("share, text", [(1.0, "every cell of"), (0.0, "no cell of"), (0.996, "over 99% of"),
                                         (0.003, "under 1% of"), (0.634, "63% of")])
def test_a_share_is_never_rounded_to_all_or_nothing(share, text):
    assert surface_view.share_text(share) == text


def test_captions_claim_only_what_the_camera_shows(views):
    for shown in views.values():
        assert "all sides" not in shown.caption and "swinging camera" in shown.caption


def test_scores_are_the_same_numbers_as_the_other_chapters(lessons, views):
    lesson = lessons["naive_bayes"]
    scores = "\n".join(views["naive_bayes"].scores)
    test = accuracy(lesson.score(lesson.data.x_test), lesson.data.y_test, lesson.kind)
    assert f"{lesson.final.train_accuracy:.0%}" in scores and f"{lesson.final.val_accuracy:.0%}" in scores
    assert f"{test:.0%}" in scores and f"{len(lesson.data.y_test):,} test passengers" in scores


def test_the_reading_says_nothing_was_refitted(views):
    assert "nothing is refitted" in views["mlp"].reading[0]


@pytest.mark.parametrize("key", ("logistic_regression", "perceptron", "decision_tree"))
def test_the_chapter_paints_and_its_notes_fit(lessons, views, key):
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_surface(stage, lessons[key], views[key], 1.0, CLASSIFICATION)
        stage.render()
    for ax in (stage.panel, stage.metric):
        assert overflowing_texts(ax) == () and colliding_texts(ax) == ()
    kind = "P(survive)" if lessons[key].kind == PROBABILITY else "score"
    assert kind in views[key].caption


def test_the_scene_prepares_the_surface(lessons):
    assert isinstance(CLASSIFICATION.prepare_surface(lessons["knn"]), Surface)
