"""The 3-D chapter: the lesson's algorithm refitted on income and house age, drawn as a price surface."""

import numpy as np
import pytest

from animkit.lessons.director import paint_surface
from animkit.lessons.fit import colliding_texts, overflowing_texts
from animkit.lessons.stage import Stage, lesson_style
from animkit.lessons.surface import Surface
from house_anim.lessons import surface_view
from house_anim.lessons.config import PRICE_TO_K, X_RANGE, Y_RANGE_K
from house_anim.lessons.data import select_columns
from house_anim.lessons.lesson import rmse_k
from house_anim.lessons.regression_scene import REGRESSION
from house_anim.lessons.stories import build_lesson, lesson_keys
from tests.lessons.conftest import TINY_DPI


@pytest.fixture(scope="module")
def lessons(income_data):
    return {key: build_lesson(key, income_data) for key in lesson_keys()}


@pytest.fixture(scope="module")
def views(lessons):
    return {key: surface_view.prepare_surface(lesson) for key, lesson in lessons.items()}


def refitted(lesson):
    model, _ = surface_view.surface_model(lesson)
    two = surface_view.two_inputs(lesson)
    return model.fit(two.X_train, two.y_train), two


def test_every_lesson_has_a_shape_sentence():
    assert set(surface_view.SHAPES) == set(lesson_keys())


def test_the_two_inputs_are_income_and_house_age_on_the_same_rows(income_data):
    full = income_data.full
    two = select_columns(full, surface_view.INPUTS)
    assert two.names == ("MedInc", "HouseAge") and two.X_train.shape == (len(full.y_train), 2)
    np.testing.assert_array_equal(two.y_train, full.y_train)
    np.testing.assert_allclose(two.X_test[:, 0] * two.scale[0] + two.mean[0], income_data.x_test)
    assert not two.X_train.flags.writeable


def test_an_unknown_column_is_refused(income_data):
    with pytest.raises(ValueError, match="Nope"):
        select_columns(income_data.full, ("MedInc", "Nope"))


@pytest.mark.parametrize("key", lesson_keys())
def test_the_sheet_is_the_refitted_model_on_the_grid(lessons, views, key):
    shown = views[key]
    model, two = refitted(lessons[key])
    rows = surface_view.grid_rows(shown.x, shown.y, two)
    expected = np.clip(model.predict(rows).reshape(len(shown.y), len(shown.x)) * PRICE_TO_K, *Y_RANGE_K)
    np.testing.assert_allclose(shown.sheets[0].heights, expected)
    assert len(shown.sheets) == 1 and shown.plane is None


def test_grid_rows_walk_the_first_input_fastest(income_data):
    two = select_columns(income_data.full, surface_view.INPUTS)
    rows = surface_view.grid_rows(np.array([2.0, 4.0, 6.0]), np.array([10.0, 20.0]), two)
    raw = rows * two.scale + two.mean
    np.testing.assert_allclose(raw[:, 0], [2.0, 4.0, 6.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(raw[:, 1], [10.0, 10.0, 10.0, 20.0, 20.0, 20.0])


def test_the_unseen_district_sits_at_the_two_input_prediction(lessons, views):
    lesson, shown = lessons["linear_regression"], views["linear_regression"]
    model, two = refitted(lesson)
    row = lesson.data.query.row
    assert shown.query.x == pytest.approx(lesson.data.query.income)
    assert shown.query.y == pytest.approx(two.X_test[row, 1] * two.scale[1] + two.mean[1])
    assert shown.query.z == pytest.approx(model.predict(two.X_test[row:row + 1])[0] * PRICE_TO_K)


def test_dots_are_training_districts_and_sticks_reach_the_surface(lessons, views):
    lesson, shown = lessons["random_forest"], views["random_forest"]
    model, two = refitted(lesson)
    dots = shown.dots
    rows = surface_view.shown_rows(two)
    income = two.X_train[:, 0] * two.scale[0] + two.mean[0]
    inside = int(np.sum((income >= X_RANGE[0]) & (income <= X_RANGE[1])))
    assert len(dots.x) == len(rows) == min(surface_view.DOTS_SHOWN, inside) and len(set(rows.tolist())) == len(rows)
    assert np.all((dots.x >= X_RANGE[0]) & (dots.x <= X_RANGE[1]))
    np.testing.assert_allclose(dots.z, two.y_train[rows] * PRICE_TO_K)
    sticks = rows[:surface_view.STICKS]
    np.testing.assert_allclose(dots.fitted, np.clip(model.predict(two.X_train[sticks]) * PRICE_TO_K, *Y_RANGE_K))


def test_linear_regression_is_a_flat_plane(views):
    heights = views["linear_regression"].sheets[0].heights
    inside = (heights > Y_RANGE_K[0]) & (heights < Y_RANGE_K[1])
    bends = np.abs(np.diff(heights, n=2, axis=1))[inside[:, 2:] & inside[:, 1:-1] & inside[:, :-2]]
    assert bends.size and bends.max() < 1e-6


def test_a_tree_has_one_flat_terrace_per_leaf(lessons, views):
    heights = views["decision_tree"].sheets[0].heights
    assert len(np.unique(np.round(heights, 9))) <= len(lessons["decision_tree"].final.detail.leaves)


@pytest.mark.parametrize("key", ("knn", "ridge"))
def test_house_age_is_judged_against_the_same_recipe_on_income_alone(lessons, views, key):
    """Ridge's lesson trains on 30 districts; the control refit uses the same rows and recipe as the
    two-input model, so the gain shown is house age's, not more data's."""
    lesson, shown = lessons[key], views[key]
    model, two = refitted(lesson)
    one = select_columns(lesson.data.full, (surface_view.INPUTS[0],))
    control = surface_view.model_for(lesson, one)[0].fit(one.X_train, one.y_train)
    y = lesson.data.y_test
    scores = "\n".join(shown.scores)
    for value in (rmse_k(control.predict(one.X_test), y), rmse_k(model.predict(two.X_test), y),
                  rmse_k(lesson.predict(lesson.data.x_test), y)):
        assert f"${value:,.0f}k" in scores
    assert f"{len(y):,} test districts" in scores and "bootstrap" in scores
    assert f"{len(two.y_train):,} training districts" in scores


def test_an_unconverged_penalised_fit_is_refused(lessons):
    from sklearn.linear_model import Lasso

    from house_anim.lessons.lesson import LessonError

    factory = surface_view._penalised_polynomial("Lasso", lambda a: Lasso(alpha=a, max_iter=1, tol=1e-12), (1e-4,))
    with pytest.raises(LessonError, match="converge"):
        factory(lessons["lasso"], surface_view.two_inputs(lessons["lasso"]))


def test_an_unconverged_final_fit_is_refused(lessons):
    from sklearn.linear_model import Lasso

    from house_anim.lessons.lesson import LessonError

    two = surface_view.two_inputs(lessons["lasso"])
    with pytest.raises(LessonError, match="converge"):
        surface_view.fit_converged(Lasso(alpha=1e-4, max_iter=1, tol=1e-12), two)


def test_clipping_anywhere_on_screen_is_disclosed():
    inside, outside = np.array([100.0, 300.0]), np.array([620.0])
    assert surface_view.clip_disclosure(inside, inside, inside) == ""
    for parts in ((outside, inside, inside), (inside, outside, inside), (inside, inside, outside)):
        assert "edge of the box" in surface_view.clip_disclosure(*parts)


def test_ridge_and_lasso_stay_penalised_polynomials_chosen_on_validation(lessons):
    for key in ("ridge", "lasso"):
        _, settings = surface_view.surface_model(lessons[key])
        assert "degree-12 polynomial of both inputs" in settings and "chosen on validation" in settings


def test_settings_tuned_for_one_input_are_re_chosen_for_two(lessons):
    assert "re-chosen on validation for 2 features" in surface_view.surface_model(lessons["knn"])[1]
    assert surface_view.surface_model(lessons["mlp"])[1].startswith("2 → ")


def test_clipping_is_said_only_when_it_happens():
    assert surface_view.clip_disclosure(np.array([10.0, 500.0])) == ""
    assert "edge of the box" in surface_view.clip_disclosure(np.array([-20.0, 300.0]))
    assert "edge of the box" in surface_view.clip_disclosure(np.array([600.0]))


@pytest.mark.parametrize("key", ("linear_regression", "lasso", "mlp"))
def test_the_chapter_paints_and_its_notes_fit(lessons, views, key):
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_surface(stage, lessons[key], views[key], 1.0, REGRESSION)
        stage.render()
    for ax in (stage.panel, stage.metric):
        assert overflowing_texts(ax) == () and colliding_texts(ax) == ()
    assert "house age" in views[key].caption


def test_the_scene_prepares_the_surface(lessons):
    assert isinstance(REGRESSION.prepare_surface(lessons["linear_regression"]), Surface)
