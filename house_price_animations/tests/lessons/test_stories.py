"""Every lesson must show genuine model states: these tests tie each frame's numbers to the model."""

import dataclasses

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from house_anim.lessons.config import POLY_SAMPLE
from house_anim.lessons.lesson import rmse_k
from house_anim.lessons.stories import build_lesson, lesson_keys
from house_anim.lessons.stories.linear_gd import run_gradient_descent
from house_anim.lessons.stories.polynomial import symlog_ticks

KEYS = lesson_keys()


@pytest.fixture(scope="module")
def lessons(income_data):
    return {key: build_lesson(key, income_data) for key in KEYS}


def test_registry_lists_the_nine_lessons():
    assert KEYS == ("linear_regression", "polynomial_regression", "ridge", "lasso", "decision_tree",
                    "random_forest", "gradient_boosting", "knn", "mlp")


def test_unknown_lesson_is_rejected(income_data):
    with pytest.raises(KeyError, match="unknown lesson"):
        build_lesson("svm", income_data)


@pytest.mark.parametrize("key", KEYS)
def test_snapshots_are_finite_read_only_and_shaped(lessons, key):
    lesson = lessons[key]
    assert len(lesson.snapshots) >= 3
    for snap in lesson.snapshots:
        assert snap.curve.shape == lesson.data.grid.shape
        assert snap.fitted.shape == lesson.data.residual.shape
        assert np.isfinite(snap.curve).all() and np.isfinite(snap.fitted).all()
        assert np.isfinite([snap.train_rmse_k, snap.val_rmse_k, snap.progress]).all()
        assert not snap.curve.flags.writeable and not snap.fitted.flags.writeable
        assert snap.label and snap.note
    assert 0 <= lesson.final_index < len(lesson.snapshots)


@pytest.mark.parametrize("key", KEYS)
def test_final_snapshot_is_the_model_that_predicts(lessons, key):
    lesson, data = lessons[key], lessons[key].data
    np.testing.assert_allclose(lesson.final.curve, lesson.predict(data.grid), rtol=1e-9, atol=1e-9)
    assert lesson.final.val_rmse_k == pytest.approx(rmse_k(lesson.predict(data.x_val), data.y_val))
    assert lesson.final.train_rmse_k == pytest.approx(rmse_k(lesson.predict(data.x_train), data.y_train))


@pytest.mark.parametrize("key", KEYS)
def test_residual_lines_use_the_snapshot_model(lessons, key):
    lesson = lessons[key]
    np.testing.assert_allclose(lesson.final.fitted, lesson.predict(lesson.data.residual_x))


@pytest.mark.parametrize("key", KEYS)
def test_worked_inference_matches_the_model(lessons, key):
    lesson = lessons[key]
    query = lesson.data.query
    explanation = lesson.explain(query.income)
    model_value = float(lesson.predict(np.array([query.income]))[0])
    assert explanation.prediction == pytest.approx(model_value, abs=1e-9)
    assert explanation.lines[-1].startswith("check") and "✓" in explanation.lines[-1]
    assert len(explanation.lines) >= 3


@pytest.mark.parametrize("key", KEYS)
def test_story_text_and_config_are_complete(lessons, key):
    lesson = lessons[key]
    for field in dataclasses.fields(lesson.story):
        value = getattr(lesson.story, field.name)
        assert value or isinstance(value, bool), f"{key}: story.{field.name} is empty"
    assert lesson.story.key == key
    assert lesson.config and all(lesson.config.values())


@pytest.mark.parametrize("key", KEYS)
def test_frame_weights_match_snapshots(lessons, key):
    weights = lessons[key].frame_weights
    assert weights is None or (len(weights) == len(lessons[key].snapshots) and min(weights) > 0)


def test_gradient_descent_converges_to_least_squares(lessons):
    lesson = lessons["linear_regression"]
    data = lesson.data
    ols = LinearRegression().fit(data.x_train.reshape(-1, 1), data.y_train)
    np.testing.assert_allclose(lesson.predict(data.grid), ols.predict(data.grid.reshape(-1, 1)), atol=2e-3)
    train = [snap.train_rmse_k for snap in lesson.snapshots]
    assert all(after <= before + 1e-9 for before, after in zip(train, train[1:]))


def test_each_gradient_descent_step_follows_the_recorded_gradient():
    z = np.array([-1.0, 0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 2.5, 4.0])
    states = run_gradient_descent(z, y, learning_rate=0.1, steps=3)
    assert len(states) == 4 and states[0].theta == (0.0, 0.0)
    for before, after in zip(states, states[1:]):
        expected = np.array(before.theta) - 0.1 * np.array(before.gradient)
        np.testing.assert_allclose(after.theta, expected)
    residual = y - (states[0].theta[0] + states[0].theta[1] * z)
    np.testing.assert_allclose(states[0].gradient, (-2 * residual.mean(), -2 * (residual * z).mean()))


@pytest.mark.parametrize("key", ["polynomial_regression", "ridge", "lasso", "knn"])
def test_sweeps_pick_the_best_validation_snapshot(lessons, key):
    validation = [snap.val_rmse_k for snap in lessons[key].snapshots]
    assert lessons[key].final_index == int(np.argmin(validation))


@pytest.mark.parametrize("key", ["polynomial_regression", "ridge", "lasso"])
def test_polynomial_lessons_train_on_a_small_labelled_sample(lessons, key):
    data = lessons[key].data
    assert data.n_train == POLY_SAMPLE and str(POLY_SAMPLE) in data.sample_note


def test_lasso_switches_terms_off_and_ridge_never_does(lessons):
    lasso_zeros = [int(np.sum(snap.detail.coef == 0.0)) for snap in lessons["lasso"].snapshots]
    ridge_zeros = [int(np.sum(snap.detail.coef == 0.0)) for snap in lessons["ridge"].snapshots]
    assert max(lasso_zeros) > 0 and max(ridge_zeros) == 0


def test_tree_grows_one_split_at_a_time_as_a_prefix(lessons):
    snaps = lessons["decision_tree"].snapshots
    assert snaps[0].detail.splits == ()
    for before, after in zip(snaps, snaps[1:]):
        assert set(before.detail.splits) < set(after.detail.splits)
        assert len(after.detail.splits) == len(before.detail.splits) + 1
        assert after.detail.newest in dict(after.detail.splits)


def test_forest_curve_is_the_average_of_its_trees(lessons):
    for snap in lessons["random_forest"].snapshots:
        assert len(snap.detail.tree_curves) == snap.detail.n_trees
        np.testing.assert_allclose(snap.curve, snap.detail.tree_curves.mean(axis=0))


def test_boosting_adds_the_shrunken_tree_to_the_previous_stage(lessons):
    for snap in lessons["gradient_boosting"].snapshots[1:]:
        detail = snap.detail
        np.testing.assert_allclose(detail.before + detail.learning_rate * detail.correction, snap.curve)


def test_knn_starts_with_its_most_flexible_model(lessons):
    snaps = lessons["knn"].snapshots
    assert snaps[0].detail.k == 1
    assert snaps[0].train_rmse_k == min(snap.train_rmse_k for snap in snaps)


def total_variation(curve: np.ndarray) -> float:
    return float(np.abs(np.diff(curve)).sum())


def test_one_tree_is_more_jagged_than_the_forest_average(lessons):
    snaps = lessons["random_forest"].snapshots
    assert total_variation(snaps[0].curve) > total_variation(snaps[-1].curve)


@pytest.mark.parametrize("count", [1, 5, 13])
def test_forest_snapshot_equals_a_cold_fit_of_that_many_trees(lessons, count):
    from sklearn.ensemble import RandomForestRegressor

    from house_anim.lessons.stories import forest

    lesson = lessons["random_forest"]
    data = lesson.data
    snap = next(s for s in lesson.snapshots if s.detail.n_trees == count)
    cold = RandomForestRegressor(n_estimators=count, min_samples_leaf=forest.min_leaf(data.n_train),
                                 n_jobs=1, random_state=42).fit(data.x_train.reshape(-1, 1), data.y_train)
    np.testing.assert_allclose(snap.curve, cold.predict(data.grid.reshape(-1, 1)), rtol=1e-9)


def test_lasso_refuses_to_show_a_solver_that_did_not_converge(income_data, monkeypatch):
    from house_anim.lessons.stories import polynomial

    monkeypatch.setattr(polynomial, "LASSO_SETTINGS", dict(max_iter=1, tol=1e-12))
    with pytest.raises(ValueError, match="did not converge"):
        polynomial.build_lasso(income_data)


@pytest.mark.parametrize("count, word, expected", [(1, "leaf", "1 leaf"), (14, "leaf", "14 leaves"),
                                                   (1, "tree", "1 tree"), (50, "tree", "50 trees")])
def test_plural_labels(count, word, expected):
    from house_anim.lessons.lesson import plural

    assert plural(count, word) == expected


def test_first_tree_and_forest_labels_are_singular(lessons):
    assert lessons["decision_tree"].snapshots[0].label == "1 leaf"
    assert lessons["random_forest"].snapshots[0].label == "1 tree"


def test_a_worked_prediction_that_disagrees_with_the_model_is_refused():
    from house_anim.lessons.lesson import LessonError, verified

    with pytest.raises(LessonError, match="disagrees"):
        verified(["ŷ = 1"], worked=1.0, model_value=2.0)


@pytest.mark.parametrize("key", KEYS)
def test_every_lesson_uses_the_full_training_average_as_its_baseline(lessons, income_data, key):
    assert lessons[key].data.baseline_mean == income_data.baseline_mean


def test_boosting_frames_name_every_tree_they_add(lessons):
    snaps = lessons["gradient_boosting"].snapshots
    for before, after in zip(snaps, snaps[1:]):
        assert after.detail.first == before.detail.stage + 1
        if after.detail.stage - before.detail.stage > 1:
            assert f"Trees {after.detail.first}–{after.detail.stage}" in after.note


@pytest.mark.parametrize(("limit", "ticks"), [(13.0, [-10.0, 0.0, 10.0]),
                                              (1_300.0, [-1_000.0, -100.0, 0.0, 100.0, 1_000.0]),
                                              (13_000.0, [-10_000.0, -100.0, 0.0, 100.0, 10_000.0])])
def test_weight_panel_keeps_few_symlog_ticks_for_the_small_side_panel(limit, ticks):
    assert symlog_ticks(limit) == ticks
