"""Every classifier lesson must show genuine model states, tied here to the models themselves."""

import dataclasses

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from titanic_anim.lessons.lesson import MARGIN, PROBABILITY, accuracy, threshold
from titanic_anim.lessons.stories import build_lesson, lesson_keys

KEYS = lesson_keys()
MARGIN_MODELS = ("perceptron", "linear_svm")


@pytest.fixture(scope="module")
def lessons(passenger_data):
    return {key: build_lesson(key, passenger_data) for key in KEYS}


def test_registry_lists_the_nine_classifier_lessons():
    assert KEYS == ("logistic_regression", "perceptron", "linear_svm", "naive_bayes", "decision_tree",
                    "random_forest", "gradient_boosting", "knn", "mlp")


def test_unknown_lesson_is_rejected(passenger_data):
    with pytest.raises(KeyError, match="unknown lesson"):
        build_lesson("svm_rbf", passenger_data)


@pytest.mark.parametrize("key", KEYS)
def test_snapshots_are_finite_read_only_and_shaped(lessons, key):
    lesson = lessons[key]
    assert len(lesson.snapshots) >= 3
    for snap in lesson.snapshots:
        assert snap.surface.shape == lesson.data.surface_shape
        assert np.isfinite(snap.surface).all() and not snap.surface.flags.writeable
        assert 0.0 <= snap.train_accuracy <= 1.0 and 0.0 <= snap.val_accuracy <= 1.0
        assert snap.wrong.shape == lesson.data.display.shape and snap.wrong.dtype == bool
        assert snap.label and snap.note
    assert 0 <= lesson.final_index < len(lesson.snapshots)


@pytest.mark.parametrize("key", KEYS)
def test_probabilities_stay_in_range(lessons, key):
    lesson = lessons[key]
    if lesson.kind == PROBABILITY:
        assert all(0.0 <= s.surface.min() and s.surface.max() <= 1.0 for s in lesson.snapshots)


@pytest.mark.parametrize("key", MARGIN_MODELS)
def test_score_only_models_never_claim_probabilities(lessons, key):
    lesson = lessons[key]
    assert lesson.kind == MARGIN
    assert "decision_function" in lesson.explain(lesson.data.query.features).lines[-1]
    assert all(np.isnan(s.val_log_loss) for s in lesson.snapshots)


@pytest.mark.parametrize("key", KEYS)
def test_final_snapshot_is_the_model_that_predicts(lessons, key):
    lesson, data = lessons[key], lessons[key].data
    np.testing.assert_allclose(lesson.final.surface.ravel(), lesson.score(data.grid), rtol=1e-9, atol=1e-12)
    expected = accuracy(lesson.score(data.x_val), data.y_val, lesson.kind)
    assert lesson.final.val_accuracy == pytest.approx(expected)
    wrong = (lesson.score(data.x_train[data.display]) > threshold(lesson.kind)) != data.y_train[data.display]
    np.testing.assert_array_equal(lesson.final.wrong, wrong)


@pytest.mark.parametrize("key", KEYS)
def test_worked_inference_matches_the_model(lessons, key):
    lesson = lessons[key]
    features = lesson.data.query.features
    explanation = lesson.explain(features)
    assert explanation.prediction == pytest.approx(float(lesson.score(features[None, :])[0]), abs=1e-9)
    assert explanation.lines[-1].startswith("check") and "✓" in explanation.lines[-1]
    assert len(explanation.lines) >= 3


@pytest.mark.parametrize("key", KEYS)
def test_story_text_and_config_are_complete(lessons, key):
    lesson = lessons[key]
    for field in dataclasses.fields(lesson.story):
        value = getattr(lesson.story, field.name)
        assert value or isinstance(value, bool), f"{key}: story.{field.name} is empty"
    assert lesson.story.key == key and lesson.config and all(lesson.config.values())


@pytest.mark.parametrize("key", KEYS)
def test_frame_weights_match_snapshots(lessons, key):
    weights = lessons[key].frame_weights
    assert weights is None or (len(weights) == len(lessons[key].snapshots) and min(weights) > 0)


def test_logistic_gradient_descent_converges_to_the_exact_fit(lessons):
    lesson = lessons["logistic_regression"]
    data = lesson.data
    detail = lesson.final.detail
    exact = LogisticRegression(penalty=None, tol=1e-10, max_iter=10_000)
    exact.fit(detail.standardise(data.x_train), data.y_train)
    np.testing.assert_allclose(lesson.score(data.grid), exact.predict_proba(detail.standardise(data.grid))[:, 1],
                               atol=2e-3)
    losses = [s.train_log_loss for s in lesson.snapshots]
    assert all(after <= before + 1e-9 for before, after in zip(losses, losses[1:]))


@pytest.mark.parametrize("key", ["perceptron", "knn"])
def test_selected_snapshot_has_the_best_validation_accuracy(lessons, key):
    accuracies = [s.val_accuracy for s in lessons[key].snapshots]
    assert lessons[key].final.val_accuracy == max(accuracies)


def test_tree_grows_one_split_at_a_time_as_a_prefix(lessons):
    snaps = lessons["decision_tree"].snapshots
    assert snaps[0].detail.splits == ()
    for before, after in zip(snaps, snaps[1:]):
        assert set(before.detail.splits) < set(after.detail.splits)
        assert len(after.detail.splits) == len(before.detail.splits) + 1


def test_forest_surface_is_the_average_of_its_trees(lessons):
    for snap in lessons["random_forest"].snapshots:
        assert len(snap.detail.tree_surfaces) == snap.detail.n_trees
        np.testing.assert_allclose(snap.surface, snap.detail.tree_surfaces.mean(axis=0))


def test_boosting_adds_shrunken_trees_in_log_odds(lessons):
    snaps = lessons["gradient_boosting"].snapshots
    for before, after in zip(snaps, snaps[1:]):
        detail = after.detail
        assert detail.first == before.detail.stage + 1
        np.testing.assert_allclose(detail.log_odds_before + detail.learning_rate * detail.correction,
                                   detail.log_odds, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(1.0 / (1.0 + np.exp(-detail.log_odds)), after.surface, rtol=1e-9)


def test_naive_bayes_counts_more_passengers_each_frame(lessons):
    counts = [s.detail.seen for s in lessons["naive_bayes"].snapshots]
    assert counts == sorted(counts) and counts[-1] == lessons["naive_bayes"].data.n_train


def test_knn_starts_with_one_neighbour(lessons):
    assert lessons["knn"].snapshots[0].detail.k == 1


@pytest.mark.parametrize("key", MARGIN_MODELS)
def test_every_linear_frame_is_scored_by_a_frozen_copy_of_the_real_model(lessons, key):
    lesson = lessons[key]
    data = lesson.data
    for snap in lesson.snapshots:
        live = snap.detail.fitted.decision_function(_scaled(data, data.grid))
        np.testing.assert_allclose(snap.surface.ravel(), live)
        np.testing.assert_allclose(snap.detail.theta, np.r_[snap.detail.fitted.intercept_, snap.detail.fitted.coef_[0]])
    first, last = lesson.snapshots[0].detail.fitted, lesson.snapshots[-1].detail.fitted
    assert first is not last and not np.allclose(first.coef_, last.coef_)


def _scaled(data, x):
    from titanic_anim.lessons.stories.common import Standardiser

    return Standardiser.fit(data.x_train)(x)


def test_a_lesson_whose_worked_check_does_not_add_up_is_refused_at_build(lessons):
    from animkit.lessons.lesson import LessonError
    from titanic_anim.lessons.lesson import ClassLesson, verified_score

    good = lessons["knn"]
    with pytest.raises(LessonError, match="disagrees"):
        ClassLesson(story=good.story, data=good.data, snapshots=good.snapshots, final_index=good.final_index,
                    score=good.score, kind=good.kind, panel=good.panel, config=good.config,
                    explain=lambda features: verified_score(["wrong"], 0.0, 1.0, PROBABILITY))


def test_tree_on_a_tiny_dataset_never_repeats_or_mislabels_a_frame():
    from animkit.lessons.lesson import plural
    from tests.conftest import make_titanic_frame
    from titanic_anim.lessons.data import passenger_view

    lesson = build_lesson("decision_tree", passenger_view(make_titanic_frame(), max_rows=60))
    leaves = [len(s.detail.leaves) for s in lesson.snapshots]
    assert leaves == sorted(set(leaves))
    assert lesson.story.final.startswith(plural(leaves[-1], "leaf"))


def test_mlp_inference_caption_matches_the_real_forward_pass(lessons):
    from titanic_anim.lessons.stories.mlp import forward

    lesson = lessons["mlp"]
    first_layer = forward(lesson.final.detail, _scaled(lesson.data, lesson.data.query.features))[1]
    silent = not np.any(first_layer > 0)
    assert ("no first-layer unit fires" in lesson.story.inference) == silent


def test_logistic_names_its_largest_weights_from_the_weights(lessons):
    theta = lessons["logistic_regression"].final.detail.theta
    names = ("age", "fare", "woman")
    largest = names[int(np.argmax(np.abs(theta[1:])))]
    assert f"Largest weight: {largest}" in lessons["logistic_regression"].story.final
    assert any("scikit-learn's exact fit" in line for line in
               lessons["logistic_regression"].explain(lessons["logistic_regression"].data.query.features).lines)
