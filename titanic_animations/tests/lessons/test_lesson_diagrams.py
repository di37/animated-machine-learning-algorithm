"""Every classifier lesson shows how its algorithm computes; the diagram's numbers are the model's."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text
from sklearn.naive_bayes import GaussianNB

from animkit.lessons import diagrams
from animkit.lessons.fit import colliding_texts, overflowing_texts
from animkit.lessons.stage import Stage
from titanic_anim.lessons.lesson import MARGIN, percent, signed, signed_small
from titanic_anim.lessons.lesson_diagrams import log_terms, weight_scale
from titanic_anim.lessons.stories import build_lesson, lesson_keys

WITH_DIAGRAM = ("logistic_regression", "perceptron", "linear_svm", "naive_bayes", "random_forest",
                "gradient_boosting", "knn")


@pytest.fixture(scope="module")
def lessons(passenger_data):
    return {key: build_lesson(key, passenger_data) for key in lesson_keys()}


def texts(ax) -> str:
    return "\n".join(t.get_text() for t in ax.findobj(match=Text))


@pytest.fixture
def ax():
    return plt.figure(figsize=(6, 2)).add_subplot()


def test_every_lesson_shows_its_algorithm(lessons):
    assert {key for key, lesson in lessons.items() if lesson.diagram is not None} == set(WITH_DIAGRAM)
    assert lessons["decision_tree"].diagram is None and lessons["mlp"].diagram is None


@pytest.mark.parametrize("key", WITH_DIAGRAM)
def test_diagram_draws_every_training_state(lessons, key, ax):
    lesson = lessons[key]
    for snapshot in (lesson.snapshots[0], lesson.final):
        lesson.diagram.draw_training(ax, snapshot)
        assert texts(ax)


@pytest.mark.parametrize("key", WITH_DIAGRAM)
def test_finished_inference_diagram_ends_on_the_models_output(lessons, key, ax):
    lesson = lessons[key]
    score = lesson.score_one(lesson.data.query.features)
    lesson.diagram.draw_inference(ax, lesson.data.query, 1.0)
    shown = signed(score) if lesson.kind == MARGIN else percent(score)
    assert shown in texts(ax)


def test_logistic_diagram_carries_the_current_weights(lessons, ax):
    lesson = lessons["logistic_regression"]
    middle = lesson.snapshots[len(lesson.snapshots) // 4]
    for snapshot in (middle, lesson.final):  # mid-training, not just the end
        lesson.diagram.draw_training(ax, snapshot)
        theta = snapshot.detail.theta
        assert signed_small(theta[3]) in texts(ax) and f"+ bias {theta[0]:+.2f}" in texts(ax)
    assert not np.allclose(middle.detail.theta, lesson.final.detail.theta)


def test_forest_grid_is_handed_the_snapshots_own_trees(lessons, passenger_data, monkeypatch):
    lesson = lessons["random_forest"]
    handed = []
    monkeypatch.setattr(diagrams, "tree_grid", lambda ax, trees, *args, **kwargs: handed.append(trees))
    grid = np.ascontiguousarray(passenger_data.grid, dtype=np.float32)
    for snapshot in lesson.snapshots:
        lesson.diagram.draw_training(ax=None, snapshot=snapshot)
        trees = handed[-1]
        assert len(trees) == snapshot.detail.n_trees
        values = np.array([tree.predict(grid).reshape(len(grid), -1) for tree in trees[:diagrams.GRID_TREES]])
        maps = (values[..., 1] / values.sum(axis=2)).reshape(-1, *snapshot.detail.tree_surfaces.shape[1:])
        np.testing.assert_allclose(maps, snapshot.detail.tree_surfaces[:diagrams.GRID_TREES], rtol=1e-6)


def test_score_models_never_show_a_probability_output(lessons, ax):
    for key in ("perceptron", "linear_svm"):
        lessons[key].diagram.draw_training(ax, lessons[key].final)
        assert "σ" not in texts(ax) and "P(survive)" not in texts(ax)
        lessons[key].diagram.draw_inference(ax, lessons[key].data.query, 1.0)
        assert "σ" not in texts(ax) and "P(survive)" not in texts(ax) and "%" not in texts(ax)


@pytest.fixture(scope="module")
def stage():
    shared = Stage(dpi=100.0)  # the video's own resolution
    shared.share_panel(True)
    return shared


@pytest.mark.parametrize("key", WITH_DIAGRAM)
def test_diagram_fits_the_shared_panel_at_every_moment_shown(lessons, key, stage):
    lesson = lessons[key]
    ax = stage.diagram
    picks = {0, len(lesson.snapshots) // 2, lesson.final_index, len(lesson.snapshots) - 1}
    for index in sorted(picks):
        lesson.diagram.draw_training(ax, lesson.snapshots[index])
        assert overflowing_texts(ax) == () and colliding_texts(ax) == (), (key, index)
    for t in (0.0, 0.5, 1.0):
        lesson.diagram.draw_inference(ax, lesson.data.query, t)
        assert overflowing_texts(ax) == () and colliding_texts(ax) == (), (key, t)


def test_forest_draws_exactly_the_trees_behind_each_snapshot(lessons, passenger_data):
    lesson = lessons["random_forest"]
    for snapshot in lesson.snapshots:
        drawn = lesson.diagram.drawn(snapshot.detail.n_trees)
        maps = np.array([tree.predict_proba(passenger_data.grid)[:, 1] for tree in drawn])
        np.testing.assert_allclose(maps.reshape(snapshot.detail.tree_surfaces.shape), snapshot.detail.tree_surfaces)
        np.testing.assert_allclose(maps.mean(axis=0).reshape(snapshot.surface.shape), snapshot.surface)


def test_forest_title_counts_trees_in_good_english(lessons, ax):
    lesson = lessons["random_forest"]
    lesson.diagram.draw_training(ax, lesson.snapshots[0])
    assert "The forest: 1 tree," in texts(ax)


def test_boosting_corrections_add_up_like_sklearns_own_stages(lessons, ax):
    lesson = lessons["gradient_boosting"]
    booster = lesson.diagram.booster
    x = lesson.data.query.features[None, :]
    staged = np.array([float(np.ravel(f)[0]) for f in booster.model.staged_decision_function(x)])  # F_1 … F_M
    lesson.diagram.draw_inference(ax, lesson.data.query, 1.0)
    shown, drawn = texts(ax), diagrams.CHAIN_TREES
    assert signed_small(staged[-drawn - 1] - float(booster.base(x)[0])) in shown  # every earlier tree, summed
    assert all(signed_small(step) in shown for step in np.diff(staged[-drawn - 1:]))


def test_bayes_log_terms_sum_to_scikit_learns_joint_log_likelihood(lessons, passenger_data):
    lesson = lessons["naive_bayes"]
    x = lesson.data.query.features
    reference = GaussianNB().fit(passenger_data.x_train, passenger_data.y_train)
    np.testing.assert_allclose(log_terms(lesson.snapshots[-1].detail, x).sum(axis=1),
                               reference.predict_joint_log_proba(x[None, :])[0], rtol=1e-6)


def test_bayes_inference_adds_logs_while_training_multiplies(lessons, ax):
    lesson = lessons["naive_bayes"]
    lesson.diagram.draw_inference(ax, lesson.data.query, 1.0)
    assert "+" in texts(ax).split("\n") and "×" not in texts(ax).split("\n")
    lesson.diagram.draw_training(ax, lesson.final)
    assert "×" in texts(ax).split("\n")


def test_small_signed_small_never_read_as_minus_zero():
    assert signed_small(0.314) == "+0.31" and signed_small(-0.0042) == "-0.004" and signed_small(-0.0001) == "0.000"


def test_weight_scale_ignores_the_bias():
    assert weight_scale([np.array([9.0, 1.0, -3.0]), np.array([0.0, 2.0, 0.5])]) == 3.0


def test_only_the_logistic_diagram_may_show_a_probability(lessons):
    from titanic_anim.lessons.lesson_diagrams import LinearDiagram
    with pytest.raises(ValueError):
        LinearDiagram("perceptron", lambda x: x, lambda s: s, np.zeros(4), 1.0, probability=float)
    with pytest.raises(ValueError):
        LinearDiagram("logistic", lambda x: x, lambda s: s, np.zeros(4), 1.0)
