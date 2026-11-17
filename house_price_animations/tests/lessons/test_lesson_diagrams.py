"""Every regression lesson shows how its algorithm computes; the diagram's numbers are the model's."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

from animkit.lessons import diagrams
from animkit.lessons.fit import colliding_texts, overflowing_texts
from animkit.lessons.stage import Stage
from house_anim.lessons.lesson import money_k, predict_one
from house_anim.lessons.lesson_diagrams import correction
from house_anim.lessons.stories import build_lesson, lesson_keys

WITH_DIAGRAM = ("linear_regression", "polynomial_regression", "ridge", "lasso", "random_forest",
                "gradient_boosting", "knn")


@pytest.fixture(scope="module")
def lessons(income_data):
    return {key: build_lesson(key, income_data) for key in lesson_keys()}


def texts(ax) -> str:
    return "\n".join(t.get_text() for t in ax.findobj(match=Text))


@pytest.fixture
def ax():
    return plt.figure(figsize=(6, 2)).add_subplot()


def test_every_lesson_shows_its_algorithm(lessons):
    assert {key for key, lesson in lessons.items() if lesson.diagram is not None} == set(WITH_DIAGRAM)
    # The tree and the network need no extra diagram: their side panel already draws the model's structure.
    assert lessons["decision_tree"].diagram is None and lessons["mlp"].diagram is None


@pytest.mark.parametrize("key", WITH_DIAGRAM)
def test_diagram_draws_every_training_state(lessons, key, ax):
    lesson = lessons[key]
    for snapshot in (lesson.snapshots[0], lesson.final):
        lesson.diagram.draw_training(ax, snapshot)
        assert texts(ax)


@pytest.mark.parametrize("key", WITH_DIAGRAM)
def test_finished_inference_diagram_ends_on_the_models_prediction(lessons, key, ax):
    lesson = lessons[key]
    lesson.diagram.draw_inference(ax, lesson.data.query, 1.0)
    assert money_k(predict_one(lesson.predict, lesson.data.query.income)) in texts(ax)


def test_linear_diagram_carries_the_current_weights(lessons, ax):
    lesson = lessons["linear_regression"]
    for snapshot in (lesson.snapshots[len(lesson.snapshots) // 4], lesson.final):  # mid-training, not just the end
        lesson.diagram.draw_training(ax, snapshot)
        theta = snapshot.detail.theta
        assert money_k(theta[1], signed=True) in texts(ax) and money_k(theta[0]) in texts(ax)
    assert lesson.snapshots[len(lesson.snapshots) // 4].detail.theta != lesson.final.detail.theta


def test_forest_grid_is_handed_the_snapshots_own_trees(lessons, income_data, monkeypatch):
    lesson = lessons["random_forest"]
    handed = []
    monkeypatch.setattr(diagrams, "tree_grid", lambda ax, trees, *args, **kwargs: handed.append(trees))
    grid = np.ascontiguousarray(income_data.grid.reshape(-1, 1), dtype=np.float32)
    for snapshot in lesson.snapshots:
        lesson.diagram.draw_training(ax=None, snapshot=snapshot)
        trees = handed[-1]
        assert len(trees) == snapshot.detail.n_trees
        curves = np.array([tree.predict(grid).reshape(len(grid), -1)[:, 0] for tree in trees[:diagrams.GRID_TREES]])
        np.testing.assert_allclose(curves, snapshot.detail.tree_curves[:diagrams.GRID_TREES], rtol=1e-6)


def test_lasso_diagram_dashes_switched_off_terms(lessons, ax):
    lesson = lessons["lasso"]
    snap = max(lesson.snapshots, key=lambda s: int(np.sum(s.detail.coef == 0.0)))
    lesson.diagram.draw_training(ax, snap)
    dashed = sum(line.get_linestyle() == "--" for line in ax.get_lines())
    assert dashed == int(np.sum(snap.detail.coef == 0.0)) > 0


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


def test_forest_draws_exactly_the_trees_behind_each_snapshot(lessons, income_data):
    lesson = lessons["random_forest"]
    grid = income_data.grid.reshape(-1, 1)
    for snapshot in lesson.snapshots:
        drawn = lesson.diagram.drawn(snapshot.detail.n_trees)
        curves = np.array([tree.predict(grid) for tree in drawn])
        np.testing.assert_allclose(curves, snapshot.detail.tree_curves)
        np.testing.assert_allclose(curves.mean(axis=0), snapshot.curve)


def test_forest_title_counts_trees_in_good_english(lessons, ax):
    lesson = lessons["random_forest"]
    lesson.diagram.draw_training(ax, lesson.snapshots[0])
    assert "The forest: 1 tree," in texts(ax)


def test_boosting_corrections_add_up_like_sklearns_own_stages(lessons, ax):
    lesson = lessons["gradient_boosting"]
    model = lesson.diagram.booster.model
    x = np.array([[lesson.data.query.income]])
    staged = np.array([float(f[0]) for f in model.staged_predict(x)])  # F_1 … F_M
    f0 = float(lesson.diagram.booster.base(x)[0])
    lesson.diagram.draw_inference(ax, lesson.data.query, 1.0)
    shown = texts(ax)
    shown_trees = diagrams.CHAIN_TREES
    assert correction(staged[-shown_trees - 1] - f0) in shown  # every earlier tree, summed
    steps = np.diff(staged[-shown_trees - 1:])
    assert all(correction(step) in shown for step in steps)
    assert f"… {model.n_estimators - shown_trees} earlier" in shown


def test_small_corrections_never_read_as_zero_thousand():
    assert correction(0.5) == "+$50k" and correction(-0.042) == "−$4.2k"
    assert correction(0.00037) == "+$37" and correction(-0.000004) == "$0"
