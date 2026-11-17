"""The forest, boosting and kNN diagrams draw the model's own trees and numbers, in a task's own words."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from animkit.algorithms.common import signed_rgba
from animkit.lessons import diagrams, model_diagrams
from animkit.lessons.model_diagrams import BoostingDiagram, ForestDiagram, NeighboursDiagram, Reading, leaf_colours

READING = Reading(subject="thing", row=lambda query: np.array([[query.x]]),
                  predict=lambda model, row: float(model.predict(row)[0]), show="{:.3f}".format)
QUERY = SimpleNamespace(x=0.4)


def texts(ax) -> str:
    return "\n".join(t.get_text() for t in ax.findobj(match=Text))


def colour(value):
    return (0.2, 0.4, 0.8, 1.0)


@pytest.fixture
def ax():
    return plt.figure(figsize=(6, 2)).add_subplot()


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(300, 1))
    return X, np.sin(3 * X[:, 0]) + rng.normal(scale=0.1, size=300)


@pytest.fixture(scope="module")
def forest(data):
    return RandomForestRegressor(n_estimators=14, max_depth=4, random_state=0).fit(*data)


class TinyBooster:
    """The booster interface the diagram needs: the fitted model, and Σ h_i(x) over a run of trees."""

    def __init__(self, model):
        self.model = model

    def trees(self, first: int, last: int, x: np.ndarray) -> np.ndarray:
        return sum((self.model.estimators_[i - 1, 0].predict(x) for i in range(first, last + 1)), np.zeros(len(x)))


@pytest.fixture(scope="module")
def booster(data):
    return TinyBooster(GradientBoostingRegressor(n_estimators=9, max_depth=2, learning_rate=0.1,
                                                 random_state=0).fit(*data))


@pytest.fixture
def handed(monkeypatch):
    """What the diagrams hand the kit's drawing helpers."""
    calls = []
    for name in ("tree_grid", "chain"):
        monkeypatch.setattr(diagrams, name, lambda ax, trees, colour, **kwargs: calls.append((trees, kwargs)))
    return calls


def test_forest_training_draws_the_trees_it_had_then(forest, handed):
    diagram = ForestDiagram(forest, READING, colour=colour, summary="average\nof all trees")
    for count in (1, 5):
        diagram.draw_training(None, SimpleNamespace(detail=SimpleNamespace(n_trees=count)))
        trees, kwargs = handed[-1]
        assert trees == [tree.tree_ for tree in forest.estimators_[:count]] == [t.tree_ for t in diagram.drawn(count)]
        assert kwargs["total"] == count and kwargs["summary"] == "average\nof all trees"
    assert handed[0][1]["title"] == "The forest: 1 tree, top 3 levels shown"
    assert handed[1][1]["title"] == "The forest: 5 trees, top 3 levels shown"


def test_forest_inference_shows_each_trees_vote_and_the_forests_answer(forest, ax):
    diagram = ForestDiagram(forest, READING, colour=colour, summary="average", show_vote="{:.1f}".format)
    diagram.draw_inference(ax, QUERY, 1.0)
    row = READING.row(QUERY)
    shown = texts(ax).split("\n")
    assert all(f"{tree.predict(row)[0]:.1f}" in shown for tree in forest.estimators_[:diagrams.GRID_TREES])
    assert READING.show(float(forest.predict(row)[0])) in shown
    assert "Every tree routes the thing to a leaf and votes" in shown
    diagram.draw_inference(ax, QUERY, 0.5)
    assert READING.show(float(forest.predict(row)[0])) not in texts(ax).split("\n")


def test_forest_inference_highlights_each_drawn_trees_own_path(forest, handed):
    ForestDiagram(forest, READING, colour=colour, summary="average").draw_inference(None, QUERY, 1.0)
    _, kwargs = handed[-1]
    row = READING.row(QUERY)
    expected = [tree.decision_path(row).indices for tree in forest.estimators_[:diagrams.GRID_TREES]]
    assert all(np.array_equal(path, want) for path, want in zip(kwargs["paths"], expected, strict=True))
    assert kwargs["votes"][0] == READING.show(float(forest.estimators_[0].predict(row)[0]))  # default vote format


def test_boosting_training_draws_the_newest_trees_in_the_tasks_words(booster, handed):
    diagram = BoostingDiagram(booster, READING, f0=1.5, show_score="{:.2f}".format, show_step="{:+.2f}".format,
                              link="σ")
    diagram.draw_training(None, SimpleNamespace(detail=SimpleNamespace(stage=0)))
    _, kwargs = handed[-1]
    assert kwargs["title"] == "Boosting: F₀ + 0 small corrections, then σ" and kwargs["end"] == "σ(F₀)"
    diagram.draw_training(None, SimpleNamespace(detail=SimpleNamespace(stage=6)))
    trees, kwargs = handed[-1]
    assert trees == [tree.tree_ for tree in booster.model.estimators_[:6, 0]]
    assert kwargs["end"] == "σ(F6)" and kwargs["start"] == "F₀\n1.50" and kwargs["step_label"] == "0.1 ×"
    assert diagram.booster is booster


def test_boosting_inference_adds_up_to_scikit_learns_own_stages(booster, ax):
    show_step = "{:+.4f}".format
    diagram = BoostingDiagram(booster, READING, f0=float(booster.model.init_.predict(np.zeros((1, 1)))[0]),
                              show_score="{:.2f}".format, show_step=show_step, step_noun="push")
    diagram.draw_inference(ax, QUERY, 1.0)
    row = READING.row(QUERY)
    staged = np.array([float(f[0]) for f in booster.model.staged_predict(row)])
    newest = diagrams.CHAIN_TREES
    shown = texts(ax)
    assert show_step(staged[-newest - 1] - diagram.f0) in shown  # all the earlier trees, summed
    assert all(show_step(step) in shown for step in np.diff(staged[-newest - 1:]))
    assert f"ŷ\n{READING.show(staged[-1])}" in shown and "This thing: F₀ plus every tree's push" in shown


def test_leaf_colours_take_their_scale_from_leaves(booster):
    tree = booster.model.estimators_[0, 0].tree_
    biggest = float(np.max(np.abs(tree.value[tree.children_left == -1])))
    assert leaf_colours([tree])(np.array([[biggest]])) == signed_rgba(biggest, biggest)


def test_neighbours_walk_through_the_tasks_own_steps(ax):
    diagram = NeighboursDiagram(READING, 9, steps=lambda k: ["store", f"k = {k}", "average", "ŷ"],
                                values=lambda query: ["300", "9 found", "mean", f"{query.x}"], verb="decides")
    diagram.draw_training(ax, SimpleNamespace(detail=SimpleNamespace(k=4)))
    assert "How kNN decides (no training beyond storing)" in texts(ax) and "k = 4" in texts(ax)
    diagram.draw_inference(ax, QUERY, 1.0)
    shown = texts(ax).split("\n")
    assert "This thing's neighbours, step by step" in shown and "k = 9" in shown and "0.4" in shown


def test_shared_diagrams_draw_with_the_default_levels():
    assert model_diagrams.SHOWN_LEVELS == 3
