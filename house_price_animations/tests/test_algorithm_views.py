from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

from house_anim.algorithms import algorithm_view
from house_anim.algorithms.tree_view import DRAW_DEPTH
from house_anim.models import get_spec, model_keys
from house_anim.trainers import train_silently

EXPECTED_VIEWS = {
    "linear_regression": "WeightedSumView",
    "ridge": "WeightedSumView",
    "lasso": "WeightedSumView",
    "sgd": "WeightedSumView",
    "decision_tree": "TreeView",
    "random_forest": "ForestView",
    "gradient_boosting": "BoostingView",
    "knn": "NeighboursView",
    "mlp": "NetworkView",
}


def small_spec(key):
    spec = get_spec(key)
    return spec if spec.strategy == "data_growth" else replace(spec, steps=spec.steps[:3])


@pytest.fixture(scope="module")
def trained(splits):
    return {key: (small_spec(key), *train_silently(small_spec(key), splits)) for key in model_keys()}


def make_view(key, splits):
    figure = plt.figure(figsize=(8, 10))
    return figure, algorithm_view(get_spec(key), figure.add_axes((0.02, 0.02, 0.96, 0.9)), splits)


@pytest.mark.parametrize("key", model_keys())
def test_each_family_gets_its_own_view(key, splits):
    _, view = make_view(key, splits)
    assert type(view).__name__ == EXPECTED_VIEWS[key]


@pytest.mark.parametrize("key", model_keys())
def test_views_draw_training_then_prediction(key, trained, splits):
    _, model, frame = trained[key]
    figure, view = make_view(key, splits)
    view.show_training(model, frame)
    figure.canvas.draw()
    view.show_prediction(model, splits.X_test[0], house_number=1)
    figure.canvas.draw()
    assert view.ax.get_title()


def test_weighted_sum_edges_follow_the_weights(splits):
    model = Lasso(alpha=0.05).fit(splits.X_train, splits.y_train)
    _, frame = train_silently(small_spec("lasso"), splits)
    _, view = make_view("lasso", splits)
    view.show_training(model, frame)
    widths = [edge.get_linewidth() for edge in view.edges]
    assert len(view.edges) == len(splits.feature_names)
    assert int(np.argmax(widths)) == int(np.argmax(np.abs(model.coef_)))
    for edge, weight in zip(view.edges, model.coef_):
        assert (edge.get_linestyle() == "--") == (weight == 0.0)


def test_weighted_sum_prediction_shows_the_price(trained, splits):
    _, model, _ = trained["linear_regression"]
    _, view = make_view("linear_regression", splits)
    view.show_prediction(model, splits.X_test[0], house_number=1)
    price_k = model.predict(splits.X_test[:1])[0] * 100
    assert f"{price_k:,.0f}k" in view.output_text.get_text()


def test_tree_view_draws_only_the_top_levels_and_the_path(splits):
    deep = DecisionTreeRegressor(random_state=0).fit(splits.X_train, splits.y_train)
    _, frame = train_silently(small_spec("decision_tree"), splits)
    _, view = make_view("decision_tree", splits)
    view.show_training(deep, frame)
    assert 0 < view.nodes_drawn <= 2 ** (DRAW_DEPTH + 1) - 1
    view.show_prediction(deep, splits.X_test[0], house_number=1)
    assert 1 <= view.path_drawn <= DRAW_DEPTH


def test_forest_view_shows_tiles_and_every_vote(trained, splits):
    _, model, frame = trained["random_forest"]
    _, view = make_view("random_forest", splits)
    view.show_training(model, frame)
    assert view.tiles_drawn == min(len(model.estimators_), 12)
    view.show_prediction(model, splits.X_test[0], house_number=1)
    assert view.votes_drawn == len(model.estimators_)


def test_boosting_trajectory_ends_at_the_prediction(trained, splits):
    _, model, frame = trained["gradient_boosting"]
    _, view = make_view("gradient_boosting", splits)
    view.show_training(model, frame)
    assert view.bars_drawn == model.n_estimators_
    view.show_prediction(model, splits.X_test[0], house_number=1)
    path = view.trajectory.get_ydata()
    assert len(path) == model.n_estimators_ + 1
    assert path[-1] == pytest.approx(model.predict(splits.X_test[:1])[0] * 100)
    low, high = view.ax.get_ylim()
    assert low <= path.min() and path.max() <= high  # the curve is actually in view
    main_bottom = view.ax.get_position().y0
    assert view.strip.get_position().y1 < main_bottom - 0.02  # room for the x-label


def test_knn_view_links_the_house_to_its_neighbours(trained, splits):
    _, model, frame = trained["knn"]
    _, view = make_view("knn", splits)
    view.show_training(model, frame)
    assert len(view.stored.get_offsets()) == model.n_samples_fit_
    view.show_prediction(model, splits.X_test[0], house_number=1)
    assert view.links_drawn == model.n_neighbors


def test_network_view_draws_every_weight(trained, splits):
    _, model, frame = trained["mlp"]
    _, view = make_view("mlp", splits)
    view.show_training(model, frame)
    expected = sum(w.size for w in model.coefs_)
    assert len(view.edges.get_segments()) == expected
    view.show_prediction(model, splits.X_test[0], house_number=1)
    assert len(view.edges.get_segments()) == expected


def test_weight_labels_sit_beside_their_inputs_not_on_top_of_each_other(trained, splits):
    _, model, frame = trained["linear_regression"]
    _, view = make_view("linear_regression", splits)
    view.show_training(model, frame)
    label_ys = [label.get_position()[1] for label in view.edge_labels]
    input_ys = list(view.inputs.get_offsets()[:, 1])
    assert label_ys == pytest.approx(input_ys)


def test_captions_wrap_to_the_panel_width(splits):
    from house_anim.algorithms.common import wrap_to_axes

    figure = plt.figure(figsize=(10, 6))
    narrow = figure.add_axes((0.1, 0.1, 0.2, 0.8))
    wide = figure.add_axes((0.4, 0.1, 0.55, 0.8))
    text = "word " * 30
    assert wrap_to_axes(narrow, text, fontsize=11).count("\n") > wrap_to_axes(wide, text, fontsize=11).count("\n")
    assert "\n" not in wrap_to_axes(wide, "short caption", fontsize=11)
