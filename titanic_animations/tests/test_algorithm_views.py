from dataclasses import replace

import matplotlib.pyplot as plt
import pytest
from sklearn.tree import DecisionTreeClassifier

from titanic_anim.algorithms import algorithm_view
from titanic_anim.algorithms.bayes_view import NaiveBayesView
from titanic_anim.models import get_spec, model_keys
from titanic_anim.projection import fit_projection
from titanic_anim.trainers import train_silently

EXPECTED_VIEWS = {
    "logistic_regression": "LogisticView",
    "sgd": "LogisticView",
    "svm": "SupportVectorView",
    "naive_bayes": "NaiveBayesView",
    "decision_tree": "TreeView",
    "random_forest": "ForestView",
    "gradient_boosting": "BoostingView",
    "knn": "NeighboursView",
    "mlp": "ClassifierNetworkView",
}


def small_spec(key):
    spec = get_spec(key)
    return spec if spec.strategy == "data_growth" else replace(spec, steps=spec.steps[:3])


@pytest.fixture(scope="module")
def trained(splits):
    return {key: (small_spec(key), *train_silently(small_spec(key), splits)) for key in model_keys()}


def make_view(key, splits):
    figure = plt.figure(figsize=(8, 10))
    return figure, algorithm_view(get_spec(key), figure.add_axes((0.05, 0.05, 0.9, 0.85)), splits)


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
    view.show_prediction(model, splits.X_test[0], 1)
    figure.canvas.draw()
    assert view.ax.get_title() or view.ax.get_title(loc="left")


def test_tree_view_highlights_the_passengers_path(splits):
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(splits.X_train, splits.y_train)
    _, view = make_view("decision_tree", splits)
    view.show_prediction(model, splits.X_test[0], 1)
    assert view.path_drawn == 3
    assert "P(survive)" in view.ax.texts[-1].get_text()


def test_forest_shows_every_trees_probability(trained, splits):
    _, model, _ = trained["random_forest"]
    _, view = make_view("random_forest", splits)
    view.show_prediction(model, splits.X_test[0], 1)
    assert view.votes_drawn == len(model.estimators_)
    assert view.tiles_drawn == min(12, len(model.estimators_))


def test_boosting_trajectory_ends_at_predict_proba(trained, splits):
    _, model, frame = trained["gradient_boosting"]
    _, view = make_view("gradient_boosting", splits)
    view.show_training(model, frame)
    assert view.bars_drawn == model.n_estimators_
    x = splits.X_test[1]
    view.show_prediction(model, x, 2)
    assert view.trajectory.get_ydata()[-1] == pytest.approx(model.predict_proba(x[None])[0, 1])


def test_knn_links_to_k_neighbours(trained, splits):
    _, model, _ = trained["knn"]
    _, view = make_view("knn", splits)
    view.show_prediction(model, splits.X_test[0], 1)
    assert view.links_drawn == model.n_neighbors


def test_svm_rings_every_support_vector(trained, splits):
    _, model, frame = trained["svm"]
    _, view = make_view("svm", splits)
    view.show_training(model, frame)
    assert view.vectors_drawn == len(model.support_)
    view.show_prediction(model, splits.X_test[0], 1)
    assert 0 < view.links_drawn <= 10


def test_naive_bayes_has_a_panel_per_feature(trained, splits):
    _, model, _ = trained["naive_bayes"]
    _, view = make_view("naive_bayes", splits)
    assert isinstance(view, NaiveBayesView) and len(view.panels) == len(splits.feature_names)
    view.show_prediction(model, splits.X_test[0], 1)
    assert all(panel.lines for panel in view.panels)


def test_network_output_is_a_probability(trained, splits):
    _, model, _ = trained["mlp"]
    _, view = make_view("mlp", splits)
    view.show_prediction(model, splits.X_test[0], 1)
    p = model.predict_proba(splits.X_test[:1])[0, 1]
    assert view.output_text.get_text() == f"{p:.0%}"
    title = view.ax.get_title() or view.ax.get_title(loc="left")  # dashboards set titlelocation=left
    assert title.startswith("Passenger #1")


def test_projection_names_its_axes(splits):
    projection = fit_projection(splits)
    assert projection.points(splits.X_test).shape == (len(splits.y_test), 2)
    assert all(name.startswith(f"axis {i + 1} ≈") for i, name in enumerate(projection.axis_names))


def test_unknown_estimator_has_no_view(splits):
    from sklearn.dummy import DummyClassifier

    figure = plt.figure()
    with pytest.raises(ValueError):
        algorithm_view(replace(get_spec("knn"), build=DummyClassifier), figure.add_subplot(), splits)
