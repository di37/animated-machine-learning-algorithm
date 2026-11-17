"""Animated diagrams of each algorithm, redrawn from the live model during training and inference.

Every view is an animkit ``AlgorithmView``: it owns one axes (the dashboards' left column) and has
two methods. ``show_training(model, frame)`` draws what the model has learned so far;
``show_prediction(model, x, house_number)`` draws how it prices one house. The views read the
fitted estimator itself (its weights, its trees, its stored houses), so the picture is the model's
real state, never a sketch of it.

``VIEWS`` maps estimator types to views; ``algorithm_view`` picks one for a model spec.
"""

from matplotlib.axes import Axes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from house_anim.algorithms.common import AlgorithmView
from house_anim.algorithms.ensemble_views import BoostingView, ForestView
from house_anim.algorithms.knn_view import NeighboursView
from house_anim.algorithms.linear_view import WeightedSumView
from house_anim.algorithms.network_view import NetworkView
from house_anim.algorithms.tree_view import TreeView
from house_anim.data import HousingSplits
from house_anim.models import ModelSpec

__all__ = ["AlgorithmView", "algorithm_view"]

VIEWS = (
    ((LinearRegression, Ridge, Lasso, SGDRegressor), WeightedSumView),
    (DecisionTreeRegressor, TreeView),
    (RandomForestRegressor, ForestView),
    (GradientBoostingRegressor, BoostingView),
    (KNeighborsRegressor, NeighboursView),
    (MLPRegressor, NetworkView),
)


def algorithm_view(spec: ModelSpec, ax: Axes, splits: HousingSplits) -> AlgorithmView:
    """The diagram for a model, chosen by the type of estimator its spec builds.

    Args:
        spec: The model; an unfitted estimator is built only to read its type.
        ax: The axes the diagram draws in.
        splits: The data (feature names, scaler, training houses).

    Returns:
        The view, with nothing drawn yet.

    Raises:
        ValueError: If no view knows the estimator's type.
    """
    estimator = spec.build()
    for types, view in VIEWS:
        if isinstance(estimator, types):
            return view(ax, splits)
    raise ValueError(f"No algorithm diagram for {type(estimator).__name__}")
