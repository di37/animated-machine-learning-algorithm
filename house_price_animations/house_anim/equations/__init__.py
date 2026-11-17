"""Live equations: what each model computes while it trains, and worked out for a single house.

An equation card (animkit's ``EquationCard``) is a title and a column of lines: headings, maths
(matplotlib mathtext), aligned numbers and notes. Two cards are made here, for any model:

- ``training_card``, after every training step: the model's formula with its current numbers, the
  objective training minimises and its current value, and what this step changed;
- ``prediction_card``, for one test house: the formula worked through with that house's numbers,
  ending in a ``check:`` line that compares the result with ``model.predict``. The worked value
  must match the model's own prediction, so the card shows what the model really computes.

Each model family has its own explainers: ``linear`` (Linear Regression, Ridge, Lasso, SGD),
``trees`` (decision tree, random forest, gradient boosting) and ``others`` (KNN, MLP, and a
generic fallback). ``EXPLAINERS`` picks one by the estimator's type.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from house_anim.data import HousingSplits
from house_anim.equations import linear, others, trees
from house_anim.equations.cards import (
    HEADING,
    KINDS,
    MATH,
    MONO,
    NOTE,
    EquationCard,
    EquationLine,
    validate_house,
)
from house_anim.frames import TrainingFrame
from house_anim.models import STRATEGY_DATA_GROWTH, ModelSpec

__all__ = ["HEADING", "KINDS", "MATH", "MONO", "NOTE", "EquationCard", "EquationLine",
           "prediction_card", "training_card"]

LINEAR_TYPES = (LinearRegression, Ridge, Lasso, SGDRegressor)
# (estimator types, training explainer, prediction explainer); first match wins.
EXPLAINERS = (
    (DecisionTreeRegressor, trees.tree_training, trees.tree_prediction),
    (RandomForestRegressor, trees.forest_training, trees.forest_prediction),
    (GradientBoostingRegressor, trees.boosting_training, trees.boosting_prediction),
    (KNeighborsRegressor, others.knn_training, others.knn_prediction),
    (MLPRegressor, others.mlp_training, others.mlp_prediction),
)


def _explainers(model):
    """The (training, prediction) explainers for a non-linear model; the generic pair if none matches."""
    for types, train, predict in EXPLAINERS:
        if isinstance(model, types):
            return train, predict
    return others.generic_training, others.generic_prediction


def training_card(model, frame: TrainingFrame, spec: ModelSpec, splits: HousingSplits) -> EquationCard:
    """The model's formula with current numbers, its objective, and what this step changed.

    Args:
        model: The live model after this step.
        frame: The step's snapshot (its training RMSE gives the objective's value).
        spec: The model's spec; for a data-growth schedule the step is the number of rows fitted.
        splits: The data (feature names, and the scaler for real-unit thresholds).

    Returns:
        The card, titled "What <model> is computing".
    """
    if isinstance(model, LINEAR_TYPES):
        rows = int(frame.step) if spec.strategy == STRATEGY_DATA_GROWTH else splits.n_train
        lines = linear.training_lines(model, frame, splits, rows)
    else:
        lines = _explainers(model)[0](model, frame, splits)
    return EquationCard(f"What {spec.short_title} is computing", tuple(lines))


def prediction_card(model, x: np.ndarray, house_number: int, splits: HousingSplits) -> EquationCard:
    """The model's equation worked through for one house, ending in model.predict's answer.

    Args:
        model: The fitted model.
        x: The house's scaled features, shape (8,).
        house_number: The house's position among the test houses, from 1, for the title.
        splits: The data (feature names, and the scaler for real units).

    Returns:
        The card, with ``value`` set to the price the worked equation arrives at, in $100k.

    Raises:
        ValueError: If ``x`` is not one row of the right length.
    """
    row = validate_house(x, len(splits.feature_names))
    if isinstance(model, LINEAR_TYPES):
        lines, value = linear.prediction_lines(model, row, splits)
    else:
        lines, value = _explainers(model)[1](model, row, splits)
    return EquationCard(f"House #{house_number:,}: the equation at work", tuple(lines), value)
