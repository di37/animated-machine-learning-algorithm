"""Registry of animated scikit-learn regressors and how each one is trained step by step.

A model is described by animkit's ``ModelSpec``: how to build the estimator, which training
strategy shows it learning, and the schedule of steps (one training step per dashboard frame).
Most regressors have no natural "one step" of training, so each one gets the strategy that makes
its learning visible:

- data growth: refit on a growing share of the training rows (Linear Regression, KNN), which is a
  learning curve;
- parameter sweep: refit while one hyper-parameter moves (Ridge and Lasso weaken their penalty,
  the decision tree grows one level deeper);
- incremental: ``partial_fit`` chunk by chunk, epoch after epoch (SGD, the MLP);
- warm start: keep the fitted ensemble and add members (Random Forest, Gradient Boosting).

``MODEL_SPECS`` maps each key (``"ridge"``, ``"mlp"``...) to its spec; it is read-only.
"""

import math
from types import MappingProxyType

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from animkit.models import (
    STRATEGIES,
    STRATEGY_DATA_GROWTH,
    STRATEGY_INCREMENTAL,
    STRATEGY_PARAM_SWEEP,
    STRATEGY_WARM_START,
    ModelSpec,
)
from house_anim.config import RANDOM_STATE

__all__ = ["GROWTH_FRACTIONS", "MODEL_SPECS", "STRATEGIES", "STRATEGY_DATA_GROWTH", "STRATEGY_INCREMENTAL",
           "STRATEGY_PARAM_SWEEP", "STRATEGY_WARM_START", "ModelSpec", "get_spec", "model_keys"]

GROWTH_FRACTIONS = tuple(float(f) for f in np.geomspace(0.004, 1.0, 30))


def _neg_log10(alpha: float) -> float:
    """The x-axis value for a penalty ``alpha``: -log10(alpha), so weaker penalties plot further right."""
    return -math.log10(alpha)


_SPECS = (
    ModelSpec(
        key="linear_regression",
        title="Linear Regression",
        short_title="Linear Regression",
        step_format="{} houses",
        strategy=STRATEGY_DATA_GROWTH,
        step_label="training houses seen",
        note="Refit on more houses each frame: a straight-line model plateaus early (high bias).",
        build=LinearRegression,
        steps=GROWTH_FRACTIONS,
        hold_frames=2,
        log_x=True,
    ),
    ModelSpec(
        key="ridge",
        title="Ridge (L2-regularised linear)",
        short_title="Ridge",
        step_format="−log10 α = {}",
        strategy=STRATEGY_PARAM_SWEEP,
        step_label="regularisation weakens  (−log10 alpha)",
        note="Alpha shrinks each frame: coefficients grow back smoothly; unlike Lasso, none is ever exactly zero.",
        build=Ridge,
        steps=tuple(float(a) for a in np.logspace(5.0, 0.0, 31)),
        param_name="alpha",
        hold_frames=2,
        step_transform=_neg_log10,
    ),
    ModelSpec(
        key="lasso",
        title="Lasso (L1-regularised linear)",
        short_title="Lasso",
        step_format="−log10 α = {}",
        strategy=STRATEGY_PARAM_SWEEP,
        step_label="regularisation weakens  (−log10 alpha)",
        note="Alpha shrinks each frame: watch features switch on one by one as the L1 penalty relaxes.",
        build=lambda: Lasso(max_iter=10_000),
        steps=tuple(float(a) for a in np.logspace(0.0, -3.0, 31)),
        param_name="alpha",
        hold_frames=2,
        step_transform=_neg_log10,
    ),
    ModelSpec(
        key="sgd",
        title="SGD Regressor (linear, gradient descent)",
        short_title="SGD (linear)",
        step_format="epoch {}",
        strategy=STRATEGY_INCREMENTAL,
        step_label="epochs",
        note="partial_fit on 1/16 of the shuffled data per frame: averaged weights walk downhill on squared error.",
        # Averaging keeps the descent steady despite heavy feature outliers (AveOccup, AveRooms);
        # faster or decaying schedules diverge on this data.
        build=lambda: SGDRegressor(
            learning_rate="constant", eta0=3e-4, average=True, random_state=RANDOM_STATE
        ),
        steps=(1, 2, 3, 4),
        chunks_per_epoch=16,
    ),
    ModelSpec(
        key="decision_tree",
        title="Decision Tree Regressor",
        short_title="Decision Tree",
        step_format="depth {}",
        strategy=STRATEGY_PARAM_SWEEP,
        step_label="max_depth",
        note="One level deeper each frame: training error keeps falling, validation error turns back up.",
        build=lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
        steps=tuple(range(1, 19)),
        param_name="max_depth",
        hold_frames=4,
    ),
    ModelSpec(
        key="random_forest",
        title="Random Forest Regressor",
        short_title="Random Forest",
        step_format="{} trees",
        strategy=STRATEGY_WARM_START,
        step_label="trees in the forest",
        note="More bootstrapped trees each frame (warm_start): averaging cancels out individual tree noise.",
        build=lambda: RandomForestRegressor(
            n_estimators=1, warm_start=True, n_jobs=-1, random_state=RANDOM_STATE
        ),
        steps=(*range(1, 11), *range(15, 101, 5)),
        param_name="n_estimators",
        hold_frames=2,
    ),
    ModelSpec(
        key="gradient_boosting",
        title="Gradient Boosting Regressor",
        short_title="Gradient Boosting",
        step_format="{} trees",
        strategy=STRATEGY_WARM_START,
        step_label="boosting stages (trees)",
        note="Five more shallow trees each frame, each one fitted to the residuals of the ones before.",
        build=lambda: GradientBoostingRegressor(
            n_estimators=1, warm_start=True, max_depth=3, random_state=RANDOM_STATE
        ),
        steps=tuple(range(5, 305, 5)),
        param_name="n_estimators",
    ),
    ModelSpec(
        key="knn",
        title="K-Nearest Neighbours Regressor (k=10)",
        short_title="KNN (k=10)",
        step_format="{} houses",
        strategy=STRATEGY_DATA_GROWTH,
        step_label="training houses memorised",
        note="KNN 'training' is memorising houses; more neighbours to borrow from means better guesses.",
        build=lambda: KNeighborsRegressor(n_neighbors=10),
        steps=GROWTH_FRACTIONS,
        hold_frames=2,
        log_x=True,
        no_weights_note=(
            "KNN learns no per-feature weights.\n"
            "Each price is the average of the\n10 most similar training houses."
        ),
    ),
    ModelSpec(
        key="mlp",
        title="MLP Regressor (neural network 64-32)",
        short_title="MLP 64-32",
        step_format="epoch {}",
        strategy=STRATEGY_INCREMENTAL,
        step_label="epochs",
        note="One Adam epoch of back-propagation per frame through a 64-32 hidden-layer network.",
        build=lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32), learning_rate_init=1e-3, batch_size=128,
            random_state=RANDOM_STATE,
        ),
        steps=tuple(range(1, 61)),
        hold_frames=2,
    ),
)

MODEL_SPECS = MappingProxyType({spec.key: spec for spec in _SPECS})


def model_keys() -> tuple[str, ...]:
    """The keys of every animated model, in the order they are rendered.

    Returns:
        The keys, such as ``"linear_regression"`` and ``"random_forest"``.
    """
    return tuple(MODEL_SPECS)


def get_spec(key: str) -> ModelSpec:
    """Looks up one model's spec by its key.

    Args:
        key: A key from ``model_keys()``, such as ``"gradient_boosting"``.

    Returns:
        The model's spec.

    Raises:
        ValueError: If no model has that key; the message lists the valid keys.
    """
    try:
        return MODEL_SPECS[key]
    except KeyError:
        raise ValueError(f"Unknown model '{key}'. Choose one of: {', '.join(MODEL_SPECS)}") from None
