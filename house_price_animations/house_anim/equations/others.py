"""Equations for k-nearest neighbours, the MLP, and a generic fallback.

- KNN learns nothing while fitting: it stores the training houses. To price a house it finds the
  k nearest ones (Euclidean distance on the standardised features) and averages their prices.
- The MLP is two ReLU hidden layers and a linear output, trained with Adam on the squared error
  plus an L2 penalty. Its prediction card runs the forward pass by hand, layer by layer.
- The generic pair names the model and its squared error, for any regressor without its own.

Every explainer takes the same arguments, so the cards can call any of them; an explainer that
has no use for one (``splits``, say) still accepts it.
"""

import numpy as np

from house_anim.config import TARGET_TO_THOUSANDS
from house_anim.data import HousingSplits
from house_anim.equations.cards import (
    EquationLine,
    check_line,
    heading,
    math_line,
    mono,
    note,
    price_line,
    training_mse,
)

SHOWN_NEIGHBOURS = 5
KNN_RULE = r"$\hat{y}(x) = \frac{1}{k}\sum_{i \in N_k(x)} y_i$"
MLP_LAYERS = (
    r"$h_1 = \mathrm{ReLU}(W_1^T x + b_1)$",
    r"$h_2 = \mathrm{ReLU}(W_2^T h_1 + b_2)$",
    r"$\hat{y} = w_3^T h_2 + b_3$",
)


def knn_training(model, frame, splits: HousingSplits) -> list[EquationLine]:
    """KNN's training card lines: the averaging rule, k, and how many houses are stored.

    Args:
        model: A fitted ``KNeighborsRegressor``.
        frame: The step's ``TrainingFrame``, for the training and validation MSE.
        splits: Unused; every training explainer takes it.

    Returns:
        The lines, in card order.
    """
    return [
        heading("Model: average the closest houses"), math_line(KNN_RULE),
        math_line(r"$d(x, x_i) = \Vert x - x_i \Vert_2$  picks $N_k(x)$"),
        note(f"k = {model.n_neighbors} neighbours, distance on standardised features"),
        heading("Training minimises"),
        note("nothing: fitting only stores the training houses"),
        heading("This step: remember more houses"),
        note(f"{model.n_samples_fit_:,} houses stored  ·  training MSE {training_mse(frame):.4f}"),
        note(f"validation MSE {frame.val_scores.rmse ** 2:.4f}"),
    ]


def knn_prediction(model, x: np.ndarray, splits: HousingSplits) -> tuple[list[EquationLine], float]:
    """One house through KNN: its nearest training houses, their distances and prices, and their mean.

    Args:
        model: A fitted ``KNeighborsRegressor``.
        x: The house's scaled features.
        splits: The data; the neighbours' prices are read from its training targets.

    Returns:
        The card's lines (the first ``SHOWN_NEIGHBOURS`` neighbours shown) and the mean price, in $100k.
    """
    distances, indices = model.kneighbors(x[None, :])
    prices = splits.y_train[indices[0]]  # the model was fit on a prefix of the training rows
    value = float(prices.mean())
    rows = [mono(f"#{rank:<2} distance {d:5.2f}   price ${p * TARGET_TO_THOUSANDS:,.0f}k")
            for rank, (d, p) in enumerate(zip(distances[0], prices), start=1)][:SHOWN_NEIGHBOURS]
    lines = [heading(f"Find the {len(prices)} nearest training houses"), math_line(KNN_RULE), *rows,
             note(f"… and {len(prices) - SHOWN_NEIGHBOURS} more neighbours"),
             price_line(value), check_line(model, x, value)]
    return lines, value


def _parameter_count(model) -> int:
    """How many weights and biases the MLP has, over every layer."""
    return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))


def mlp_training(model, frame, splits: HousingSplits) -> list[EquationLine]:
    """The MLP's training card lines: its layers, its loss now, and Adam's update rule.

    Args:
        model: A fitted ``MLPRegressor`` with two hidden layers.
        frame: The step's ``TrainingFrame``, for the training MSE.
        splits: Unused; every training explainer takes it.

    Returns:
        The lines, in card order.
    """
    sizes = " → ".join(str(n) for n in (model.coefs_[0].shape[0], *model.hidden_layer_sizes, 1))
    return [
        heading("Model: two hidden layers"), *(math_line(layer) for layer in MLP_LAYERS),
        note(f"{_parameter_count(model):,} weights and biases  ·  layers {sizes}"),
        heading("Training minimises"),
        math_line(r"$\frac{1}{2n}\sum_i (y_i - \hat{y}_i)^2 + \frac{\alpha}{2n}\Vert W \Vert_2^2$"),
        math_line(f"$= {model.loss_:.4f}$  (training MSE {training_mse(frame):.4f})"),
        heading("This step: one epoch of Adam"),
        math_line(r"$W \leftarrow W - \eta\,\hat{m} / (\sqrt{\hat{v}} + \epsilon)$"),
        note(f"η = {model.learning_rate_init:g}  ·  gradients from {model.t_:,} houses so far"),
    ]


def mlp_prediction(model, x: np.ndarray, splits: HousingSplits) -> tuple[list[EquationLine], float]:
    """One house through the MLP's forward pass, computed by hand with numpy from its weights.

    It shows how many units each ReLU layer switched on, and the three hidden units whose
    activation × output weight pushes the price hardest.

    Args:
        model: A fitted ``MLPRegressor`` with exactly two hidden layers.
        x: The house's scaled features.
        splits: Unused; every prediction explainer takes it.

    Returns:
        The card's lines and the forward pass's output, in $100k.
    """
    (w1, w2, w3), (b1, b2, b3) = model.coefs_, model.intercepts_
    h1 = np.maximum(x @ w1 + b1, 0.0)
    h2 = np.maximum(h1 @ w2 + b2, 0.0)
    pushes = h2 * w3[:, 0]
    value = float(pushes.sum() + b3[0])
    top = np.argsort(-np.abs(pushes))[:3]
    lines = [heading("Forward pass for this house"), *(math_line(layer) for layer in MLP_LAYERS),
             mono(f"h1: {np.count_nonzero(h1)}/{h1.size} units on, largest {h1.max():.2f}"),
             mono(f"h2: {np.count_nonzero(h2)}/{h2.size} units on, largest {h2.max():.2f}"),
             mono("top pushes: " + "  ".join(f"u{i} {pushes[i]:+.2f}" for i in top)),
             mono(f"b3 = {b3[0]:+.3f}"),
             price_line(value), check_line(model, x, value)]
    return lines, value


def generic_training(model, frame, splits: HousingSplits) -> list[EquationLine]:
    """The fallback training card lines: ŷ = f(x), the model's class name and its training MSE.

    Args:
        model: Any fitted regressor.
        frame: The step's ``TrainingFrame``, for the training MSE.
        splits: Unused; every training explainer takes it.

    Returns:
        The lines, in card order.
    """
    return [heading("Model"), math_line(r"$\hat{y} = f(x)$"), note(type(model).__name__),
            heading("Training minimises"),
            math_line(f"$\\frac{{1}}{{n}}\\sum_i (y_i - \\hat{{y}}_i)^2 = {training_mse(frame):.4f}$")]


def generic_prediction(model, x: np.ndarray, splits: HousingSplits) -> tuple[list[EquationLine], float]:
    """The fallback prediction card lines: the model's own prediction, with nothing worked out.

    Args:
        model: Any fitted regressor.
        x: The house's scaled features.
        splits: Unused; every prediction explainer takes it.

    Returns:
        The card's lines and ``model.predict(x)``, in $100k.
    """
    value = float(model.predict(x[None, :])[0])
    lines = [heading("Prediction"), math_line(r"$\hat{y} = f(x)$"), note(type(model).__name__),
             price_line(value), check_line(model, x, value)]
    return lines, value
