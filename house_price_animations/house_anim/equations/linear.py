"""Equations for the linear family: LinearRegression, Ridge, Lasso, SGDRegressor.

All four predict with the same formula, ŷ = b + Σ_j w_j x_j: an intercept plus one weight per
(standardised) feature. They differ in what training minimises:

- Linear Regression: the squared error alone, solved exactly in one step;
- Ridge: the squared error plus α‖w‖², which shrinks every weight but never to exactly 0;
- Lasso: the squared error plus α‖w‖₁, which sets some weights exactly to 0;
- SGD: the squared error, walked downhill one house at a time; predictions use the average of
  the weights along the way.

The objectives' values come from the frame's training RMSE, which is scored on a sample of the
training rows, so they are shown as approximate (≈) where they scale it up.
"""

import numpy as np
from sklearn.linear_model import Lasso, Ridge, SGDRegressor

from house_anim.data import HousingSplits
from house_anim.equations.cards import (
    EquationLine,
    check_line,
    heading,
    linear_terms,
    math_line,
    mono,
    note,
    power_of_ten,
    price_line,
    training_mse,
)


def _weights(model) -> tuple[float, np.ndarray]:
    """The model's intercept b and its weight vector w."""
    return float(np.ravel(model.intercept_)[0]), np.ravel(model.coef_)


def _ridge_objective(model, mse: float, rows: int) -> list[EquationLine]:
    """Ridge's objective and its value now: the summed squared error (rows × MSE) plus α‖w‖²."""
    _, w = _weights(model)
    data, squares = rows * mse, float(np.sum(w ** 2))
    return [
        math_line(r"$\min_w\ \Vert y - Xw \Vert_2^2 + \alpha \Vert w \Vert_2^2$"),
        math_line(f"$\\approx {data:,.0f} + {power_of_ten(model.alpha)} \\times {squares:.3f}"
                  f" = {data + model.alpha * squares:,.0f}$"),
        note("the L2 penalty shrinks every weight a little, but never to exactly 0"),
    ]


def _sparsity_note(alive: int, total: int) -> str:
    """How many of Lasso's weights are still non-zero, and how many L1 has zeroed."""
    if alive == total:
        return f"all {total} weights are non-zero: the penalty is now too weak to zero any"
    return f"{alive} of {total} weights are non-zero; L1 set the other {total - alive} exactly to 0"


def _lasso_objective(model, mse: float) -> list[EquationLine]:
    """Lasso's objective (scikit-learn's 1/(2n) scaling of the squared error) and its value now."""
    _, w = _weights(model)
    l1 = float(np.sum(np.abs(w)))
    return [
        math_line(r"$\min_w\ \frac{1}{2n}\Vert y - Xw \Vert_2^2 + \alpha \Vert w \Vert_1$"),
        math_line(f"$\\approx {mse / 2:.4f} + {power_of_ten(model.alpha)} \\times {l1:.3f}"
                  f" = {mse / 2 + model.alpha * l1:.4f}$"),
        note(_sparsity_note(int(np.count_nonzero(w)), len(w))),
    ]


def _sgd_lines(model, mse: float) -> list[EquationLine]:
    """SGD's objective, its per-house update rule with η and α, the update count, and the averaging."""
    return [
        math_line(r"$\min_w\ \frac{1}{n}\sum_i \frac{1}{2}(y_i - \hat{y}_i)^2 + \frac{\alpha}{2}\Vert w \Vert_2^2$"),
        math_line(f"$\\approx {mse / 2:.4f}$  on the training houses"),
        heading("This step: stochastic gradient descent"),
        math_line(r"$w \leftarrow w - \eta\,[(\hat{y}_i - y_i)\,x_i + \alpha w]$"),
        note(f"η = {model.eta0:g}, α = {model.alpha:g}, one house at a time:"
             f" {int(model.t_) - 1:,} updates so far"),
        math_line(r"$\bar{w} = \frac{1}{t}\sum_{s \leq t} w_s$   (predictions use the average)"),
    ]


def _least_squares_lines(mse: float, rows: int) -> list[EquationLine]:
    """Least squares' objective on the rows fitted so far, and its closed-form solve."""
    return [
        math_line(r"$\min_{w,b}\ \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$"),
        math_line(f"$= {mse:.4f}$  on the {rows:,} houses seen"),
        heading("This step: exact least-squares solve"),
        math_line(r"$w = (X^T X)^{-1} X^T y$"),
        note("one closed-form solve, refit each frame on more houses"),
    ]


def training_lines(model, frame, splits: HousingSplits, rows: int) -> list[EquationLine]:
    """A linear model's training card lines: the formula with its biggest weights, then its objective.

    Args:
        model: A fitted ``LinearRegression``, ``Ridge``, ``Lasso`` or ``SGDRegressor``.
        frame: The step's ``TrainingFrame``.
        splits: The data, for the feature names.
        rows: How many training rows the model was fitted on.

    Returns:
        The lines, in card order.
    """
    b, w = _weights(model)
    mse = training_mse(frame)
    lines = [heading("Model"), math_line(r"$\hat{y} = b + \sum_j w_j x_j$"),
             *linear_terms(b, w, splits.feature_names),
             note("x = standardised (z-score) features, y in $100k"),
             heading("Training minimises")]
    if isinstance(model, Ridge):
        return lines + _ridge_objective(model, mse, rows)
    if isinstance(model, Lasso):
        return lines + _lasso_objective(model, mse)
    if isinstance(model, SGDRegressor):
        return lines + _sgd_lines(model, mse)
    return lines + _least_squares_lines(mse, rows)


def prediction_lines(model, x: np.ndarray, splits: HousingSplits) -> tuple[list[EquationLine], float]:
    """One house through the formula: each w_j × x_j term, largest push first, summed with b.

    Args:
        model: A fitted linear model.
        x: The house's scaled features.
        splits: The data, for the feature names.

    Returns:
        The card's lines (ending in the price and the check against ``model.predict``) and the
        worked prediction, in $100k.
    """
    b, w = _weights(model)
    contributions = w * x
    order = np.argsort(-np.abs(contributions))
    rows = [mono(f"{splits.feature_names[i]:<10} {w[i]:+.2f} × {x[i]:+.2f} = {contributions[i]:+.3f}")
            for i in order]
    value = b + float(contributions.sum())
    lines = [heading("Plug this house into the equation"),
             math_line(r"$\hat{y} = b + \sum_j w_j x_j$"),
             mono(f"{'b':<10} {'':>13} {b:+.3f}"), *rows,
             price_line(value), check_line(model, x, value)]
    return lines, value
