"""Equations for the linear family: LogisticRegression and SGDClassifier with logistic loss.

Both models compute z = b + Σ_j w_j·x_j and P(survive) = σ(z); they differ in how they find w and
b. LogisticRegression minimises ½‖w‖² + C·Σ log loss over every passenger at once, so a larger C
weakens the penalty and lets the weights grow. SGDClassifier walks downhill one passenger at a
time, w ← w − η·[(p − y)·x + α·w], and predicts with the average of the weights it has visited.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier

from titanic_anim.data import TitanicSplits
from titanic_anim.equations.cards import (
    P_SURVIVE,
    SIGMOID,
    EquationLine,
    accuracy_note,
    heading,
    linear_terms,
    math_line,
    mono,
    note,
    power_of_ten,
    probability_ending,
    sigmoid,
)

LOGIT = r"$z = b + \sum_j w_j x_j$"
LOG_LOSS = r"$\ell_i = -[\,y_i \log p_i + (1 - y_i)\log(1 - p_i)\,]$"
COIN_FLIP_LOSS = float(np.log(2.0))


def weights(model) -> tuple[float, np.ndarray]:
    """The intercept and the weights of a linear model.

    Args:
        model: A fitted linear classifier.

    Returns:
        ``(b, w)``: the intercept, and one weight per feature.
    """
    return float(np.ravel(model.intercept_)[0]), np.ravel(model.coef_)


def _logistic_objective(model, loss: float, rows: int) -> list[EquationLine]:
    """LogisticRegression's objective with this frame's numbers: the penalty plus C × the summed log loss."""
    _, w = weights(model)
    penalty, data = 0.5 * float(np.sum(w ** 2)), rows * loss
    return [
        math_line(r"$\min_{w,b}\ \frac{1}{2}\Vert w \Vert_2^2 + C \sum_i \ell_i$"),
        math_line(LOG_LOSS),
        math_line(f"$\\approx {penalty:.3f} + {power_of_ten(model.C)} \\times {data:,.1f}"
                  f" = {penalty + model.C * data:,.3f}$"),
        note(f"mean log loss {loss:.3f} per passenger (a coin flip scores {COIN_FLIP_LOSS:.3f})"),
    ]


def _sgd_lines(model, loss: float) -> list[EquationLine]:
    """SGD's objective and update rule, with its learning rate, penalty and number of updates so far."""
    return [
        math_line(r"$\min_w\ \frac{1}{n}\sum_i \ell_i + \frac{\alpha}{2}\Vert w \Vert_2^2$"),
        math_line(LOG_LOSS),
        math_line(f"$\\approx {loss:.4f}$  mean log loss on the passengers seen"),
        heading("This step: stochastic gradient descent"),
        math_line(r"$w \leftarrow w - \eta\,[(p_i - y_i)\,x_i + \alpha w]$"),
        note(f"η = {model.eta0:g}, α = {model.alpha:g}, one passenger at a time:"
             f" {int(model.t_) - 1:,} updates so far (predictions use the averaged weights)"),
    ]


def training_lines(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """The linear model, its weights so far, and what training minimises.

    Args:
        model: The live ``LogisticRegression`` or ``SGDClassifier``.
        frame: This step's training frame.
        splits: The splits, for the feature names.
        rows: How many training passengers the model was fit on; LogisticRegression's summed log loss counts them.

    Returns:
        The card's lines, top to bottom.
    """
    b, w = weights(model)
    loss = frame.history[-1].train_loss
    lines = [heading("Model"), math_line(LOGIT), *linear_terms(b, w, splits.feature_names),
             math_line(f"${P_SURVIVE} = \\sigma(z)$"), math_line(SIGMOID),
             heading("Training minimises")]
    body = _sgd_lines(model, loss) if isinstance(model, SGDClassifier) else _logistic_objective(model, loss, rows)
    return lines + body + [accuracy_note(frame)]


def prediction_lines(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """Works one passenger through z = b + Σ w·x, the biggest contribution first, then σ(z).

    Args:
        model: The trained ``LogisticRegression`` or ``SGDClassifier``.
        x: The passenger's standardised features.
        splits: The splits, for the feature names.

    Returns:
        The card's lines, and the P(survive) they arrive at.
    """
    b, w = weights(model)
    contributions = w * x
    order = np.argsort(-np.abs(contributions))
    rows = [mono(f"{splits.feature_names[i]:<10} {w[i]:+.2f} × {x[i]:+.2f} = {contributions[i]:+.3f}")
            for i in order]
    z = b + float(contributions.sum())
    p = sigmoid(z)
    lines = [heading("Plug this passenger into the equation"), math_line(LOGIT),
             mono(f"{'b':<10} {'':>13} {b:+.3f}"), *rows,
             math_line(f"$z = {z:+.3f}$,   $\\sigma({z:+.3f}) = {p:.3f}$"),
             *probability_ending(model, x, p)]
    return lines, p
