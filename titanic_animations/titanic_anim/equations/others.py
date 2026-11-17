"""Equations for k-nearest neighbours, the MLP, and a generic fallback.

KNN has no formula to fit: P(survive) is the share of survivors among the k training passengers
nearest to x. The MLP passes x through two ReLU hidden layers to one sigmoid output, and its
prediction card does that forward pass by hand. Any other classifier gets a generic card that still
ends with the model's own probability and the check.
"""

import numpy as np

from titanic_anim.data import TitanicSplits
from titanic_anim.equations.cards import (
    P_SURVIVE,
    EquationLine,
    accuracy_note,
    heading,
    math_line,
    mono,
    note,
    probability_ending,
    sigmoid,
)
from titanic_anim.passengers import outcome

SHOWN_NEIGHBOURS = 5
KNN_RULE = f"${P_SURVIVE} = \\frac{{1}}{{k}}\\sum_{{i \\in N_k(x)}} y_i$"
MLP_LAYERS = (
    r"$h_1 = \mathrm{ReLU}(W_1^T x + b_1)$",
    r"$h_2 = \mathrm{ReLU}(W_2^T h_1 + b_2)$",
    f"${P_SURVIVE} = \\sigma(w_3^T h_2 + b_3)$",
)


def knn_training(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """KNN's voting rule, its k, and how many passengers it has stored.

    Args:
        model: The live ``KNeighborsClassifier``.
        frame: This step's training frame.
        splits: Unused; every training explainer takes the same four arguments.
        rows: Unused; every training explainer takes the same four arguments.

    Returns:
        The card's lines, top to bottom.
    """
    return [
        heading("Model: let the closest passengers vote"), math_line(KNN_RULE),
        math_line(r"$d(x, x_i) = \Vert x - x_i \Vert_2$  picks $N_k(x)$"),
        note(f"k = {model.n_neighbors} neighbours, distance on standardised features"),
        heading("Training minimises"),
        note("nothing: fitting only stores the training passengers"),
        heading("This step: remember more passengers"),
        note(f"{model.n_samples_fit_:,} passengers stored"),
        accuracy_note(frame),
    ]


def knn_prediction(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """Lists one passenger's nearest neighbours and how many of them survived.

    Args:
        model: The trained ``KNeighborsClassifier``.
        x: The passenger's standardised features.
        splits: The splits; the training labels say what happened to each neighbour.

    Returns:
        The card's lines, and the P(survive) they arrive at.
    """
    distances, indices = model.kneighbors(x[None, :])
    fates = splits.y_train[indices[0]]  # the model was fit on a prefix of the training rows
    p = float(fates.mean())
    rows = [mono(f"#{rank:<2} distance {d:5.2f}   {outcome(int(y))}")
            for rank, (d, y) in enumerate(zip(distances[0], fates), start=1)][:SHOWN_NEIGHBOURS]
    lines = [heading(f"Find the {len(fates)} most similar passengers"), math_line(KNN_RULE), *rows,
             note(f"… and {len(fates) - SHOWN_NEIGHBOURS} more  ·  {int(fates.sum())} of {len(fates)} survived"),
             *probability_ending(model, x, p)]
    return lines, p


def _parameter_count(model) -> int:
    """How many weights and biases the network has."""
    return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))


def mlp_training(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """The network's layers, its loss this epoch, and the Adam update rule.

    Args:
        model: The live ``MLPClassifier``.
        frame: This step's training frame.
        splits: Unused; every training explainer takes the same four arguments.
        rows: Unused; every training explainer takes the same four arguments.

    Returns:
        The card's lines, top to bottom.
    """
    sizes = " → ".join(str(n) for n in (model.coefs_[0].shape[0], *model.hidden_layer_sizes, 1))
    return [
        heading("Model: two hidden layers, sigmoid output"), *(math_line(layer) for layer in MLP_LAYERS),
        note(f"{_parameter_count(model):,} weights and biases  ·  layers {sizes}"),
        heading("Training minimises"),
        math_line(r"$\frac{1}{n}\sum_i \ell_i + \frac{\alpha}{2n}\Vert W \Vert_2^2$   (log loss)"),
        math_line(f"$= {model.loss_:.4f}$  (mean over this epoch's mini-batches)"),
        heading("This step: one epoch of Adam"),
        math_line(r"$W \leftarrow W - \eta\,\hat{m} / (\sqrt{\hat{v}} + \epsilon)$"),
        note(f"η = {model.learning_rate_init:g}  ·  gradients from {model.t_:,} passengers so far"),
        accuracy_note(frame),
    ]


def forward(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(h1, h2, z) for one passenger; P(survive) = σ(z).

    It repeats the MLP's forward pass by hand, for a network with two ReLU hidden layers.

    Args:
        model: A fitted ``MLPClassifier`` with two hidden layers.
        x: The passenger's standardised features.

    Returns:
        The two hidden layers' activations and the output unit's log-odds z.
    """
    (w1, w2, w3), (b1, b2, b3) = model.coefs_, model.intercepts_
    h1 = np.maximum(x @ w1 + b1, 0.0)
    h2 = np.maximum(h1 @ w2 + b2, 0.0)
    return h1, h2, float(h2 @ w3[:, 0] + b3[0])


def mlp_prediction(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """Works one passenger through the network: units on in each layer, the strongest pushes, then σ(z).

    Args:
        model: The trained ``MLPClassifier``.
        x: The passenger's standardised features.
        splits: Unused; every prediction explainer takes the same three arguments.

    Returns:
        The card's lines, and the P(survive) they arrive at.
    """
    h1, h2, z = forward(model, x)
    pushes = h2 * model.coefs_[2][:, 0]
    p = sigmoid(z)
    top = np.argsort(-np.abs(pushes))[:3]
    lines = [heading("Forward pass for this passenger"), *(math_line(layer) for layer in MLP_LAYERS),
             mono(f"h1: {np.count_nonzero(h1)}/{h1.size} units on, largest {h1.max():.2f}"),
             mono(f"h2: {np.count_nonzero(h2)}/{h2.size} units on, largest {h2.max():.2f}"),
             mono("top pushes: " + "  ".join(f"u{i} {pushes[i]:+.2f}" for i in top)),
             mono(f"b3 = {model.intercepts_[2][0]:+.3f}"),
             math_line(f"$z = {z:+.3f}$,   $\\sigma(z) = {p:.3f}$"),
             *probability_ending(model, x, p)]
    return lines, p


def generic_training(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """A placeholder card for a classifier with no explainer of its own.

    Args:
        model: The live model.
        frame: This step's training frame.
        splits: Unused; every training explainer takes the same four arguments.
        rows: Unused; every training explainer takes the same four arguments.

    Returns:
        The card's lines, top to bottom.
    """
    return [heading("Model"), math_line(f"${P_SURVIVE} = f(x)$"), note(type(model).__name__),
            accuracy_note(frame)]


def generic_prediction(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """A placeholder card for a classifier with no explainer of its own; it still ends with the check.

    Args:
        model: The trained model; its ``predict_proba`` gives the probability.
        x: The passenger's standardised features.
        splits: Unused; every prediction explainer takes the same three arguments.

    Returns:
        The card's lines, and the P(survive) they arrive at.
    """
    p = float(model.predict_proba(x[None, :])[0, 1])
    return [heading("Prediction"), math_line(f"${P_SURVIVE} = f(x)$"), note(type(model).__name__),
            *probability_ending(model, x, p)], p
