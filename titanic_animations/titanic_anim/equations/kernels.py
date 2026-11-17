"""Equations for the RBF-kernel SVM and Gaussian Naive Bayes.

The SVM scores a passenger by comparing it with its support vectors: f(x) = Σ α_i·y_i·K(x_i, x) + b.
The RBF kernel K = exp(−γ‖x − x_i‖²) is 1 for an identical passenger and fades towards 0 with
distance. f(x) > 0 means survived; this SVM gives no probability.

Gaussian Naive Bayes multiplies the prior π_y by one bell-curve density per feature. In log-odds
that is a sum: log(π_1/π_0), plus each feature's log-likelihood ratio, the log of the survivors'
density over the others'.
"""

import numpy as np

from titanic_anim.data import TitanicSplits
from titanic_anim.equations.cards import (
    P_SURVIVE,
    EquationLine,
    accuracy_note,
    heading,
    margin_check_line,
    math_line,
    mono,
    note,
    probability_ending,
    sigmoid,
    verdict_line,
)
from titanic_anim.passengers import outcome
from titanic_anim.trainers import class_separation

SHOWN_SUPPORT_VECTORS = 5
SHOWN_FEATURES = 8
SVM_RULE = r"$f(x) = \sum_{i \in SV} \alpha_i y_i\, K(x_i, x) + b$"
RBF = r"$K(x_i, x) = e^{-\gamma \Vert x - x_i \Vert^2}$"
NB_RULE = r"$P(y \mid x) \propto \pi_y \prod_j \mathcal{N}(x_j;\ \mu_{yj}, \sigma_{yj}^2)$"
NB_LOG_ODDS = r"$\log\frac{P(1 \mid x)}{P(0 \mid x)} = \log\frac{\pi_1}{\pi_0} + \sum_j \log\frac{\mathcal{N}_{1j}}{\mathcal{N}_{0j}}$"


def kernel_terms(model, x: np.ndarray) -> np.ndarray:
    """α_i y_i K(x_i, x) for every support vector; their sum plus b is decision_function(x).

    Args:
        model: A fitted RBF-kernel ``SVC``.
        x: The passenger's standardised features.

    Returns:
        One term per support vector: positive pulls towards survived, negative away.
    """
    distances = np.sum((model.support_vectors_ - x) ** 2, axis=1)
    return model.dual_coef_[0] * np.exp(-model.gamma * distances)


def svm_training(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """The SVM's decision function and kernel, its hinge-loss objective, and its support vectors.

    Args:
        model: The live ``SVC``.
        frame: This step's training frame.
        splits: Unused; every training explainer takes the same four arguments.
        rows: How many training passengers the model was fit on.

    Returns:
        The card's lines, top to bottom.
    """
    per_class = model.n_support_
    return [
        heading("Model"), math_line(SVM_RULE), math_line(RBF),
        note(f"γ = {model.gamma:g}, C = {model.C:g}  ·  f(x) > 0 → survived"),
        heading("Training minimises (hinge loss)"),
        math_line(r"$\frac{1}{2}\Vert w \Vert^2 + C \sum_i \max(0,\ 1 - y_i f(x_i))$"),
        note(f"mean hinge loss {frame.history[-1].train_loss:.3f} on the {rows:,} passengers seen"),
        heading("This step: refit on more passengers"),
        note(f"{int(per_class.sum()):,} of {rows:,} passengers are support vectors "
             f"({per_class[1]} survivors, {per_class[0]} others)"),
        note("only passengers on or inside the margin get α > 0; the rest could be deleted"),
        accuracy_note(frame),
    ]


def svm_prediction(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """Works one passenger through f(x): the strongest support-vector terms, the rest summed, and b.

    Args:
        model: The trained ``SVC``.
        x: The passenger's standardised features.
        splits: Unused; every prediction explainer takes the same three arguments.

    Returns:
        The card's lines, and the margin f(x) they arrive at.
    """
    terms = kernel_terms(model, x)
    value = float(terms.sum() + model.intercept_[0])
    order = np.argsort(-np.abs(terms))
    survived = model.dual_coef_[0] > 0  # α_i y_i is positive exactly for the survivors
    rows = [mono(f"SV {rank}: {outcome(int(survived[i])):<15} → {terms[i]:+.3f}")
            for rank, i in enumerate(order[:SHOWN_SUPPORT_VECTORS], start=1)]
    rest = float(terms[order[SHOWN_SUPPORT_VECTORS:]].sum())
    lines = [heading(f"Compare with all {len(terms)} support vectors"), math_line(SVM_RULE), math_line(RBF),
             *rows, note(f"… {max(len(terms) - SHOWN_SUPPORT_VECTORS, 0)} more, together {rest:+.3f}"),
             mono(f"b = {model.intercept_[0]:+.3f}"),
             math_line(f"$f(x) = {value:+.3f}$"),
             verdict_line(model, x, f"{value:+.3f} {'>' if value > 0 else '≤'} 0"),
             margin_check_line(model, x, value)]
    return lines, value


def log_likelihood_ratios(model, x: np.ndarray) -> np.ndarray:
    """Per feature: log N(x_j; survivors) − log N(x_j; others).

    Args:
        model: A fitted ``GaussianNB``.
        x: The passenger's standardised features.

    Returns:
        One ratio per feature: positive where the passenger's value is likelier among survivors.
    """
    log_densities = -0.5 * (np.log(2.0 * np.pi * model.var_) + (x - model.theta_) ** 2 / model.var_)
    return log_densities[1] - log_densities[0]


def nb_training(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """Naive Bayes' rule, its prior, and the features whose two means sit furthest apart.

    Args:
        model: The live ``GaussianNB``.
        frame: This step's training frame.
        splits: The splits, for the feature names.
        rows: Unused; every training explainer takes the same four arguments.

    Returns:
        The card's lines, top to bottom.
    """
    counts = model.class_count_.astype(int)
    gaps = class_separation(model)
    order = np.argsort(-np.abs(gaps))[:4]
    means = [mono(f"{splits.feature_names[i]:<10} μ survived {model.theta_[1, i]:+.2f}   "
                  f"others {model.theta_[0, i]:+.2f}") for i in order]
    return [
        heading("Model"), math_line(NB_RULE),
        note("'naive': every feature is treated as independent once you know the outcome"),
        heading("Training is counting"),
        note(f"π_survived = {model.class_prior_[1]:.3f}  ({counts[1]} of {counts.sum()} passengers)"),
        *means,
        note("a mean and a variance per feature and outcome: no loss, no iterations"),
        heading("This step: count more passengers"),
        accuracy_note(frame),
    ]


def nb_prediction(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """Adds up one passenger's evidence: the prior log-odds plus every feature's log-likelihood ratio.

    Args:
        model: The trained ``GaussianNB``.
        x: The passenger's standardised features.
        splits: The splits, for the feature names.

    Returns:
        The card's lines, and the P(survive) they arrive at.
    """
    ratios = log_likelihood_ratios(model, x)
    prior = float(np.log(model.class_prior_[1] / model.class_prior_[0]))
    log_odds = prior + float(ratios.sum())
    p = sigmoid(log_odds)
    order = np.argsort(-np.abs(ratios))[:SHOWN_FEATURES]
    rows = [mono(f"{splits.feature_names[i]:<10} {ratios[i]:+.3f}") for i in order]
    lines = [heading("Weigh the evidence, feature by feature"), math_line(NB_LOG_ODDS),
             mono(f"{'prior':<10} {prior:+.3f}"), *rows,
             math_line(f"$\\mathrm{{log\\ odds}} = {log_odds:+.3f}$,   ${P_SURVIVE} = \\sigma({log_odds:+.2f})$"),
             *probability_ending(model, x, p)]
    return lines, p
