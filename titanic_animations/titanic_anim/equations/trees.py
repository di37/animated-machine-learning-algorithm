"""Equations for tree models: a decision tree, a random forest, gradient boosting (log-odds).

A decision tree's answer for x is the share of survivors in the leaf x lands in. It grows split by
split, each split chosen to make its two sides as pure as it can: it minimises the passengers'
weighted Gini impurity, G = 1 − Σ p_k², which is 0 for a node of one outcome. A random forest
averages many such trees' answers. Gradient boosting adds trees in log-odds,
F_M(x) = F_0 + ν·Σ h_m(x), starting from the base rate's log-odds F_0, and P(survive) = σ(F_M(x)).
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
    raw_value,
    sigmoid,
)
from titanic_anim.passengers import feature_text, split_label

MAX_PATH_ROWS = 7
VOTES_PER_ROW = 6
SHOWN_CORRECTIONS = 4
GINI_RULE = r"$\frac{n_L}{n}\,G_L + \frac{n_R}{n}\,G_R$,   $G = 1 - \sum_k p_k^2$"
LEAF_SHARE = f"${P_SURVIVE} = \\frac{{\\mathrm{{survivors\\ in}}\\ L(x)}}{{|L(x)|}}$"
FOREST_AVERAGE = f"${P_SURVIVE} = \\frac{{1}}{{T}}\\sum_{{t=1}}^{{T}} p_t(x)$"
BOOSTED_LOG_ODDS = r"$F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$"


def leaf_note(survivors: int, passengers: int) -> str:
    """What a leaf's training passengers say, in words.

    Args:
        survivors: How many of the leaf's training passengers survived.
        passengers: How many training passengers the leaf holds.

    Returns:
        For example "leaf: 12 of the 40 training passengers here survived". A leaf of one passenger is
        called a sign of overfitting.
    """
    if passengers == 1:
        fate = "survived" if survivors else "did not survive"
        return f"leaf: the single training passenger here {fate} (a sign of overfitting)"
    return f"leaf: {survivors} of the {passengers} training passengers here survived"


def survival_share(t, node: int) -> float:
    """The share of a node's training passengers who survived.

    Args:
        t: A fitted tree's ``tree_`` structure.
        node: The node's id.

    Returns:
        The share, from 0 to 1.
    """
    return float(t.value[node].ravel()[1])


def _root_split(tree, splits: TitanicSplits) -> list[EquationLine]:
    """The root's question and how much it lowers the Gini impurity, or a note that there is no split yet."""
    t = tree.tree_
    if t.node_count < 3:
        return [note("no split yet: one leaf holds every passenger")]
    left, right = t.children_left[0], t.children_right[0]
    n = t.n_node_samples
    after = (n[left] * t.impurity[left] + n[right] * t.impurity[right]) / n[0]
    feature = int(t.feature[0])
    question = split_label(splits.feature_names[feature], raw_value(splits, feature, t.threshold[0]))
    return [note(f"root split: {question}   Gini {t.impurity[0]:.3f} → {after:.3f}")]


def tree_training(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """The leaf-share rule, the Gini criterion with the root split, and the tree's depth and leaves.

    Args:
        model: The live ``DecisionTreeClassifier``.
        frame: This step's training frame.
        splits: The splits, to word the root's question in real units.
        rows: Unused; every training explainer takes the same four arguments.

    Returns:
        The card's lines, top to bottom.
    """
    return [
        heading("Model"), math_line(LEAF_SHARE),
        note("the share of survivors among the training passengers in the leaf L(x) that x lands in"),
        heading("Each split minimises (Gini impurity)"), math_line(GINI_RULE), *_root_split(model, splits),
        heading("This step: grow one level deeper"),
        note(f"depth {model.get_depth()}  ·  {model.get_n_leaves():,} leaves"),
        accuracy_note(frame),
        note("a widening gap between the two is overfitting"),
    ]


def _path_rows(model, x: np.ndarray, splits: TitanicSplits) -> list[EquationLine]:
    """One row per question on x's path, with the answer and x's value; at most ``MAX_PATH_ROWS``."""
    t = model.tree_
    path = model.decision_path(x[None, :]).indices  # node ids, root first
    rows = []
    for node in path[:-1]:
        feature = int(t.feature[node])
        name = splits.feature_names[feature]
        answer = "yes" if x[feature] <= t.threshold[node] else "no"
        question = split_label(name, raw_value(splits, feature, t.threshold[node]))
        rows.append(mono(f"{question:<18} {answer:<4} ({feature_text(name, raw_value(splits, feature, x[feature]))})"))
    shown = rows[:MAX_PATH_ROWS]
    if len(rows) > MAX_PATH_ROWS:
        shown.append(note(f"… {len(rows) - MAX_PATH_ROWS} more splits"))
    return shown


def tree_prediction(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """Walks one passenger down the tree, question by question, to its leaf's share of survivors.

    Args:
        model: The trained ``DecisionTreeClassifier``.
        x: The passenger's standardised features.
        splits: The splits, to word the questions and the passenger's values in real units.

    Returns:
        The card's lines, and the P(survive) they arrive at.
    """
    t = model.tree_
    leaf = int(model.apply(x[None, :])[0])
    passengers = int(t.n_node_samples[leaf])
    p = survival_share(t, leaf)
    survivors = round(p * passengers)
    lines = [heading("Answer the questions, top to bottom"), *_path_rows(model, x, splits),
             note(leaf_note(survivors, passengers)),
             math_line(LEAF_SHARE), *probability_ending(model, x, p)]
    return lines, p


def forest_training(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """The averaging rule, the Gini criterion, and how many trees, how deep, the forest has.

    Args:
        model: The live ``RandomForestClassifier``.
        frame: This step's training frame.
        splits: Unused; every training explainer takes the same four arguments.
        rows: Unused; every training explainer takes the same four arguments.

    Returns:
        The card's lines, top to bottom.
    """
    depths = [tree.get_depth() for tree in model.estimators_]
    leaves = [tree.get_n_leaves() for tree in model.estimators_]
    return [
        heading("Model: average many trees"), math_line(FOREST_AVERAGE),
        note("each p_t is one full-depth tree's leaf share, grown on its own bootstrap sample"),
        heading("Each tree's splits minimise"), math_line(GINI_RULE),
        heading("This step: add more trees"),
        note(f"T = {len(model.estimators_)} trees  ·  depth ≈ {np.mean(depths):.0f}  ·  "
             f"{np.mean(leaves):,.0f} leaves each"),
        accuracy_note(frame),
        note("averaging cancels each tree's noise, so validation accuracy steadies"),
    ]


def forest_prediction(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """Lists every tree's P(survive) for one passenger (the first twelve) and their average.

    Args:
        model: The trained ``RandomForestClassifier``.
        x: The passenger's standardised features.
        splits: Unused; every prediction explainer takes the same three arguments.

    Returns:
        The card's lines, and the P(survive) they arrive at.
    """
    votes = np.array([tree.predict_proba(x[None, :])[0, 1] for tree in model.estimators_])
    p = float(votes.mean())
    shown = votes[:2 * VOTES_PER_ROW]
    rows = [mono("  ".join(f"{v:.2f}" for v in shown[i:i + VOTES_PER_ROW]))
            for i in range(0, len(shown), VOTES_PER_ROW)]
    more = f"… {len(votes) - len(shown)} more  ·  " if len(votes) > len(shown) else ""
    lines = [heading(f"Average the {len(votes)} trees' P(survive)"), math_line(FOREST_AVERAGE), *rows,
             note(f"{more}{int(np.sum(votes > 0.5))} of {len(votes)} trees lean towards survived"),
             *probability_ending(model, x, p)]
    return lines, p


def initial_log_odds(model) -> float:
    """Boosting's starting point F_0: the log-odds of the training passengers' survival rate.

    Args:
        model: A fitted ``GradientBoostingClassifier``.

    Returns:
        log(p / (1 − p)), p being the share of training passengers who survived.
    """
    prior = float(model.init_.class_prior_[1])
    return float(np.log(prior / (1.0 - prior)))


def boosting_training(model, frame, splits: TitanicSplits, rows: int) -> list[EquationLine]:
    """The additive log-odds model, how each new tree is fit, and the training log loss so far.

    Args:
        model: The live ``GradientBoostingClassifier``.
        frame: This step's training frame.
        splits: Unused; every training explainer takes the same four arguments.
        rows: Unused; every training explainer takes the same four arguments.

    Returns:
        The card's lines, top to bottom.
    """
    start = initial_log_odds(model)
    return [
        heading("Model: start at the base rate, add corrections"),
        math_line(BOOSTED_LOG_ODDS), math_line(f"${P_SURVIVE} = \\sigma(F_M(x))$"),
        note(f"F₀ = log-odds of the base rate {sigmoid(start):.1%} = {start:+.3f}  ·  "
             f"ν = {model.learning_rate:g}  ·  trees of depth {model.max_depth}"),
        heading("Each new tree fits the residuals"),
        math_line(r"$r_i = y_i - \sigma(F_{m-1}(x_i))$"),
        note("leaf values are Newton steps on the log loss: Σ r / Σ p(1 − p)"),
        heading("This step: add more trees"),
        note(f"M = {model.n_estimators_} trees  ·  training log loss {frame.history[-1].train_loss:.4f}"),
        accuracy_note(frame),
    ]


def boosting_prediction(model, x: np.ndarray, splits: TitanicSplits) -> tuple[list[EquationLine], float]:
    """Adds up one passenger's log-odds: F_0, then every tree's correction times ν, then σ.

    Args:
        model: The trained ``GradientBoostingClassifier``.
        x: The passenger's standardised features.
        splits: Unused; every prediction explainer takes the same three arguments.

    Returns:
        The card's lines, and the P(survive) they arrive at.
    """
    start, rate = initial_log_odds(model), float(model.learning_rate)
    corrections = np.array([tree.predict(x[None, :])[0] for tree in model.estimators_[:, 0]])
    total = float(corrections.sum())
    log_odds = start + rate * total
    p = sigmoid(log_odds)
    first = "  ".join(f"h{m + 1} {c:+.3f}" for m, c in enumerate(corrections[:SHOWN_CORRECTIONS]))
    lines = [heading("Start at the base rate, add every correction"), math_line(BOOSTED_LOG_ODDS),
             mono(f"F0 = {start:+.3f}"), mono(first),
             note(f"… and {len(corrections) - SHOWN_CORRECTIONS} more trees")
             if len(corrections) > SHOWN_CORRECTIONS else note("(all trees shown)"),
             math_line(f"$\\nu \\sum_m h_m(x) = {rate:g} \\times ({total:+.3f}) = {rate * total:+.3f}$"),
             math_line(f"$F_M(x) = {log_odds:+.3f}$,   $\\sigma(F_M) = {p:.3f}$"),
             *probability_ending(model, x, p)]
    return lines, p
