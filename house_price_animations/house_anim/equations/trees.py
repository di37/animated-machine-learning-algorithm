"""Equations for tree models: a single decision tree, a random forest, gradient boosting.

- A regression tree predicts the mean price of the training houses in the leaf a house falls
  into. Each split picks the feature and threshold that most reduce the children's weighted
  variance (``SPLIT_RULE``); for squared error, a node's impurity is its variance.
- A random forest averages many full-depth trees, each grown on its own bootstrap sample; the
  average cancels much of each tree's noise.
- Gradient boosting starts at the mean price F₀ and adds shallow trees one by one, each fitted to
  the residuals of the ones before and scaled by the learning rate ν.

Thresholds are stored on the scaled axes, so the cards convert them (and the house's values)
back to real units with ``raw_value``.

Every explainer takes the same arguments, so the cards can call any of them; an explainer that
has no use for one (``splits``, say) still accepts it.
"""

import numpy as np

from house_anim.data import HousingSplits
from house_anim.equations.cards import (
    EquationLine,
    check_line,
    heading,
    math_line,
    mono,
    note,
    price_line,
    raw_value,
    training_mse,
)

MAX_PATH_ROWS = 7
VOTES_PER_ROW = 6
SHOWN_CORRECTIONS = 4
SPLIT_RULE = r"$\frac{n_L}{n}\,\mathrm{Var}(y_L) + \frac{n_R}{n}\,\mathrm{Var}(y_R)$"
LEAF_MEAN = r"$\hat{y}(x) = \frac{1}{|L(x)|}\sum_{i \in L(x)} y_i$"


def _leaf_note(houses: int) -> str:
    """What a leaf's value is: the mean of its training houses, or a single house (overfitting)."""
    if houses == 1:
        return "leaf: the single training house that landed here (a sign of overfitting)"
    return f"leaf: mean of the {houses:,} training houses that landed here"


def _root_split(tree, splits: HousingSplits) -> list[EquationLine]:
    """The root's split in real units, and the variance before and after it (the children weighted by size)."""
    t = tree.tree_
    if t.node_count < 3:
        return [note("no split yet: one leaf holds every house")]
    left, right = t.children_left[0], t.children_right[0]
    n = t.n_node_samples
    after = (n[left] * t.impurity[left] + n[right] * t.impurity[right]) / n[0]
    feature = int(t.feature[0])
    threshold = raw_value(splits, feature, t.threshold[0])
    name = splits.feature_names[feature]
    return [math_line(f"root: $\\mathrm{{{name}}} \\leq {threshold:.2f}$,  "
                      f"$\\mathrm{{Var}}$ {t.impurity[0]:.3f} → {after:.3f}")]


def tree_training(model, frame, splits: HousingSplits) -> list[EquationLine]:
    """A decision tree's training card lines: leaf means, the split rule, the root split, depth and fit.

    Args:
        model: A fitted ``DecisionTreeRegressor``.
        frame: The step's ``TrainingFrame``, for the training and validation MSE.
        splits: The data, to read the root split in real units.

    Returns:
        The lines, in card order.
    """
    return [
        heading("Model"), math_line(LEAF_MEAN),
        note("the mean price of the training houses in the leaf L(x) that x falls into"),
        heading("Each split minimises"), math_line(SPLIT_RULE), *_root_split(model, splits),
        heading("This step: grow one level deeper"),
        note(f"depth {model.get_depth()}  ·  {model.get_n_leaves():,} leaves  ·  "
             f"training MSE {training_mse(frame):.4f}"),
        note(f"validation MSE {frame.val_scores.rmse ** 2:.4f}: the gap is overfitting"),
    ]


def tree_prediction(model, x: np.ndarray, splits: HousingSplits) -> tuple[list[EquationLine], float]:
    """One house down the tree: each split on its path in real units, then its leaf's mean price.

    Args:
        model: A fitted ``DecisionTreeRegressor``.
        x: The house's scaled features.
        splits: The data, for feature names and real units.

    Returns:
        The card's lines (at most ``MAX_PATH_ROWS`` splits shown) and the leaf's value, in $100k.
    """
    t = model.tree_
    path = model.decision_path(x[None, :]).indices  # node ids, root first
    rows = []
    for node in path[:-1]:
        feature = int(t.feature[node])
        goes_left = x[feature] <= t.threshold[node]
        rows.append(mono(f"{splits.feature_names[feature]:<10} {raw_value(splits, feature, x[feature]):8.2f} "
                         f"{'≤' if goes_left else '>'} {raw_value(splits, feature, t.threshold[node]):8.2f}"))
    leaf = path[-1]
    value = float(t.value[leaf].ravel()[0])
    shown = rows[:MAX_PATH_ROWS]
    if len(rows) > MAX_PATH_ROWS:
        shown.append(note(f"… {len(rows) - MAX_PATH_ROWS} more splits"))
    lines = [heading("Follow the splits (real units)"), *shown,
             note(_leaf_note(int(t.n_node_samples[leaf]))),
             price_line(value), check_line(model, x, value)]
    return lines, value


def forest_training(model, frame, splits: HousingSplits) -> list[EquationLine]:
    """A random forest's training card lines: the average of T trees, their size, and the fit.

    Args:
        model: A fitted ``RandomForestRegressor``.
        frame: The step's ``TrainingFrame``, for the training and validation MSE.
        splits: Unused; every training explainer takes it.

    Returns:
        The lines, in card order.
    """
    depths = [tree.get_depth() for tree in model.estimators_]
    leaves = [tree.get_n_leaves() for tree in model.estimators_]
    return [
        heading("Model: average many trees"),
        math_line(r"$\hat{y}(x) = \frac{1}{T}\sum_{t=1}^{T} h_t(x)$"),
        note("each h_t is a full-depth tree on its own bootstrap sample"),
        heading("Each tree's splits minimise"), math_line(SPLIT_RULE),
        heading("This step: add more trees"),
        note(f"T = {len(model.estimators_)} trees  ·  depth ≈ {np.mean(depths):.0f}  ·  "
             f"{np.mean(leaves):,.0f} leaves each"),
        note(f"training MSE {training_mse(frame):.4f}  ·  validation MSE {frame.val_scores.rmse ** 2:.4f}"),
        note("averaging cancels each tree's noise, so validation error falls"),
    ]


def forest_prediction(model, x: np.ndarray, splits: HousingSplits) -> tuple[list[EquationLine], float]:
    """One house through the forest: every tree's vote (the first dozen shown), their spread and average.

    Args:
        model: A fitted ``RandomForestRegressor``.
        x: The house's scaled features.
        splits: Unused; every prediction explainer takes it.

    Returns:
        The card's lines and the mean vote, in $100k.
    """
    votes = np.array([tree.predict(x[None, :])[0] for tree in model.estimators_])
    value = float(votes.mean())
    shown = votes[:2 * VOTES_PER_ROW]
    rows = [mono("  ".join(f"{v:.2f}" for v in shown[i:i + VOTES_PER_ROW]))
            for i in range(0, len(shown), VOTES_PER_ROW)]
    lines = [heading(f"Average the {len(votes)} tree votes"),
             math_line(r"$\hat{y}(x) = \frac{1}{T}\sum_{t=1}^{T} h_t(x)$"), *rows,
             note(f"… {len(votes) - len(shown)} more  ·  min {votes.min():.2f}  ·  max {votes.max():.2f}"
                  f"  ·  spread σ {votes.std():.2f}") if len(votes) > len(shown) else
             note(f"min {votes.min():.2f}  ·  max {votes.max():.2f}  ·  spread σ {votes.std():.2f}"),
             price_line(value), check_line(model, x, value)]
    return lines, value


def _initial(model) -> float:
    """Gradient boosting's starting guess F₀: the constant its initial estimator predicts (the mean price)."""
    return float(np.ravel(model.init_.constant_)[0])


def boosting_training(model, frame, splits: HousingSplits) -> list[EquationLine]:
    """Gradient boosting's training card lines: F₀ plus ν times the trees, fitted to residuals.

    It also prints the variance of the training prices: the MSE of F₀ alone, to compare with the
    MSE now.

    Args:
        model: A fitted ``GradientBoostingRegressor``.
        frame: The step's ``TrainingFrame``, for the training MSE.
        splits: The data, for the training prices.

    Returns:
        The lines, in card order.
    """
    return [
        heading("Model: start at the mean, add corrections"),
        math_line(r"$F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$"),
        note(f"F₀ = {_initial(model):.3f} (mean price)  ·  ν = {model.learning_rate:g}  ·  "
             f"trees of depth {model.max_depth}"),
        heading("Each new tree fits the residuals"),
        math_line(r"$r_i = y_i - F_{m-1}(x_i)$"),
        math_line(r"$h_m \leftarrow$ least-squares tree fit to $r$"),
        heading("This step: add more trees"),
        math_line(f"$\\frac{{1}}{{n}}\\sum_i (y_i - F_M(x_i))^2 = {training_mse(frame):.4f}$"),
        note(f"M = {model.n_estimators_} trees  ·  F₀ alone scores {np.var(splits.y_train):.4f}"),
    ]


def boosting_prediction(model, x: np.ndarray, splits: HousingSplits) -> tuple[list[EquationLine], float]:
    """One house through boosting: F₀, every tree's correction (the first few shown), times ν, summed.

    Args:
        model: A fitted ``GradientBoostingRegressor``.
        x: The house's scaled features.
        splits: Unused; every prediction explainer takes it.

    Returns:
        The card's lines and F_M(x), in $100k.
    """
    initial, rate = _initial(model), float(model.learning_rate)
    corrections = np.array([tree.predict(x[None, :])[0] for tree in model.estimators_[:, 0]])
    total = float(corrections.sum())
    value = initial + rate * total
    first = "  ".join(f"h{m + 1} {c:+.3f}" for m, c in enumerate(corrections[:SHOWN_CORRECTIONS]))
    lines = [heading("Start at the mean, add every correction"),
             math_line(r"$F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$"),
             mono(f"F0 = {initial:.3f}"), mono(first),
             note(f"… and {len(corrections) - SHOWN_CORRECTIONS} more trees")
             if len(corrections) > SHOWN_CORRECTIONS else note("(all trees shown)"),
             math_line(f"$\\nu \\sum_m h_m(x) = {rate:g} \\times ({total:+.3f}) = {rate * total:+.3f}$"),
             price_line(value, symbol="F_M(x)"), check_line(model, x, value)]
    return lines, value
