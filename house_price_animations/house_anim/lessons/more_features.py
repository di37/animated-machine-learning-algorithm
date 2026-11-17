"""The more-features chapter: the lesson's algorithm given all eight features, against its income-only model.

The same algorithm is refit on the full training split with all eight features, then scored on the same
test districts as the lesson's one-feature model. The result is animkit's ``Comparison``: one bar for
always guessing the average, one for the lesson's model, one for the all-features model.

Settings carry over where they mean the same thing with eight inputs (tree size, number of trees,
learning rate, network shape). Choices tuned for one input (polynomial degree, Ridge/Lasso α, k) are
re-chosen on validation, never on test. The headline claims a difference only when a paired bootstrap
says it is not noise. The factories take the features to use, so the 3-D view reuses them for two inputs.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from animkit.lessons.evidence import paired_rmse
from animkit.lessons.fitting import EpochTrainer
from animkit.lessons.lesson import Comparison, ComparisonRow
from house_anim.config import RANDOM_STATE
from house_anim.lessons.config import PRICE_TO_K
from house_anim.lessons.data import FullFeatures
from house_anim.lessons.lesson import Lesson, baseline_rmse_k, rmse_k
from house_anim.lessons.stories import boosting, forest, knn, mlp, tree

FEATURE_LIST = "income, house age, rooms, bedrooms, population, occupancy, latitude, longitude"
RIDGE_ALPHAS = tuple(float(a) for a in np.logspace(-3, 3, 13))
LASSO_ALPHAS = tuple(float(a) for a in np.logspace(-4, 0, 9))

Factory = Callable[[Lesson, FullFeatures], tuple[Any, str]]  # (lesson, features to use) -> unfitted model, settings


def chosen_on_validation(make: Callable[[Any], Any], options: tuple, view: FullFeatures) -> tuple[Any, Any]:
    """Fit one model per option on the training rows; keep the option with the lowest validation RMSE.

    Args:
        make: Builds an unfitted model from one option (a degree, an α, a k).
        options: The options to try, in order; the first wins a tie.
        view: The features to fit on and validate with.

    Returns:
        A fresh, unfitted model built from the best option, and that option.
    """
    errors = [rmse_k(make(option).fit(view.X_train, view.y_train).predict(view.X_val), view.y_val)
              for option in options]
    best = options[int(np.argmin(errors))]
    return make(best), best


def _polynomial(lesson: Lesson, view: FullFeatures) -> tuple[Any, str]:
    """Polynomial regression with its degree re-chosen on validation, up to the lesson's own choice.

    A degree picked for one feature lets eight features' extra terms extrapolate wildly on outlier districts.
    """
    degrees = tuple(range(1, int(lesson.final.progress) + 1))
    model, degree = chosen_on_validation(lambda d: make_pipeline(PolynomialFeatures(d), LinearRegression()),
                                         degrees, view)
    return model, f"degree {degree}, chosen on validation from 1 to {degrees[-1]}"


def _penalised(name: str, make: Callable[[float], Any], alphas: tuple[float, ...]) -> Factory:
    """A factory for Ridge or Lasso on the plain features (no polynomial terms), α chosen on validation."""
    def factory(lesson: Lesson, view: FullFeatures) -> tuple[Any, str]:
        """Chooses α on ``view``'s validation rows and returns the unfitted model.

        Args:
            lesson: The lesson (unused: the α grid is this module's own).
            view: The features to use.

        Returns:
            The unfitted model with the chosen α, and a line saying how it was set up.
        """
        model, alpha = chosen_on_validation(make, alphas, view)
        return model, f"{name} on linear terms only (no polynomial expansion), α = {alpha:g} chosen on validation"
    return factory


def _knn(lesson: Lesson, view: FullFeatures) -> tuple[Any, str]:
    """A kNN model with k re-chosen on validation from the lesson's k values no larger than the training set."""
    options = tuple(k for k in knn.K_VALUES if k <= len(view.y_train))
    model, k = chosen_on_validation(lambda k: KNeighborsRegressor(n_neighbors=k), options, view)
    return model, f"k = {k}, re-chosen on validation for {len(view.names)} features"


RICHER: dict[str, Factory] = {
    "linear_regression": lambda lesson, view: (LinearRegression(), "ordinary least squares"),
    "polynomial_regression": _polynomial,
    "ridge": _penalised("Ridge", lambda a: Ridge(alpha=a), RIDGE_ALPHAS),
    "lasso": _penalised("Lasso", lambda a: Lasso(alpha=a, max_iter=100_000), LASSO_ALPHAS),
    "decision_tree": lambda lesson, view: (
        DecisionTreeRegressor(max_leaf_nodes=len(lesson.final.detail.leaves),
                              min_samples_leaf=tree.min_leaf(len(view.y_train)), random_state=RANDOM_STATE),
        f"{len(lesson.final.detail.leaves)} leaves, grown best-first"),
    "random_forest": lambda lesson, view: (
        RandomForestRegressor(n_estimators=forest.TREE_COUNTS[-1], min_samples_leaf=forest.min_leaf(len(view.y_train)),
                              n_jobs=1, random_state=RANDOM_STATE),
        f"{forest.TREE_COUNTS[-1]} trees"),
    "gradient_boosting": lambda lesson, view: (
        GradientBoostingRegressor(n_estimators=boosting.N_TREES, learning_rate=boosting.LEARNING_RATE,
                                  max_depth=boosting.TREE_DEPTH, min_samples_leaf=boosting.min_leaf(len(view.y_train)),
                                  random_state=RANDOM_STATE),
        f"{boosting.N_TREES} trees at learning rate {boosting.LEARNING_RATE}"),
    "knn": _knn,
    "mlp": lambda lesson, view: (
        EpochTrainer(MLPRegressor(**mlp.MLP_SETTINGS), mlp.EPOCHS, seed=RANDOM_STATE),
        f"{len(view.names)} → {' → '.join(map(str, mlp.HIDDEN))} → 1, {mlp.EPOCHS} epochs of Adam, one run"),
}


def richer_model(lesson: Lesson, view: FullFeatures | None = None) -> tuple[Any, str]:
    """An unfitted model of the lesson's algorithm for ``view``'s features, and how it was set up.

    Args:
        lesson: The lesson; its story's key names the algorithm, and its final model supplies the
            settings that carry over (tree size, number of trees, ...).
        view: The features the model will see; None uses all eight.

    Returns:
        The unfitted model, and a line describing its settings.
    """
    return RICHER[lesson.story.key](lesson, lesson.data.full if view is None else view)


def _headline(own: float, richer: float, better: bool, worse: bool) -> str:
    """The chapter's headline; it claims a gain or a loss only when the bootstrap says so and the $k differ."""
    shown_own, shown_richer = f"${own:,.0f}k", f"${richer:,.0f}k"
    if better and shown_own != shown_richer:
        return f"Seven more inputs cut the typical error from {shown_own} to {shown_richer}."
    if worse and shown_own != shown_richer:
        return f"Here the extra inputs make it worse: {shown_richer} against {shown_own} with income alone."
    return f"About the same: {shown_richer} with all 8 features against {shown_own} with income alone."


def compare(lesson: Lesson) -> Comparison:
    """Fits the lesson's algorithm on all eight features and scores it against the lesson's own model.

    Both models are scored on the same test districts, and a paired bootstrap of their errors decides
    whether the difference is more than noise.

    Args:
        lesson: The lesson being played.

    Returns:
        animkit's ``Comparison``: the baseline, lesson and all-features rows, with the headline and note.
    """
    data, full = lesson.data, lesson.data.full
    model, settings = richer_model(lesson)
    model.fit(full.X_train, full.y_train)
    richer_pred, own_pred = model.predict(full.X_test), lesson.predict(data.x_test)
    richer, own = rmse_k(richer_pred, data.y_test), rmse_k(own_pred, data.y_test)
    baseline = baseline_rmse_k(data, data.y_test)
    verdict = paired_rmse(own_pred - data.y_test, richer_pred - data.y_test,
                          show=lambda gain: f"${gain * PRICE_TO_K:,.1f}k", seed=RANDOM_STATE)
    trained = f"median income only, {data.n_train:,} districts" if data.sample_note else "median income only"
    rows = (ComparisonRow("Always guess the average", baseline, f"${baseline:,.0f}k", "no inputs at all", "baseline"),
            ComparisonRow("This lesson's model", own, f"${own:,.0f}k", trained, "lesson"),
            ComparisonRow("Same algorithm, all 8 features", richer, f"${richer:,.0f}k", FEATURE_LIST, "richer"))
    sample = f" (the lesson itself fitted only {data.n_train})" if data.sample_note else ""
    note = (f"Both are scored on the same {len(data.y_test):,} test districts; {verdict.evidence}. The all-features "
            f"model ({settings}) trains on all {len(full.y_train):,} training districts{sample}.")
    return Comparison(metric="Test RMSE: typical error in $ (lower is better)", higher_is_better=False,
                      rows=rows, headline=_headline(own, richer, verdict.better, verdict.worse), note=note,
                      takeaway=f"Same algorithm with all 8 features: ${richer:,.0f}k test error "
                               f"(income only: ${own:,.0f}k).")
