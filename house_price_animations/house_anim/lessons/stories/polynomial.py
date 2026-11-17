"""Polynomial regression and its regularised forms, Ridge (L2) and Lasso (L1), on 30 districts.

All three use the same features: x, x², …, x¹² of standardised income, each rescaled. Every frame
is a separate least-squares (or penalised) fit; the sweep is the honest model-building sequence.

A polynomial is still a linear model: a weighted sum, but of the powers of the input rather than the
input alone, so the curve can bend. More powers fit the training districts ever better and, past
some degree, fit their noise. To make that over-fitting visible the three lessons train on a
subsample of 30 districts, one per income band. The polynomial lesson sweeps the degree from 1 to 12.
Ridge and Lasso keep degree 12 and sweep a penalty α on the weights instead: Ridge adds α × the sum
of squared weights to the error, which shrinks every weight smoothly; Lasso adds α × the sum of
|weights|, which switches weak terms off (their weight becomes exactly 0). In each sweep the
validation districts choose the final model. A fit whose solver does not converge raises
``LessonError`` rather than being shown.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, NullLocator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from animkit.lessons import theme
from animkit.lessons.timing import ease
from house_anim.data import read_only
from house_anim.lessons.config import POLY_SAMPLE, PRICE_TO_K
from house_anim.lessons.data import IncomeData, Query, subsample
from house_anim.lessons.lesson import (
    Explanation,
    Lesson,
    LessonError,
    Panel,
    Snapshot,
    Story,
    best_validation,
    income_k,
    make_snapshot,
    money_k,
    predict_one,
    verified,
)
from house_anim.lessons.lesson_diagrams import PolynomialDiagram
from house_anim.lessons.plots import K_FORMAT

MAX_DEGREE = 12
DEGREES = tuple(range(1, MAX_DEGREE + 1))
RIDGE_ALPHAS = tuple(float(a) for a in np.logspace(-5, 3, 25))
LASSO_ALPHAS = tuple(float(a) for a in np.logspace(-4, 0, 21))
LASSO_SETTINGS = dict(max_iter=2_000_000, tol=1e-4)
SYMLOG_K = 10.0  # weights within ±$10k are drawn on a linear scale, larger ones on a log scale
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
DIAGRAM_TITLES = {
    "polynomial_regression": "The model: each power of income, weighted and summed",
    "ridge": "Powers of income, weighted (L2 keeps weights small)",
    "lasso": "Powers of income, weighted (dashed = switched off)",
}
TERM_NAMES = tuple("x" if d == 1 else f"x{str(d).translate(SUPERSCRIPT)}" for d in DEGREES)


def symlog_ticks(limit: float) -> list[float]:
    """The weight panel's y ticks on its symlog scale: few enough labels for the small side panel.

    They are 0, the largest whole decade inside ±limit and, when there is room, one decade roughly
    halfway to it on the log part.

    Args:
        limit: The axis runs from −limit to +limit, in $k.

    Returns:
        The tick values, in increasing order, symmetric about 0.
    """
    top = float(10.0 ** np.floor(np.log10(max(limit, SYMLOG_K))))
    middle = float(10.0 ** np.floor((np.log10(top) + np.log10(SYMLOG_K)) / 2))
    levels = [top] if middle in (SYMLOG_K, top) else [middle, top]
    return sorted([0.0, *levels, *(-level for level in levels)])


def poly_pipeline(degree: int, regressor) -> Pipeline:
    """The polynomial model: standardise, expand into powers, standardise each power, then regress.

    Standardising each power puts the terms on one scale, so a penalty treats every weight alike.

    Args:
        degree: The highest power.
        regressor: The final linear model (``LinearRegression``, ``Ridge`` or ``Lasso``).

    Returns:
        The unfitted scikit-learn pipeline; its last step holds the weights.
    """
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), regressor)


@dataclass(frozen=True)
class CoefState:
    """One fit's weights, for the side panel and the diagram.

    Attributes:
        coef: One weight per term up to ``MAX_DEGREE`` ($100k); NaN where the term is not in the model.
        intercept: The fit's intercept ($100k).
    """

    coef: np.ndarray  # one weight per term up to MAX_DEGREE; NaN where the term is not in the model
    intercept: float


def _fit(model: Pipeline, data: IncomeData) -> Pipeline:
    """Fit, refusing to show a solver state that has not converged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            return model.fit(data.x_train.reshape(-1, 1), data.y_train)
        except ConvergenceWarning as exc:
            raise LessonError(f"{model[-1]!r} did not converge: {exc}") from exc


def _predictor(model: Pipeline) -> Callable[[np.ndarray], np.ndarray]:
    """The fitted pipeline as a predictor: raw incomes in, prices out."""
    return lambda x: model.predict(np.asarray(x, dtype=float).reshape(-1, 1))


def _coef_state(model: Pipeline) -> CoefState:
    """The fit's weights padded with NaN up to ``MAX_DEGREE``, and its intercept."""
    weights = model[-1].coef_
    coef = np.full(MAX_DEGREE, np.nan)
    coef[:len(weights)] = weights
    return CoefState(coef=read_only(coef), intercept=float(model[-1].intercept_))


def contributions(model: Pipeline, income: float) -> tuple[float, np.ndarray]:
    """Intercept and each term's push w_j·φ_j(x); together they are exactly the prediction.

    φ_j(x) is term j as the regressor sees it: the j-th power of the standardised income, itself
    standardised.

    Args:
        model: A fitted polynomial pipeline.
        income: The income ($10k).

    Returns:
        The intercept ($100k), and one push per term ($100k).
    """
    features = model[:-1].transform(np.array([[income]]))[0]
    return float(model[-1].intercept_), model[-1].coef_ * features


def _alpha_range(alphas: tuple[float, ...]) -> str:
    """The α sweep for the summary's config: its first and last α and how many steps."""
    return f"{_alpha(alphas[0])} to {_alpha(alphas[-1])}, {len(alphas)} log steps"


def _alpha(value: float) -> str:
    """An α in short scientific notation: 0.001 -> '1.0e-3'."""
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e")


class WeightsPanel(Panel):
    """One bar per polynomial term: its weight while training, its push on the price at inference.

    Weights span orders of magnitude, so the y-axis is symlog: linear within ±``SYMLOG_K``, logarithmic
    beyond. A weight Lasso set to exactly 0 is an empty circle.
    """

    def __init__(self, snapshots: tuple[Snapshot, ...], final: Pipeline):
        """Fixes the training axis to the largest weight any snapshot reaches, so the bars visibly shrink.

        Args:
            snapshots: Every fit in the sweep; their ``detail`` is a ``CoefState``.
            final: The fitted pipeline kept for inference.
        """
        self._final = final
        weights = np.concatenate([s.detail.coef for s in snapshots]) * PRICE_TO_K
        self._limit = float(np.nanmax(np.abs(weights))) * 1.3

    def _axes(self, ax: Axes, title: str, limit: float) -> None:
        """Sets up the symlog bar axes from −limit to +limit ($k), one slot per term."""
        theme.style_axes(ax, grid_axis="y")
        ax.set_yscale("symlog", linthresh=SYMLOG_K)
        ax.set_ylim(-limit, limit)
        ax.yaxis.set_major_locator(FixedLocator(symlog_ticks(limit)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.3, MAX_DEGREE + 0.7)
        ax.set_xticks(DEGREES, TERM_NAMES)
        ax.yaxis.set_major_formatter(K_FORMAT)
        ax.axhline(0.0, color=theme.INK_MUTED, linewidth=1.0)
        ax.set_title(title)

    @staticmethod
    def _bars(ax: Axes, values: np.ndarray) -> None:
        """Draws one bar per term (blue up, red down), skipping NaN and marking exact zeros with a circle."""
        for degree, value in zip(DEGREES, values):
            if np.isnan(value):
                continue
            if value == 0.0:  # Lasso switched this term off: an empty marker, not an invisible bar
                ax.scatter([degree], [0.0], s=70, facecolors=theme.PANEL, edgecolors=theme.INK_MUTED,
                           linewidths=1.6, zorder=4)
                continue
            ax.bar(degree, value, width=0.6, color=theme.MODEL if value > 0 else theme.ERROR, zorder=3)

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws one fit's weights, and how many are exactly 0.

        Args:
            ax: The side panel.
            snapshot: The fit being shown; its ``detail`` is a ``CoefState``.
        """
        state: CoefState = snapshot.detail
        self._axes(ax, "Weight on each term (log scale)", self._limit)
        self._bars(ax, state.coef * PRICE_TO_K)
        zeros = int(np.sum(state.coef == 0.0))
        note = f"{zeros} weights exactly 0 (○)" if zeros else "blue: pushes price up · red: down"
        ax.text(0.99, 0.04, note, transform=ax.transAxes, ha="right", fontsize=12,
                color=theme.INK_SECONDARY)

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Draws each term's push on the unseen district's price, appearing one term at a time.

        Args:
            ax: The side panel.
            query: The unseen district.
            t: How far through the reveal, from 0 to 1.
        """
        intercept, pushes = contributions(self._final, query.income)
        values = np.full(MAX_DEGREE, np.nan)
        shown = int(np.ceil(len(pushes) * ease(t)))
        values[:shown] = pushes[:shown] * PRICE_TO_K
        limit = max(float(np.max(np.abs(pushes))) * PRICE_TO_K * 1.3, SYMLOG_K * 2)
        self._axes(ax, "Each term's push on this house's price", limit)
        self._bars(ax, values)
        ax.text(0.99, 0.04, f"starts from the intercept {money_k(intercept)}", transform=ax.transAxes,
                ha="right", fontsize=12, color=theme.INK_SECONDARY)


def _explainer(model: Pipeline) -> Callable[[float], Explanation]:
    """The lesson's ``explain`` for the final pipeline: intercept plus every term's push."""
    predict = _predictor(model)
    degree = model[1].degree

    def explain(income: float) -> Explanation:
        """Works the polynomial's price for one income: the intercept, the biggest pushes, then the sum.

        Args:
            income: The income ($10k).

        Returns:
            The worked lines, checked against the model's own prediction.
        """
        intercept, pushes = contributions(model, income)
        worked = intercept + float(pushes.sum())
        biggest = np.argsort(-np.abs(pushes))[:3]
        lines = [f"terms x … {TERM_NAMES[degree - 1]} of income {income_k(income)}, each rescaled",
                 f"start: intercept {money_k(intercept)}",
                 "biggest pushes: " + ", ".join(f"{TERM_NAMES[j]} {money_k(pushes[j], signed=True)}"
                                                for j in biggest),
                 f"all {degree} terms together: {money_k(float(pushes.sum()), signed=True)}",
                 f"ŷ = {money_k(intercept)} {money_k(float(pushes.sum()), signed=True)} = {money_k(worked)}"]
        zeros = int(np.sum(model[-1].coef_ == 0.0))
        if zeros:
            lines.insert(1, f"{zeros} of {degree} weights are exactly zero, so those terms drop out")
        return verified(lines, worked, predict_one(predict, income))

    return explain


def _sweep(data: IncomeData, models: list[Pipeline], label: Callable[[int], str],
           progress: Callable[[int], float], note: Callable[[int, Pipeline], str]) -> tuple[Snapshot, ...]:
    """One snapshot per fitted model in the sweep, labelled, placed and noted by its index."""
    return tuple(make_snapshot(_predictor(model), data, label=label(i), progress=progress(i),
                               note=note(i, model), detail=_coef_state(model))
                 for i, model in enumerate(models))


def _largest_weight(snapshots: tuple[Snapshot, ...]) -> float:
    """The diagram's edge scale: the largest |weight| any snapshot reaches (missing terms are NaN)."""
    return max(float(np.nanmax(np.abs(s.detail.coef), initial=0.0)) for s in snapshots) or 1.0


def _lesson(story: Story, data: IncomeData, snapshots: tuple[Snapshot, ...], models: list[Pipeline],
            config: dict[str, str]) -> Lesson:
    """The lesson for a sweep: the fit with the best validation score is the final model."""
    best = best_validation(snapshots)
    final = models[best]
    return Lesson(story=story, data=data, snapshots=snapshots, final_index=best,
                  predict=_predictor(final), explain=_explainer(final),
                  panel=WeightsPanel(snapshots, final), config=config,
                  diagram=PolynomialDiagram(final, TERM_NAMES, DIAGRAM_TITLES[story.key], _largest_weight(snapshots)))


def _common_config(data: IncomeData) -> dict[str, str]:
    """The summary settings all three lessons share: the features, the sample and how the model is chosen."""
    return {"features": f"x … x^{MAX_DEGREE} of standardised MedInc, each standardised",
            "training_rows": f"{data.n_train} (one per income band, seed 42)",
            "selection": "lowest validation RMSE"}


def build_polynomial(full: IncomeData) -> Lesson:
    """Builds the polynomial-regression lesson: one least-squares fit per degree, 1 to ``MAX_DEGREE``.

    Args:
        full: The full income view; the lesson trains on a ``POLY_SAMPLE``-district subsample of it.

    Returns:
        The lesson, its final model the degree with the lowest validation RMSE.

    Raises:
        LessonError: If a fit did not converge.
    """
    data = subsample(full, POLY_SAMPLE)
    models = [_fit(poly_pipeline(d, LinearRegression()), data) for d in DEGREES]

    def note(i: int, model: Pipeline) -> str:
        """The callout for degree ``DEGREES[i]``: its shape and how many weights it fits.

        Args:
            i: The fit's index in the sweep.
            model: The fitted pipeline (unused: the note depends only on the degree).

        Returns:
            Two lines of callout text.
        """
        shape = "a straight line" if DEGREES[i] == 1 else f"a curve with {DEGREES[i]} bends allowed"
        return f"Degree {DEGREES[i]}: {shape}\n{DEGREES[i]} weights + an intercept, fitted by least squares"

    snaps = _sweep(data, models, lambda i: f"Degree {DEGREES[i]} of {MAX_DEGREE}",
                   lambda i: DEGREES[i], note)
    best = snaps[best_validation(snaps)]
    story = Story(
        key="polynomial_regression", title="Polynomial Regression",
        method="least squares on x, x², … x¹²",
        objective=("Give a straight line extra powers of income and it can bend. Watch what happens "
                   "as the degree climbs from 1 to 12: the curve chases the training districts, and "
                   "the validation districts reveal when bending turns into over-fitting."),
        initial=("Degree 1 is ordinary linear regression: one weight and an intercept. The red lines "
                 f"are its errors on the {data.n_train} training districts."),
        training=("Each frame is a fresh least-squares fit with one more power of income. Training "
                  "error can only fall; watch the orange validation curve on the right."),
        final=(f"Validation chose degree {best.progress:.0f}: train RMSE ${best.train_rmse_k:,.0f}k, "
               f"validation ${best.val_rmse_k:,.0f}k. Higher degrees fit the {data.n_train} districts "
               f"better but new districts worse."),
        inference=("The chosen polynomial prices an unseen district: every power of its income adds "
                   "a push up or down, starting from the intercept."),
        takeaway=("More flexibility always lowers training error. Only held-out data can tell you "
                  "when the extra bends are fitting noise."),
        limitation=(f"with only {data.n_train} districts the curve is fragile, and polynomials swing "
                    "wildly outside the incomes they were trained on."),
        progress_label="polynomial degree")
    return _lesson(story, data, snaps, models, {**_common_config(data), "model": "LinearRegression",
                                                "degrees": f"1-{MAX_DEGREE}"})


def build_ridge(full: IncomeData) -> Lesson:
    """Builds the Ridge lesson: one degree-``MAX_DEGREE`` fit per α in ``RIDGE_ALPHAS``, weakest first.

    Args:
        full: The full income view; the lesson trains on a ``POLY_SAMPLE``-district subsample of it.

    Returns:
        The lesson, its final model the α with the lowest validation RMSE.

    Raises:
        LessonError: If a fit did not converge.
    """
    data = subsample(full, POLY_SAMPLE)
    models = [_fit(poly_pipeline(MAX_DEGREE, Ridge(alpha=a)), data) for a in RIDGE_ALPHAS]

    def note(i: int, model: Pipeline) -> str:
        """The callout for α = ``RIDGE_ALPHAS[i]``: the penalty and the largest weight left.

        Args:
            i: The fit's index in the sweep.
            model: The fitted pipeline.

        Returns:
            Two lines of callout text.
        """
        largest = float(np.max(np.abs(model[-1].coef_))) * PRICE_TO_K
        return (f"α = {_alpha(RIDGE_ALPHAS[i])}: penalty on the sum of squared weights\n"
                f"largest weight ${largest:,.1f}k, none exactly zero")

    snaps = _sweep(data, models, lambda i: f"α = {_alpha(RIDGE_ALPHAS[i])}",
                   lambda i: RIDGE_ALPHAS[i], note)
    best = best_validation(snaps)
    story = Story(
        key="ridge", title="Ridge Regression",
        method="degree-12 polynomial with an L2 penalty",
        objective=("Keep all twelve powers of income, but charge the model for large weights. "
                   "Watch the penalty α grow: the wiggly curve calms down, the weights shrink "
                   "smoothly, and validation finds the sweet spot."),
        initial=(f"Almost no penalty (α = {_alpha(RIDGE_ALPHAS[0])}): the degree-12 curve bends to "
                 f"chase the {data.n_train} training districts."),
        training=("Each frame refits with a stronger penalty. Ridge minimises squared error + α × "
                  "(sum of squared weights), so every weight shrinks but never quite reaches zero."),
        final=(f"Validation chose α = {_alpha(RIDGE_ALPHAS[best])}: train RMSE "
               f"${snaps[best].train_rmse_k:,.0f}k, validation ${snaps[best].val_rmse_k:,.0f}k. "
               "Too much penalty flattens the curve towards the average."),
        inference=("The tamed polynomial prices an unseen district. Small weights mean small pushes, "
                   "so no single power of income can swing the price wildly."),
        takeaway=("Ridge trades a little training error for a smoother curve: shrinking every "
                  "weight reduces over-fitting without removing any feature."),
        limitation=("α must be tuned on held-out data, and Ridge never sets a weight to exactly zero, "
                    "so it cannot tell you which terms are unnecessary."),
        progress_label="penalty strength α (log scale)", log_progress=True)
    return _lesson(story, data, snaps, models, {**_common_config(data), "model": "Ridge",
                                                "alphas": _alpha_range(RIDGE_ALPHAS)})


def build_lasso(full: IncomeData) -> Lesson:
    """Builds the Lasso lesson: one degree-``MAX_DEGREE`` fit per α in ``LASSO_ALPHAS``, weakest first.

    Lasso's coordinate-descent solver runs with a high iteration limit, and a fit that still does not
    converge stops the lesson.

    Args:
        full: The full income view; the lesson trains on a ``POLY_SAMPLE``-district subsample of it.

    Returns:
        The lesson, its final model the α with the lowest validation RMSE.

    Raises:
        LessonError: If a fit did not converge.
    """
    data = subsample(full, POLY_SAMPLE)
    models = [_fit(poly_pipeline(MAX_DEGREE, Lasso(alpha=a, **LASSO_SETTINGS)), data)
              for a in LASSO_ALPHAS]

    def note(i: int, model: Pipeline) -> str:
        """The callout for α = ``LASSO_ALPHAS[i]``: the penalty and how many terms it switched off.

        Args:
            i: The fit's index in the sweep.
            model: The fitted pipeline.

        Returns:
            Two lines of callout text.
        """
        zeros = int(np.sum(model[-1].coef_ == 0.0))
        return (f"α = {_alpha(LASSO_ALPHAS[i])}: penalty on the sum of |weights|\n"
                f"{zeros} of {MAX_DEGREE} terms switched off (weight exactly 0)")

    snaps = _sweep(data, models, lambda i: f"α = {_alpha(LASSO_ALPHAS[i])}",
                   lambda i: LASSO_ALPHAS[i], note)
    best = best_validation(snaps)
    kept = int(np.sum(models[best][-1].coef_ != 0.0))
    story = Story(
        key="lasso", title="Lasso Regression",
        method="degree-12 polynomial with an L1 penalty",
        objective=("Lasso charges the model for the absolute size of its weights. As the penalty α "
                   "grows, whole terms are switched off, one by one: the model chooses its own features."),
        initial=(f"A tiny penalty (α = {_alpha(LASSO_ALPHAS[0])}): most of the twelve powers of "
                 "income are still in use and the curve bends freely."),
        training=("Each frame refits with a stronger penalty. Lasso minimises squared error + α × "
                  "(sum of |weights|), which pushes weak weights all the way to zero (○ on the right)."),
        final=(f"Validation chose α = {_alpha(LASSO_ALPHAS[best])}: only {kept} of {MAX_DEGREE} terms "
               f"survive. Train RMSE ${snaps[best].train_rmse_k:,.0f}k, validation "
               f"${snaps[best].val_rmse_k:,.0f}k."),
        inference=("The sparse polynomial prices an unseen district using only the terms Lasso kept; "
                   "the zeroed terms contribute nothing."),
        takeaway="Lasso shrinks weights and selects features at the same time: weak terms get exactly zero.",
        limitation=("with correlated features (x³ and x⁵ move together) Lasso may keep one almost "
                    "arbitrarily, and α still needs tuning on held-out data."),
        progress_label="penalty strength α (log scale)", log_progress=True)
    return _lesson(story, data, snaps, models, {**_common_config(data), "model": "Lasso",
                                                "alphas": _alpha_range(LASSO_ALPHAS),
                                                "solver": "coordinate descent, tol 1e-4, converged"})
