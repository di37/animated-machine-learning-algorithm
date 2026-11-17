"""The 3-D chapter for a house-price lesson: the lesson's algorithm refitted on income and house age.

The same algorithm is refitted on two inputs, median income and house age, and its predicted price is
drawn as height over both: a sheet over a floor of (income, age), as animkit's ``Surface`` draws it.

The lesson itself learns from income alone, so this is a second, real model. It trains on every training
district with the settings of the all-features comparison (more_features.RICHER), re-chosen on
validation where they were tuned for one input. Ridge and Lasso keep the lesson's degree-12 polynomial,
now of both inputs, so they stay penalised polynomials rather than flat planes. House age is the second
input because, of the seven others, it lowers a straight line's validation error the most.

Its score is judged against a control: the same recipe refitted on income alone, on the same rows. Some
lessons train on only 30 districts, so comparing with the lesson's own model would credit house age with
the gain from more data.

``prepare_surface`` builds the chapter: it fits both models, evaluates the sheet on a grid, picks the
dots and their error sticks, places the unseen district, and writes the side notes and scores. A model
whose solver does not converge raises ``LessonError`` rather than being drawn.
"""

import warnings
from typing import Any

import numpy as np
from matplotlib.colors import Normalize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge

from animkit.lessons import theme
from animkit.lessons.evidence import paired_rmse
from animkit.lessons.surface import Axis, Dots, Marker, Sheet, Surface
from house_anim.config import RANDOM_STATE
from house_anim.lessons.config import FEATURE, INCOME_TO_K, PRICE_TO_K, X_RANGE, Y_RANGE_K
from house_anim.lessons.data import FullFeatures, select_columns
from house_anim.lessons.lesson import Lesson, LessonError, money_k, predict_one, rmse_k
from house_anim.lessons.more_features import RICHER, Factory, chosen_on_validation
from house_anim.lessons.stories import polynomial

SECOND_FEATURE = "HouseAge"
INPUTS = (FEATURE, SECOND_FEATURE)
AGE_RANGE = (1.0, 52.0)  # years; the census tops house age out at 52
GRID_SIZE = 44  # grid points per input
DOTS_SHOWN = 320  # training districts drawn as dots
STICKS = 70  # of those, how many get an error stick
INCOME_TICKS = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
AGE_TICKS = (10.0, 20.0, 30.0, 40.0, 50.0)
PRICE_TICKS_K = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0)
DEGREE = polynomial.MAX_DEGREE

SHAPES = {
    "linear_regression": ("With two inputs, linear regression is a flat, tilted plane: a fixed step in price for "
                          "every $10k of income and for every year of house age."),
    "polynomial_regression": ("The polynomial of both inputs is a sheet; its degree, chosen on validation, sets "
                              "how much it can curve."),
    "ridge": (f"Ridge fits a degree-{DEGREE} polynomial of both inputs; the penalty α, chosen on validation, "
              "decides how smooth the sheet stays."),
    "lasso": (f"Lasso fits the same degree-{DEGREE} polynomial of both inputs and switches weak terms off; the "
              "penalty α, chosen on validation, decides how many."),
    "decision_tree": ("A tree cuts the floor into rectangles and gives each one a flat height, so the sheet is a "
                      "set of flat terraces."),
    "random_forest": "A forest averages many trees' terraces, so the sheet is a staircase of many small steps.",
    "gradient_boosting": "Boosting adds up many small trees' corrections, so the sheet is built from small steps.",
    "knn": ("kNN prices each point by averaging its k nearest training districts, so the sheet is made of flat "
            "patches, one for each set of neighbours."),
    "mlp": ("The network's ReLU units fold the sheet along straight creases, so it is made of flat facets "
            "joined at angles."),
}


def _refusing_unconverged(what: str, fit):
    """Run ``fit()``, refusing to show a solver state that has not converged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            return fit()
        except ConvergenceWarning as exc:
            raise LessonError(f"{what} did not converge: {exc}") from exc


def fit_converged(model: Any, view: FullFeatures) -> Any:
    """``model`` fitted on ``view``'s training rows, or a LessonError if its solver did not converge.

    Args:
        model: An unfitted scikit-learn style model.
        view: The features to fit on.

    Returns:
        The fitted model.

    Raises:
        LessonError: If the solver warns that it did not converge.
    """
    return _refusing_unconverged(f"{model!r} on {view.names}", lambda: model.fit(view.X_train, view.y_train))


def _penalised_polynomial(name: str, make, alphas: tuple[float, ...]) -> Factory:
    """A factory for Ridge or Lasso on the polynomial lessons' degree-``DEGREE`` features, α chosen on validation."""
    def factory(lesson: Lesson, view: FullFeatures) -> tuple[Any, str]:
        """Chooses α on ``view``'s validation rows and returns the unfitted model.

        Args:
            lesson: The lesson (unused: the degree is the lesson family's own).
            view: The features to use.

        Returns:
            The unfitted model with the chosen α, and a line saying how it was set up.

        Raises:
            LessonError: If a fit during the choice did not converge.
        """
        model, alpha = _refusing_unconverged(
            f"{name} on {view.names}",
            lambda: chosen_on_validation(lambda a: polynomial.poly_pipeline(DEGREE, make(a)), alphas, view))
        inputs = "both inputs" if len(view.names) == 2 else " and ".join(view.names)
        return model, f"{name}, the lesson's degree-{DEGREE} polynomial of {inputs}, α = {alpha:.2g} chosen on validation"
    return factory


TWO_INPUT_OVERRIDES: dict[str, Factory] = {
    "ridge": _penalised_polynomial("Ridge", lambda a: Ridge(alpha=a), polynomial.RIDGE_ALPHAS),
    "lasso": _penalised_polynomial("Lasso", lambda a: Lasso(alpha=a, **polynomial.LASSO_SETTINGS),
                                   polynomial.LASSO_ALPHAS),
}


def two_inputs(lesson: Lesson) -> FullFeatures:
    """The lesson's districts with just median income and house age, the 3-D view's two inputs.

    Args:
        lesson: The lesson being played.

    Returns:
        Its full features narrowed to ``INPUTS``, in that order.
    """
    return select_columns(lesson.data.full, INPUTS)


def model_for(lesson: Lesson, view: FullFeatures) -> tuple[Any, str]:
    """An unfitted model of the lesson's algorithm for ``view``'s inputs, and how it was set up.

    Ridge and Lasso use this module's polynomial factories; every other algorithm uses the
    all-features comparison's (``more_features.RICHER``).

    Args:
        lesson: The lesson being played; its story's key names the algorithm.
        view: The features the model will see.

    Returns:
        The unfitted model, and a line describing its settings.
    """
    key = lesson.story.key
    return TWO_INPUT_OVERRIDES.get(key, RICHER[key])(lesson, view)


def surface_model(lesson: Lesson) -> tuple[Any, str]:
    """The model drawn in 3-D: the lesson's algorithm for income and house age.

    Args:
        lesson: The lesson being played.

    Returns:
        The unfitted model, and a line describing its settings.
    """
    return model_for(lesson, two_inputs(lesson))


def grid_rows(incomes: np.ndarray, ages: np.ndarray, view: FullFeatures) -> np.ndarray:
    """Scaled feature rows for every (age, income) grid point, income running fastest.

    So the predictions reshape to (len(ages), len(incomes)).

    Args:
        incomes: The grid's incomes, in raw ``MedInc`` units ($10k).
        ages: The grid's house ages, in years.
        view: The two-input features, whose mean and scale standardise the rows.

    Returns:
        One standardised (income, age) row per grid point, shape (len(ages) * len(incomes), 2).
    """
    income, age = np.meshgrid(incomes, ages)
    return (np.column_stack([income.ravel(), age.ravel()]) - view.mean) / view.scale


def shown_rows(view: FullFeatures) -> np.ndarray:
    """Which training districts are drawn as dots, from those inside the box.

    The first ``STICKS`` of them get error sticks. The choice is random but seeded, so it is the same
    every render.

    Args:
        view: The two-input features; the first column is income.

    Returns:
        Up to ``DOTS_SHOWN`` training row indices, of districts whose income is within ``X_RANGE``.
    """
    income = view.X_train[:, 0] * view.scale[0] + view.mean[0]
    inside = np.flatnonzero((income >= X_RANGE[0]) & (income <= X_RANGE[1]))
    return np.random.default_rng(RANDOM_STATE).choice(inside, size=min(DOTS_SHOWN, len(inside)), replace=False)


def clip_disclosure(*heights_k: np.ndarray) -> str:
    """A note when anything drawn (sheet, error sticks, the unseen district) had to be clipped to the box.

    Args:
        *heights_k: The unclipped heights of everything drawn, in $k, in any shape.

    Returns:
        A sentence saying predictions outside the box are drawn at its edge, or "" if none are.
    """
    low, high = Y_RANGE_K
    values = np.concatenate([np.ravel(part) for part in heights_k])
    if np.all((values >= low) & (values <= high)):
        return ""
    return f"Where the model predicts below ${low:,.0f}k or above ${high:,.0f}k, it is drawn at the edge of the box."


def _in_k(model: Any, rows: np.ndarray) -> np.ndarray:
    """The model's prices for some scaled rows, in $k."""
    return np.asarray(model.predict(rows)) * PRICE_TO_K


def _sheet(model: Any, view: FullFeatures, incomes: np.ndarray, ages: np.ndarray) -> tuple[Sheet, np.ndarray]:
    """The model's sheet over the grid, clipped to the box and coloured by price; also its unclipped heights."""
    raw = _in_k(model, grid_rows(incomes, ages, view)).reshape(len(ages), len(incomes))
    heights = np.clip(raw, *Y_RANGE_K)
    return Sheet(heights, theme.value_colormap()(Normalize(*Y_RANGE_K)(heights))), raw


def _dots(model: Any, view: FullFeatures, rows: np.ndarray) -> tuple[Dots, np.ndarray]:
    """The training dots at their true prices, the first ``STICKS`` with error sticks; also the sticks' raw ends."""
    raw = view.X_train[rows] * view.scale + view.mean
    fitted = _in_k(model, view.X_train[rows[:STICKS]])
    dots = Dots(raw[:, 0], raw[:, 1], np.clip(view.y_train[rows] * PRICE_TO_K, *Y_RANGE_K),
                (theme.NAVY,) * len(rows), fitted=np.clip(fitted, *Y_RANGE_K))
    return dots, fitted


def _query(lesson: Lesson, model: Any, view: FullFeatures) -> tuple[Marker, float, float]:
    """The Prediction chapter's district, at this model's price; also its house age and that raw price."""
    row = lesson.data.query.row
    age = float(view.X_test[row, 1] * view.scale[1] + view.mean[1])
    price = float(_in_k(model, view.X_test[row:row + 1])[0])
    shown = float(np.clip(price, *Y_RANGE_K))
    return Marker(lesson.data.query.income, age, shown, f"${price:,.0f}k"), age, price


def _reading(lesson: Lesson, view: FullFeatures, settings: str, dots: int, age: float, price: float,
             clipped: str) -> tuple[str, ...]:
    """The "Reading the 3-D view" notes: what the sheet, the dots and the orange district show."""
    query = lesson.data.query
    own = predict_one(lesson.predict, query.income)
    notes = (f"The same algorithm, refitted on two inputs, median income and house age ({settings}).",
             "Height is the predicted price; the floor holds the two inputs.",
             f"Dots: {dots:,} of the {len(view.y_train):,} training districts it learned from. Red sticks join "
             f"{min(STICKS, dots)} of them to the sheet: their errors.",
             f"Orange: the unseen district from the Prediction chapter, {age:.0f} years old. This model says "
             f"${price:,.0f}k; the lesson's income-only model said {money_k(own)}; its census value is "
             f"{money_k(query.price)}.")
    return (*notes, clipped) if clipped else notes


def _scores(lesson: Lesson, model: Any, view: FullFeatures, control: Any, alone: FullFeatures) -> tuple[str, ...]:
    """The "How this model scores" notes: test RMSE with and without house age, and a paired bootstrap verdict.

    The income-only control is refitted the same way on the same rows, so any gain is house age's alone.
    """
    y = lesson.data.y_test
    both_pred, alone_pred = model.predict(view.X_test), control.predict(alone.X_test)
    verdict = paired_rmse(alone_pred - y, both_pred - y, show=lambda gain: f"${gain * PRICE_TO_K:,.1f}k",
                          seed=RANDOM_STATE)
    effect = ("Adding house age helps beyond chance." if verdict.better else
              "Adding house age makes it worse beyond chance." if verdict.worse else
              "Adding house age makes no clear difference.")
    own = rmse_k(lesson.predict(lesson.data.x_test), y)
    return (f"Test RMSE (typical error) on the same {len(y):,} test districts, both refitted the same way on "
            f"all {len(view.y_train):,} training districts:",
            f"Income only: ${rmse_k(alone_pred, y):,.0f}k",
            f"Income and house age, shown here: ${rmse_k(both_pred, y):,.0f}k",
            f"{effect} {verdict.evidence}.",
            f"This lesson's own income-only model scored ${own:,.0f}k. Next: all 8 features.")


def _axes() -> tuple[Axis, Axis, Axis]:
    """The box's three axes: income, house age and predicted price, with their ticks and labels."""
    income = Axis("Median income", X_RANGE, INCOME_TICKS, tuple(f"${t * INCOME_TO_K:,.0f}k" for t in INCOME_TICKS))
    age = Axis("House age (years)", AGE_RANGE, AGE_TICKS, tuple(f"{t:.0f}" for t in AGE_TICKS))
    price = Axis("Predicted price", Y_RANGE_K, PRICE_TICKS_K, tuple(f"${t:,.0f}k" for t in PRICE_TICKS_K))
    return income, age, price


def prepare_surface(lesson: Lesson) -> Surface:
    """Builds the 3-D chapter: the lesson's algorithm on income and house age, drawn as a sheet.

    It fits two models on every training district: the one drawn (income and house age) and the
    income-only control its score is compared with.

    Args:
        lesson: The lesson being played.

    Returns:
        animkit's ``Surface``: the sheet, the dots, the unseen district, the caption and the side notes.

    Raises:
        LessonError: If either model's solver did not converge.
    """
    view, alone = two_inputs(lesson), select_columns(lesson.data.full, INPUTS[:1])
    model, settings = model_for(lesson, view)
    model = fit_converged(model, view)
    control = fit_converged(model_for(lesson, alone)[0], alone)
    incomes, ages = np.linspace(*X_RANGE, GRID_SIZE), np.linspace(*AGE_RANGE, GRID_SIZE)
    sheet, grid_raw = _sheet(model, view, incomes, ages)
    rows = shown_rows(view)
    dots, sticks_raw = _dots(model, view, rows)
    query, age, price = _query(lesson, model, view)
    clipped = clip_disclosure(grid_raw, sticks_raw, np.array([price]))
    x_axis, y_axis, z_axis = _axes()
    return Surface(
        x=incomes, y=ages, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, sheets=(sheet,), dots=dots, query=query,
        title="Income × house age, in 3-D",
        caption=(f"{SHAPES[lesson.story.key]} Same algorithm, now given house age as well as income; height is "
                 "its predicted price."),
        reading=_reading(lesson, view, settings, len(rows), age, price, clipped),
        scores=_scores(lesson, model, view, control, alone))
