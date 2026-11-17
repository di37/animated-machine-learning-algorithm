"""One-feature view of the housing splits: median income (raw units) against median house value.

The lessons reuse the dashboards' exact 60/20/20 split, so the same districts are held out.

Every lesson learns a price from one input, a district's median income (``MedInc``, in $10k), so its
model can be drawn as a curve over a 2-D scatter. The splits store standardised features; this module
undoes the standardisation for ``MedInc`` so the x-axis reads in dollars. It also decides what the video
shows of the data: a random subset of training districts drawn as dots (the model still trains on all
of them), a few of those with residual lines, the grid of incomes where each curve is evaluated, and the
one unseen test district (the query) every lesson predicts at the end. ``FullFeatures`` keeps all eight
scaled features for the chapters that look past the single input. Every array is read-only.
"""

from dataclasses import dataclass, replace

import numpy as np

from animkit.lessons.data import read_only_indices
from house_anim.config import RANDOM_STATE
from house_anim.data import HousingSplits, read_only
from house_anim.lessons.config import (
    DISPLAY_POINTS,
    FEATURE,
    GRID_POINTS,
    QUERY_TARGET_INCOME,
    RESIDUAL_POINTS,
    X_RANGE,
)


@dataclass(frozen=True)
class Query:
    """The unseen test district every lesson predicts at the end.

    Attributes:
        row: Its index in the test split.
        income: Its median income, in ``MedInc`` units ($10k).
        price: Its true median house value, in ``MedHouseVal`` units ($100k).
    """

    row: int  # index into the test split
    income: float  # MedInc ($10k)
    price: float  # true MedHouseVal ($100k)


@dataclass(frozen=True)
class FullFeatures:
    """All eight scaled features for the same districts, for the 'more features' comparison and the 3-D view.

    ``mean`` and ``scale`` are the training rows' standardisation, so raw values can be put on the scaled axes.

    Attributes:
        names: The feature names, in column order.
        X_train: The training districts' scaled features, one row per district.
        y_train: The training districts' prices ($100k).
        X_val: The validation districts' scaled features.
        y_val: The validation districts' prices ($100k).
        X_test: The test districts' scaled features.
        mean: Each feature's training mean, in raw units.
        scale: Each feature's training standard deviation, in raw units.
    """

    names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


def select_columns(full: FullFeatures, names: tuple[str, ...]) -> FullFeatures:
    """The same districts with only the features ``names``, in that order.

    Args:
        full: Every feature for the districts.
        names: The features to keep, in the order wanted.

    Returns:
        A ``FullFeatures`` with just those columns (and their mean and scale); the prices are shared.

    Raises:
        ValueError: If a name is not one of ``full.names``.
    """
    missing = [name for name in names if name not in full.names]
    if missing:
        raise ValueError(f"unknown features {missing}; there are {full.names}")
    columns = [full.names.index(name) for name in names]
    return FullFeatures(names=tuple(names), X_train=read_only(full.X_train[:, columns]), y_train=full.y_train,
                        X_val=read_only(full.X_val[:, columns]), y_val=full.y_val,
                        X_test=read_only(full.X_test[:, columns]), mean=read_only(full.mean[columns]),
                        scale=read_only(full.scale[columns]))


@dataclass(frozen=True)
class IncomeData:
    """Everything a lesson knows about the data: income against price for each split, and what to draw.

    Attributes:
        x_train: The training districts' median incomes ($10k).
        y_train: Their prices ($100k).
        x_val: The validation districts' incomes, for choosing between model states.
        y_val: Their prices.
        x_test: The test districts' incomes, for the final exam.
        y_test: Their prices.
        display: The training rows drawn as dots, in the order they appear.
        residual: The training rows that also get a residual line (the first few of ``display``).
        grid: The evenly spaced incomes where each model's curve is evaluated.
        query: The unseen test district predicted in the inference chapter.
        sample_note: Says so when the lesson trains on a subsample; empty otherwise.
        baseline_mean: The full training split's average price ($100k), the predict-the-average baseline.
        test_coords: Each test district's (longitude, latitude), for the exam's error map.
        full: Every scaled feature for the full splits, for the 3-D view and the more-features comparison.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    display: np.ndarray  # training rows drawn as dots, in reveal order
    residual: np.ndarray  # training rows that also get a residual line
    grid: np.ndarray  # incomes where each model's curve is evaluated
    query: Query
    sample_note: str  # non-empty when the lesson trains on a subsample
    baseline_mean: float  # average price of the full training split (kept by subsample)
    test_coords: np.ndarray  # (longitude, latitude) of each test district, for the exam map
    full: FullFeatures  # every feature, all training districts (kept by subsample)

    @property
    def n_train(self) -> int:
        """How many districts the model trains on.

        Returns:
            The number of training rows.
        """
        return len(self.y_train)

    @property
    def display_x(self) -> np.ndarray:
        """The incomes of the districts drawn as dots.

        Returns:
            Their incomes ($10k), in reveal order.
        """
        return self.x_train[self.display]

    @property
    def display_y(self) -> np.ndarray:
        """The prices of the districts drawn as dots.

        Returns:
            Their prices ($100k), in reveal order.
        """
        return self.y_train[self.display]

    @property
    def residual_x(self) -> np.ndarray:
        """The incomes of the districts that get a residual line.

        Returns:
            Their incomes ($10k).
        """
        return self.x_train[self.residual]

    @property
    def residual_y(self) -> np.ndarray:
        """The prices of the districts that get a residual line.

        Returns:
            Their prices ($100k).
        """
        return self.y_train[self.residual]


def pick_query(x_test: np.ndarray, y_test: np.ndarray, target: float = QUERY_TARGET_INCOME) -> Query:
    """The test district whose income is nearest ``target``, chosen by income alone (never its price).

    Args:
        x_test: The test districts' incomes ($10k).
        y_test: Their prices ($100k).
        target: The income wanted ($10k).

    Returns:
        The chosen district, with its row, income and true price.

    Raises:
        ValueError: If the test split is empty.
    """
    if len(x_test) == 0:
        raise ValueError("the test split is empty; there is no district to predict")
    row = int(np.argmin(np.abs(x_test - target)))
    return Query(row=row, income=float(x_test[row]), price=float(y_test[row]))


def _feature_column(splits: HousingSplits, X: np.ndarray, column: int) -> np.ndarray:
    """Undo the training-set standardisation, back to raw MedInc."""
    return read_only(X[:, column] * splits.feature_scale[column] + splits.feature_mean[column])


def income_view(splits: HousingSplits, *, display_points: int = DISPLAY_POINTS,
                residual_points: int = RESIDUAL_POINTS, seed: int = RANDOM_STATE) -> IncomeData:
    """The lessons' view of the housing splits: raw median income against price, and what to draw.

    Args:
        splits: The standardised train, validation and test splits.
        display_points: How many training districts to draw as dots (all of them if there are fewer).
        residual_points: How many of those dots also get a residual line.
        seed: Seeds the random choice of the dots.

    Returns:
        The income view, with the query district picked from the test split.

    Raises:
        ValueError: If the splits have no ``MedInc`` feature.
    """
    if FEATURE not in splits.feature_names:
        raise ValueError(f"the lessons need the {FEATURE} feature, got {splits.feature_names}")
    column = splits.feature_names.index(FEATURE)
    x_train = _feature_column(splits, splits.X_train, column)
    x_test = _feature_column(splits, splits.X_test, column)
    rng = np.random.default_rng(seed)
    display = rng.choice(len(x_train), size=min(display_points, len(x_train)), replace=False)
    return IncomeData(
        x_train=x_train,
        y_train=read_only(splits.y_train),
        x_val=_feature_column(splits, splits.X_val, column),
        y_val=read_only(splits.y_val),
        x_test=x_test,
        y_test=read_only(splits.y_test),
        display=read_only_indices(display),
        residual=read_only_indices(display[:residual_points]),
        grid=read_only(np.linspace(*X_RANGE, GRID_POINTS)),
        query=pick_query(x_test, splits.y_test),
        sample_note="",
        baseline_mean=float(splits.y_train.mean()),
        test_coords=read_only(splits.coords_test),
        full=FullFeatures(splits.feature_names, read_only(splits.X_train), read_only(splits.y_train),
                          read_only(splits.X_val), read_only(splits.y_val), read_only(splits.X_test),
                          read_only(splits.feature_mean), read_only(splits.feature_scale)),
    )


def subsample(data: IncomeData, size: int, seed: int = RANDOM_STATE) -> IncomeData:
    """Train on one random district from each of ``size`` equal-width income bands.

    Spreading the sample across the whole income range means validation never asks a polynomial
    to extrapolate past its data, so the lesson shows over-fitting rather than extrapolation.
    Validation, test and the query stay the same, as do ``baseline_mean`` and ``full``. Every training
    row of the sample is drawn and gets a residual line. A band with no district is skipped.

    Args:
        data: The full income view.
        size: How many income bands to sample from, between 2 and the number of training rows.
        seed: Seeds the choice within each band.

    Returns:
        A copy of ``data`` that trains on the sample, with a ``sample_note`` saying so.

    Raises:
        ValueError: If ``size`` is out of range.
    """
    if not 2 <= size <= data.n_train:
        raise ValueError(f"subsample size must be between 2 and {data.n_train}, got {size}")
    rng = np.random.default_rng(seed)
    edges = np.linspace(data.x_train.min(), data.x_train.max(), size + 1)
    band = np.clip(np.searchsorted(edges, data.x_train, side="right") - 1, 0, size - 1)
    rows = np.array([rng.choice(np.flatnonzero(band == b)) for b in range(size) if (band == b).any()])
    every_row = read_only_indices(np.arange(len(rows)))
    return replace(
        data,
        x_train=read_only(data.x_train[rows]),
        y_train=read_only(data.y_train[rows]),
        display=every_row,
        residual=every_row,
        sample_note=(f"trained on just {len(rows)} districts, one from each income band, "
                     "so over-fitting is easy to see"),
    )

