"""Load California Housing and produce scaled, read-only train/validation/test splits.

The dataset has 20,640 census block groups ("districts" or "houses" here), 8 features and one
target: the median house value, in $100k. The rows are shuffled and split three ways:

- train (60%): the only rows a model learns from;
- validation (20%): scored after every training step, to draw the learning curve;
- test (20%): seen only once training is over, to score the finished model on unseen houses.

Features are standardised (each column to mean 0, standard deviation 1) with the training rows'
mean and scale only, so nothing about the validation or test rows leaks into training. The scaler's
mean and scale are kept, so a scaled value (a tree threshold, a map point) can be converted to or
from real units. Every array is read-only, so no loop or chart can change the data another one reads.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from animkit.data import read_only
from house_anim.config import MIN_ROWS, RANDOM_STATE, TARGET_COLUMN, TEST_FRACTION, VAL_FRACTION

FEATURE_COLUMNS = (
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
)
COORD_COLUMNS = ("Longitude", "Latitude")


class DataLoadError(RuntimeError):
    """The dataset could not be fetched (usually no network on first download)."""


@dataclass(frozen=True)
class HousingSplits:
    """The scaled train/validation/test splits, plus what is needed to read them in real units.

    Attributes:
        feature_names: The 8 feature columns, in the order of the arrays' columns.
        X_train: Training features, standardised, shape (n_train, 8).
        y_train: Training targets: median house value in $100k.
        X_val: Validation features, standardised.
        y_val: Validation targets, in $100k.
        X_test: Test features, standardised.
        y_test: Test targets, in $100k.
        coords_test: Raw (longitude, latitude) of each test house, in degrees, for the map.
        coords_train: Raw (longitude, latitude) of each training house, in degrees.
        feature_mean: The scaler's per-feature mean (from the training rows), in real units.
        feature_scale: The scaler's per-feature standard deviation, in real units.
    """

    feature_names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    coords_test: np.ndarray  # raw (longitude, latitude) degrees, for the map
    coords_train: np.ndarray  # raw (longitude, latitude) of training houses
    feature_mean: np.ndarray  # StandardScaler parameters, to place new points on the scaled axes
    feature_scale: np.ndarray

    @property
    def n_train(self) -> int:
        """The number of training rows.

        Returns:
            ``len(y_train)``.
        """
        return len(self.y_train)


def load_housing_frame(fetcher: Callable = fetch_california_housing) -> pd.DataFrame:
    """Fetches the dataset as one table: the 8 feature columns and ``MedHouseVal``.

    Args:
        fetcher: scikit-learn's ``fetch_california_housing``, or a stand-in with the same
            ``as_frame=True`` call (the tests pass synthetic data).

    Returns:
        The dataset's ``frame``: one row per district.

    Raises:
        DataLoadError: If the download fails (usually no network on the first run, before
            scikit-learn has cached it in ~/scikit_learn_data).
    """
    try:
        return fetcher(as_frame=True).frame
    except OSError as exc:
        raise DataLoadError(
            "Could not download the California Housing dataset. Check your internet "
            "connection; scikit-learn caches it in ~/scikit_learn_data after the first run."
        ) from exc


def validate_frame(frame: pd.DataFrame) -> None:
    """Checks the table has every column, enough rows, and only finite numbers.

    Args:
        frame: The table from ``load_housing_frame``.

    Raises:
        ValueError: If a feature or the target column is missing, there are fewer than ``MIN_ROWS``
            rows, or a value is missing or infinite.
    """
    missing = [c for c in (*FEATURE_COLUMNS, TARGET_COLUMN) if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if len(frame) < MIN_ROWS:
        raise ValueError(f"Dataset needs at least {MIN_ROWS} rows, got {len(frame)}")
    values = frame[[*FEATURE_COLUMNS, TARGET_COLUMN]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains missing or infinite values")


def _validate_fractions(test_fraction: float, val_fraction: float) -> None:
    """Refuses split fractions outside (0, 1), or ones that leave no rows for training."""
    if not (0.0 < test_fraction < 1.0 and 0.0 < val_fraction < 1.0):
        raise ValueError("test_fraction and val_fraction must each be between 0 and 1")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction + val_fraction must leave rows for training")


def split_and_scale(
    frame: pd.DataFrame,
    *,
    test_fraction: float = TEST_FRACTION,
    val_fraction: float = VAL_FRACTION,
    random_state: int = RANDOM_STATE,
    max_rows: int | None = None,
) -> HousingSplits:
    """Shuffle-split into train/val/test and standardise features using training rows only.

    The test rows are split off first; the validation rows then come from the rest, so each
    fraction is a share of the whole table.

    Args:
        frame: The table from ``load_housing_frame``.
        test_fraction: The share of rows held out for the test set.
        val_fraction: The share of rows held out for validation.
        random_state: Seeds the subsample and both shuffles, so a run is repeatable.
        max_rows: Subsamples the table to this many rows first, for a quick run; None keeps all.

    Returns:
        The read-only splits, with the scaler's mean and scale.

    Raises:
        ValueError: If the fractions are out of range, ``max_rows`` is not positive, or the table
            fails ``validate_frame``.
    """
    _validate_fractions(test_fraction, val_fraction)
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    validate_frame(frame)
    rows = frame if max_rows is None or max_rows >= len(frame) else frame.sample(
        n=max_rows, random_state=random_state
    )

    features = rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    target = rows[TARGET_COLUMN].to_numpy(dtype=float)
    X_rest, X_test, y_rest, y_test = train_test_split(
        features, target, test_size=test_fraction, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction / (1.0 - test_fraction), random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    coord_index = [FEATURE_COLUMNS.index(c) for c in COORD_COLUMNS]
    return HousingSplits(
        feature_names=FEATURE_COLUMNS,
        X_train=read_only(scaler.transform(X_train)),
        y_train=read_only(y_train),
        X_val=read_only(scaler.transform(X_val)),
        y_val=read_only(y_val),
        X_test=read_only(scaler.transform(X_test)),
        y_test=read_only(y_test),
        coords_test=read_only(X_test[:, coord_index]),
        coords_train=read_only(X_train[:, coord_index]),
        feature_mean=read_only(scaler.mean_),
        feature_scale=read_only(scaler.scale_),
    )


def load_splits(
    fetcher: Callable = fetch_california_housing, max_rows: int | None = None
) -> HousingSplits:
    """Fetches, checks, splits and scales the dataset in one call, with the default fractions.

    Args:
        fetcher: Loads the dataset (see ``load_housing_frame``).
        max_rows: Subsamples the table to this many rows; None keeps all.

    Returns:
        The read-only splits.

    Raises:
        DataLoadError: If the dataset cannot be downloaded.
        ValueError: If the data or ``max_rows`` is invalid.
    """
    return split_and_scale(load_housing_frame(fetcher), max_rows=max_rows)
