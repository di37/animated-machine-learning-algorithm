"""Load the Titanic passenger list and produce encoded, scaled, read-only train/validation/test splits.

The data is OpenML's ``titanic`` (version 1): 1,309 passengers. ``load_splits`` runs every step:

1. Drop ``boat`` and ``body`` (``LEAKY_COLUMNS``). A lifeboat number or a recovered body is only
   known after the sinking, so either would give the answer away.
2. Check the columns and values, then split the passengers into train, validation and test
   (60/20/20 by default), stratified so every split keeps the same share of survivors.
3. Learn fill values for missing ages, fares and ports from the training rows only, so nothing
   about the validation or test passengers leaks into training.
4. Encode each passenger as the eight numbers in ``FEATURE_COLUMNS``: class, female (1 or 0), age,
   siblings/spouses, parents/children, fare, and one-hot flags for Cherbourg and Queenstown
   (Southampton, the most common port, is the baseline with both flags 0).
5. Standardise every feature with the training rows' mean and standard deviation, so models that
   use distances or gradients treat the features alike.

Every array is read-only, so no later step can change the data another step relies on.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from animkit.data import read_only
from titanic_anim.config import (
    MIN_ROWS,
    OPENML_NAME,
    OPENML_VERSION,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_FRACTION,
    VAL_FRACTION,
)

FEATURE_COLUMNS = ("pclass", "female", "age", "sibsp", "parch", "fare", "embarked_C", "embarked_Q")
SOURCE_COLUMNS = ("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", TARGET_COLUMN)
# Lifeboat number and body-recovery tag are only known after the sinking: they would leak the answer.
LEAKY_COLUMNS = ("boat", "body")
PORTS = ("C", "Q", "S")
SEXES = ("female", "male")
CLASSES_OF_TRAVEL = (1, 2, 3)
NUMERIC_COLUMNS = ("age", "sibsp", "parch", "fare")


class DataLoadError(RuntimeError):
    """The dataset could not be fetched (usually no network on first download)."""


@dataclass(frozen=True)
class TitanicSplits:
    """The passengers in three splits, ready for scikit-learn, plus what it takes to read them back.

    Each ``X`` has one row per passenger and one column per name in ``feature_names``. Every array is
    read-only.

    Attributes:
        feature_names: The eight feature columns, in ``FEATURE_COLUMNS`` order.
        X_train: The training passengers' standardised features.
        y_train: Their outcomes: 1 = survived, 0 = did not survive.
        X_val: The validation passengers' standardised features. No model trains on them; every
            training step is scored on them.
        y_val: Their outcomes.
        X_test: The held-out test passengers' standardised features, classified only after training.
        y_test: Their outcomes.
        raw_train: The training passengers' encoded but unscaled features, for wording in real units.
        raw_test: The test passengers' encoded but unscaled features.
        age_known_test: Per test passenger, False where the age was missing and filled with the median.
        feature_mean: The scaler's mean per feature, to place new points on the scaled axes.
        feature_scale: The scaler's standard deviation per feature.

    Example:
        >>> x = (raw_row - splits.feature_mean) / splits.feature_scale  # doctest: +SKIP
    """

    feature_names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray  # 1 = survived, 0 = did not survive
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    raw_train: np.ndarray  # encoded but unscaled features, for labels in real units
    raw_test: np.ndarray
    age_known_test: np.ndarray  # False where the age was missing and filled with the median
    feature_mean: np.ndarray  # StandardScaler parameters, to place new points on the scaled axes
    feature_scale: np.ndarray

    @property
    def n_train(self) -> int:
        """How many training passengers there are.

        Returns:
            The length of ``y_train``.
        """
        return len(self.y_train)


@dataclass(frozen=True)
class Imputation:
    """Fill values learned from the training passengers.

    Attributes:
        age: The median age, in years.
        fare: The median fare, in pounds.
        port: The most common port of embarkation, ``"C"``, ``"Q"`` or ``"S"``.
    """

    age: float
    fare: float
    port: str


def read_only_labels(values) -> np.ndarray:
    """A read-only integer copy of class labels.

    Args:
        values: The labels, as any array-like (floats are truncated to ints).

    Returns:
        A new int array that cannot be written to; ``values`` is left as it is.
    """
    labels = np.array(values, dtype=int, copy=True)
    labels.setflags(write=False)
    return labels


def fetch_titanic(as_frame: bool = True):
    """Downloads OpenML's Titanic dataset, or reads it from scikit-learn's cache.

    scikit-learn keeps the download in ``~/scikit_learn_data``, so only the first run needs the
    network. Tests pass a small stand-in fetcher instead of this one.

    Args:
        as_frame: Whether to return the data as a pandas frame.

    Returns:
        scikit-learn's dataset bunch; its ``frame`` holds the passenger table.
    """
    return fetch_openml(name=OPENML_NAME, version=OPENML_VERSION, as_frame=as_frame)


def load_titanic_frame(fetcher: Callable = fetch_titanic) -> pd.DataFrame:
    """The passenger table, with the columns that would leak the answer dropped.

    Args:
        fetcher: Fetches the dataset: ``fetcher(as_frame=True).frame`` is the passenger table.

    Returns:
        One row per passenger, without ``boat`` and ``body``.

    Raises:
        DataLoadError: If the download fails (an ``OSError``) or the cached copy cannot be read
            (a ``ValueError``); the message says what to try.
    """
    try:
        frame = fetcher(as_frame=True).frame
    except OSError as exc:
        raise DataLoadError(
            f"Could not load the Titanic dataset from OpenML ({exc}). Check your internet connection "
            "and that ~/scikit_learn_data is writable; scikit-learn caches it there after the first run."
        ) from exc
    except ValueError as exc:  # e.g. a truncated download in the cache
        raise DataLoadError(
            f"The cached Titanic download looks corrupt ({exc}). Delete ~/scikit_learn_data/openml and retry."
        ) from exc
    return frame.drop(columns=[c for c in LEAKY_COLUMNS if c in frame.columns])


def _labels(frame: pd.DataFrame) -> np.ndarray:
    """The ``survived`` column as floats; anything that is not a number becomes NaN."""
    return pd.to_numeric(frame[TARGET_COLUMN].astype(str), errors="coerce").to_numpy(dtype=float)


def read_only_flags(values) -> np.ndarray:
    """A read-only boolean copy of flags.

    Args:
        values: The flags, as any array-like.

    Returns:
        A new bool array that cannot be written to.
    """
    flags = np.array(values, dtype=bool, copy=True)
    flags.setflags(write=False)
    return flags


def _check_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Refuses text or infinities in numeric columns; gaps are allowed, since they are imputed."""
    original = frame[list(columns)]
    numbers = original.apply(pd.to_numeric, errors="coerce")
    garbled = (numbers.isna() & original.notna()).to_numpy()
    if garbled.any() or np.isinf(numbers.to_numpy(dtype=float)).any():
        raise ValueError(f"Columns {', '.join(columns)} must hold finite numbers")


def _check_values(frame: pd.DataFrame, column: str, allowed: tuple) -> None:
    """Refuses a value of ``column`` outside ``allowed``; missing values are skipped."""
    present = frame[column].dropna()
    values = present.astype(int) if column == "pclass" else present.astype(str)
    unexpected = set(values) - set(allowed)
    if unexpected:
        raise ValueError(f"Column '{column}' has unexpected values: {sorted(map(str, unexpected))}")


def validate_frame(frame: pd.DataFrame) -> None:
    """Checks the passenger table before anything is computed from it.

    Args:
        frame: The passenger table, as ``load_titanic_frame`` returns it.

    Raises:
        ValueError: If a needed column is missing; there are fewer than ``MIN_ROWS`` passengers;
            ``survived`` holds anything but 0 and 1, or only one of them; class, sex or a family count
            has gaps; a numeric column holds text or infinities; or class, sex or port has a value
            outside the known ones.
    """
    missing = [c for c in SOURCE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if len(frame) < MIN_ROWS:
        raise ValueError(f"Dataset needs at least {MIN_ROWS} rows, got {len(frame)}")
    labels = _labels(frame)
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("Column 'survived' must contain only 0 and 1")
    if len(np.unique(labels)) < 2:
        raise ValueError("Dataset needs both survivors and non-survivors")
    if frame[["pclass", "sex", "sibsp", "parch"]].isna().any().any():
        raise ValueError("Columns 'pclass', 'sex', 'sibsp' and 'parch' must not have missing values")
    _check_numeric(frame, NUMERIC_COLUMNS)
    _check_values(frame, "pclass", CLASSES_OF_TRAVEL)
    _check_values(frame, "sex", SEXES)
    _check_values(frame, "embarked", PORTS)


def learn_imputation(rows: pd.DataFrame) -> Imputation:
    """The values that fill missing ages, fares and ports.

    Args:
        rows: The training passengers only, so nothing about the others leaks in.

    Returns:
        Their median age and fare, and their most common port (Southampton if none is known).
    """
    ports = rows["embarked"].dropna().astype(str)
    return Imputation(
        age=float(rows["age"].median()),
        fare=float(rows["fare"].median()),
        port=str(ports.mode().iloc[0]) if len(ports) else "S",
    )


def encode(rows: pd.DataFrame, fill: Imputation) -> np.ndarray:
    """Turns passengers into the eight numbers a model reads, filling gaps from ``fill``.

    Sex becomes ``female`` (1.0 or 0.0). The port becomes two one-hot flags, ``embarked_C`` and
    ``embarked_Q``; a Southampton passenger has both at 0. Nothing is scaled yet.

    Args:
        rows: Passengers, with at least the source feature columns.
        fill: The fill values learned from the training passengers.

    Returns:
        A float array with one row per passenger, columns in ``FEATURE_COLUMNS`` order.
    """
    port = rows["embarked"].astype(object).fillna(fill.port).astype(str)
    return np.column_stack([
        rows["pclass"].astype(int),
        (rows["sex"].astype(str) == "female").astype(float),
        rows["age"].astype(float).fillna(fill.age),
        rows["sibsp"].astype(float),
        rows["parch"].astype(float),
        rows["fare"].astype(float).fillna(fill.fare),
        (port == "C").astype(float),
        (port == "Q").astype(float),
    ]).astype(float)


def _validate_fractions(test_fraction: float, val_fraction: float) -> None:
    """Refuses split fractions outside 0 … 1, or two that leave no rows for training."""
    if not (0.0 < test_fraction < 1.0 and 0.0 < val_fraction < 1.0):
        raise ValueError("test_fraction and val_fraction must each be between 0 and 1")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction + val_fraction must leave rows for training")


def _three_way_split(rows: pd.DataFrame, test_fraction: float, val_fraction: float, random_state: int):
    """Stratified train/val/test, so every split keeps the same share of survivors."""
    rest, test = train_test_split(rows, test_size=test_fraction, stratify=_labels(rows),
                                  random_state=random_state)
    train, val = train_test_split(rest, test_size=val_fraction / (1.0 - test_fraction),
                                  stratify=_labels(rest), random_state=random_state)
    return train, val, test


def split_and_scale(
    frame: pd.DataFrame,
    *,
    test_fraction: float = TEST_FRACTION,
    val_fraction: float = VAL_FRACTION,
    random_state: int = RANDOM_STATE,
    max_rows: int | None = None,
) -> TitanicSplits:
    """Checks, splits, encodes and standardises the passenger table.

    Args:
        frame: The passenger table, leaky columns already dropped.
        test_fraction: The share of all passengers held out for the test set.
        val_fraction: The share of all passengers kept for validation.
        random_state: The seed for the subsample and the split.
        max_rows: How many passengers to keep, chosen at random, for a quick run; None keeps them all.

    Returns:
        The read-only splits, standardised by a ``StandardScaler`` fit on the training rows only.

    Raises:
        ValueError: If the fractions leave no rows for training, ``max_rows`` is below ``MIN_ROWS``,
            or the table fails ``validate_frame``.
    """
    _validate_fractions(test_fraction, val_fraction)
    if max_rows is not None and max_rows < MIN_ROWS:
        raise ValueError(f"max_rows must be at least {MIN_ROWS}")
    validate_frame(frame)
    rows = frame if max_rows is None or max_rows >= len(frame) else frame.sample(
        n=max_rows, random_state=random_state
    )
    train, val, test = _three_way_split(rows, test_fraction, val_fraction, random_state)
    fill = learn_imputation(train)
    raw_train, raw_val, raw_test = (encode(part, fill) for part in (train, val, test))
    scaler = StandardScaler().fit(raw_train)
    return TitanicSplits(
        feature_names=FEATURE_COLUMNS,
        X_train=read_only(scaler.transform(raw_train)),
        y_train=read_only_labels(_labels(train)),
        X_val=read_only(scaler.transform(raw_val)),
        y_val=read_only_labels(_labels(val)),
        X_test=read_only(scaler.transform(raw_test)),
        y_test=read_only_labels(_labels(test)),
        raw_train=read_only(raw_train),
        raw_test=read_only(raw_test),
        age_known_test=read_only_flags(test["age"].notna()),
        feature_mean=read_only(scaler.mean_),
        feature_scale=read_only(scaler.scale_),
    )


def load_splits(fetcher: Callable = fetch_titanic, max_rows: int | None = None) -> TitanicSplits:
    """Fetches the Titanic dataset and returns its splits: the one call the command line makes.

    Args:
        fetcher: Fetches the dataset, as ``load_titanic_frame`` takes it.
        max_rows: How many passengers to keep for a quick run; None keeps them all.

    Returns:
        The read-only train, validation and test splits.

    Raises:
        DataLoadError: If the dataset cannot be fetched or read.
        ValueError: If the table or ``max_rows`` fails the checks in ``split_and_scale``.
    """
    return split_and_scale(load_titanic_frame(fetcher), max_rows=max_rows)
