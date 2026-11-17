"""Three-feature view of the Titanic splits: age, log fare and sex, for passengers with a known age.

Ages are missing for about one passenger in five. Filling them with the median would paint an
artificial stripe at one age on every map, so the lessons leave those passengers out and say so.
The split itself is titanic_anim's stratified 60/20/20 split.

titanic_anim scales its eight inputs for its own models; the lessons undo that scaling and keep three
columns in real units (age in years, log(1 + fare), and 1 for a woman), so the maps can be drawn over
real ages and fares. Each lesson's model scales them again on its own if it needs to.

Every lesson reads the same ``PassengerData``: the three-feature rows of each split, the grid of points
the two maps are scored on, the one unseen test passenger every lesson predicts, and the same passengers
with all eight inputs, for the "more features" chapter.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from animkit.data import read_only
from animkit.lessons.data import read_only_indices
from titanic_anim.config import RANDOM_STATE
from titanic_anim.data import (
    TitanicSplits,
    fetch_titanic,
    load_titanic_frame,
    read_only_labels,
    split_and_scale,
)
from titanic_anim.lessons.config import AGE_RANGE, FARE_MAX, FEMALE, GRID_AGES, GRID_FARES, QUERY_TARGET

QUERY_SCALE = (AGE_RANGE[1], np.log1p(FARE_MAX))  # puts age and log fare on comparable footing


@dataclass(frozen=True)
class Passenger:
    """The unseen test passenger every lesson predicts at the end.

    Attributes:
        row: The passenger's index into the test split.
        features: Their (age, log fare, female) row, as the lessons' models read it.
        survived: The true answer, 1 if they survived: revealed only after the prediction.
    """

    row: int  # index into the test split
    features: np.ndarray  # (age, log fare, female)
    survived: int


@dataclass(frozen=True)
class FullFeatures:
    """All eight scaled inputs for the same passengers, for the 'more features' comparison.

    Attributes:
        names: The column names, with ``fare`` renamed ``log_fare``.
        X_train: The training passengers' inputs, standardised on the training rows.
        y_train: Their labels, 1 = survived.
        X_val: The validation passengers' inputs, scaled with the training rows' mean and SD.
        y_val: Their labels.
        X_test: The test passengers' inputs, scaled the same way (their labels are ``PassengerData.y_test``).
    """

    names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray


@dataclass(frozen=True)
class PassengerData:
    """Everything a Titanic lesson reads about the passengers; every array is read-only.

    Attributes:
        x_train: The training passengers, shape (n, 3): age, log(1 + fare), female.
        y_train: Their labels, 1 = survived.
        x_val: The validation passengers, in the same columns: used to compare settings.
        y_val: Their labels.
        x_test: The test passengers, in the same columns: untouched until the final exam.
        y_test: Their labels.
        display: Every training row's index, in the shuffled order the passengers appear on screen.
        ages: The map grid's ages, left to right (the x axis).
        log_fares: The map grid's log fares, bottom to top (the y axis).
        grid: Every map cell as a feature row, shape (2 · fares · ages, 3), the women's map first.
        query: The unseen test passenger every lesson predicts.
        baseline_rate: The share of training passengers who survived.
        dropped: How many passengers were left out because their age is unknown.
        test_class: Each test passenger's class of travel (1 to 3); never a lesson input.
        full: The same passengers with all eight inputs.
    """

    x_train: np.ndarray  # (n, 3): age, log(1 + fare), female
    y_train: np.ndarray  # 1 = survived
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    display: np.ndarray  # training rows in reveal order (all of them)
    ages: np.ndarray  # map grid, x
    log_fares: np.ndarray  # map grid, y
    grid: np.ndarray  # (2 · fares · ages, 3): every map cell, women's map first
    query: Passenger
    baseline_rate: float  # share of training passengers who survived
    dropped: int  # passengers left out because their age is unknown
    test_class: np.ndarray  # class of travel (1-3) of each test passenger; never a lesson input
    full: FullFeatures  # every input, same passengers

    @property
    def n_train(self) -> int:
        """How many passengers the models train on.

        Returns:
            The number of training rows.
        """
        return len(self.y_train)

    @property
    def surface_shape(self) -> tuple[int, int, int]:
        """The shape of a model's scores on ``grid``, once reshaped into the two maps.

        Returns:
            ``(2, fares, ages)``: one image per sex, the women's first, rows running up the fare axis.
        """
        return 2, len(self.log_fares), len(self.ages)


def map_grid(ages: np.ndarray, log_fares: np.ndarray) -> np.ndarray:
    """Feature rows for both maps, ordered so ``scores.reshape(2, fares, ages)`` gives the images.

    Args:
        ages: The grid's ages, in years.
        log_fares: The grid's fares, as log(1 + fare).

    Returns:
        A read-only array, shape (2 · fares · ages, 3): every (age, fare) pair for women, then for men.
    """
    age, fare = np.meshgrid(ages, log_fares)
    maps = [np.column_stack([age.ravel(), fare.ravel(), np.full(age.size, female)]) for female in (1.0, 0.0)]
    return read_only(np.vstack(maps))


def pick_query(x_test: np.ndarray, y_test: np.ndarray, target=QUERY_TARGET) -> Passenger:
    """The test passenger of the target's sex nearest the target age and fare.

    Labels are never read to choose (``y_test`` only supplies the answer that is revealed at the end).
    Age and log fare are each divided by the top of their axis (``QUERY_SCALE``) first, so the distance
    is measured in shares of each axis and neither input outweighs the other.

    Args:
        x_test: The test passengers' (age, log fare, female) rows.
        y_test: Their labels, 1 = survived.
        target: The (age, log fare, female) row to look near: by default a 30-year-old woman on a $15 ticket.

    Returns:
        The chosen passenger.

    Raises:
        ValueError: If no test passenger has the target's sex.
    """
    same_sex = np.flatnonzero(x_test[:, FEMALE] == target[FEMALE])
    if same_sex.size == 0:
        raise ValueError("there is no test passenger of the target's sex to predict")
    gaps = (x_test[same_sex, :2] - np.asarray(target[:2])) / np.asarray(QUERY_SCALE)
    row = int(same_sex[np.argmin(np.hypot(gaps[:, 0], gaps[:, 1]))])
    return Passenger(row=row, features=read_only(x_test[row]), survived=int(y_test[row]))


def _features(splits: TitanicSplits, X: np.ndarray) -> np.ndarray:
    """Undo the scaling and keep age, log(1 + fare) and the female flag."""
    raw = X * splits.feature_scale + splits.feature_mean
    names = splits.feature_names
    columns = [raw[:, names.index("age")], np.log1p(np.clip(raw[:, names.index("fare")], 0.0, None)),
               np.round(raw[:, names.index("female")])]
    return read_only(np.column_stack(columns))  # columns in AGE, LOG_FARE, FEMALE order


def full_features(splits: TitanicSplits) -> FullFeatures:
    """All eight inputs with the lessons' fare transform, log(1 + fare), standardised on the training rows.

    Args:
        splits: titanic_anim's split, whose inputs are scaled on the raw fare.

    Returns:
        The same passengers' eight inputs, rescaled after the fare is logged.
    """
    fare = splits.feature_names.index("fare")

    def logged(X: np.ndarray) -> np.ndarray:
        """Undoes titanic_anim's scaling and replaces the fare column with log(1 + fare).

        Args:
            X: One split's eight inputs, as titanic_anim scaled them.

        Returns:
            The same inputs in real units, with the fare logged.
        """
        raw = X * splits.feature_scale + splits.feature_mean
        return np.column_stack([np.log1p(np.clip(raw[:, j], 0.0, None)) if j == fare else raw[:, j]
                                for j in range(raw.shape[1])])

    train, val, test = (logged(X) for X in (splits.X_train, splits.X_val, splits.X_test))
    mean, spread = train.mean(axis=0), train.std(axis=0)
    scale = np.where(spread > 0, spread, 1.0)
    names = tuple("log_fare" if name == "fare" else name for name in splits.feature_names)
    return FullFeatures(names, read_only((train - mean) / scale), read_only_labels(splits.y_train),
                        read_only((val - mean) / scale), read_only_labels(splits.y_val),
                        read_only((test - mean) / scale))


def passenger_view(frame: pd.DataFrame, *, max_rows: int | None = None,
                   seed: int = RANDOM_STATE) -> PassengerData:
    """The lessons' view of a passenger list: known ages only, three features, the map grid and the query.

    Args:
        frame: The passenger list, one row per passenger, as titanic_anim loads it.
        max_rows: Subsample this many passengers (before the split) for a quick run; None keeps them all.
        seed: Seeds the order the training passengers appear in on screen.

    Returns:
        The passenger data every lesson reads.

    Raises:
        ValueError: If ``max_rows`` is below titanic_anim's minimum, the frame is not a passenger list, or
            no test passenger has the query's sex.
    """
    known = frame[frame["age"].notna()]
    splits = split_and_scale(known, max_rows=max_rows)
    x_test = _features(splits, splits.X_test)
    ages = np.linspace(*AGE_RANGE, GRID_AGES)
    log_fares = np.linspace(0.0, np.log1p(FARE_MAX), GRID_FARES)
    return PassengerData(
        x_train=_features(splits, splits.X_train),
        y_train=read_only_labels(splits.y_train),
        x_val=_features(splits, splits.X_val),
        y_val=read_only_labels(splits.y_val),
        x_test=x_test,
        y_test=read_only_labels(splits.y_test),
        display=read_only_indices(np.random.default_rng(seed).permutation(splits.n_train)),
        ages=read_only(ages),
        log_fares=read_only(log_fares),
        grid=map_grid(ages, log_fares),
        query=pick_query(x_test, splits.y_test),
        baseline_rate=float(np.mean(splits.y_train)),
        dropped=len(frame) - len(known),
        test_class=read_only_indices(splits.raw_test[:, splits.feature_names.index("pclass")]),
        full=full_features(splits),
    )


def load_passengers(fetcher: Callable = fetch_titanic, max_rows: int | None = None) -> PassengerData:
    """Loads the passenger list and builds the lessons' view of it: the command line's ``load_data``.

    Args:
        fetcher: Loads the Titanic passenger list (``titanic_anim.data.fetch_titanic``); tests pass a
            small stand-in.
        max_rows: Subsample this many passengers for a quick run; None keeps them all.

    Returns:
        The passenger data every lesson reads.

    Raises:
        DataLoadError: If the passenger list cannot be fetched or read.
        ValueError: If ``max_rows`` is too small or the data cannot be split.
    """
    return passenger_view(load_titanic_frame(fetcher), max_rows=max_rows)
