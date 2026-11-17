import numpy as np
import pandas as pd
import pytest

from tests.conftest import fake_fetcher, make_titanic_frame
from titanic_anim.data import (
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    DataLoadError,
    encode,
    learn_imputation,
    load_splits,
    load_titanic_frame,
    split_and_scale,
    validate_frame,
)


def test_loader_drops_columns_that_leak_the_outcome(titanic_frame):
    frame = load_titanic_frame(fake_fetcher(titanic_frame))
    assert not set(LEAKY_COLUMNS) & set(frame.columns)


def test_loader_explains_network_failures():
    def offline(as_frame=True):
        raise OSError("no route to host")

    with pytest.raises(DataLoadError, match="internet"):
        load_titanic_frame(offline)


def test_splits_are_stratified_scaled_and_read_only(splits, titanic_frame):
    survivors = pd.to_numeric(titanic_frame["survived"].astype(str)).mean()
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert abs(y.mean() - survivors) < 0.03
    assert splits.feature_names == FEATURE_COLUMNS
    np.testing.assert_allclose(splits.X_train.mean(axis=0), 0.0, atol=1e-9)
    total = splits.n_train + len(splits.y_val) + len(splits.y_test)
    assert total == len(titanic_frame)
    for array in (splits.X_train, splits.y_train, splits.X_test, splits.raw_test, splits.age_known_test):
        assert not array.flags.writeable


def test_raw_features_round_trip_through_the_scaler(splits):
    restored = splits.X_test * splits.feature_scale + splits.feature_mean
    np.testing.assert_allclose(restored, splits.raw_test)


def test_gaps_are_filled_from_training_rows_only(titanic_frame):
    fill = learn_imputation(titanic_frame.iloc[:100])
    encoded = encode(titanic_frame, fill)
    assert np.isfinite(encoded).all()
    missing_age = titanic_frame["age"].isna().to_numpy()
    assert (encoded[missing_age, FEATURE_COLUMNS.index("age")] == fill.age).all()


def test_encoding_one_hot_ports_and_sex():
    rows = pd.DataFrame({"pclass": [1, 3], "sex": ["female", "male"], "age": [30.0, None],
                         "sibsp": [0, 1], "parch": [0, 2], "fare": [80.0, 7.25], "embarked": ["C", None]})
    fill = learn_imputation(rows.assign(embarked=["Q", "Q"]))
    encoded = encode(rows, fill)
    np.testing.assert_allclose(encoded[0], [1, 1, 30, 0, 0, 80, 1, 0])
    np.testing.assert_allclose(encoded[1], [3, 0, 30, 1, 2, 7.25, 0, 1])  # missing port -> Q, age -> median


def test_age_known_flags_track_missing_ages(splits):
    assert splits.age_known_test.dtype == bool
    assert not splits.age_known_test.all()


@pytest.mark.parametrize("change, message", [
    (lambda f: f.drop(columns=["fare"]), "missing required columns"),
    (lambda f: f.head(10), "at least"),
    (lambda f: f.assign(survived=2), "only 0 and 1"),
    (lambda f: f.assign(survived=1), "both survivors"),
    (lambda f: f.assign(pclass=4), "unexpected values"),
    (lambda f: f.assign(sex="unknown"), "unexpected values"),
    (lambda f: f.assign(embarked="X"), "unexpected values"),
])
def test_validation_rejects_bad_frames(titanic_frame, change, message):
    with pytest.raises(ValueError, match=message):
        validate_frame(change(titanic_frame.copy()))


@pytest.mark.parametrize("column, bad", [("fare", np.inf), ("age", "unknown")])
def test_validation_rejects_non_numbers(titanic_frame, column, bad):
    frame = titanic_frame.copy().astype({column: object})
    frame.loc[1, column] = bad
    with pytest.raises(ValueError, match="finite numbers"):
        validate_frame(frame)


def test_corrupt_cache_is_explained():
    def corrupt(as_frame=True):
        raise ValueError("Expecting value: line 1 column 1")

    with pytest.raises(DataLoadError, match="corrupt"):
        load_titanic_frame(corrupt)


def test_validation_rejects_missing_sex(titanic_frame):
    frame = titanic_frame.copy().astype({"sex": object})
    frame.loc[0, "sex"] = None
    with pytest.raises(ValueError, match="must not have missing"):
        validate_frame(frame)


@pytest.mark.parametrize("kwargs", [dict(test_fraction=0.0), dict(val_fraction=1.2),
                                    dict(test_fraction=0.5, val_fraction=0.5), dict(max_rows=0),
                                    dict(max_rows=3)])
def test_split_rejects_bad_arguments(titanic_frame, kwargs):
    with pytest.raises(ValueError):
        split_and_scale(titanic_frame, **kwargs)


def test_load_splits_subsamples(titanic_frame):
    small = load_splits(fake_fetcher(make_titanic_frame(300, seed=3)), max_rows=150)
    assert small.n_train + len(small.y_val) + len(small.y_test) == 150
