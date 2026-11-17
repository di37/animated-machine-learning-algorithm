import numpy as np
import pytest

from house_anim.data import (
    DataLoadError,
    FEATURE_COLUMNS,
    load_housing_frame,
    load_splits,
    split_and_scale,
)
from tests.conftest import fake_fetcher, make_housing_frame


def test_split_sizes_follow_fractions(housing_frame):
    result = split_and_scale(housing_frame, test_fraction=0.2, val_fraction=0.2)
    assert len(result.y_test) == 100
    assert len(result.y_val) == 100
    assert result.n_train == 300
    assert result.X_train.shape == (300, len(FEATURE_COLUMNS))


def test_scaler_is_fit_on_training_rows_only(splits):
    np.testing.assert_allclose(splits.X_train.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(splits.X_train.std(axis=0), 1.0, atol=1e-9)
    assert not np.allclose(splits.X_val.mean(axis=0), 0.0, atol=1e-9)


def test_test_coordinates_stay_in_raw_degrees(splits):
    longitude, latitude = splits.coords_test[:, 0], splits.coords_test[:, 1]
    assert longitude.min() >= -124.3 and longitude.max() <= -114.3
    assert latitude.min() >= 32.5 and latitude.max() <= 42.0
    assert len(splits.coords_test) == len(splits.y_test)


def test_arrays_are_read_only(splits):
    for array in (splits.X_train, splits.y_train, splits.X_test, splits.coords_test):
        assert not array.flags.writeable


def test_split_does_not_mutate_input_frame(housing_frame):
    before = housing_frame.copy()
    split_and_scale(housing_frame)
    assert housing_frame.equals(before)


def test_max_rows_subsamples(housing_frame):
    result = split_and_scale(housing_frame, max_rows=100)
    assert result.n_train + len(result.y_val) + len(result.y_test) == 100


@pytest.mark.parametrize("test_fraction, val_fraction", [(0.0, 0.2), (0.5, 0.5), (0.2, -0.1)])
def test_invalid_fractions_raise(housing_frame, test_fraction, val_fraction):
    with pytest.raises(ValueError, match="fraction"):
        split_and_scale(housing_frame, test_fraction=test_fraction, val_fraction=val_fraction)


def test_missing_column_raises(housing_frame):
    with pytest.raises(ValueError, match="AveOccup"):
        split_and_scale(housing_frame.drop(columns=["AveOccup"]))


def test_missing_values_raise(housing_frame):
    broken = housing_frame.assign(MedInc=housing_frame["MedInc"].where(housing_frame.index != 3))
    with pytest.raises(ValueError, match="missing or infinite"):
        split_and_scale(broken)


def test_too_few_rows_raise():
    with pytest.raises(ValueError, match="at least"):
        split_and_scale(make_housing_frame(n_rows=10))


def test_max_rows_must_be_positive(housing_frame):
    with pytest.raises(ValueError, match="max_rows"):
        split_and_scale(housing_frame, max_rows=0)


def test_load_housing_frame_uses_fetcher(housing_frame):
    loaded = load_housing_frame(fetcher=fake_fetcher(housing_frame))
    assert loaded.equals(housing_frame)


def test_load_housing_frame_wraps_network_errors():
    def failing_fetcher(as_frame=True):
        raise OSError("no network")

    with pytest.raises(DataLoadError, match="California Housing"):
        load_housing_frame(fetcher=failing_fetcher)


def test_load_splits_end_to_end(housing_frame):
    result = load_splits(fetcher=fake_fetcher(housing_frame), max_rows=200)
    assert result.feature_names == FEATURE_COLUMNS
    assert result.n_train == 120


def test_training_coordinates_and_scaler_parameters_are_kept(splits):
    lon = FEATURE_COLUMNS.index("Longitude")
    restored = splits.X_train[:, lon] * splits.feature_scale[lon] + splits.feature_mean[lon]
    np.testing.assert_allclose(restored, splits.coords_train[:, 0])
    assert splits.coords_train.shape == (splits.n_train, 2)
    assert not splits.feature_mean.flags.writeable
