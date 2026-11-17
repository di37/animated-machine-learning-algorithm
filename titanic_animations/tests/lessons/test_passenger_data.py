import numpy as np
import pytest

from tests.conftest import make_titanic_frame
from titanic_anim.lessons.config import AGE, FEMALE, LOG_FARE, QUERY_TARGET
from titanic_anim.lessons.data import passenger_view, pick_query


def test_passengers_without_a_recorded_age_are_left_out():
    frame = make_titanic_frame()
    data = passenger_view(frame)
    known = int(frame["age"].notna().sum())
    assert data.dropped == len(frame) - known
    assert data.n_train + len(data.y_val) + len(data.y_test) == known


def test_features_are_age_log_fare_and_sex_in_real_units(passenger_data):
    frame = make_titanic_frame()
    ages = set(np.round(frame["age"].dropna(), 6))
    assert set(np.round(passenger_data.x_train[:, AGE], 6)) <= ages
    assert set(np.unique(passenger_data.x_train[:, FEMALE])) <= {0.0, 1.0}
    fares = np.expm1(passenger_data.x_train[:, LOG_FARE])
    assert fares.min() >= 0.0 and np.isfinite(fares).all()


def test_arrays_are_read_only(passenger_data):
    arrays = (passenger_data.x_train, passenger_data.y_train, passenger_data.x_val, passenger_data.y_val,
              passenger_data.x_test, passenger_data.y_test, passenger_data.grid, passenger_data.display)
    assert not any(array.flags.writeable for array in arrays)


def test_grid_covers_both_sexes_in_map_order(passenger_data):
    sexes, rows, cols = passenger_data.surface_shape
    grid = passenger_data.grid
    assert sexes == 2 and grid.shape == (2 * rows * cols, 3)
    assert (grid[:rows * cols, FEMALE] == 1.0).all() and (grid[rows * cols:, FEMALE] == 0.0).all()


def test_display_holds_every_training_passenger_once(passenger_data):
    assert sorted(passenger_data.display.tolist()) == list(range(passenger_data.n_train))


def test_baseline_is_the_training_survival_rate(passenger_data):
    assert passenger_data.baseline_rate == pytest.approx(passenger_data.y_train.mean())


def test_query_is_chosen_from_the_features_alone(passenger_data):
    query = passenger_data.query
    np.testing.assert_array_equal(passenger_data.x_test[query.row], query.features)
    assert query.survived == passenger_data.y_test[query.row]
    flipped = pick_query(passenger_data.x_test, 1 - passenger_data.y_test, QUERY_TARGET)
    assert flipped.row == query.row


def test_query_matches_the_target_sex(passenger_data):
    assert passenger_data.query.features[FEMALE] == QUERY_TARGET[FEMALE]


def test_pick_query_needs_a_passenger_of_that_sex():
    men_only = np.array([[30.0, 2.0, 0.0]])
    with pytest.raises(ValueError, match="no test passenger"):
        pick_query(men_only, np.array([0]), (30.0, 2.0, 1.0))
