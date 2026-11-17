import dataclasses

import numpy as np
import pytest

from house_anim.lessons.config import QUERY_TARGET_INCOME
from house_anim.lessons.data import income_view, pick_query, subsample


def test_income_view_recovers_raw_income(splits, housing_frame):
    data = income_view(splits)
    raw = np.round(housing_frame["MedInc"].to_numpy(), 6)
    assert np.isin(np.round(data.x_train, 6), raw).all()
    assert np.isin(np.round(data.x_test, 6), raw).all()
    assert data.x_train.shape == data.y_train.shape == (splits.n_train,)
    assert len(data.x_val) == len(splits.y_val)
    assert len(data.x_test) == len(splits.y_test)


def test_arrays_are_read_only(income_data):
    arrays = (income_data.x_train, income_data.y_train, income_data.x_val, income_data.y_val,
              income_data.x_test, income_data.y_test, income_data.grid, income_data.display,
              income_data.residual)
    assert not any(array.flags.writeable for array in arrays)


def test_display_and_residual_rows_are_deterministic_training_rows(splits):
    first, second = income_view(splits), income_view(splits)
    np.testing.assert_array_equal(first.display, second.display)
    assert len(set(first.display.tolist())) == len(first.display) <= first.n_train
    assert set(first.residual.tolist()) <= set(first.display.tolist())
    np.testing.assert_array_equal(first.residual_x, first.x_train[first.residual])


def test_query_is_the_test_district_nearest_the_target_income(income_data):
    query = income_data.query
    assert income_data.x_test[query.row] == query.income
    assert income_data.y_test[query.row] == query.price
    assert abs(query.income - QUERY_TARGET_INCOME) == np.abs(income_data.x_test - QUERY_TARGET_INCOME).min()


def test_pick_query_never_looks_at_the_price():
    incomes = np.array([3.0, 5.9, 6.4])
    first = pick_query(incomes, np.array([1.0, 2.0, 3.0]), target=6.0)
    second = pick_query(incomes, np.array([9.0, 5.0, 0.1]), target=6.0)
    assert first.row == second.row == 1


def test_pick_query_needs_a_test_district():
    with pytest.raises(ValueError, match="empty"):
        pick_query(np.array([]), np.array([]), target=3.5)


def test_subsample_keeps_the_full_training_average(income_data):
    small = subsample(income_data, 30)
    assert small.baseline_mean == income_data.baseline_mean == pytest.approx(income_data.y_train.mean())


def test_subsample_keeps_validation_test_and_query(income_data):
    small = subsample(income_data, 30)
    assert small.n_train == 30
    np.testing.assert_array_equal(small.x_val, income_data.x_val)
    np.testing.assert_array_equal(small.x_test, income_data.x_test)
    assert small.query == income_data.query
    assert sorted(small.display.tolist()) == list(range(30))
    assert sorted(small.residual.tolist()) == list(range(30))
    assert set(np.round(small.x_train, 9)) <= set(np.round(income_data.x_train, 9))
    assert "30" in small.sample_note


def test_subsample_is_deterministic(income_data):
    np.testing.assert_array_equal(subsample(income_data, 25).x_train, subsample(income_data, 25).x_train)


@pytest.mark.parametrize("size", [1, 10_000])
def test_subsample_rejects_bad_sizes(income_data, size):
    with pytest.raises(ValueError, match="subsample"):
        subsample(income_data, size)


def test_income_view_requires_the_income_feature(splits):
    renamed = tuple("Other" if name == "MedInc" else name for name in splits.feature_names)
    with pytest.raises(ValueError, match="MedInc"):
        income_view(dataclasses.replace(splits, feature_names=renamed))
