import pytest
from sklearn.base import RegressorMixin

from house_anim.models import (
    MODEL_SPECS,
    STRATEGIES,
    get_spec,
    model_keys,
)

EXPECTED_KEYS = {
    "linear_regression",
    "ridge",
    "lasso",
    "sgd",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "knn",
    "mlp",
}


def test_registry_covers_expected_algorithms():
    assert set(model_keys()) == EXPECTED_KEYS


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_each_spec_builds_a_fresh_regressor(key):
    spec = get_spec(key)
    first, second = spec.build(), spec.build()
    assert isinstance(first, RegressorMixin)
    assert first is not second
    assert spec.strategy in STRATEGIES
    assert len(spec.steps) >= 2
    assert spec.hold_frames >= 1


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_param_strategies_name_a_real_parameter(key):
    spec = get_spec(key)
    if spec.param_name is not None:
        assert spec.param_name in spec.build().get_params()


def test_unknown_key_lists_choices():
    with pytest.raises(ValueError, match="random_forest"):
        get_spec("xgboost")


def test_registry_is_read_only():
    with pytest.raises(TypeError):
        MODEL_SPECS["new"] = get_spec("lasso")


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_specs_have_short_titles_and_step_formats(key):
    spec = get_spec(key)
    assert 0 < len(spec.short_title) <= 18
    assert "{}" in spec.step_format
