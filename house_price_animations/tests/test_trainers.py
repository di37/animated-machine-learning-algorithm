from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from house_anim.frames import KIND_COEFFICIENT, KIND_IMPORTANCE, KIND_SENSITIVITY
from house_anim.models import get_spec, model_keys
from house_anim.trainers import (
    NumericalError,
    checked_predict,
    growth_sizes,
    planned_steps,
    train_silently,
    training_frames,
)


def small_spec(key: str):
    """Trim each schedule so the whole suite stays fast on the synthetic data."""
    spec = get_spec(key)
    if spec.strategy == "data_growth":
        return spec
    return replace(spec, steps=spec.steps[:3])


def drain(generator):
    frames = []
    while True:
        try:
            frames.append(next(generator))
        except StopIteration as stop:
            return tuple(frames), stop.value


@pytest.mark.parametrize("key", model_keys())
def test_every_model_streams_frames_and_returns_fitted_model(key, splits):
    spec = small_spec(key)
    frames, model = drain(training_frames(spec, splits))

    assert len(frames) == frames[0].total_steps == len(planned_steps(spec, splits))
    assert [len(f.history) for f in frames] == list(range(1, len(frames) + 1))
    assert frames[-1].is_final and not frames[0].is_final or len(frames) == 1
    assert frames[-1].val_pred.shape == splits.y_val.shape
    assert model.predict(splits.X_test).shape == splits.y_test.shape


@pytest.mark.parametrize("key", model_keys())
def test_frames_are_immutable(key, splits):
    frames, _ = drain(training_frames(small_spec(key), splits))
    frame = frames[-1]
    with pytest.raises(FrozenInstanceError):
        frame.step = 99.0
    assert not frame.val_pred.flags.writeable
    if frame.weights is not None:
        assert not frame.weights.values.flags.writeable


@pytest.mark.parametrize(
    "key, kind",
    [
        ("linear_regression", KIND_COEFFICIENT),
        ("ridge", KIND_COEFFICIENT),
        ("lasso", KIND_COEFFICIENT),
        ("sgd", KIND_COEFFICIENT),
        ("decision_tree", KIND_IMPORTANCE),
        ("random_forest", KIND_IMPORTANCE),
        ("gradient_boosting", KIND_IMPORTANCE),
        ("mlp", KIND_SENSITIVITY),
        ("knn", None),
    ],
)
def test_weight_kind_matches_model_family(key, kind, splits):
    frames, _ = drain(training_frames(small_spec(key), splits))
    weights = frames[-1].weights
    if kind is None:
        assert weights is None
    else:
        assert weights.kind == kind
        assert weights.names == splits.feature_names
        assert weights.values.shape == (len(splits.feature_names),)


def test_boosting_validation_error_falls(splits):
    spec = replace(get_spec("gradient_boosting"), steps=(1, 20, 60))
    frames, _ = drain(training_frames(spec, splits))
    val_rmse = [point.val_rmse for point in frames[-1].history]
    assert val_rmse[-1] < val_rmse[0]


def test_incremental_steps_are_fractional_epochs(splits):
    spec = replace(get_spec("sgd"), steps=(1, 2), chunks_per_epoch=4)
    frames, _ = drain(training_frames(spec, splits))
    assert [f.step for f in frames] == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    assert frames[0].x_range == (0.0, 2.0)


def test_ridge_shrinks_coefficients_smoothly_without_zeroing_them(splits):
    spec = replace(get_spec("ridge"), steps=(1e5, 1e3, 1.0))
    frames, _ = drain(training_frames(spec, splits))
    sizes = [np.abs(f.weights.values).sum() for f in frames]
    assert sizes == sorted(sizes)
    assert np.all(frames[0].weights.values != 0.0)


def test_param_sweep_applies_step_transform(splits):
    spec = replace(get_spec("lasso"), steps=(1.0, 0.1, 0.01))
    frames, _ = drain(training_frames(spec, splits))
    np.testing.assert_allclose([f.step for f in frames], [0.0, 1.0, 2.0])


def test_fit_seconds_only_grow(splits):
    frames, _ = drain(training_frames(small_spec("random_forest"), splits))
    elapsed = [f.fit_seconds for f in frames]
    assert elapsed == sorted(elapsed) and elapsed[0] > 0.0


def test_first_epoch_only_counts_rows_already_consumed(splits):
    from house_anim.trainers import _incremental_fits

    spec = replace(get_spec("sgd"), steps=(1, 2), chunks_per_epoch=4)
    seen = [rows for _, _, rows in _incremental_fits(spec, splits)]
    quarter = splits.n_train // 4
    assert seen[:4] == [quarter, 2 * quarter, 3 * quarter, splits.n_train]
    assert seen[4:] == [splits.n_train] * 4


@pytest.mark.parametrize("key", ["decision_tree", "lasso", "knn"])
def test_refit_models_report_final_fit_time_not_the_sweep_total(key, splits):
    frames, _ = drain(training_frames(small_spec(key), splits))
    final = frames[-1]
    assert 0.0 < final.model_fit_seconds < final.fit_seconds


@pytest.mark.parametrize("key", ["random_forest", "sgd"])
def test_cumulative_models_report_total_fit_time(key, splits):
    frames, _ = drain(training_frames(small_spec(key), splits))
    assert frames[-1].model_fit_seconds == frames[-1].fit_seconds


def test_growth_sizes_are_unique_clamped_and_end_at_full_set():
    assert growth_sizes((0.001, 0.002, 0.5, 1.0), n_train=1000, minimum=20) == (20, 500, 1000)


def test_train_silently_returns_final_frame(splits):
    model, final = train_silently(small_spec("decision_tree"), splits)
    assert final.is_final
    assert model.predict(splits.X_val).shape == splits.y_val.shape


def test_checked_predict_rejects_non_finite_output(splits):
    class Exploding:
        def predict(self, X):
            return np.full(len(X), np.inf)

    with pytest.raises(NumericalError, match="non-finite"):
        checked_predict(Exploding(), splits.X_val)


def test_unknown_strategy_raises(splits):
    spec = replace(get_spec("lasso"), strategy="genetic")
    with pytest.raises(ValueError, match="genetic"):
        next(training_frames(spec, splits))
