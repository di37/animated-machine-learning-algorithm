from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from house_anim.inference import inference_frames
from house_anim.metrics import score


@pytest.fixture(scope="module")
def model(splits):
    return LinearRegression().fit(splits.X_train, splits.y_train)


def test_frames_cover_the_whole_test_set(model, splits):
    frames = tuple(inference_frames(model, splits, batch_size=30))
    assert [f.n_done for f in frames] == [30, 60, 90, 100]
    assert frames[-1].is_final and not frames[0].is_final
    assert all(f.n_total == 100 for f in frames)


def test_final_scores_match_full_test_set(model, splits):
    final = tuple(inference_frames(model, splits, batch_size=25))[-1]
    expected = score(splits.y_test, model.predict(splits.X_test))
    assert final.scores.rmse == pytest.approx(expected.rmse)
    assert final.scores.r2 == pytest.approx(expected.r2)
    np.testing.assert_allclose(final.y_true, splits.y_test)


def test_cumulative_arrays_grow_and_stay_read_only(model, splits):
    frames = tuple(inference_frames(model, splits, batch_size=40))
    for frame in frames:
        assert len(frame.y_pred) == len(frame.y_true) == len(frame.coords) == frame.n_done
        assert not frame.y_pred.flags.writeable
        assert len(frame.history) == frames.index(frame) + 1
    with pytest.raises(FrozenInstanceError):
        frames[0].n_done = 5


def test_batch_bounds_and_throughput(model, splits):
    frames = tuple(inference_frames(model, splits, batch_size=40))
    assert frames[1].batch_start == 40
    assert all(f.batch_latency_ms >= 0.0 for f in frames)
    assert frames[-1].throughput_per_s > 0.0


def test_prepare_for_serving_predicts_single_threaded(splits):
    from sklearn.ensemble import RandomForestRegressor

    from house_anim.inference import prepare_for_serving

    forest = RandomForestRegressor(n_estimators=3, n_jobs=-1, random_state=0)
    forest.fit(splits.X_train, splits.y_train)
    assert prepare_for_serving(forest).n_jobs == 1
    linear = LinearRegression().fit(splits.X_train, splits.y_train)
    assert prepare_for_serving(linear) is linear


@pytest.mark.parametrize("batch_size", [0, -5])
def test_invalid_batch_size_raises(model, splits, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        next(inference_frames(model, splits, batch_size=batch_size))
