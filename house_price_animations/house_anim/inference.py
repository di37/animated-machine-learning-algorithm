"""Stream the held-out test houses through a fitted model in batches, timing each predict call.

This is the model at work on houses it has never seen. Each batch is one real ``predict`` call,
timed on its own, so the dashboard's latency and throughput are the model's cost and nothing else.
Call ``prepare_for_serving`` (from animkit) on the model first: it makes ``predict`` single-threaded,
since on small batches thread dispatch would cost more than the prediction.
"""

from collections.abc import Iterator
from time import perf_counter

import numpy as np

from animkit.inference import prepare_for_serving
from house_anim.config import DEFAULT_INFERENCE_BATCH
from house_anim.data import HousingSplits, read_only
from house_anim.frames import InferenceFrame, MetricPoint
from house_anim.metrics import score
from house_anim.trainers import checked_predict

__all__ = ["inference_frames", "prepare_for_serving"]


def inference_frames(
    model, splits: HousingSplits, batch_size: int = DEFAULT_INFERENCE_BATCH
) -> Iterator[InferenceFrame]:
    """Predicts the test houses in order, batch by batch, with running scores after each batch.

    Args:
        model: A fitted regressor.
        splits: The data; only its test rows are used.
        batch_size: Test houses per ``predict`` call (and so per frame).

    Yields:
        One ``InferenceFrame`` per batch, holding every prediction so far.

    Raises:
        ValueError: If ``batch_size`` is less than 1.
        NumericalError: If the model predicts NaN or infinite values.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    n_total = len(splits.y_test)
    y_pred = read_only(np.empty(0))
    history: tuple[MetricPoint, ...] = ()
    predict_seconds = 0.0

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        started = perf_counter()
        batch_pred = checked_predict(model, splits.X_test[start:stop])
        latency = perf_counter() - started
        predict_seconds += latency

        y_pred = read_only(np.concatenate([y_pred, batch_pred]))
        y_true = splits.y_test[:stop]
        scores = score(y_true, y_pred)
        history = history + (MetricPoint(stop, scores.rmse, scores.mae, scores.r2),)
        yield InferenceFrame(
            n_done=stop,
            n_total=n_total,
            batch_start=start,
            y_true=y_true,
            y_pred=y_pred,
            coords=splits.coords_test[:stop],
            history=history,
            batch_latency_ms=latency * 1_000.0,
            throughput_per_s=stop / predict_seconds if predict_seconds > 0 else float("inf"),
        )
