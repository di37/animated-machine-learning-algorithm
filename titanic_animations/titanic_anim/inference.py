"""Stream the held-out test passengers through a fitted classifier in batches, timing each call.

The trained model sees the test passengers for the first time here, a small batch per frame, the
way a served model would. Only the ``predict`` call is timed, so the latency and throughput on the
dashboard are the model's own. animkit's ``prepare_for_serving`` (re-exported here) first switches
the model to one thread, since handing a small batch to threads costs more than predicting it.
"""

from collections.abc import Iterator
from time import perf_counter

import numpy as np

from animkit.data import read_only
from animkit.inference import prepare_for_serving
from titanic_anim.config import DEFAULT_INFERENCE_BATCH
from titanic_anim.data import TitanicSplits, read_only_labels
from titanic_anim.frames import InferenceFrame, MetricPoint
from titanic_anim.metrics import score
from titanic_anim.trainers import survival_scores

__all__ = ["inference_frames", "prepare_for_serving"]


def inference_frames(
    model, splits: TitanicSplits, score_kind: str, batch_size: int = DEFAULT_INFERENCE_BATCH
) -> Iterator[InferenceFrame]:
    """Classifies the test passengers batch by batch, yielding a frame after each batch.

    Each batch is one timed ``model.predict``. Its scores (``predict_proba`` or ``decision_function``)
    are computed outside the timer, so latency and throughput measure ``predict`` alone. The running
    scores cover every passenger classified so far.

    Args:
        model: A fitted classifier, best prepared with ``prepare_for_serving``.
        splits: The splits; only the test passengers are classified.
        score_kind: ``SCORE_PROBABILITY`` or ``SCORE_MARGIN``: which score to collect.
        batch_size: Test passengers per ``predict`` call, and so per frame.

    Yields:
        An ``InferenceFrame`` per batch, the last covering every test passenger.

    Raises:
        ValueError: If ``batch_size`` is below 1.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    n_total = len(splits.y_test)
    y_pred, y_score = read_only_labels(np.empty(0)), read_only(np.empty(0))
    history: tuple[MetricPoint, ...] = ()
    predict_seconds = 0.0

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = splits.X_test[start:stop]
        started = perf_counter()
        raw_pred = model.predict(batch)
        latency = perf_counter() - started
        batch_pred = read_only_labels(raw_pred)
        batch_score = survival_scores(model, batch, score_kind)
        predict_seconds += latency

        y_pred = read_only_labels(np.concatenate([y_pred, batch_pred]))
        y_score = read_only(np.concatenate([y_score, batch_score]))
        y_true = splits.y_test[:stop]
        scores = score(y_true, y_pred)
        history = history + (MetricPoint(stop, scores.accuracy, scores.precision, scores.recall, scores.f1),)
        yield InferenceFrame(
            n_done=stop,
            n_total=n_total,
            batch_start=start,
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            history=history,
            batch_latency_ms=latency * 1_000.0,
            throughput_per_s=stop / predict_seconds if predict_seconds > 0 else float("inf"),
        )
