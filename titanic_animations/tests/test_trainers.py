from dataclasses import replace

import numpy as np
import pytest

from titanic_anim.data import read_only_labels
from titanic_anim.frames import KIND_COEFFICIENT, KIND_IMPORTANCE, KIND_SENSITIVITY, KIND_SEPARATION
from titanic_anim.inference import inference_frames, prepare_for_serving
from titanic_anim.models import SCORE_MARGIN, get_spec, model_keys
from titanic_anim.trainers import (
    NumericalError,
    planned_steps,
    rows_with_both_classes,
    survival_scores,
    train_silently,
    training_frames,
    training_loss,
)

EXPECTED_WEIGHTS = {
    "logistic_regression": KIND_COEFFICIENT, "sgd": KIND_COEFFICIENT, "svm": None,
    "naive_bayes": KIND_SEPARATION, "decision_tree": KIND_IMPORTANCE, "random_forest": KIND_IMPORTANCE,
    "gradient_boosting": KIND_IMPORTANCE, "knn": None, "mlp": KIND_SENSITIVITY,
}


def short(key):
    spec = get_spec(key)
    return spec if spec.strategy == "data_growth" else replace(spec, steps=spec.steps[:3])


@pytest.mark.parametrize("key", model_keys())
def test_every_model_trains_frame_by_frame(key, splits):
    spec = short(key)
    frames = list(training_frames(spec, splits))
    assert [f.step for f in frames] == list(planned_steps(spec, splits))
    final = frames[-1]
    assert final.is_final and len(final.history) == len(frames)
    assert final.val_pred.shape == splits.y_val.shape == final.val_score.shape
    assert 0.0 <= final.val_scores.accuracy <= 1.0
    kind = EXPECTED_WEIGHTS[key]
    assert (final.weights is None) if kind is None else final.weights.kind == kind
    assert final.model_fit_seconds <= final.fit_seconds + 1e-12


def test_training_returns_the_final_model(splits):
    model, frame = train_silently(short("decision_tree"), splits)
    assert model.max_depth == 3 and frame.is_final


def test_logistic_sweep_starts_near_the_majority_vote(splits):
    frames = list(training_frames(get_spec("logistic_regression"), splits))
    assert frames[-1].val_scores.accuracy > frames[0].val_scores.accuracy
    assert frames[-1].history[-1].train_loss < frames[0].history[0].train_loss


def test_growth_starts_once_both_outcomes_are_present(splits):
    steps = planned_steps(get_spec("knn"), splits)
    assert steps[0] >= max(20, rows_with_both_classes(splits.y_train))
    assert steps[-1] == splits.n_train


def test_rows_with_both_classes():
    assert rows_with_both_classes(np.array([0, 0, 1, 0])) == 3
    with pytest.raises(ValueError):
        rows_with_both_classes(np.array([1, 1]))


def test_svm_scores_are_margins(splits):
    model, frame = train_silently(get_spec("svm"), splits)
    scores = survival_scores(model, splits.X_val, SCORE_MARGIN)
    np.testing.assert_array_equal(scores > 0, frame.val_pred == 1)
    assert np.isnan(frame.val_scores.log_loss)


def test_training_loss_matches_definitions():
    y = np.array([1, 0])
    assert training_loss(y, np.array([2.0, -0.5]), SCORE_MARGIN) == pytest.approx(0.25)
    assert training_loss(y, np.array([0.5, 0.5]), "probability") == pytest.approx(np.log(2))


def test_non_finite_scores_raise():
    class Broken:
        def predict_proba(self, X):
            return np.full((len(X), 2), np.nan)

    with pytest.raises(NumericalError):
        survival_scores(Broken(), np.zeros((2, 8)), "probability")


def test_labels_outside_the_classes_raise():
    from titanic_anim.trainers import checked_labels

    class Odd:
        def predict(self, X):
            return np.full(len(X), 7)

    with pytest.raises(NumericalError):
        checked_labels(Odd(), np.zeros((2, 8)))


def test_unknown_strategy_is_rejected(splits):
    with pytest.raises(ValueError):
        planned_steps(replace(get_spec("knn"), strategy="nope"), splits)
    with pytest.raises(ValueError):
        next(training_frames(replace(get_spec("knn"), strategy="nope"), splits))


def test_inference_streams_every_test_passenger(splits):
    spec = get_spec("random_forest")
    model, _ = train_silently(replace(spec, steps=(5,)), splits)
    frames = list(inference_frames(prepare_for_serving(model), splits, spec.score_kind, batch_size=25))
    final = frames[-1]
    assert final.is_final and final.n_done == len(splits.y_test)
    np.testing.assert_array_equal(final.y_pred, model.predict(splits.X_test))
    assert final.scores.accuracy == pytest.approx(np.mean(final.y_pred == splits.y_test))
    assert frames[1].batch_start == 25
    assert all(f.batch_latency_ms >= 0 for f in frames)


def test_inference_rejects_bad_batches(splits):
    with pytest.raises(ValueError):
        next(inference_frames(object(), splits, "probability", batch_size=0))


def test_read_only_labels_are_ints():
    labels = read_only_labels([1.0, 0.0])
    assert labels.dtype.kind == "i" and not labels.flags.writeable
