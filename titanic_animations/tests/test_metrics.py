import numpy as np
import pytest

from titanic_anim.metrics import roc_points, score


def test_scores_match_hand_counts():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    p = np.array([0.9, 0.4, 0.2, 0.6, 0.8])
    s = score(y_true, y_pred, p)
    assert (s.tn, s.fp, s.fn, s.tp) == (1, 1, 1, 2)
    assert s.accuracy == pytest.approx(0.6)
    assert s.error_rate == pytest.approx(0.4)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)
    assert s.auc == pytest.approx(5 / 6)
    assert np.isfinite(s.log_loss)
    np.testing.assert_array_equal(s.confusion, [[1, 1], [1, 2]])


def test_margins_give_auc_but_no_log_loss():
    s = score(np.array([0, 1]), np.array([0, 1]), np.array([-1.5, 2.0]), probabilities=False)
    assert s.auc == 1.0
    assert np.isnan(s.log_loss)


def test_single_class_has_no_auc():
    s = score(np.array([1, 1]), np.array([1, 0]), np.array([0.7, 0.4]))
    assert np.isnan(s.auc)


def test_labels_only_scores():
    s = score(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert np.isnan(s.auc) and np.isnan(s.log_loss)
    assert s.precision == 1.0


@pytest.mark.parametrize("y_true, y_pred, y_score", [
    (np.array([0, 1]), np.array([0]), None),
    (np.array([[0, 1]]), np.array([[0, 1]]), None),
    (np.array([], dtype=int), np.array([], dtype=int), None),
    (np.array([0, 2]), np.array([0, 1]), None),
    (np.array([0, 1]), np.array([0, 1]), np.array([0.2, np.nan])),
])
def test_score_validates_inputs(y_true, y_pred, y_score):
    with pytest.raises(ValueError):
        score(y_true, y_pred, y_score)


def test_roc_points_handle_one_class():
    fpr, tpr = roc_points(np.array([1, 1]), np.array([0.3, 0.6]))
    np.testing.assert_array_equal(fpr, [0.0, 1.0])
    fpr, tpr = roc_points(np.array([0, 1, 1]), np.array([0.1, 0.8, 0.6]))
    assert fpr[0] == 0.0 and tpr[-1] == 1.0
