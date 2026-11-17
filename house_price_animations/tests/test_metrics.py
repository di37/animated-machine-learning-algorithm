import math

import numpy as np
import pytest

from house_anim.metrics import score, share_within


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    result = score(y, y)
    assert result.rmse == 0.0
    assert result.mae == 0.0
    assert result.r2 == 1.0


def test_known_values():
    result = score(np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert result.rmse == pytest.approx(1.0)
    assert result.mae == pytest.approx(1.0)
    assert result.r2 == pytest.approx(0.0)


def test_single_point_has_undefined_r2():
    result = score(np.array([2.0]), np.array([1.5]))
    assert result.rmse == pytest.approx(0.5)
    assert math.isnan(result.r2)


def test_scores_are_frozen():
    result = score(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(AttributeError):
        result.rmse = 3.0


@pytest.mark.parametrize(
    "y_true, y_pred, message",
    [
        (np.array([1.0, 2.0]), np.array([1.0]), "same shape"),
        (np.array([]), np.array([]), "at least one"),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), "finite"),
        (np.ones((2, 2)), np.ones((2, 2)), "1-D"),
    ],
)
def test_invalid_inputs_raise(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        score(y_true, y_pred)


def test_share_within_tolerance():
    residuals = np.array([-0.6, -0.2, 0.0, 0.4, 1.0])
    assert share_within(residuals, 0.5) == pytest.approx(0.6)
    assert share_within(np.array([]), 0.5) == 0.0


def test_format_step_uses_typographic_minus():
    from house_anim.panels import format_step

    assert format_step(-3.333) == "−3.33"
    assert format_step(1240.0) == "1,240"
