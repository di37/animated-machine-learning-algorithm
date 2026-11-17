from dataclasses import replace

import numpy as np
import pytest

from titanic_anim.equations import prediction_card, training_card
from titanic_anim.frames import KIND_SEPARATION, FeatureWeights
from titanic_anim.inference import inference_frames
from titanic_anim.inference_dashboard import InferenceDashboard, format_rate
from titanic_anim.manifest import manifest_layout
from titanic_anim.models import get_spec
from titanic_anim.passengers import describe, feature_text, ordinal, person, split_label
from titanic_anim.trainers import train_silently, training_steps
from titanic_anim.training_dashboard import TrainingDashboard

SMALL = (0.48, 0.96)  # even a small canvas must lay out without errors


@pytest.mark.parametrize("key", ["logistic_regression", "svm", "naive_bayes", "knn"])
def test_training_dashboard_follows_the_frames(key, splits):
    spec = get_spec(key)
    spec = spec if spec.strategy == "data_growth" else replace(spec, steps=spec.steps[:3])
    dashboard = TrainingDashboard(spec, splits)
    for step in training_steps(spec, splits):
        dashboard.render(step.frame, training_card(step.model, step.frame, spec, splits), model=step.model)
    dashboard.figure.canvas.draw()
    assert len(dashboard.train_line.get_xdata()) == step.frame.total_steps
    assert "validation accuracy" in dashboard.status_text.get_text()
    if spec.no_weights_note and step.frame.weights is None:
        assert dashboard.weights_note.get_visible()


def test_training_dashboard_rejects_mismatched_frames(splits):
    spec = replace(get_spec("decision_tree"), steps=(2,))
    _, frame = train_silently(spec, splits)
    dashboard = TrainingDashboard(spec, splits)
    with pytest.raises(ValueError):
        dashboard.render(replace(frame, val_pred=frame.val_pred[:-1]))


def test_separation_bars_are_signed(splits):
    dashboard = TrainingDashboard(get_spec("naive_bayes"), splits)
    values = np.linspace(-1, 1, len(splits.feature_names))
    dashboard.weights.update(FeatureWeights(splits.feature_names, values, KIND_SEPARATION))
    assert dashboard.weights.ax.get_xlim()[0] < 0
    assert "σ" in dashboard.weights.ax.get_title(loc="left")


@pytest.mark.parametrize("key", ["random_forest", "svm"])
def test_inference_dashboard_fills_the_manifest(key, splits):
    spec = get_spec(key)
    model, _ = train_silently(replace(spec, steps=spec.steps[:3]) if spec.strategy != "data_growth" else spec,
                              splits)
    dashboard = InferenceDashboard(spec, splits)
    for frame in inference_frames(model, splits, spec.score_kind, batch_size=40):
        x = splits.X_test[frame.n_done - 1]
        dashboard.render(frame, prediction_card(model, x, frame.n_done, splits), model=model, passenger=x)
    dashboard.figure.canvas.draw()
    assert len(dashboard.manifest.filled.get_offsets()) == len(splits.y_test)
    wrong = int(np.sum(frame.y_pred != frame.y_true))
    assert len(dashboard.manifest.wrong.get_offsets()) == wrong
    assert "Latest: passenger" in dashboard.ticker_text.get_text()
    assert ("f(x)" in dashboard.ticker_text.get_text()) == (spec.score_kind == "margin")


def test_manifest_groups_by_class_and_sex(splits):
    positions, labels, width, height = manifest_layout(splits.raw_test)
    assert len({tuple(p) for p in positions}) == len(positions)  # one square per passenger
    women = splits.raw_test[:, 1] >= 0.5
    assert positions[women, 0].max() < positions[~women, 0].min()
    assert [text.split()[0] for _, text in labels] == ["1st", "2nd", "3rd"]
    assert positions[:, 1].max() < height and positions[:, 0].max() < width


def test_wording_helpers():
    raw = np.array([3, 0, 22, 1, 0, 7.25, 0, 0], dtype=float)
    assert describe(raw) == "3rd class · man · age 22 · fare £7.25 · 1 sibling/spouse, 0 parent/child aboard"
    assert "age unknown" in describe(raw, age_known=False)
    assert person(np.array([1, 1, 8, 0, 2, 30, 1, 0], dtype=float)) == "girl"
    assert ordinal(2) == "2nd" and ordinal(5) == "5th"
    assert split_label("female", 0.5) == "male?"
    assert split_label("embarked_C", 0.5) == "not Cherbourg?"
    assert split_label("pclass", 2.5) == "class ≤ 2?"
    assert split_label("fare", 52.3) == "fare ≤ £52?"
    assert split_label("parch", 1.5) == "parch ≤ 1?"
    assert split_label("age", 9.5) == "age ≤ 9.5?"
    assert feature_text("female", 1.0) == "female = yes"
    assert feature_text("fare", 7.25) == "fare = £7.25"
    assert feature_text("age", 30.0) == "age = 30"
    assert feature_text("sibsp", 2.0) == "sibsp = 2"
    assert format_rate(float("inf")) == "—"
    baby = np.array([3, 1, 0.42, 0, 2, 15.0, 0, 0], dtype=float)
    assert "age 5 months" in describe(baby) and feature_text("age", 0.42) == "age = 5 months"


def test_early_noisy_rates_do_not_set_the_axis_floor():
    import matplotlib.pyplot as plt

    from titanic_anim.panels import RateLines

    chart = RateLines(plt.figure().add_subplot(), ("accuracy", "recall"))
    x = np.array([4.0, 40.0, 80.0])
    chart.update(x, (np.array([0.5, 0.8, 0.82]), np.array([0.0, 0.7, 0.72])), floor_from=30)
    assert chart.ax.get_ylim() == (0.5, 1.0)
    chart.update(x, (np.array([0.5, 0.8, 0.82]), np.array([0.0, 0.3, 0.72])), floor_from=30)
    assert chart.ax.get_ylim()[0] == pytest.approx(0.3)
