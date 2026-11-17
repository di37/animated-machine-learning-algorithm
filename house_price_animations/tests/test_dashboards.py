from dataclasses import replace

import pytest

from house_anim.inference import inference_frames
from house_anim.inference_dashboard import InferenceDashboard
from house_anim.models import get_spec
from house_anim.trainers import train_silently, training_frames
from house_anim.training_dashboard import TrainingDashboard


def short_spec(key):
    return replace(get_spec(key), steps=get_spec(key).steps[:3])


@pytest.mark.parametrize("key", ["lasso", "random_forest", "knn", "mlp"])
def test_training_dashboard_renders_every_frame(key, splits):
    spec = get_spec(key) if key == "knn" else short_spec(key)
    dashboard = TrainingDashboard(spec, splits)
    last = None
    for frame in training_frames(spec, splits):
        dashboard.render(frame)
        last = frame
    dashboard.figure.canvas.draw()

    assert len(dashboard.scatter.get_offsets()) == len(splits.y_val)
    assert f"{last.step_index + 1}/{last.total_steps}" in dashboard.status_text.get_text()
    assert len(dashboard.train_line.get_xdata()) == len(last.history)
    if last.weights is None:
        assert dashboard.weights_note.get_visible()


def test_training_dashboard_rejects_mismatched_frame(splits):
    dashboard = TrainingDashboard(short_spec("lasso"), splits)
    frame = next(training_frames(short_spec("lasso"), splits))
    with pytest.raises(ValueError, match="validation"):
        dashboard.render(replace(frame, val_pred=frame.val_pred[:5]))


def test_inference_dashboard_renders_stream(splits):
    spec = short_spec("decision_tree")
    model, _ = train_silently(spec, splits)
    dashboard = InferenceDashboard(spec, splits)
    frames = tuple(inference_frames(model, splits, batch_size=30))
    for frame in frames:
        dashboard.render(frame)
    dashboard.figure.canvas.draw()

    final = frames[-1]
    assert len(dashboard.map_points.get_offsets()) == final.n_done
    assert len(dashboard.rmse_line.get_xdata()) == len(final.history)
    assert f"{final.n_done:,}/{final.n_total:,}" in dashboard.status_text.get_text()
    assert "predicted" in dashboard.ticker_text.get_text()
    assert "actual" in dashboard.ticker_text.get_text()


def test_dashboards_draw_the_equation_card(splits):
    from house_anim.equations import prediction_card, training_card
    from house_anim.trainers import training_steps

    spec = short_spec("lasso")
    dashboard = TrainingDashboard(spec, splits)
    step = next(training_steps(spec, splits))
    dashboard.render(step.frame, training_card(step.model, step.frame, spec, splits))
    dashboard.figure.canvas.draw()
    assert "Lasso" in dashboard.equations.title.get_text()

    inference = InferenceDashboard(spec, splits)
    frame = next(inference_frames(step.model, splits, batch_size=30))
    inference.render(frame, prediction_card(step.model, splits.X_test[29], 30, splits))
    inference.figure.canvas.draw()
    assert "#30" in inference.equations.title.get_text()
