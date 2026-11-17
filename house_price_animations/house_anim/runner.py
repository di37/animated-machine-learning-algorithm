"""Drive the regression dashboards from live frame streams, using animkit's sinks and playback.

The training loop and the drawing are interleaved: animkit's ``play`` pulls one step from the loop,
draws it, and sends the picture to the sinks (the live window, the video writer, or both) before
the next real training step runs. So the video shows the model as it is, step by step. One model's
run is the training animation (or silent training), then the inference animation on the test houses.
``RunOptions``, ``ModelResult`` and the sink helpers are animkit's, re-exported here.
"""

from pathlib import Path

from sklearn.base import RegressorMixin

from animkit.runner import (
    PHASE_INFERENCE,
    PHASE_TRAINING,
    ModelResult,
    RunOptions,
    finish,
    open_sinks_or_close,
    play,
)
from house_anim.config import DASHBOARD_DPI_SCALE, FINAL_HOLD_SECONDS
from house_anim.data import HousingSplits
from house_anim.equations import prediction_card, training_card
from house_anim.frames import InferenceFrame, TrainingFrame
from house_anim.inference import inference_frames, prepare_for_serving
from house_anim.inference_dashboard import InferenceDashboard
from house_anim.models import ModelSpec
from house_anim.trainers import TrainingStep, train_silently, training_steps
from house_anim.training_dashboard import TrainingDashboard

__all__ = ["PHASE_INFERENCE", "PHASE_TRAINING", "ModelResult", "RunOptions", "animate_inference",
           "animate_training", "finish", "open_sinks_or_close", "play", "run_model"]


def animate_training(spec: ModelSpec, splits: HousingSplits,
                     options: RunOptions) -> tuple[RegressorMixin, TrainingFrame, Path | None]:
    """Trains one model on screen: the training dashboard, with its equation card, after every step.

    Args:
        spec: The model and its schedule.
        splits: The data.
        options: Live window, video file, format, fps, dpi and output folder.

    Returns:
        The final model, the last training frame, and the video's path (None when not saving).
    """
    dashboard = TrainingDashboard(spec, splits)
    sinks, path = open_sinks_or_close(dashboard, options, f"{spec.key}_training", DASHBOARD_DPI_SCALE)
    try:
        def draw(step: TrainingStep) -> None:
            """Draws one training step, with the equation card for the model as it is now.

            Args:
                step: The step's frame and live model.
            """
            card = training_card(step.model, step.frame, spec, splits)
            dashboard.render(step.frame, card, model=step.model)

        last, model = play(training_steps(spec, splits), draw, sinks, spec.hold_frames)
        sinks.hold(FINAL_HOLD_SECONDS)
    finally:
        finish(dashboard, sinks, keep_open=False)
    return model, last.frame, path


def animate_inference(model: RegressorMixin, spec: ModelSpec, splits: HousingSplits,
                      options: RunOptions,
                      keep_open: bool = False) -> tuple[InferenceFrame, Path | None]:
    """Predicts the test houses on screen, batch by batch, explaining the newest house each time.

    The model is first made single-threaded (``prepare_for_serving``), so batch latency measures
    the prediction rather than thread start-up.

    Args:
        model: The fitted model.
        spec: Its spec (title and key).
        splits: The data; only its test rows are predicted.
        options: Live window, video file, format, fps, dpi, output folder and batch size.
        keep_open: Leave the live window open at the end.

    Returns:
        The last inference frame and the video's path (None when not saving).
    """
    dashboard = InferenceDashboard(spec, splits)
    sinks, path = open_sinks_or_close(dashboard, options, f"{spec.key}_inference", DASHBOARD_DPI_SCALE)
    try:
        serving = prepare_for_serving(model)

        def draw(frame: InferenceFrame) -> None:
            """Draws one batch, with the prediction card worked through for its last house.

            Args:
                frame: The snapshot after the batch.
            """
            latest = frame.n_done - 1  # explain the newest house the moment it is priced
            house = splits.X_test[latest]
            card = prediction_card(serving, house, latest + 1, splits)
            dashboard.render(frame, card, model=serving, house=house)

        final, _ = play(inference_frames(serving, splits, options.batch_size), draw, sinks)
        sinks.hold(FINAL_HOLD_SECONDS)
    finally:
        finish(dashboard, sinks, keep_open=keep_open)
    return final, path


def run_model(spec: ModelSpec, splits: HousingSplits, options: RunOptions,
              keep_open: bool = False) -> ModelResult:
    """Training animation (or silent training), then the inference animation, per options.phases.

    Inference alone still needs a trained model, so without the training phase the same loop runs
    without drawing.

    Args:
        spec: The model and its schedule.
        splits: The data.
        options: What to show and record, including which phases to run.
        keep_open: Leave the last live window open at the end.

    Returns:
        The model's key and title, its last training and inference frames (the latter None when
        inference did not run), and the videos written.
    """
    if PHASE_TRAINING in options.phases:
        model, final_training, training_video = animate_training(spec, splits, options)
    else:
        (model, final_training), training_video = train_silently(spec, splits), None

    final_inference, inference_video = None, None
    if PHASE_INFERENCE in options.phases:
        final_inference, inference_video = animate_inference(model, spec, splits, options, keep_open)

    return ModelResult(
        key=spec.key,
        title=spec.title,
        final_training=final_training,
        final_inference=final_inference,
        videos=tuple(p for p in (training_video, inference_video) if p is not None),
    )
