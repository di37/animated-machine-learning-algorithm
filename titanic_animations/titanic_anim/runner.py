"""Drive the classifier dashboards from live frame streams, using animkit's sinks and playback.

A model's run has two phases. Training plays ``training_steps`` into the training dashboard: each
step is drawn, with its equation card and diagram, the moment training produces it. Inference then
plays ``inference_frames`` into the inference dashboard, explaining the newest passenger in every
frame. With only the inference phase, the model is first trained silently, the same way.
animkit's ``play`` draws and emits the frames, and ``finish`` closes the sinks and the figure even
if the run is stopped.
"""

from pathlib import Path

from sklearn.base import ClassifierMixin

from animkit.config import DASHBOARD_DPI_SCALE, FINAL_HOLD_SECONDS
from animkit.runner import (
    PHASE_INFERENCE,
    PHASE_TRAINING,
    ModelResult,
    RunOptions,
    finish,
    open_sinks_or_close,
    play,
)
from titanic_anim.config import INFERENCE_HOLD_FRAMES
from titanic_anim.data import TitanicSplits
from titanic_anim.equations import prediction_card, training_card
from titanic_anim.frames import InferenceFrame, TrainingFrame
from titanic_anim.inference import inference_frames, prepare_for_serving
from titanic_anim.inference_dashboard import InferenceDashboard
from titanic_anim.models import ClassifierSpec
from titanic_anim.trainers import TrainingStep, train_silently, training_steps
from titanic_anim.training_dashboard import TrainingDashboard

__all__ = ["PHASE_INFERENCE", "PHASE_TRAINING", "ModelResult", "RunOptions", "animate_inference",
           "animate_training", "run_model"]


def animate_training(spec: ClassifierSpec, splits: TitanicSplits,
                     options: RunOptions) -> tuple[ClassifierMixin, TrainingFrame, Path | None]:
    """Plays one classifier's training into its dashboard, then holds the last frame.

    Args:
        spec: The classifier.
        splits: The passengers.
        options: Where the frames go: live window, video file, frame rate and so on.

    Returns:
        The final model, the final training frame, and the video's path (None when nothing was saved).
    """
    dashboard = TrainingDashboard(spec, splits)
    sinks, path = open_sinks_or_close(dashboard, options, f"{spec.key}_training", DASHBOARD_DPI_SCALE)
    try:
        def draw(step: TrainingStep) -> None:
            """Draws one training step: the charts, the algorithm's diagram and the equation card.

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


def animate_inference(model: ClassifierMixin, spec: ClassifierSpec, splits: TitanicSplits,
                      options: RunOptions, keep_open: bool = False) -> tuple[InferenceFrame, Path | None]:
    """Plays the test passengers through a trained classifier into its dashboard, then holds the last frame.

    Args:
        model: The trained classifier; it is switched to one thread first.
        spec: Its spec.
        splits: The passengers; the test passengers are classified.
        options: Where the frames go, and how many passengers make a batch.
        keep_open: Whether to leave the live window open afterwards.

    Returns:
        The final inference frame, and the video's path (None when nothing was saved).
    """
    dashboard = InferenceDashboard(spec, splits)
    sinks, path = open_sinks_or_close(dashboard, options, f"{spec.key}_inference", DASHBOARD_DPI_SCALE)
    try:
        serving = prepare_for_serving(model)

        def draw(frame: InferenceFrame) -> None:
            """Draws one batch, explaining its last passenger in the diagram and the equation card.

            Args:
                frame: The batch's inference frame.
            """
            latest = frame.n_done - 1  # explain the newest passenger the moment it is classified
            passenger = splits.X_test[latest]
            card = prediction_card(serving, passenger, latest + 1, splits)
            dashboard.render(frame, card, model=serving, passenger=passenger)

        frames = inference_frames(serving, splits, spec.score_kind, options.batch_size)
        final, _ = play(frames, draw, sinks, INFERENCE_HOLD_FRAMES)
        sinks.hold(FINAL_HOLD_SECONDS)
    finally:
        finish(dashboard, sinks, keep_open=keep_open)
    return final, path


def run_model(spec: ClassifierSpec, splits: TitanicSplits, options: RunOptions,
              keep_open: bool = False) -> ModelResult:
    """Training animation (or silent training), then the inference animation, per ``options.phases``.

    Args:
        spec: The classifier.
        splits: The passengers.
        options: The run's options; ``phases`` says which animations to play.
        keep_open: Whether to leave the last live window open afterwards.

    Returns:
        The run's result: the final frames and every video saved.
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
