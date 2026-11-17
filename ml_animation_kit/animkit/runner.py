"""Play a live frame stream into a dashboard, sending every rendered frame to the sinks.

A project's training (or inference) loop is a generator: it yields a frame each time there is
something new to show, and returns a summary when it is done. ``play`` takes one frame at a time,
draws it and sends the drawn figure to every sink, so the video is written while the model trains,
not afterwards. The other helpers open a run's sinks, and close them and the figure whatever happens.
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from animkit.recorder import LiveSink, SinkGroup, VideoSink, fit_to_screen

PHASE_TRAINING = "training"
PHASE_INFERENCE = "inference"


class Dashboard(Protocol):
    """Anything that draws into one matplotlib ``figure`` (and has a ``render(frame, ...)`` method)."""

    @property
    def figure(self) -> Figure:
        """The figure the dashboard draws into.

        Returns:
            The matplotlib figure every sink records.
        """
        ...


@dataclass(frozen=True)
class RunOptions:
    """How a run was asked to go, resolved from the command line.

    Attributes:
        live: Whether to show a live window.
        save: Whether to write video files.
        video_format: ``"mp4"`` or ``"gif"``.
        fps: Video frames per second.
        dpi: The video's resolution, in dots per inch of the figure.
        output_dir: The folder the videos are written to.
        batch_size: How many rows each inference step predicts at once.
        phases: The phases to run, in order: ``PHASE_TRAINING``, ``PHASE_INFERENCE`` or both.
    """

    live: bool
    save: bool
    video_format: str
    fps: int
    dpi: int
    output_dir: Path
    batch_size: int
    phases: tuple[str, ...]


@dataclass(frozen=True)
class ModelResult:
    """What one model's run left behind, for the summary.

    Attributes:
        key: The model's key.
        title: The model's name.
        final_training: The project's last training frame.
        final_inference: The project's last inference frame, or ``None`` when inference did not run.
        videos: The video files written for this model.
    """

    key: str
    title: str
    final_training: Any  # the project's last TrainingFrame
    final_inference: Any | None  # the project's last InferenceFrame, when inference ran
    videos: tuple[Path, ...]


def open_sinks(figure, options: RunOptions, stem: str,
               dpi_scale: float = 1.0) -> tuple[SinkGroup, Path | None]:
    """Opens the sinks a run asked for: a live window (shrunk to fit the screen), a video file, or both.

    Args:
        figure: The dashboard's figure.
        options: The run's options.
        stem: The video's file name, without its extension.
        dpi_scale: Multiplies the run's dpi for this video, for a dashboard that records sharper than the
            plain video dpi.

    Returns:
        The sinks as one group, and the video's path (``None`` when nothing is saved).
    """
    sinks, path = [], None
    if options.live:
        fit_to_screen(figure)
        sinks.append(LiveSink(figure))
    if options.save:
        path = options.output_dir / f"{stem}.{options.video_format}"
        sinks.append(VideoSink(figure, path, fps=options.fps, dpi=round(options.dpi * dpi_scale)))
    return SinkGroup(sinks), path


def play(stream: Generator, draw: Callable[[object], None], sinks: SinkGroup,
         hold_frames: int = 1) -> tuple[object, object]:
    """Draw each item the moment the loop produces it; returns (last item, loop return value).

    Args:
        stream: The project's loop, as a generator: it yields frames and returns a summary.
        draw: Draws one yielded item into the dashboard.
        sinks: Where each drawn frame goes.
        hold_frames: How many video frames each item is held for.

    Returns:
        The last item drawn (``None`` if there was none) and the value the generator returned.
    """
    last = None
    while True:
        try:
            item = next(stream)
        except StopIteration as stop:
            return last, stop.value
        draw(item)
        sinks.emit(repeat=hold_frames)
        last = item


def open_sinks_or_close(dashboard: Dashboard, options: RunOptions, stem: str,
                        dpi_scale: float = 1.0) -> tuple[SinkGroup, Path | None]:
    """Like ``open_sinks``, but closes the dashboard's figure if a sink cannot be opened.

    Args:
        dashboard: The dashboard whose figure is recorded.
        options: The run's options.
        stem: The video's file name, without its extension.
        dpi_scale: Multiplies the run's dpi for this video.

    Returns:
        The sinks as one group, and the video's path (``None`` when nothing is saved).
    """
    try:
        return open_sinks(dashboard.figure, options, stem, dpi_scale)
    except Exception:
        plt.close(dashboard.figure)  # don't leak the window if a writer can't be created
        raise


def finish(dashboard: Dashboard, sinks: SinkGroup, keep_open: bool) -> None:
    """Closes every sink, then closes the figure unless its window should stay open.

    The figure is closed even if closing a sink fails.

    Args:
        dashboard: The dashboard whose run is ending.
        sinks: The run's sinks.
        keep_open: Leave the figure open, for a live window the user closes at the end.
    """
    try:
        sinks.close()
    finally:
        if not keep_open:
            plt.close(dashboard.figure)
