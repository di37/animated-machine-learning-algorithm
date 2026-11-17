"""Frame sinks: a live window, a video file, or both at once, fed from the same render loop.

A sink is anything with ``emit`` (send the figure's current state as one or more frames), ``hold``
(keep it showing for some seconds) and ``close``. The render loop draws each frame once and hands it
to a ``SinkGroup``, which passes it to every sink, so watching and recording never draw twice.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure

from animkit.config import LIVE_MAX_WIDTH_PX, LIVE_PAUSE_S

# matplotlib treats an event-loop timeout <= 0 as "run forever", so never pause for less.
MIN_PAUSE_S = 1e-3


def fit_to_screen(figure: Figure, max_width_px: int = LIVE_MAX_WIDTH_PX) -> None:
    """Shrink the on-screen window of a big figure; video files keep their own dpi and full size.

    Args:
        figure: The dashboard's figure; its dpi is lowered in place when it is too wide.
        max_width_px: The widest the window may open, in screen pixels.
    """
    width_in = figure.get_size_inches()[0]
    if width_in * figure.get_dpi() > max_width_px:
        figure.set_dpi(max_width_px / width_in)


class RenderingStopped(Exception):
    """The user closed the live window, so the run should end early."""


class LiveSink:
    """Redraws the on-screen window and lets the GUI event loop breathe between frames."""

    def __init__(self, figure: Figure, pause_s: float = LIVE_PAUSE_S):
        """Watches one figure's window.

        Args:
            figure: The figure shown on screen.
            pause_s: How long the GUI event loop runs after each frame, in seconds (at least ``MIN_PAUSE_S``).
        """
        self._figure = figure
        self._pause_s = max(pause_s, MIN_PAUSE_S)

    def emit(self, repeat: int = 1) -> None:
        """Redraws the window and lets the GUI handle its events.

        Args:
            repeat: How many video frames this frame stands for; a window shows it once whatever the count.

        Raises:
            RenderingStopped: If the user has closed the window.
        """
        if not plt.fignum_exists(self._figure.number):
            raise RenderingStopped("live window was closed")
        plt.figure(self._figure.number)  # plt.pause drives the *active* figure
        self._figure.canvas.draw_idle()
        plt.pause(self._pause_s)

    def hold(self, seconds: float) -> None:
        """Keeps the window responsive for ``seconds`` while the frame stays on screen.

        Args:
            seconds: How long to hold; a window that has been closed is not waited on.
        """
        if plt.fignum_exists(self._figure.number):
            plt.pause(max(seconds, MIN_PAUSE_S))

    def close(self) -> None:
        """Nothing to release; the window lifetime is managed by the runner."""


class VideoSink:
    """Appends frames to an .mp4 (ffmpeg) or .gif (Pillow) file as they are rendered.

    Attributes:
        path: The video file being written.
        frames_written: How many frames have been written so far.
    """

    def __init__(self, figure: Figure, path: Path, fps: int, dpi: int):
        """Chooses the writer for the file's extension; the file itself is opened on the first frame.

        Args:
            figure: The figure to record.
            path: The video file to write, ending in .mp4 or .gif.
            fps: Frames per second.
            dpi: The video's resolution, in dots per inch of the figure.

        Raises:
            ValueError: If the extension is neither .mp4 nor .gif, or .mp4 is asked for without ffmpeg.
        """
        self.path = Path(path)
        self._figure = figure
        self._fps = fps
        self._dpi = dpi
        self._size_inches = tuple(figure.get_size_inches())
        self._writer = self._make_writer(self.path.suffix.lower(), fps)
        self._started = False
        self.frames_written = 0

    def _start(self) -> None:
        """Open the file on the first frame, so a run stopped early leaves no broken video."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer.setup(self._figure, str(self.path), dpi=self._dpi)
        self._started = True

    @staticmethod
    def _make_writer(suffix: str, fps: int) -> animation.AbstractMovieWriter:
        """The movie writer for an extension: Pillow for .gif, H.264 through ffmpeg (even-sized) for .mp4."""
        if suffix == ".gif":
            return animation.PillowWriter(fps=fps)
        if suffix == ".mp4":
            if not animation.writers.is_available("ffmpeg"):
                raise ValueError("Saving .mp4 needs ffmpeg on PATH; install it or use --format gif")
            # yuv420p needs even width/height; figsize x dpi is often odd, so round down to even.
            return animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=[
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
            ])
        raise ValueError(f"Unsupported video extension '{suffix}'; use .mp4 or .gif")

    def emit(self, repeat: int = 1) -> None:
        """Writes the figure's current state as ``repeat`` identical frames.

        The figure is set back to its first size if a live window has resized it, since every frame of a
        video must be the same size.

        Args:
            repeat: How many frames to write.

        Raises:
            ValueError: If ``repeat`` is below 1.
        """
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        # Every frame must share one size, even if the live window was resized meanwhile.
        if tuple(self._figure.get_size_inches()) != self._size_inches:
            self._figure.set_size_inches(self._size_inches)
        if not self._started:
            self._start()
        for _ in range(repeat):
            self._writer.grab_frame()
        self.frames_written += repeat

    def hold(self, seconds: float) -> None:
        """Holds the current frame for ``seconds`` by writing it repeatedly (at least once).

        Args:
            seconds: How long to hold, in seconds of video.
        """
        self.emit(repeat=max(1, round(seconds * self._fps)))

    def close(self) -> None:
        """Finishes the file, if a frame was ever written; closing twice does nothing."""
        if self._started:
            self._writer.finish()
            self._started = False


class SinkGroup:
    """Fans every frame out to all sinks and always closes all of them."""

    def __init__(self, sinks):
        """Groups the sinks.

        Args:
            sinks: The sinks to feed, in order.
        """
        self._sinks = tuple(sinks)

    def emit(self, repeat: int = 1) -> None:
        """Sends the current frame to every sink.

        Args:
            repeat: How many video frames it stands for.
        """
        for sink in self._sinks:
            sink.emit(repeat=repeat)

    def hold(self, seconds: float) -> None:
        """Holds the current frame in every sink.

        Args:
            seconds: How long to hold, in seconds.
        """
        for sink in self._sinks:
            sink.hold(seconds)

    def close(self) -> None:
        """Closes every sink, even when one fails, then raises the first failure.

        Raises:
            Exception: The first error a sink raised while closing, once all the others are closed.
        """
        errors = []
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as exc:  # keep closing the rest so no file is left half-written
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> "SinkGroup":
        """Enters a ``with`` block; the group itself is the value."""
        return self

    def __exit__(self, *exc_info) -> None:
        """Closes every sink on leaving the ``with`` block, whether or not it raised."""
        self.close()
