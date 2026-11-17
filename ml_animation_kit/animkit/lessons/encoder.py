"""Pipe raw RGBA frames straight into ffmpeg (H.264, yuv420p) and probe the result with ffprobe.

Held frames are written as repeated bytes, so a 2-second pause costs one render, not sixty.
The video is written to a temporary name and only renamed into place once ffmpeg succeeds.

No image files are written along the way: each frame goes from matplotlib's canvas to ffmpeg's
standard input as raw bytes. ``probe`` reads a finished video back, so the command line and the
tests can report what a viewer would get.
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from animkit.lessons.config import CRF


class EncoderError(RuntimeError):
    """ffmpeg/ffprobe is missing or failed."""


@dataclass(frozen=True)
class VideoInfo:
    """Decode-level facts about a video file, as ffprobe reports them.

    Attributes:
        codec: The video codec, for example "h264".
        pix_fmt: The pixel format, for example "yuv420p".
        width: Width in pixels.
        height: Height in pixels.
        fps: Average frames per second.
        frames: How many frames were decoded.
        duration_s: The container's duration, in seconds.
    """

    codec: str
    pix_fmt: str
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float


def _require(tool: str) -> str:
    """The path of a command-line tool, or an ``EncoderError`` saying how to install ffmpeg."""
    path = shutil.which(tool)
    if path is None:
        raise EncoderError(f"{tool} not found on PATH; install ffmpeg (macOS: brew install ffmpeg)")
    return path


class Mp4Encoder:
    """Writes raw RGBA frames to an H.264 MP4 through an ffmpeg process; use it as a context manager.

    The video appears at ``path`` only when ffmpeg finishes cleanly. Until then it is written to a
    temporary ``.partial.mp4`` file beside it, which is removed if anything goes wrong.

    Attributes:
        path: Where the finished video goes.
        frames_written: How many frames have been written so far, repeats included.

    Example:
        Two seconds of one still frame cost one render::

            with Mp4Encoder(Path("lesson.mp4"), width, height, fps=30) as video:
                video.write(stage.render(), repeat=60)
    """

    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int = CRF):
        """Sets up the encoder; ffmpeg starts when the ``with`` block is entered.

        Args:
            path: Where the finished video goes.
            width: Frame width in pixels; it must be even for yuv420p.
            height: Frame height in pixels; it must be even too.
            fps: Frames per second.
            crf: H.264's constant rate factor: lower is better quality and a bigger file.

        Raises:
            ValueError: If the width or height is odd.
        """
        if width % 2 or height % 2:
            raise ValueError(f"H.264 yuv420p needs even dimensions, got {width}x{height}")
        self.path = Path(path)
        self._size = (width, height)
        self._fps = fps
        self._crf = crf
        self._frame_bytes = width * height * 4
        # Per-process name, so two renders into the same folder never share a temporary file.
        self._partial = self.path.with_name(f"{self.path.stem}.{os.getpid()}.partial.mp4")
        self._process: subprocess.Popen | None = None
        self._stderr = None
        self.frames_written = 0

    def _command(self, ffmpeg: str) -> list[str]:
        """The ffmpeg command line: raw RGBA frames on standard input, H.264 yuv420p into the partial file."""
        width, height = self._size
        return [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-r", str(self._fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(self._crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self._partial),
        ]

    def __enter__(self) -> "Mp4Encoder":
        """Starts ffmpeg, keeping what it writes to stderr in a temporary file for error messages."""
        ffmpeg = _require("ffmpeg")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(self._command(ffmpeg), stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL, stderr=self._stderr)
        return self

    def write(self, frame: bytes, repeat: int = 1) -> None:
        """Writes one frame, ``repeat`` times in a row.

        Args:
            frame: The frame's raw RGBA bytes, width × height × 4 of them.
            repeat: How many frames it lasts.

        Raises:
            ValueError: If ``repeat`` is below 1 or the frame has the wrong number of bytes.
            EncoderError: If the encoder is not open, or ffmpeg stopped accepting frames.
        """
        if repeat < 1:
            raise ValueError(f"repeat must be at least 1, got {repeat}")
        if len(frame) != self._frame_bytes:
            raise ValueError(f"frame has {len(frame)} bytes; expected {self._frame_bytes} "
                             f"for {self._size[0]}x{self._size[1]} RGBA")
        if self._process is None:
            raise EncoderError("encoder is not open; use it as a context manager")
        try:
            for _ in range(repeat):
                self._process.stdin.write(frame)
        except BrokenPipeError as exc:
            raise EncoderError(f"ffmpeg stopped accepting frames: {self._error_text()}") from exc
        self.frames_written += repeat

    def _error_text(self) -> str:
        """The end of what ffmpeg wrote to stderr (its last 800 characters), for error messages."""
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace").strip()[-800:]

    def _finish(self) -> None:
        """Closes ffmpeg's input, waits for it, and moves the finished video into place."""
        try:
            self._process.stdin.close()  # flushes the last buffered frame
        except BrokenPipeError:
            pass  # ffmpeg already exited; its return code and stderr say why
        if self._process.wait() != 0:
            raise EncoderError(f"ffmpeg failed: {self._error_text()}")
        os.replace(self._partial, self.path)

    def _abort(self) -> None:
        """Stops ffmpeg and deletes the partial file."""
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass  # ffmpeg already exited; we are cleaning up anyway
        self._process.kill()
        self._process.wait()
        self._partial.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Finishes the video, or aborts it if the block raised; a failure never leaves the partial file behind."""
        if self._process is None:
            return
        try:
            if exc_type is None:
                self._finish()
            else:
                self._abort()
        except BaseException:  # including Ctrl-C while waiting: stop ffmpeg, drop the partial file
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait()
            self._partial.unlink(missing_ok=True)
            raise
        finally:
            self._stderr.close()
            self._process = None


def probe(path: Path) -> VideoInfo:
    """Decode-level facts about a video: codec, pixel format, size, fps, counted frames, duration.

    Frames are counted by decoding the whole stream, so the count is what a player would show.

    Args:
        path: The video file.

    Returns:
        The facts about its first video stream.

    Raises:
        EncoderError: If ffprobe is missing or cannot read the file.
    """
    ffprobe = _require("ffprobe")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames"
        ":format=duration", "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise EncoderError(f"ffprobe could not read {path}: {result.stderr.strip()}")
    report = json.loads(result.stdout)
    stream = report["streams"][0]
    numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
    return VideoInfo(
        codec=stream["codec_name"],
        pix_fmt=stream["pix_fmt"],
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=numerator / denominator if denominator else 0.0,
        frames=int(stream["nb_read_frames"]),
        duration_s=float(report["format"]["duration"]),
    )
