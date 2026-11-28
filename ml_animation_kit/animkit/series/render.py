"""Plays a series video into an MP4 with matplotlib, and writes its chapters file beside it.

Each chapter is painted as a lesson card is (``Recorder.animate``): repainted every second frame while
it moves, then its last picture held. The Manim player (``manimkit.series``) plays the same chapters
frame for frame, so both videos have the same length and the same chapters file.
"""

from pathlib import Path

from animkit.lessons.config import DEFAULT_RESOLUTION, RESOLUTIONS
from animkit.lessons.director import Recorder, RenderReport
from animkit.lessons.encoder import Mp4Encoder
from animkit.lessons.stage import Stage, lesson_style
from animkit.lessons.timing import Timing
from animkit.series.chapters import Series, write_chapters

CHAPTERS_SUFFIX = ".chapters.json"


def chapters_path(video: Path) -> Path:
    """Where a series video's chapters file is: beside it, ``<name>.chapters.json``."""
    return Path(video).with_suffix(CHAPTERS_SUFFIX)


def render_series(series: Series, path: Path, *, timing: Timing = Timing(),
                  dpi: float = RESOLUTIONS[DEFAULT_RESOLUTION]) -> RenderReport:
    """Renders every chapter of ``series`` to ``path``, then writes the chapters file.

    Args:
        series: The series video.
        path: The MP4 to write.
        timing: The frame rate and duration scale.
        dpi: The resolution (``RESOLUTIONS``).

    Returns:
        Where the video went, and how many frames and seconds it has.
    """
    with lesson_style():
        stage = Stage(dpi=dpi, source_note=series.source_note)
        width, height = stage.size_px
        with Mp4Encoder(Path(path), width, height, timing.fps) as video:
            recorder = Recorder(stage, video)
            for chapter in series.chapters:
                recorder.animate(chapter.frames(timing), lambda t, paint=chapter.paint: paint(stage, t),
                                 moving=chapter.moving)
    write_chapters(series, timing, chapters_path(path))
    return RenderReport(path=Path(path), frames=video.frames_written, seconds=video.frames_written / timing.fps)
