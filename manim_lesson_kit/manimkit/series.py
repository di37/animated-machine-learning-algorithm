"""Plays a series video (a playlist's introduction or conclusion) with Manim, as the lessons are played.

A series chapter is one of animkit's cards (``animkit.series``), painted on a matplotlib ``Stage`` that
is never rasterized. It is sampled exactly where animkit's ``Recorder.animate`` renders it (every second
frame while it moves, then its last picture), recorded and followed, and Manim plays the changes on every
frame: text is written, shapes fade or are drawn in, and the chapter ends on animkit's own picture. The
video has the matplotlib version's frame count, and the same chapters file is written beside it, so the
series voice-over works on either.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from animkit.lessons import theme
from animkit.lessons.config import DEFAULT_RESOLUTION, RESOLUTIONS
from animkit.lessons.director import RenderReport
from animkit.lessons.encoder import Mp4Encoder
from animkit.lessons.stage import Stage, lesson_style
from animkit.lessons.timing import Timing
from animkit.series.chapters import Series, write_chapters
from animkit.series.render import chapters_path
from manimkit.director import Plan, animated, play_chapter
from manimkit.frame import Frame
from manimkit.player import Player, manim_stage
from manimkit.timeline import Chapter, Tracker


def series_plans(series: Series, stage: Stage, timing: Timing) -> list[Plan]:
    """Each chapter's plan: its frames, and the frames on which its card is painted.

    Args:
        series: The series video.
        stage: The stage the cards are painted on.
        timing: The frame rate and duration scale.

    Returns:
        One plan per chapter, in order.
    """
    return [animated(chapter.key, chapter.frames(timing), chapter.moving,
                     lambda t, paint=chapter.paint: paint(stage, t)) for chapter in series.chapters]


def render_series(series: Series, path: Path, *, timing: Timing = Timing(),
                  dpi: float = RESOLUTIONS[DEFAULT_RESOLUTION],
                  on_chapter: Callable[[str, Player, Chapter], None] | None = None) -> RenderReport:
    """Renders ``series`` to ``path`` with Manim, then writes the chapters file beside it.

    Args:
        series: The series video.
        path: The MP4 to write.
        timing: The frame rate and duration scale.
        dpi: The resolution (animkit's ``RESOLUTIONS``).
        on_chapter: Called as ``on_chapter(name, player, chapter)`` after each chapter's last frame, for
            tests that look at the screen.

    Returns:
        Where the video went, and how many frames and seconds it has.
    """
    with lesson_style():
        stage = Stage(dpi=dpi, source_note=series.source_note)
        frame = Frame.of(stage.fig)
        width, height = stage.size_px
        with manim_stage(frame, theme.CANVAS, timing.fps) as player, \
                Mp4Encoder(Path(path), width, height, timing.fps) as video:
            tracker = Tracker()
            for plan in series_plans(series, stage, timing):
                chapter = play_chapter(plan, stage, tracker, player, video.write, timing.fps)
                if on_chapter is not None:
                    on_chapter(plan.name, player, chapter)
    write_chapters(series, timing, chapters_path(path))
    return RenderReport(path=Path(path), frames=video.frames_written, seconds=video.frames_written / timing.fps)


def render_playlist(videos: Sequence[Series], output_dir: Path, *, timing: Timing = Timing(),
                    dpi: float = RESOLUTIONS[DEFAULT_RESOLUTION]) -> int:
    """Renders each series video to ``output_dir/<name>.mp4`` and prints where it went.

    Args:
        videos: The series videos (a playlist's introduction and conclusion).
        output_dir: The folder they go in.
        timing: The frame rate and duration scale.
        dpi: The resolution.

    Returns:
        0, for a command line's exit code.
    """
    for series in videos:
        print(f"{series.name} …", flush=True)
        report = render_series(series, Path(output_dir) / f"{series.name}.mp4", timing=timing, dpi=dpi)
        print(f"    saved {report.path} · {report.seconds:.1f} s · {report.frames} frames", flush=True)
    return 0
