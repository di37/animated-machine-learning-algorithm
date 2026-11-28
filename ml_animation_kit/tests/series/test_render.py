"""Rendering a series video with matplotlib: every chapter for its frames, and the chapters file beside it."""

import shutil

import pytest

from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.encoder import probe
from animkit.lessons.timing import Timing
from animkit.series.cards import Line, paint_text
from animkit.series.chapters import Series, card, read_chapters
from animkit.series.render import chapters_path, render_series

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")
TIMING = Timing(fps=10, scale=0.2)


def tiny_series() -> Series:
    return Series(name="introduction", title="Tiny: Introduction", source_note="Data: somewhere", chapters=(
        card("title", "Introduction", 5.0, "A tiny series.", paint_text, kicker="SERIES", headline="Tiny"),
        card("audience", "Who it's for", 4.0, "Not a first course.", paint_text, kicker="WHO IT IS FOR",
             headline="Not a first course.", lines=(Line("You know the basics.", marker="✓"),)),
    ))


def test_a_card_chapter_paints_with_its_content():
    chapter = tiny_series().chapters[1]
    assert chapter.paint.keywords["headline"] == "Not a first course."
    assert chapter.frames(TIMING) == 8


@needs_ffmpeg
def test_the_video_has_every_chapters_frames_and_its_chapters_file(tmp_path):
    out = tmp_path / "series" / "introduction.mp4"
    report = render_series(tiny_series(), out, timing=TIMING, dpi=RESOLUTIONS["preview"])
    assert probe(out).frames == report.frames == tiny_series().total_frames(TIMING)
    record = read_chapters(chapters_path(out))
    assert [row["chapter"] for row in record["chapters"]] == ["title", "audience"]
    assert record["chapters"][1]["narration"] == "Not a first course."


def test_the_chapters_file_sits_beside_the_video(tmp_path):
    assert chapters_path(tmp_path / "conclusion.mp4") == tmp_path / "conclusion.chapters.json"
