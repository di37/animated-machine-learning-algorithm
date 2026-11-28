"""Series videos played with Manim: animkit's cards, frame for frame as long as the matplotlib video."""

import shutil

import numpy as np
import pytest

from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.encoder import probe
from animkit.lessons.timing import Timing
from animkit.series.cards import Algorithm, Line, paint_lineup, paint_text
from animkit.series.charts import Bar, paint_bars
from animkit.series.chapters import Series, card, read_chapters
from animkit.series.render import chapters_path
from manimkit.series import render_playlist, render_series, series_plans

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")
TIMING = Timing(fps=10, scale=0.2)


def tiny_series() -> Series:
    return Series(name="introduction", title="Tiny: Introduction", source_note="Data: somewhere", chapters=(
        card("title", "Introduction", 5.0, "A tiny series.", paint_text, kicker="SERIES", headline="Tiny",
             lines=(Line("Not a first course.", marker="✗"),)),
        card("lineup", "The algorithms", 5.0, "Three models.", paint_lineup, kicker="MODELS", headline="Three",
             algorithms=tuple(Algorithm(f"Model {n}", "learns") for n in range(1, 4))),
        card("board", "Scoreboard", 4.0, "A tie.", paint_bars, kicker="SCOREBOARD", headline="Scores",
             metric="Test error", bars=(Bar("A", 2.0, "2"), Bar("B", 1.0, "1"), Bar("Base", 3.0, "3", "baseline")),
             verdict="A tie."),
    ))


def test_each_chapter_is_planned_at_its_length_and_moving_share():
    from animkit.lessons.stage import Stage
    stage = Stage(dpi=RESOLUTIONS["preview"])
    plans = series_plans(tiny_series(), stage, TIMING)
    assert [(plan.name, plan.frames) for plan in plans] == list(tiny_series().frames(TIMING))
    assert all(plan.samples[0][0] == 0 for plan in plans)


@needs_ffmpeg
def test_the_manim_video_has_the_matplotlib_videos_frames_and_settles_every_chapter(tmp_path):
    out = tmp_path / "introduction.mp4"
    unsettled = {}

    def check(name, player, chapter):
        unsettled[name] = player.mismatches(chapter)
        unsettled[f"{name} ink"] = float(np.asarray(player.last_frame()).std()) > 0  # something is drawn

    report = render_series(tiny_series(), out, timing=TIMING, dpi=RESOLUTIONS["preview"], on_chapter=check)
    assert probe(out).frames == report.frames == tiny_series().total_frames(TIMING)
    assert all(unsettled[key] == [] for key in ("title", "lineup", "board"))
    assert all(unsettled[f"{key} ink"] for key in ("title", "lineup", "board"))
    assert [row["chapter"] for row in read_chapters(chapters_path(out))["chapters"]] == ["title", "lineup", "board"]


@needs_ffmpeg
def test_a_playlist_renders_each_video_under_its_own_name(tmp_path, capsys):
    assert render_playlist((tiny_series(),), tmp_path / "series", timing=TIMING, dpi=RESOLUTIONS["preview"]) == 0
    assert (tmp_path / "series" / "introduction.mp4").is_file()
    assert "introduction" in capsys.readouterr().out
