"""The command line: the matplotlib lessons' options, the same lessons and summary, played with Manim."""

import csv
import shutil

import pytest

from animkit.lessons.encoder import probe
from animkit.lessons.timing import Timing
from house_anim.lessons import cli as matplotlib_cli
from house_anim.data import load_splits
from house_anim.lessons.data import income_view
from house_anim.lessons.stories import build_lesson, lesson_keys
from house_manim.cli import main, parse_args
from manimkit.director import lesson_frames
from tests.conftest import fake_fetcher, make_housing_frame

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
QUICK = ["--duration-scale", "0.05", "--resolution", "preview"]


def test_list_prints_every_lesson(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(key in out for key in lesson_keys())


def test_the_options_and_defaults_are_the_matplotlib_lessons():
    ours, theirs = parse_args([]), matplotlib_cli.parse_args([])
    assert vars(ours) == vars(theirs)
    assert ours.output_dir.parts[-2:] == ("california_housing", "lessons")


@needs_ffmpeg
def test_renders_one_lesson_as_long_as_matplotlib_and_writes_the_summary(tmp_path, capsys):
    fetcher = fake_fetcher(make_housing_frame(n_rows=400))
    assert main(["--model", "linear_regression", "--output-dir", str(tmp_path), *QUICK], fetcher=fetcher) == 0
    info = probe(tmp_path / "linear_regression_lesson.mp4")
    assert (info.codec, info.pix_fmt, info.width, info.height) == ("h264", "yuv420p", 384, 216)
    lesson = build_lesson("linear_regression", income_view(load_splits(fetcher=fetcher)))
    assert info.frames == lesson_frames(lesson, Timing(scale=0.05))  # as many as animkit's own video
    with open(tmp_path / "lessons_summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["lesson"] for row in rows] == ["linear_regression"] and float(rows[0]["test_rmse_k"]) > 0
    assert "(Manim)" in capsys.readouterr().out


def test_reports_a_dataset_download_failure(tmp_path, capsys):
    def offline(as_frame=True):
        raise OSError("no network")

    assert main(["--output-dir", str(tmp_path)], fetcher=offline) == 1
    assert "error" in capsys.readouterr().err


@needs_ffmpeg
def test_a_lesson_that_fails_is_reported_and_the_rest_go_on(tmp_path, capsys, monkeypatch):
    from house_anim.lessons.lesson import LessonError
    from house_manim import cli

    def failing(*args, **kwargs):
        raise LessonError("a solver did not converge")

    monkeypatch.setattr(cli, "render_lesson", failing)
    fetcher = fake_fetcher(make_housing_frame(n_rows=400))
    assert main(["--model", "knn", "--output-dir", str(tmp_path), *QUICK], fetcher=fetcher) == 1
    assert "did not converge" in capsys.readouterr().err
