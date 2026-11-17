import csv
import shutil

import pytest

from animkit.lessons.encoder import probe
from tests.conftest import fake_fetcher, make_titanic_frame
from titanic_anim.lessons.cli import main, parse_args
from titanic_anim.lessons.stories import lesson_keys

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
QUICK = ["--duration-scale", "0.05", "--resolution", "preview"]


def test_list_prints_every_lesson(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(key in out for key in lesson_keys())


def test_defaults_are_full_hd_lessons_in_the_titanic_folder():
    args = parse_args([])
    assert args.model == "all" and args.resolution == "1440p" and args.duration_scale == 1.0
    assert args.output_dir.parts[-2:] == ("titanic", "lessons")


@needs_ffmpeg
def test_renders_one_lesson_and_writes_the_summary(tmp_path, capsys):
    status = main(["--model", "knn", "--output-dir", str(tmp_path), *QUICK],
                  fetcher=fake_fetcher(make_titanic_frame()))
    assert status == 0
    assert probe(tmp_path / "knn_lesson.mp4").codec == "h264"
    with open(tmp_path / "lessons_summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["lesson"] for row in rows] == ["knn"]
    assert 0.0 < float(rows[0]["test_accuracy"]) <= 1.0 and rows[0]["score_kind"] == "probability"
    assert "knn_lesson.mp4" in capsys.readouterr().out


def test_reports_a_dataset_download_failure(tmp_path, capsys):
    def offline(as_frame=True):
        raise OSError("no network")

    assert main(["--model", "knn", "--output-dir", str(tmp_path)], fetcher=offline) == 1
    assert "Could not load the Titanic dataset" in capsys.readouterr().err
