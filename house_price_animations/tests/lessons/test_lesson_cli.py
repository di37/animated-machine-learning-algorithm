import csv
import shutil

import pytest

from animkit.lessons.encoder import probe
from house_anim.lessons.cli import main, parse_args
from house_anim.lessons.stories import lesson_keys
from tests.conftest import fake_fetcher, make_housing_frame

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_list_prints_every_lesson(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(key in out for key in lesson_keys())


def test_defaults_are_full_hd_lessons_in_the_dataset_folder():
    args = parse_args([])
    assert args.model == "all" and args.resolution == "1440p" and args.duration_scale == 1.0
    assert args.output_dir.parts[-2:] == ("california_housing", "lessons")


@pytest.mark.parametrize("value", ["0", "5", "abc"])
def test_rejects_bad_duration_scale(value):
    with pytest.raises(SystemExit):
        parse_args(["--duration-scale", value])


@needs_ffmpeg
def test_renders_one_lesson_and_writes_the_summary(tmp_path, capsys):
    fetcher = fake_fetcher(make_housing_frame(n_rows=400))
    status = main(["--model", "knn", "--output-dir", str(tmp_path), "--duration-scale", "0.05",
                   "--resolution", "preview"], fetcher=fetcher)
    assert status == 0
    video = tmp_path / "knn_lesson.mp4"
    assert probe(video).codec == "h264"
    with open(tmp_path / "lessons_summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["lesson"] for row in rows] == ["knn"]
    assert float(rows[0]["test_rmse_k"]) > 0 and rows[0]["video"].endswith("knn_lesson.mp4")
    assert "knn_lesson.mp4" in capsys.readouterr().out


@needs_ffmpeg
def test_summary_rows_accumulate_across_runs(tmp_path):
    fetcher = fake_fetcher(make_housing_frame(n_rows=400))
    common = ["--output-dir", str(tmp_path), "--duration-scale", "0.05", "--resolution", "preview"]
    assert main(["--model", "knn", *common], fetcher=fetcher) == 0
    assert main(["--model", "linear_regression", *common], fetcher=fetcher) == 0
    with open(tmp_path / "lessons_summary.csv", newline="") as handle:
        assert [row["lesson"] for row in csv.DictReader(handle)] == ["linear_regression", "knn"]


def test_reports_a_dataset_download_failure(tmp_path, capsys):
    def offline(as_frame=True):
        raise OSError("no network")

    assert main(["--model", "knn", "--output-dir", str(tmp_path)], fetcher=offline) == 1
    assert "Could not download" in capsys.readouterr().err


@needs_ffmpeg
def test_one_failing_lesson_does_not_stop_the_others(tmp_path, capsys, monkeypatch):
    from house_anim.lessons import cli
    from house_anim.lessons.lesson import LessonError

    real_build = cli.build_lesson

    def build(key, data):
        if key == "knn":
            raise LessonError("solver did not converge")
        return real_build(key, data)

    monkeypatch.setattr(cli, "lesson_keys", lambda: ("knn", "linear_regression"))
    monkeypatch.setattr(cli, "build_lesson", build)
    fetcher = fake_fetcher(make_housing_frame(n_rows=400))
    status = main(["--output-dir", str(tmp_path), "--duration-scale", "0.05", "--resolution", "preview"],
                  fetcher=fetcher)
    assert status == 1
    assert (tmp_path / "linear_regression_lesson.mp4").exists()
    assert not (tmp_path / "knn_lesson.mp4").exists()
    assert "1 of 2 lessons failed: knn" in capsys.readouterr().err
