import csv
from dataclasses import replace

import pytest

from animkit.runner import ModelResult
from tests.conftest import fake_fetcher, make_titanic_frame
from titanic_anim import cli
from titanic_anim.models import get_spec
from titanic_anim.summary import format_table, merge_summary_csv, result_row
from titanic_anim.trainers import train_silently

FRAME = make_titanic_frame(300, seed=1)


def run(argv, tmp_path):
    return cli.main([*argv, "--no-live", "--output-dir", str(tmp_path), "--format", "gif", "--dpi", "20",
                     "--fps", "4"], fetcher=fake_fetcher(FRAME))


@pytest.fixture
def fast_specs(monkeypatch):
    """Shorter schedules so end-to-end runs stay quick."""
    original = cli.get_spec

    def quick(key):
        spec = original(key)
        return spec if spec.strategy == "data_growth" else replace(spec, steps=spec.steps[:2])

    monkeypatch.setattr(cli, "get_spec", quick)


def test_list_prints_every_model(capsys):
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "logistic_regression" in out and "naive_bayes" in out and "svm" in out


def test_one_model_records_both_videos_and_a_summary(tmp_path, fast_specs, capsys):
    assert run(["--model", "decision_tree"], tmp_path) == 0
    assert (tmp_path / "decision_tree_training.gif").stat().st_size > 0
    assert (tmp_path / "decision_tree_inference.gif").stat().st_size > 0
    with open(tmp_path / "metrics_summary.csv") as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["model"] == "decision_tree" and float(row["test_accuracy"]) > 0
    assert "test accuracy" in capsys.readouterr().out


def test_inference_only_trains_silently(tmp_path, fast_specs):
    assert run(["--model", "knn", "--phase", "inference"], tmp_path) == 0
    assert not (tmp_path / "knn_training.gif").exists()
    assert (tmp_path / "knn_inference.gif").exists()


def test_nothing_to_do_is_an_error(capsys):
    assert cli.main(["--no-live", "--no-save"], fetcher=fake_fetcher(FRAME)) == 2


def test_bad_data_is_reported(tmp_path, capsys):
    broken = FRAME.drop(columns=["fare"])
    assert cli.main(["--no-live", "--output-dir", str(tmp_path)], fetcher=fake_fetcher(broken)) == 1
    assert "missing required columns" in capsys.readouterr().err


def test_scenes_record_a_video(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("titanic_anim.scenes.RACE_TICKS", 2)
    monkeypatch.setattr("titanic_anim.scenes.RESULTS_HOLD_SECONDS", 0.5)
    monkeypatch.setattr("titanic_anim.scenes.LANDSCAPE_TICKS", 2)
    assert run(["--scene", "race"], tmp_path) == 0
    assert "Final exam" in capsys.readouterr().out
    assert run(["--scene", "landscape"], tmp_path) == 0
    assert (tmp_path / "algorithm_race.gif").exists() and (tmp_path / "survival_landscape.gif").exists()


def test_a_failing_model_keeps_the_ones_before_it(tmp_path, monkeypatch, fast_specs, capsys):
    calls = []

    def flaky(spec, splits, options, keep_open=False):
        calls.append(spec.key)
        if len(calls) == 2:
            raise KeyboardInterrupt
        model, frame = train_silently(spec if spec.strategy == "data_growth" else
                                      replace(spec, steps=spec.steps[:2]), splits)
        return ModelResult(spec.key, spec.title, frame, None, ())

    monkeypatch.setattr(cli, "run_model", flaky)
    assert run(["--model", "all"], tmp_path) == 130
    with open(tmp_path / "metrics_summary.csv") as handle:
        assert [row["model"] for row in csv.DictReader(handle)] == [calls[0]]


def test_summary_rows_merge_by_model(tmp_path, splits):
    spec = replace(get_spec("decision_tree"), steps=(2,))
    _, frame = train_silently(spec, splits)
    row = result_row(ModelResult("decision_tree", spec.title, frame, None, ()), rows_used=10, batch_size=4)
    assert row["test_accuracy"] == "" and float(row["val_accuracy"]) > 0
    path = tmp_path / "summary.csv"
    merge_summary_csv(path, [{**row, "test_accuracy": "0.8000"}])
    merge_summary_csv(path, [row])  # a later training-only run keeps the earlier test score
    with open(path) as handle:
        (saved,) = list(csv.DictReader(handle))
    assert saved["test_accuracy"] == "0.8000"
    assert "decision_tree" in format_table([row])
