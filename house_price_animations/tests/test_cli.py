import csv

import pytest

from house_anim.cli import main, parse_args
from tests.conftest import fake_fetcher, make_housing_frame

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


def test_defaults():
    args = parse_args([])
    assert args.model == "random_forest"
    assert args.phase == "both"
    assert args.fps == 12
    assert not args.no_live


@pytest.mark.parametrize(
    "argv",
    [["--fps", "0"], ["--fps", "500"], ["--batch-size", "0"], ["--model", "xgboost"], ["--dpi", "5"]],
)
def test_invalid_arguments_exit(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_list_prints_models(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "gradient_boosting" in out and "Lasso" in out


def test_nothing_to_do_is_an_error(capsys):
    assert main(["--no-live", "--no-save"]) == 2
    assert "nothing to do" in capsys.readouterr().err


def test_headless_without_saving_is_nothing_to_do(capsys):
    # The test backend is Agg, so a "live" request falls back to no window at all.
    assert main(["--no-save"]) == 2
    assert "nothing to do" in capsys.readouterr().err


def test_completed_models_are_reported_when_a_later_one_stops(monkeypatch, tmp_path, capsys):
    from animkit.recorder import RenderingStopped
    from house_anim import cli

    real_run_model = cli.run_model

    def first_then_stop(spec, *args, **kwargs):
        if spec.key != "linear_regression":
            raise RenderingStopped("closed")
        return real_run_model(spec, *args, **kwargs)

    monkeypatch.setattr(cli, "run_model", first_then_stop)
    fetcher = fake_fetcher(make_housing_frame(n_rows=300))
    argv = ["--model", "all", "--no-live", "--format", "gif", "--fps", "2", "--dpi", "20",
            "--output-dir", str(tmp_path)]
    assert main(argv, fetcher=fetcher) == 130
    with open(tmp_path / "metrics_summary.csv", newline="") as handle:
        assert [row["model"] for row in csv.DictReader(handle)] == ["linear_regression"]
    captured = capsys.readouterr()
    assert "linear_regression" in captured.out and "Stopped early" in captured.err


def test_unwritable_output_dir_is_reported(tmp_path, capsys):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    fetcher = fake_fetcher(make_housing_frame(n_rows=300))
    argv = ["--model", "linear_regression", "--no-live", "--format", "gif", "--dpi", "20",
            "--output-dir", str(blocker)]
    assert main(argv, fetcher=fetcher) == 1
    assert "error:" in capsys.readouterr().err


def test_numerical_errors_are_reported(monkeypatch, capsys):
    from house_anim import cli
    from house_anim.trainers import NumericalError

    def diverge(*args, **kwargs):
        raise NumericalError("SGDRegressor produced non-finite predictions")

    monkeypatch.setattr(cli, "run_model", diverge)
    fetcher = fake_fetcher(make_housing_frame(n_rows=100))
    assert main(["--no-live"], fetcher=fetcher) == 1
    assert "non-finite" in capsys.readouterr().err


def test_data_errors_are_reported(capsys):
    def failing_fetcher(as_frame=True):
        raise OSError("offline")

    assert main(["--no-live"], fetcher=failing_fetcher) == 1
    assert "California Housing" in capsys.readouterr().err


def test_non_integer_argument_exits():
    with pytest.raises(SystemExit):
        parse_args(["--fps", "fast"])


def test_closing_the_live_window_exits_cleanly(monkeypatch, capsys):
    from animkit.recorder import RenderingStopped
    from house_anim import cli

    def stop(*args, **kwargs):
        raise RenderingStopped("closed")

    monkeypatch.setattr(cli, "run_model", stop)
    fetcher = fake_fetcher(make_housing_frame(n_rows=100))
    assert main(["--no-live"], fetcher=fetcher) == 130
    assert "Stopped early" in capsys.readouterr().err


def test_end_to_end_records_videos_and_summary(tmp_path, capsys):
    fetcher = fake_fetcher(make_housing_frame(n_rows=300))
    argv = [
        "--model", "linear_regression",
        "--no-live",
        "--format", "gif",
        "--fps", "2",
        "--dpi", "20",
        "--batch-size", "30",
        "--output-dir", str(tmp_path),
    ]
    assert main(argv, fetcher=fetcher) == 0

    assert (tmp_path / "linear_regression_training.gif").stat().st_size > 0
    assert (tmp_path / "linear_regression_inference.gif").stat().st_size > 0
    with open(tmp_path / "metrics_summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model"] == "linear_regression"
    assert float(rows[0]["test_rmse_k"]) > 0.0
    assert "linear_regression" in capsys.readouterr().out


def test_inference_only_phase_trains_silently(tmp_path):
    fetcher = fake_fetcher(make_housing_frame(n_rows=300))
    argv = [
        "--model", "linear_regression", "--phase", "inference", "--no-live",
        "--format", "gif", "--fps", "2", "--dpi", "20", "--output-dir", str(tmp_path),
    ]
    assert main(argv, fetcher=fetcher) == 0
    assert not (tmp_path / "linear_regression_training.gif").exists()
    assert (tmp_path / "linear_regression_inference.gif").exists()


@pytest.mark.parametrize("scene, filename", [("race", "algorithm_race.gif"), ("map", "california_map.gif")])
def test_scenes_record_their_video(monkeypatch, tmp_path, capsys, scene, filename):
    from house_anim import scenes

    monkeypatch.setattr(scenes, "RACE_TICKS", 3)
    monkeypatch.setattr(scenes, "MAP_TICKS", 2)
    monkeypatch.setattr(scenes, "MAP_GRID_STEP_DEG", 0.5)
    monkeypatch.setattr(scenes, "MAP_MASK_RADIUS_DEG", 0.6)
    fetcher = fake_fetcher(make_housing_frame(n_rows=300))
    argv = ["--scene", scene, "--no-live", "--format", "gif", "--fps", "2", "--dpi", "20",
            "--output-dir", str(tmp_path)]
    assert main(argv, fetcher=fetcher) == 0
    assert (tmp_path / filename).stat().st_size > 0
    if scene == "race":
        assert "Final exam" in capsys.readouterr().out


def test_scene_defaults_to_dashboards():
    assert parse_args([]).scene == "dashboards"
