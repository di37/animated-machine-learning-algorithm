"""The lesson command line every project shares: the standard options, the listing, the loading, the failures."""

import shutil
from dataclasses import fields, replace
from pathlib import Path

import pytest

from animkit.lessons import cli
from animkit.lessons.config import DEFAULT_RESOLUTION, RESOLUTIONS
from animkit.lessons.director import RenderReport
from animkit.lessons.encoder import Mp4Encoder
from animkit.lessons.lesson import LessonError

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
KEYS = ("first", "second")
VIDEO_SIZE = (64, 36)


class LoadFailed(RuntimeError):
    """The stand-in project's data error."""


def project(summary: list | None = None, **changes) -> cli.MatplotlibProject:
    """A stand-in project with two lessons; ``summary`` collects the rows it would write."""
    rows = summary if summary is not None else []
    stand_in = cli.MatplotlibProject(
        description="Narrated lessons.", load_data=lambda fetcher, max_rows: "data",
        data_errors=(LoadFailed, ValueError), lesson_keys=lambda: KEYS, describe=lambda key: f"the {key} lesson",
        build_lesson=lambda key, data: (key, data), scene="the scene", summary_file="summary.csv",
        summary_row=lambda lesson, report, info, resolution: {"lesson": lesson[0], "frames": info.frames},
        merge_summary=lambda path, new: rows.extend((path.name, row) for row in new),
        default_output_dir=Path("outputs/lessons"), max_rows=(50, 1000), max_rows_help="subsample for a quick run")
    return replace(stand_in, **changes)


def tiny_video(calls: list):
    """A renderer that writes a real three-frame MP4, noting what it was asked for."""
    def render(lesson, path, *, scene, timing, dpi):
        calls.append((lesson[0], scene, timing.scale, dpi))
        width, height = VIDEO_SIZE
        with Mp4Encoder(path, width, height, timing.fps) as video:
            video.write(bytes(width * height * 4), repeat=3)
        return RenderReport(path=path, frames=video.frames_written, seconds=video.frames_written / timing.fps)

    return render


def failing(error: BaseException, calls: list | None = None):
    """A renderer that raises ``error``, noting each lesson and file it was asked for."""
    def render(lesson, path, **options):
        if calls is not None:
            calls.append((lesson[0], path.name))
        raise error

    return render


def test_a_lesson_project_must_say_where_its_options_come_from():
    shared = {field.name: getattr(project(), field.name) for field in fields(cli.LessonProject)}
    with pytest.raises(TypeError):
        cli.LessonProject(**shared)


def test_the_standard_options_and_their_defaults():
    parser = project().parser()
    assert parser.description == "Narrated lessons."
    assert vars(parser.parse_args([])) == {"model": "all", "list": False, "resolution": DEFAULT_RESOLUTION,
                                           "duration_scale": 1.0, "max_rows": None,
                                           "output_dir": Path("outputs/lessons")}


def test_model_takes_a_lesson_key_or_all():
    parser = project().parser()
    assert parser.parse_args(["--model", "second"]).model == "second"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "third"])


@pytest.mark.parametrize("option", [["--max-rows", "49"], ["--max-rows", "1001"], ["--duration-scale", "0"],
                                    ["--duration-scale", "abc"], ["--resolution", "8k"]])
def test_rejects_values_outside_the_projects_ranges(option):
    with pytest.raises(SystemExit):
        project().parser().parse_args(option)


def test_list_prints_every_lesson_and_loads_nothing(capsys):
    def no_loading(fetcher, max_rows):
        raise AssertionError("listing the lessons loads no data")

    assert cli.run(project(load_data=no_loading), ["--list"], fetcher=None, render=failing(LessonError("no"))) == 0
    out = capsys.readouterr().out
    assert all(f"{key:<22} the {key} lesson" in out for key in KEYS)


def test_the_data_is_loaded_once_with_the_fetcher_and_the_row_limit():
    loaded, built = [], []

    def load(fetcher, max_rows):
        loaded.append((fetcher, max_rows))
        return "data"

    def build(key, data):
        built.append(data)
        return key, data

    fetch = object()
    cli.run(project(load_data=load, build_lesson=build), ["--max-rows", "60"], fetcher=fetch,
            render=failing(LessonError("stop")))
    assert loaded == [(fetch, 60)] and built == ["data", "data"]


def test_a_data_error_is_reported_and_nothing_is_rendered(capsys):
    def unreachable(fetcher, max_rows):
        raise LoadFailed("no network")

    calls = []
    assert cli.run(project(load_data=unreachable), [], fetcher=None, render=failing(LessonError("no"), calls)) == 1
    assert "error: no network" in capsys.readouterr().err and calls == []


def test_stopping_while_loading_exits_as_interrupted(capsys):
    def interrupted(fetcher, max_rows):
        raise KeyboardInterrupt

    assert cli.run(project(load_data=interrupted), [], fetcher=None, render=failing(LessonError("no"))) == 130
    assert "Stopped while loading the dataset." in capsys.readouterr().err


def test_model_renders_only_that_lesson_to_its_own_file():
    calls = []
    cli.run(project(), ["--model", "second"], fetcher=None, render=failing(LessonError("stop"), calls))
    assert calls == [("second", "second_lesson.mp4")]


def test_the_progress_lines_carry_the_label(capsys):
    cli.run(project(), [], fetcher=None, render=failing(LessonError("stop")))
    cli.run(project(), [], fetcher=None, render=failing(LessonError("stop")), label=" (by hand)")
    out = capsys.readouterr().out
    assert "[1/2] first …" in out and "[2/2] second (by hand) …" in out


def test_every_failed_lesson_is_reported_and_the_rest_go_on(capsys):
    status = cli.run(project(), [], fetcher=None, render=failing(LessonError("a solver did not converge")))
    err = capsys.readouterr().err
    assert status == 1
    assert "error: first: a solver did not converge" in err and "error: second: a solver did not converge" in err
    assert "2 of 2 lessons failed: first, second" in err


def test_stopping_during_a_lesson_exits_as_interrupted_and_renders_no_more(capsys):
    calls = []
    assert cli.run(project(), [], fetcher=None, render=failing(KeyboardInterrupt(), calls)) == 130
    assert [key for key, _ in calls] == ["first"]
    assert "the unfinished video was discarded" in capsys.readouterr().err


@needs_ffmpeg
def test_renders_every_lesson_with_the_projects_scene_and_writes_its_summary(tmp_path, capsys):
    calls, rows = [], []
    status = cli.run(project(rows), ["--output-dir", str(tmp_path), "--duration-scale", "0.5", "--resolution",
                                     "preview"], fetcher=None, render=tiny_video(calls))
    assert status == 0
    assert calls == [(key, "the scene", 0.5, RESOLUTIONS["preview"]) for key in KEYS]
    assert all((tmp_path / f"{key}_lesson.mp4").is_file() for key in KEYS)
    assert rows == [("summary.csv", {"lesson": key, "frames": 3}) for key in KEYS]
    out = capsys.readouterr().out
    assert "[1/2] first …" in out and "[2/2] second …" in out and out.count("    saved ") == 2


@needs_ffmpeg
def test_a_lesson_that_fails_does_not_stop_the_next_one(tmp_path, capsys):
    rows = []
    video = tiny_video([])

    def first_fails(lesson, path, **options):
        if lesson[0] == "first":
            raise OSError("disk full")
        return video(lesson, path, **options)

    assert cli.run(project(rows), ["--output-dir", str(tmp_path)], fetcher=None, render=first_fails) == 1
    assert [row["lesson"] for _, row in rows] == ["second"]
    assert "1 of 2 lessons failed: first" in capsys.readouterr().err
