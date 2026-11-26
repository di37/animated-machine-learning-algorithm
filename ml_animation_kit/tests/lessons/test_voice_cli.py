"""The voice-over command: script, synthesise, dub, for each chosen lesson; failures do not stop the rest."""

import json
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from animkit.lessons import cli, voice_cli
from animkit.lessons.encoder import probe
from animkit.lessons.timing import CHAPTERS, Timing
from animkit.lessons.tts import ServiceError, TtsError

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")
KEYS = ("first", "second")
TIMING = Timing(fps=10, scale=0.05)


def lesson_for(key: str, data) -> SimpleNamespace:
    story = SimpleNamespace(title=f"The {key} model", method="trained somehow", objective="Watch it learn.",
                            initial="A first guess.", training="It learns.", final="RMSE $83k.",
                            inference="A new example.", takeaway="It works.", limitation="not always.")
    return SimpleNamespace(key=key, story=story)


SCENE = SimpleNamespace(
    data_caption=lambda lesson: "Each dot is one example.", exam_caption=lambda lesson, exam: "The final exam.",
    prepare_exam=lambda lesson: "exam", compare=lambda lesson: SimpleNamespace(headline="More inputs help."),
    prepare_surface=lambda lesson: SimpleNamespace(caption="A smooth sheet."))


def project(source_dir: Path) -> cli.MatplotlibProject:
    return cli.MatplotlibProject(
        description="Narrated lessons.", load_data=lambda fetcher, max_rows: "data", data_errors=(ValueError,),
        lesson_keys=lambda: KEYS, describe=lambda key: key, build_lesson=lesson_for, scene=SCENE,
        summary_file="summary.csv", summary_row=lambda *a: {}, merge_summary=lambda *a: None,
        default_output_dir=source_dir, max_rows=(50, 1000), max_rows_help="")


@dataclass
class ToneEngine:
    identity: str = "tone:v1"
    suffix: str = ".wav"
    calls: list = field(default_factory=list)

    def synthesize(self, text: str, path) -> None:
        self.calls.append(text)
        seconds = len(text.split()) / 10
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"sine=frequency=330:sample_rate=24000:duration={seconds}", str(path)], check=True)


def make_video(path: Path, frames: int, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=64x36:rate={fps}",
                    "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def run(tmp_path, argv, engine=None):
    engine = engine or ToneEngine()
    code = voice_cli.run(project(tmp_path / "lessons"), [*argv, "--cache-dir", str(tmp_path / "cache")],
                         fetcher=None, engines={"tone": lambda voice: engine}, timing=TIMING)
    return code, engine


def test_script_only_writes_the_narration_and_calls_no_voice(tmp_path):
    code, engine = run(tmp_path, ["--engine", "tone", "--script-only"])
    assert code == 0 and engine.calls == []
    script = json.loads((tmp_path / "lessons_voiced" / "first_lesson.narration.json").read_text())
    assert [row["chapter"] for row in script["chapters"]] == list(CHAPTERS)
    assert script["chapters"][4]["spoken"] == "Root mean squared error 83 thousand dollars."


@needs_ffmpeg
def test_each_lesson_is_dubbed_with_its_narration(tmp_path):
    for key in KEYS:
        make_video(tmp_path / "lessons" / f"{key}_lesson.mp4", TIMING.total_frames)
    code, engine = run(tmp_path, ["--engine", "tone"])
    assert code == 0
    out = tmp_path / "lessons_voiced" / "first_lesson.mp4"
    script = json.loads((tmp_path / "lessons_voiced" / "first_lesson.narration.json").read_text())
    assert probe(out).frames == round(script["seconds"] * TIMING.fps)
    assert probe(out).frames > TIMING.total_frames  # some chapters were held for their voice
    assert len(set(engine.calls)) == len(engine.calls)  # each distinct line was asked for once
    assert script["engine"] == "tone:v1"


@needs_ffmpeg
def test_a_second_run_reuses_every_clip(tmp_path):
    make_video(tmp_path / "lessons" / "first_lesson.mp4", TIMING.total_frames)
    run(tmp_path, ["--engine", "tone", "--model", "first"])
    code, engine = run(tmp_path, ["--engine", "tone", "--model", "first"])
    assert code == 0 and engine.calls == []


@needs_ffmpeg
def test_a_missing_source_video_fails_that_lesson_only(tmp_path, capsys):
    make_video(tmp_path / "lessons" / "second_lesson.mp4", TIMING.total_frames)
    code, _ = run(tmp_path, ["--engine", "tone"])
    assert code == cli.FAILED
    assert (tmp_path / "lessons_voiced" / "second_lesson.mp4").is_file()
    assert "first" in capsys.readouterr().err


def test_a_voice_service_error_stops_the_run(tmp_path, capsys):
    class NoKey(ToneEngine):
        def synthesize(self, text, path):
            raise ServiceError("OPENAI_API_KEY is not set")

    for key in KEYS:
        (tmp_path / "lessons").mkdir(exist_ok=True)
        (tmp_path / "lessons" / f"{key}_lesson.mp4").write_bytes(b"")
    code, engine = run(tmp_path, ["--engine", "tone"], engine=NoKey())
    assert code == cli.FAILED
    assert len(engine.calls) == 0
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and err.count("OPENAI_API_KEY") == 1  # reported once, not per lesson


def test_output_dir_can_be_chosen(tmp_path):
    code, _ = run(tmp_path, ["--engine", "tone", "--script-only", "--output-dir", str(tmp_path / "elsewhere")])
    assert code == 0 and (tmp_path / "elsewhere" / "second_lesson.narration.json").is_file()


@needs_ffmpeg
def test_a_clip_that_fails_fails_that_lesson_only(tmp_path, capsys):
    class FailsOnFirst(ToneEngine):
        def synthesize(self, text, path):
            if text.startswith("The first model"):
                raise TtsError("this clip could not be trimmed")
            super().synthesize(text, path)

    for key in KEYS:
        make_video(tmp_path / "lessons" / f"{key}_lesson.mp4", TIMING.total_frames)
    code, _ = run(tmp_path, ["--engine", "tone"], engine=FailsOnFirst())
    assert code == cli.FAILED
    assert (tmp_path / "lessons_voiced" / "second_lesson.mp4").is_file()
    assert "first: this clip could not be trimmed" in capsys.readouterr().err


@needs_ffmpeg
def test_a_lesson_saying_an_abbreviation_with_no_full_name_is_not_voiced(tmp_path, capsys):
    """The voice never spells out an abbreviation: the lesson fails before any line is paid for."""
    def with_unknown(key, data):
        lesson = lesson_for(key, data)
        story = SimpleNamespace(**{**vars(lesson.story), "takeaway": "The XYZ score works."})
        return SimpleNamespace(key=key, story=story) if key == "first" else lesson

    for key in KEYS:
        make_video(tmp_path / "lessons" / f"{key}_lesson.mp4", TIMING.total_frames)
    engine = ToneEngine()
    proj = replace(project(tmp_path / "lessons"), build_lesson=with_unknown)
    code = voice_cli.run(proj, ["--engine", "tone", "--cache-dir", str(tmp_path / "cache")],
                         fetcher=None, engines={"tone": lambda voice: engine}, timing=TIMING)
    assert code == cli.FAILED
    assert "first: no full name for XYZ" in capsys.readouterr().err
    assert not any(text.startswith("The first model") for text in engine.calls)
    assert (tmp_path / "lessons_voiced" / "second_lesson.mp4").is_file()


def test_the_output_folder_cannot_be_the_source_folder(tmp_path, capsys):
    code, _ = run(tmp_path, ["--engine", "tone", "--output-dir", str(tmp_path / "lessons")])
    assert code == cli.FAILED
    assert "source" in capsys.readouterr().err
