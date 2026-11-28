"""Voicing a series video: each chapter's words spoken over it, held where the voice runs long."""

import json
import shutil
import subprocess
from dataclasses import dataclass, field

import pytest

from animkit.lessons import cli
from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.encoder import probe
from animkit.lessons.lesson import LessonError
from animkit.lessons.timing import Timing
from animkit.lessons.tts import ServiceError, SpeechCache
from animkit.series.cards import paint_text
from animkit.series.chapters import Series, card
from animkit.series.render import render_series
from animkit.series.voice import run_series_voice, series_cues, voice_series

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")
TIMING = Timing(fps=10, scale=0.2)


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


def tiny_series(narration="The test error is $83k with an MLP.") -> Series:
    return Series(name="conclusion", title="Tiny: Conclusion", source_note="Data: somewhere", chapters=(
        card("title", "Conclusion", 5.0, "What we learned.", paint_text, kicker="SERIES", headline="Tiny"),
        card("board", "Scoreboard", 4.0, narration, paint_text, kicker="SCOREBOARD", headline="Scores"),
    ))


def rendered(tmp_path, series=None):
    video = tmp_path / "series" / f"{(series or tiny_series()).name}.mp4"
    render_series(series or tiny_series(), video, timing=TIMING, dpi=RESOLUTIONS["preview"])
    return video


def test_the_words_are_said_as_the_voice_should_say_them(tmp_path):
    record = {"chapters": [{"chapter": "board", "label": "Scoreboard", "frames": 8,
                            "narration": "The test error is $83k with an MLP."}]}
    [cue] = series_cues(record)
    assert cue.spoken == "The test error is 83 thousand dollars with a multi-layer perceptron."
    assert cue.screen == "Scoreboard"


@needs_ffmpeg
def test_a_voiced_series_holds_frames_and_records_its_narration(tmp_path):
    video = rendered(tmp_path)
    out = tmp_path / "series_voiced" / "conclusion.mp4"
    engine = ToneEngine()
    record = voice_series(video, out, SpeechCache(engine, tmp_path / "cache"))
    assert probe(out).frames == round(record["seconds"] * TIMING.fps)
    assert probe(out).frames > tiny_series().total_frames(TIMING)  # the 1 s chapters were held for the voice
    saved = json.loads(out.with_suffix(".narration.json").read_text())
    assert [row["chapter"] for row in saved["chapters"]] == ["title", "board"]
    assert saved["chapters"][1]["screen"] == "Scoreboard" and saved["engine"] == "tone:v1"
    assert {"start_s", "voice_at_s", "voice_s", "hold_s"} <= set(saved["chapters"][0])


@needs_ffmpeg
def test_words_with_an_abbreviation_it_cannot_say_are_refused_before_any_voice(tmp_path):
    video = rendered(tmp_path, tiny_series("The XYZ score."))
    engine = ToneEngine()
    with pytest.raises(LessonError, match="XYZ"):
        voice_series(video, tmp_path / "out" / "conclusion.mp4", SpeechCache(engine, tmp_path / "cache"))
    assert engine.calls == []


@needs_ffmpeg
def test_the_voiced_video_never_replaces_its_silent_source(tmp_path):
    video = rendered(tmp_path)
    with pytest.raises(Exception, match="source"):
        voice_series(video, video, SpeechCache(ToneEngine(), tmp_path / "cache"))


def run(tmp_path, argv, engine):
    return run_series_voice([*argv, "--cache-dir", str(tmp_path / "cache")],
                            engines={"tone": lambda voice: engine})


@needs_ffmpeg
def test_the_command_voices_every_series_video_in_the_folder(tmp_path, capsys):
    video = rendered(tmp_path)
    code = run(tmp_path, ["--engine", "tone", "--source-dir", str(video.parent)], ToneEngine())
    assert code == 0
    assert (tmp_path / "series_voiced" / "conclusion.mp4").is_file()
    assert "conclusion" in capsys.readouterr().out


def test_the_command_reports_a_folder_with_no_series_videos(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    code = run(tmp_path, ["--engine", "tone", "--source-dir", str(tmp_path / "empty")], ToneEngine())
    assert code == cli.FAILED and "no series video" in capsys.readouterr().err


@needs_ffmpeg
def test_a_voice_service_failure_stops_the_command(tmp_path, capsys):
    class NoKey(ToneEngine):
        def synthesize(self, text, path):
            raise ServiceError("ELEVENLABS_API_KEY is not set")

    video = rendered(tmp_path)
    code = run(tmp_path, ["--engine", "tone", "--source-dir", str(video.parent)], NoKey())
    assert code == cli.FAILED and "ELEVENLABS_API_KEY" in capsys.readouterr().err
