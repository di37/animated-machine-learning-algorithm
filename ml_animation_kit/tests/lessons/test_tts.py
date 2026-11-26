"""Speech engines and the clip cache: one paid request per distinct line, trimmed and measured."""

import shutil
import subprocess
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import pytest

from animkit.lessons import tts
from animkit.lessons.tts import OpenAISpeech, SpeechCache, TtsError, audio_seconds

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")


@dataclass
class ToneEngine:
    """A stand-in engine: 0.4 s of silence, a tone as long as the text has words / 10, then silence."""

    identity: str = "tone:v1"
    suffix: str = ".wav"
    calls: list = field(default_factory=list)

    def synthesize(self, text: str, path) -> None:
        self.calls.append(text)
        seconds = len(text.split()) / 10
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"sine=frequency=330:sample_rate=24000:duration={seconds}",
                        "-af", "adelay=400:all=1,apad=pad_dur=0.5", str(path)], check=True)


@needs_ffmpeg
def test_a_line_is_synthesised_once_then_read_from_the_cache(tmp_path):
    engine = ToneEngine()
    cache = SpeechCache(engine, tmp_path / "cache")
    first = cache.clip("one two three four five six seven eight nine ten")
    again = cache.clip("one two three four five six seven eight nine ten")
    assert first == again and first.suffix == ".flac"
    assert engine.calls == ["one two three four five six seven eight nine ten"]


@needs_ffmpeg
def test_silence_around_the_speech_is_trimmed(tmp_path):
    clip = SpeechCache(ToneEngine(), tmp_path).clip(" ".join(["word"] * 10))  # 1.0 s of tone, 0.9 s of silence
    assert audio_seconds(clip) == pytest.approx(1.0, abs=0.08)


@needs_ffmpeg
def test_a_different_voice_is_a_different_clip(tmp_path):
    a = SpeechCache(ToneEngine(identity="tone:a"), tmp_path).clip("hello there")
    b = SpeechCache(ToneEngine(identity="tone:b"), tmp_path).clip("hello there")
    assert a != b


def test_an_engine_failure_leaves_nothing_in_the_cache(tmp_path):
    class Broken(ToneEngine):
        def synthesize(self, text, path):
            raise TtsError("the service said no")

    with pytest.raises(TtsError, match="said no"):
        SpeechCache(Broken(), tmp_path).clip("hello")
    assert not list(tmp_path.rglob("*.flac"))


def test_empty_text_is_refused(tmp_path):
    with pytest.raises(ValueError):
        SpeechCache(ToneEngine(), tmp_path).clip("   ")


def test_audio_seconds_of_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(TtsError):
        audio_seconds(tmp_path / "missing.flac")


# ---- OpenAI --------------------------------------------------------------------------------------------

class FakeStream:
    def __init__(self, sink, kwargs):
        self.sink, self.kwargs = sink, kwargs

    def __enter__(self):
        self.sink.append(self.kwargs)
        return SimpleNamespace(stream_to_file=lambda path: path.write_bytes(b"RIFF"))

    def __exit__(self, *exc):
        return False


def fake_client(sink):
    create = lambda **kwargs: FakeStream(sink, kwargs)  # noqa: E731
    return SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(
        with_streaming_response=SimpleNamespace(create=create))))


def test_openai_request_carries_model_voice_instructions_and_text(tmp_path):
    sent = []
    engine = OpenAISpeech(voice="marin", client_factory=lambda: fake_client(sent))
    engine.synthesize("Hello there.", tmp_path / "raw.wav")
    assert sent == [{"model": tts.OPENAI_MODEL, "voice": "marin", "input": "Hello there.",
                     "instructions": tts.OPENAI_INSTRUCTIONS, "response_format": "wav"}]
    assert (tmp_path / "raw.wav").read_bytes() == b"RIFF"


def test_openai_identity_changes_with_voice_and_instructions():
    base = OpenAISpeech(voice="marin")
    assert base.identity != OpenAISpeech(voice="cedar").identity
    assert base.identity != OpenAISpeech(voice="marin", instructions="Whisper.").identity


def test_openai_without_a_key_says_where_to_put_one(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here or above
    monkeypatch.setattr(tts, "find_dotenv", lambda usecwd: "")
    with pytest.raises(TtsError, match="OPENAI_API_KEY"):
        tts.openai_client()


def test_openai_service_errors_become_tts_errors(tmp_path):
    def refuse(**kwargs):
        raise tts.openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/audio/speech"))

    client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(
        with_streaming_response=SimpleNamespace(create=refuse))))
    with pytest.raises(TtsError, match="OpenAI"):
        OpenAISpeech(client_factory=lambda: client).synthesize("hi", tmp_path / "raw.wav")
    assert not (tmp_path / "raw.wav").exists()


def test_openai_network_failures_mid_stream_are_service_errors(tmp_path):
    def drop(**kwargs):
        raise httpx.RemoteProtocolError("connection dropped")

    client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(
        with_streaming_response=SimpleNamespace(create=drop))))
    with pytest.raises(tts.ServiceError, match="OpenAI"):
        OpenAISpeech(client_factory=lambda: client).synthesize("hi", tmp_path / "raw.wav")


def test_a_missing_key_is_a_service_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(tts, "find_dotenv", lambda usecwd: "")
    with pytest.raises(tts.ServiceError):
        tts.openai_api_key()


def test_the_key_is_read_from_env_file_without_changing_the_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test-key-from-file\nOTHER_SETTING=1\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_SETTING", raising=False)
    monkeypatch.setattr(tts, "find_dotenv", lambda usecwd: str(env_file))
    assert tts.openai_api_key() == "test-key-from-file"
    assert "OPENAI_API_KEY" not in tts.os.environ and "OTHER_SETTING" not in tts.os.environ


@needs_ffmpeg
def test_a_clip_that_is_all_silence_is_refused_and_not_cached(tmp_path):
    class Silent(ToneEngine):
        def synthesize(self, text, path):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "anullsrc=r=24000:cl=mono", "-t", "1", str(path)], check=True)

    with pytest.raises(TtsError, match="silent"):
        SpeechCache(Silent(), tmp_path).clip("hello")
    assert not list(tmp_path.rglob("*.flac"))


def test_a_change_to_the_trimming_is_a_different_clip(monkeypatch, tmp_path):
    cache = SpeechCache(ToneEngine(), tmp_path)
    before = cache.path_for("hello")
    monkeypatch.setattr(tts, "TRIM_SETTINGS", "trim:something-else")
    assert cache.path_for("hello") != before


@needs_ffmpeg
def test_audio_seconds_of_a_file_that_is_not_audio_is_an_error(tmp_path):
    junk = tmp_path / "junk.flac"
    junk.write_bytes(b"not audio at all")
    with pytest.raises(TtsError):
        audio_seconds(junk)
