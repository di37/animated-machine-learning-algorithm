"""The ElevenLabs voice: the request it sends, what changes a clip, and which failures stop a run."""

import httpx
import pytest

from animkit.lessons import tts, voice_cli
from animkit.lessons.tts import ElevenLabsSpeech, ServiceError

KEY = "test-key-not-real"


class FakePost:
    """Stands in for ``httpx.post``: records each request and answers with ``response``."""

    def __init__(self, response=None, error=None):
        self.response = response or httpx.Response(200, content=b"ID3 fake mp3")
        self.error = error
        self.sent = []

    def __call__(self, url, **kwargs):
        self.sent.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response


def engine(post, **settings):
    return ElevenLabsSpeech(post=post, key_source=lambda: KEY, **settings)


def test_the_request_carries_voice_model_settings_and_text(tmp_path):
    post = FakePost()
    engine(post, voice="abc123").synthesize("Hello there.", tmp_path / "raw.mp3")
    [sent] = post.sent
    assert sent["url"] == "https://api.elevenlabs.io/v1/text-to-speech/abc123"
    assert sent["params"] == {"output_format": tts.ELEVENLABS_FORMAT}
    assert sent["headers"]["xi-api-key"] == KEY
    assert sent["json"] == {
        "text": "Hello there.", "model_id": tts.ELEVENLABS_MODEL, "seed": tts.ELEVENLABS_SEED,
        "voice_settings": {"stability": tts.ELEVENLABS_STABILITY, "similarity_boost": tts.ELEVENLABS_SIMILARITY,
                           "style": 0.0, "use_speaker_boost": True, "speed": tts.ELEVENLABS_SPEED}}
    assert (tmp_path / "raw.mp3").read_bytes() == b"ID3 fake mp3"


def test_the_defaults_are_the_mixture_of_experts_voice():
    voice = ElevenLabsSpeech()
    assert (voice.voice, voice.model, voice.suffix) == ("TX3LPaxmHKxFdv7VOQHJ", "eleven_multilingual_v2", ".mp3")


def test_a_voice_id_with_a_slash_cannot_change_the_endpoint(tmp_path):
    post = FakePost()
    engine(post, voice="../../v1/user").synthesize("hi", tmp_path / "raw.mp3")
    assert post.sent[0]["url"].endswith("/text-to-speech/..%2F..%2Fv1%2Fuser")


@pytest.mark.parametrize("change", [{"voice": "other"}, {"model": "eleven_flash_v2_5"}, {"speed": 1.1},
                                    {"stability": 0.3}, {"similarity": 0.9}, {"seed": 8}])
def test_anything_that_changes_the_sound_changes_the_identity(change):
    assert ElevenLabsSpeech(**change).identity != ElevenLabsSpeech().identity


def test_the_identity_names_the_engine_model_and_voice():
    assert ElevenLabsSpeech().identity.startswith("elevenlabs:eleven_multilingual_v2:TX3LPaxmHKxFdv7VOQHJ:")


@pytest.mark.parametrize("bad", [{"speed": 0.5}, {"speed": 1.3}, {"stability": -0.1}, {"similarity": 1.5},
                                 {"seed": -1}, {"seed": tts.ELEVENLABS_MAX_SEED + 1}])
def test_settings_outside_what_the_api_accepts_are_refused(bad):
    with pytest.raises(ValueError):
        ElevenLabsSpeech(**bad)


@pytest.mark.parametrize("status", [400, 401, 402, 404, 422, 429, 500])
def test_every_refusal_stops_the_run(tmp_path, status):
    """A refused key, credit, voice or model would refuse every later line too, so none is asked for."""
    refusal = httpx.Response(status, json={"detail": {"status": "quota_exceeded", "message": "Not enough credits"}})
    with pytest.raises(ServiceError, match="ElevenLabs") as caught:
        engine(FakePost(refusal)).synthesize("hi", tmp_path / "raw.mp3")
    assert "Not enough credits" in str(caught.value)
    assert not (tmp_path / "raw.mp3").exists()


def test_a_validation_error_is_reported_by_its_messages(tmp_path):
    refusal = httpx.Response(422, json={"detail": [{"loc": ["body", "model_id"], "msg": "unknown model",
                                                    "input": "x" * 1000}]})
    with pytest.raises(ServiceError) as caught:
        engine(FakePost(refusal)).synthesize("hi", tmp_path / "raw.mp3")
    assert "unknown model" in str(caught.value) and "xxxx" not in str(caught.value)


def test_a_request_that_cannot_be_sent_never_quotes_its_headers(tmp_path):
    leak = httpx.LocalProtocolError(f"Illegal header value b'{KEY}\\n'")
    with pytest.raises(ServiceError, match="ELEVENLABS_API_KEY") as caught:
        engine(FakePost(error=leak)).synthesize("hi", tmp_path / "raw.mp3")
    assert KEY not in str(caught.value)


@pytest.mark.parametrize("pasted", ["sk_one\nsk_two", "sk one", "sk_été", "sk_\x07bell"])
def test_a_key_that_is_not_one_key_is_refused_without_echoing_it(monkeypatch, pasted):
    monkeypatch.setenv("ELEVENLABS_API_KEY", pasted)
    with pytest.raises(ServiceError, match="the key alone") as caught:
        tts.elevenlabs_api_key()
    assert pasted not in str(caught.value)


def test_an_error_without_a_json_body_still_says_what_happened(tmp_path):
    with pytest.raises(ServiceError, match="503"):
        engine(FakePost(httpx.Response(503, text="<html>down</html>"))).synthesize("hi", tmp_path / "raw.mp3")


def test_no_connection_stops_the_run(tmp_path):
    with pytest.raises(ServiceError, match="could not reach ElevenLabs"):
        engine(FakePost(error=httpx.ConnectError("no route"))).synthesize("hi", tmp_path / "raw.mp3")


def test_the_key_is_never_part_of_an_error_message(tmp_path):
    with pytest.raises(ServiceError) as caught:
        engine(FakePost(httpx.Response(401, json={"detail": "invalid key"}))).synthesize("hi", tmp_path / "raw.mp3")
    assert KEY not in str(caught.value)


def test_a_missing_key_says_where_to_put_one(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(tts, "find_dotenv", lambda usecwd: "")
    with pytest.raises(ServiceError, match="ELEVENLABS_API_KEY.*\\.env"):
        tts.elevenlabs_api_key()


def test_the_key_is_read_from_the_environment_or_a_env_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-environment")
    assert tts.elevenlabs_api_key() == "from-environment"
    env_file = tmp_path / ".env"
    env_file.write_text("ELEVENLABS_API_KEY=from-file\n")
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    monkeypatch.setattr(tts, "find_dotenv", lambda usecwd: str(env_file))
    assert tts.elevenlabs_api_key() == "from-file"
    assert "ELEVENLABS_API_KEY" not in tts.os.environ


def test_the_voice_command_uses_elevenlabs_by_default():
    assert next(iter(voice_cli.ENGINES)) == "elevenlabs"
    assert voice_cli.ENGINES["elevenlabs"]("abc").voice == "abc"
    assert voice_cli.ENGINES["elevenlabs"](None).voice == tts.ELEVENLABS_VOICE
