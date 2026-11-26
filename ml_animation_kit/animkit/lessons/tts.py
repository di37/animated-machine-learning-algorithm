"""Text-to-speech: the engines that make a voice clip, and the cache that makes each line only once.

``ElevenLabsSpeech`` is the default online voice (ElevenLabs' eleven_multilingual_v2, in the same voice and
settings as the Mixture-of-Experts videos), and ``OpenAISpeech`` the other (OpenAI's gpt-4o-mini-tts). Their
keys are read from ``ELEVENLABS_API_KEY`` and ``OPENAI_API_KEY``, from the environment or a ``.env`` file in
the working folder or above it, and are never written anywhere. ``SaySpeech`` is macOS's built-in ``say``:
free and offline, for trying the pipeline before spending anything.

``SpeechCache`` keys every clip on the engine (model, voice, instructions) and the exact text, so a lesson
can be dubbed again, or its Manim twin dubbed, without paying twice. Clips are stored as FLAC with the
silence around the speech trimmed, so a clip's length is the length of its words.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx
import openai
from dotenv import dotenv_values, find_dotenv

OPENAI_MODEL = "gpt-4o-mini-tts"
OPENAI_VOICE = "marin"
OPENAI_INSTRUCTIONS = (
    "Voice: a clear, friendly teacher narrating a short machine-learning animation. "
    "Tone: warm, confident and precise, never salesy. "
    "Pace: brisk but unhurried, with a short natural pause at each full stop. "
    "Read numbers naturally, and say letters joined by hyphens, such as R-M-S-E, one letter at a time."
)
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
ELEVENLABS_VOICE = "TX3LPaxmHKxFdv7VOQHJ"  # the Mixture-of-Experts videos' voice
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_FORMAT = "mp3_44100_128"
ELEVENLABS_SPEED = 1.0  # the voice's own pace
ELEVENLABS_SPEEDS = (0.7, 1.2)  # the speeds the API accepts
ELEVENLABS_STABILITY = 0.5  # how even the delivery is, 0 to 1
ELEVENLABS_SIMILARITY = 0.75  # how closely the voice keeps to its speaker, 0 to 1
ELEVENLABS_SEED = 7  # fixes the sampling, so the same request gives the same audio
ELEVENLABS_MAX_SEED = 4_294_967_295  # the largest seed the API accepts
ELEVENLABS_TIMEOUT_S = 60
DETAIL_CHARS = 300  # of an API refusal's message, in an error
SAY_VOICE = "Samantha"
SAY_RATE = 185  # words per minute
SAMPLE_RATE = 48000
SILENCE_THRESHOLD = "-50dB"
KEEP_SILENCE_S = 0.02  # left at each end of a trimmed clip, so the first and last sounds are not clipped
MIN_CLIP_S = 0.05  # a trimmed clip shorter than this came back silent
TRIM_SETTINGS = f"trim:{SILENCE_THRESHOLD}:{KEEP_SILENCE_S}:{SAMPLE_RATE}"  # part of every clip's cache key
IDENTITY_DIGEST_CHARS = 12  # of the instructions' digest, in an engine's identity
CLIP_DIGEST_CHARS = 32  # of a clip's file name
KEY_HELP = ("{variable} is not set: add a line {variable}=... to a .env file in this folder or a "
            "folder above it (the repo's .gitignore keeps .env out of git), or export it in your shell")


class TtsError(RuntimeError):
    """A voice clip could not be made, for example it came back silent or could not be trimmed."""


class ServiceError(TtsError):
    """The voice service cannot be used at all: no key, refused, unreachable, or the engine is missing.

    Every later clip would fail the same way, so a run stops on this one.
    """


class SpeechEngine(Protocol):
    """Makes one voice clip.

    Attributes:
        identity: Everything that changes the sound (engine, model, voice, instructions), for the cache key.
        suffix: The file suffix of the audio ``synthesize`` writes, for example ".wav".
    """

    identity: str
    suffix: str

    def synthesize(self, text: str, path: Path) -> None:
        """Writes ``text``, spoken, to ``path``."""


def api_key(variable: str) -> str:
    """A service's key: from the environment, else from the nearest ``.env`` file.

    Only the key is read from the file; nothing is added to the environment.

    Args:
        variable: The key's name, for example "ELEVENLABS_API_KEY".

    Returns:
        The key.

    Raises:
        ServiceError: If no key is set, with where to put one, or it holds more than one key's characters
            (never echoing it: a request carrying it would fail with the key in its error).
    """
    key = os.environ.get(variable, "").strip()
    if not key:
        env_file = find_dotenv(usecwd=True)
        key = (dotenv_values(env_file).get(variable) or "").strip() if env_file else ""
    if not key:
        raise ServiceError(KEY_HELP.format(variable=variable))
    if not (key.isascii() and key.isprintable()) or any(char.isspace() for char in key):
        raise ServiceError(f"{variable} holds spaces, line breaks or other characters no key has: "
                           f"set it to the key alone")
    return key


def openai_api_key() -> str:
    """The OpenAI key, from ``OPENAI_API_KEY``; see ``api_key``."""
    return api_key("OPENAI_API_KEY")


def elevenlabs_api_key() -> str:
    """The ElevenLabs key, from ``ELEVENLABS_API_KEY``; see ``api_key``."""
    return api_key("ELEVENLABS_API_KEY")


def openai_client() -> Any:
    """An OpenAI client with the key from ``openai_api_key``.

    Returns:
        ``openai.OpenAI(api_key=...)``.

    Raises:
        ServiceError: If no key is set.
    """
    return openai.OpenAI(api_key=openai_api_key())


@dataclass(frozen=True)
class OpenAISpeech:
    """OpenAI's text-to-speech.

    Attributes:
        voice: The voice, for example "marin" or "cedar".
        model: The speech model.
        instructions: How to speak: tone, pace, pronunciation.
        client_factory: Makes the client; tests pass a stand-in.
        suffix: The audio format requested.
    """

    voice: str = OPENAI_VOICE
    model: str = OPENAI_MODEL
    instructions: str = OPENAI_INSTRUCTIONS
    client_factory: Callable[[], Any] = openai_client
    suffix: str = ".wav"

    @property
    def identity(self) -> str:
        """The engine, model, voice and a digest of the instructions.

        Returns:
            For example "openai:gpt-4o-mini-tts:marin:1a2b3c4d5e6f".
        """
        digest = hashlib.sha256(self.instructions.encode()).hexdigest()[:IDENTITY_DIGEST_CHARS]
        return f"openai:{self.model}:{self.voice}:{digest}"

    def synthesize(self, text: str, path: Path) -> None:
        """Asks OpenAI to speak ``text`` and streams the audio to ``path``.

        Args:
            text: What to say.
            path: Where the WAV audio goes.

        Raises:
            ServiceError: If there is no key, or the service fails or drops mid-stream.
        """
        client = self.client_factory()
        try:
            with client.audio.speech.with_streaming_response.create(
                    model=self.model, voice=self.voice, input=text, instructions=self.instructions,
                    response_format=self.suffix.lstrip(".")) as response:
                response.stream_to_file(path)
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise ServiceError(f"OpenAI text-to-speech failed: {exc}") from exc


def _detail(response: httpx.Response) -> str:
    """Why the API refused a request, in its own words where its JSON body gives any, kept short."""
    try:
        detail = response.json().get("detail", "")
    except (ValueError, AttributeError):  # no JSON body, or not an object
        return response.reason_phrase or str(response.status_code)
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("status") or ""
    if isinstance(detail, list):  # a validation error: one entry per field
        detail = "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail)[:DETAIL_CHARS] or response.reason_phrase


@dataclass(frozen=True)
class ElevenLabsSpeech:
    """ElevenLabs' text-to-speech, with the Mixture-of-Experts videos' voice and settings by default.

    Attributes:
        voice: The ElevenLabs voice ID.
        model: The speech model.
        speed: Speaking speed, within ``ELEVENLABS_SPEEDS``.
        stability: How even the delivery is, 0 to 1.
        similarity: How closely the voice keeps to its speaker, 0 to 1.
        seed: Fixes the sampling, so the same request gives the same audio.
        output_format: The audio format, sample rate and bitrate asked for.
        post: Sends the request (``httpx.post``); tests pass a stand-in.
        key_source: Returns the API key; asked only when a clip is made.
        suffix: The file suffix of what the API returns.
    """

    voice: str = ELEVENLABS_VOICE
    model: str = ELEVENLABS_MODEL
    speed: float = ELEVENLABS_SPEED
    stability: float = ELEVENLABS_STABILITY
    similarity: float = ELEVENLABS_SIMILARITY
    seed: int = ELEVENLABS_SEED
    output_format: str = ELEVENLABS_FORMAT
    post: Callable[..., httpx.Response] = httpx.post
    key_source: Callable[[], str] = elevenlabs_api_key
    suffix: str = ".mp3"

    def __post_init__(self) -> None:
        """Refuses settings the API would refuse, before anything is sent."""
        low, high = ELEVENLABS_SPEEDS
        if not low <= self.speed <= high:
            raise ValueError(f"speed must be from {low} to {high}, got {self.speed}")
        for name in ("stability", "similarity"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be from 0 to 1, got {getattr(self, name)}")
        if not 0 <= self.seed <= ELEVENLABS_MAX_SEED:
            raise ValueError(f"seed must be from 0 to {ELEVENLABS_MAX_SEED}, got {self.seed}")

    def _body(self, text: str) -> dict[str, Any]:
        """The JSON body that asks for ``text`` in this voice."""
        return {"text": text, "model_id": self.model, "seed": self.seed,
                "voice_settings": {"stability": self.stability, "similarity_boost": self.similarity,
                                   "style": 0.0, "use_speaker_boost": True, "speed": self.speed}}

    @property
    def identity(self) -> str:
        """The engine, model, voice and a digest of every setting that changes the sound.

        Returns:
            For example "elevenlabs:eleven_multilingual_v2:TX3LPaxmHKxFdv7VOQHJ:1a2b3c4d5e6f".
        """
        settings = {"format": self.output_format, **self._body("")}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:IDENTITY_DIGEST_CHARS]
        return f"elevenlabs:{self.model}:{self.voice}:{digest}"

    def synthesize(self, text: str, path: Path) -> None:
        """Asks ElevenLabs to speak ``text`` and writes the audio to ``path``.

        Args:
            text: What to say.
            path: Where the MP3 goes.

        Raises:
            ServiceError: If there is no key, or ElevenLabs refuses the request or cannot be reached. Every
                refusal stops the run: the lines are short and plain, so one refused is almost always the
                key, the credit, the voice or the model, which every later line would meet too.
        """
        url = ELEVENLABS_URL.format(voice=quote(self.voice, safe=""))
        headers = {"xi-api-key": self.key_source(), "Content-Type": "application/json", "Accept": "audio/mpeg"}
        try:
            response = self.post(url, params={"output_format": self.output_format}, json=self._body(text),
                                 headers=headers, timeout=ELEVENLABS_TIMEOUT_S)
        except httpx.LocalProtocolError as exc:  # its message can quote the headers, key included
            raise ServiceError("the request to ElevenLabs could not be sent: check ELEVENLABS_API_KEY") from exc
        except httpx.HTTPError as exc:
            raise ServiceError(f"could not reach ElevenLabs: {exc}") from exc
        if response.status_code != httpx.codes.OK:
            raise ServiceError(f"ElevenLabs answered {response.status_code}: {_detail(response)} "
                               f"(check ELEVENLABS_API_KEY and the account's credits)")
        Path(path).write_bytes(response.content)


@dataclass(frozen=True)
class SaySpeech:
    """macOS's ``say``: offline and free, for trying the pipeline.

    Attributes:
        voice: A voice ``say -v '?'`` lists, for example "Samantha".
        rate: Words per minute.
        suffix: ``say`` writes AIFF.
    """

    voice: str = SAY_VOICE
    rate: int = SAY_RATE
    suffix: str = ".aiff"

    @property
    def identity(self) -> str:
        """The engine, voice and rate.

        Returns:
            For example "say:Samantha:185".
        """
        return f"say:{self.voice}:{self.rate}"

    def synthesize(self, text: str, path: Path) -> None:
        """Speaks ``text`` into ``path`` with ``say``.

        Args:
            text: What to say.
            path: Where the AIFF audio goes.

        Raises:
            ServiceError: If ``say`` is missing (not macOS).
            TtsError: If ``say`` fails on this text.
        """
        say = shutil.which("say")
        if say is None:
            raise ServiceError("the 'say' engine needs macOS's say command")
        script = Path(path).with_suffix(".txt")
        script.write_text(text, encoding="utf-8")
        result = subprocess.run([say, "-v", self.voice, "-r", str(self.rate), "-o", str(path), "-f", str(script)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise TtsError(f"say failed: {result.stderr.strip() or result.returncode}")


def _ffmpeg(tool: str = "ffmpeg") -> str:
    """The path to ffmpeg or ffprobe; raises ``ServiceError`` if it is not installed."""
    found = shutil.which(tool)
    if found is None:
        raise ServiceError(f"{tool} is not installed")
    return found


def trim_silence(source: Path, target: Path) -> None:
    """Writes ``source`` to ``target`` as mono FLAC, with the silence before and after the speech removed.

    Args:
        source: Any audio ffmpeg reads.
        target: The FLAC file to write.

    Raises:
        TtsError: If ffmpeg fails.
    """
    edge = (f"silenceremove=start_periods=1:start_threshold={SILENCE_THRESHOLD}:"
            f"start_silence={KEEP_SILENCE_S}")
    wav_input = ["-ignore_length", "1"] if Path(source).suffix == ".wav" else []  # streamed WAVs have no size
    result = subprocess.run(
        [_ffmpeg(), "-y", "-loglevel", "error", *wav_input, "-i", str(source),
         "-af", f"{edge},areverse,{edge},areverse", "-ac", "1", "-ar", str(SAMPLE_RATE), str(target)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise TtsError(f"could not trim the voice clip: {result.stderr.strip()}")


def audio_seconds(path: Path) -> float:
    """How long an audio file plays.

    Args:
        path: The audio file.

    Returns:
        Its length in seconds.

    Raises:
        TtsError: If the file is missing or ffprobe cannot read it.
    """
    if not Path(path).is_file():
        raise TtsError(f"audio file not found: {path}")
    result = subprocess.run([_ffmpeg("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json",
                             str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise TtsError(f"could not read {path}: {result.stderr.strip()}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, TypeError) as exc:  # json.JSONDecodeError is a ValueError
        raise TtsError(f"{path} has no readable length") from exc


def _trimmed_seconds(path: Path) -> float:
    """A just-trimmed clip's length; 0 when trimming left nothing (ffprobe finds no length in an empty file)."""
    try:
        return audio_seconds(path)
    except TtsError:
        return 0.0


@dataclass(frozen=True)
class SpeechCache:
    """Voice clips on disk, one per (engine, text): a line is synthesised once, then read back.

    Attributes:
        engine: Makes a clip the cache does not have.
        directory: Where the clips are kept.
    """

    engine: SpeechEngine
    directory: Path

    def path_for(self, text: str) -> Path:
        """Where the clip for ``text`` is, or will be, kept.

        Args:
            text: The words.

        Returns:
            A FLAC path named after a digest of the engine's identity, the trimming and the text.
        """
        key = f"{self.engine.identity}\n{TRIM_SETTINGS}\n{text}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:CLIP_DIGEST_CHARS]
        return Path(self.directory) / f"{digest}.flac"

    def clip(self, text: str) -> Path:
        """The trimmed clip of ``text``, made now if the cache does not have it.

        Args:
            text: The words to say.

        Returns:
            The clip's path.

        Raises:
            ValueError: If ``text`` is blank.
            TtsError: If the engine fails or the clip comes back silent; nothing is cached then.
        """
        if not text.strip():
            raise ValueError("there is no text to speak")
        target = self.path_for(text)
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".clip-") as work:
            raw = Path(work) / f"raw{self.engine.suffix}"
            self.engine.synthesize(text, raw)
            trimmed = Path(work) / "clip.flac"
            trim_silence(raw, trimmed)
            if _trimmed_seconds(trimmed) < MIN_CLIP_S:
                raise TtsError(f"the voice for {text[:40]!r} came back silent")
            trimmed.replace(target)
        return target
