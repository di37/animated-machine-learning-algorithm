"""Laying a voice over a rendered lesson: where each chapter's clip goes, and which frames wait for it.

A chapter's clip starts a moment (``LEAD_S``) after the chapter does. When the clip, with a short pause
after it (``TAIL_S``), is longer than the chapter, the chapter's last frame is held until the voice has
finished; every later chapter moves along by the frames held. No frame is drawn, and none is changed:
the held frame is the model state the chapter ended on.

``plan_voiceover`` works this out; ``dub`` makes the voiced video from the silent one in one ffmpeg run.
"""

import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from animkit.lessons.config import CRF
from animkit.lessons.encoder import EncoderError, probe

LEAD_S = 0.35  # from the start of a chapter to its voice
TAIL_S = 0.6  # from the end of a voice to the next chapter, at least
SAMPLE_RATE = 48000
AUDIO_BITRATE = "128k"
LOUDNESS = "loudnorm=I=-16:TP=-1.5:LRA=11"  # a steady speech level, the same in every lesson
X264_PRESET = "medium"  # the lesson encoder's
MS_PER_S = 1000
STDERR_TAIL = 600  # characters of ffmpeg's error output kept in a DubError


class DubError(RuntimeError):
    """A voiced video could not be made: a file is missing, the lengths disagree, or ffmpeg failed."""


@dataclass(frozen=True)
class Slot:
    """One chapter of the voiced video.

    Attributes:
        chapter: The chapter's key.
        source_start: Its first frame in the silent video.
        source_frames: How many frames it has there.
        hold: How many times its last frame is repeated, so the voice can finish.
        start: Its first frame in the voiced video.
        voice_at: When its voice starts, in seconds into the voiced video.
        voice_seconds: How long its voice lasts; 0 for a chapter with no voice.
    """

    chapter: str
    source_start: int
    source_frames: int
    hold: int
    start: int
    voice_at: float
    voice_seconds: float

    @property
    def frames(self) -> int:
        """Its length in the voiced video: its own frames and the held ones.

        Returns:
            The frame count.
        """
        return self.source_frames + self.hold


def _frames_needed(seconds: float, fps: int, lead_s: float, tail_s: float) -> int:
    """Frames a chapter needs for a voice ``seconds`` long; 0 if it has no voice."""
    if seconds == 0:
        return 0
    return math.ceil(round((lead_s + seconds + tail_s) * fps, 6))  # round first: 20.000000000000004 is 20


def plan_voiceover(chapters: Sequence[tuple[str, int]], voice_seconds: Mapping[str, float], fps: int,
                   lead_s: float = LEAD_S, tail_s: float = TAIL_S) -> tuple[Slot, ...]:
    """Where each chapter's voice goes, and how long each chapter's last frame is held for it.

    Args:
        chapters: Each chapter's key and frame count in the silent video, in order.
        voice_seconds: Chapter key: its voice clip's length; a chapter not listed has no voice.
        fps: The video's frame rate.
        lead_s: From a chapter's start to its voice.
        tail_s: From the end of a voice to the next chapter, at least.

    Returns:
        One slot per chapter, in order.

    Raises:
        ValueError: If ``fps`` is below 1, a voice is for an unknown chapter, or a length is negative or
            not finite.
    """
    if fps < 1:
        raise ValueError(f"fps must be at least 1, got {fps}")
    keys = [key for key, _ in chapters]
    unknown = set(voice_seconds) - set(keys)
    if unknown:
        raise ValueError(f"voice for unknown chapters {sorted(unknown)}")
    if any(not math.isfinite(s) or s < 0 for s in voice_seconds.values()):
        raise ValueError("voice lengths must be finite and not negative")
    seconds = tuple(float(voice_seconds.get(key, 0.0)) for key in keys)
    holds = tuple(max(0, _frames_needed(s, fps, lead_s, tail_s) - n) for s, (_, n) in zip(seconds, chapters))
    source_starts = tuple(accumulate((n for _, n in chapters[:-1]), initial=0))
    starts = tuple(accumulate((n + h for (_, n), h in zip(chapters[:-1], holds)), initial=0))
    return tuple(Slot(chapter=key, source_start=source_start, source_frames=n, hold=hold, start=start,
                      voice_at=start / fps + lead_s, voice_seconds=s)
                 for (key, n), s, hold, source_start, start in zip(chapters, seconds, holds, source_starts, starts))


def _video_part(index: int, slot: Slot) -> str:
    """The chapter's frames, cut from the silent video, with its last frame held ``slot.hold`` times."""
    end = slot.source_start + slot.source_frames
    return (f"[0:v]trim=start_frame={slot.source_start}:end_frame={end},setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop={slot.hold}[v{index}]")


def _audio_part(index: int, slot: Slot, inputs: Mapping[str, int], fps: int) -> str:
    """The chapter's sound, exactly as long as its frames: its voice after the lead, or silence."""
    samples = slot.frames * SAMPLE_RATE // fps
    if slot.chapter not in inputs:
        return f"anullsrc=r={SAMPLE_RATE}:cl=mono,atrim=end_sample={samples}[a{index}]"
    delay_ms = round((slot.voice_at - slot.start / fps) * MS_PER_S)
    return (f"[{inputs[slot.chapter]}:a]aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=mono,"
            f"adelay={delay_ms}:all=1,apad=whole_len={samples},atrim=end_sample={samples}[a{index}]")


def _filter_graph(slots: Sequence[Slot], inputs: Mapping[str, int], fps: int) -> str:
    """The ffmpeg filter graph: cut and hold each chapter's frames, place and pad each chapter's voice."""
    count = len(slots)
    total = sum(slot.frames for slot in slots) * SAMPLE_RATE // fps  # loudnorm may nudge the length; pin it
    joined_video = "".join(f"[v{i}]" for i in range(count)) + f"concat=n={count}:v=1:a=0[vout]"
    joined_audio = ("".join(f"[a{i}]" for i in range(count)) +
                    f"concat=n={count}:v=0:a=1,{LOUDNESS},aresample={SAMPLE_RATE},"
                    f"apad=whole_len={total},atrim=end_sample={total}[aout]")
    return ";".join((*(_video_part(i, slot) for i, slot in enumerate(slots)),
                     *(_audio_part(i, slot, inputs, fps) for i, slot in enumerate(slots)),
                     joined_video, joined_audio))


def _check_inputs(source: Path, slots: Sequence[Slot], clips: Mapping[str, Path]) -> None:
    """Refuses a missing source or clip, and a source whose length is not the plan's."""
    if not Path(source).is_file():
        raise DubError(f"source video not found: {source}")
    missing = [slot.chapter for slot in slots
               if slot.voice_seconds > 0 and not Path(clips.get(slot.chapter, "")).is_file()]
    if missing:
        raise DubError(f"no voice clip for chapter(s) {', '.join(missing)}")
    expected = sum(slot.source_frames for slot in slots)
    try:
        frames = probe(source).frames
    except EncoderError as exc:
        raise DubError(str(exc)) from exc
    if frames != expected:
        raise DubError(f"{source} has {frames} frames, but the lesson's chapters add up to {expected}; "
                       "voice a video rendered at duration scale 1")


def dub(source: Path, slots: Sequence[Slot], clips: Mapping[str, Path], out: Path, fps: int,
        crf: int = CRF) -> Path:
    """Makes the voiced video: the silent one with held frames, and each chapter's voice on its chapter.

    The video is written to a ``.partial.mp4`` first and renamed when ffmpeg has finished and the frame
    count is checked, so ``out`` is never a broken video.

    Args:
        source: The silent lesson video.
        slots: The plan, from ``plan_voiceover``.
        clips: Chapter key: its voice clip, for every slot with a voice.
        out: The voiced MP4 to write.
        fps: The video's frame rate.
        crf: H.264's quality setting, the lesson encoder's by default.

    Returns:
        ``out``.

    Raises:
        DubError: If a file is missing, the source's length is not the plan's, or ffmpeg fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DubError("ffmpeg is not installed")
    if Path(out).resolve() == Path(source).resolve():
        raise DubError(f"the voiced video would overwrite its silent source {source}; choose another folder")
    _check_inputs(source, slots, clips)
    voiced = [slot.chapter for slot in slots if slot.voice_seconds > 0]
    inputs = {chapter: index for index, chapter in enumerate(voiced, start=1)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f"{out.stem}.partial.mp4")
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
               *(arg for chapter in voiced for arg in ("-i", str(clips[chapter]))),
               "-filter_complex", _filter_graph(slots, inputs, fps), "-map", "[vout]", "-map", "[aout]",
               "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps),
               "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", str(SAMPLE_RATE), "-movflags", "+faststart", str(partial)]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise DubError(f"ffmpeg failed: {result.stderr.strip()[-STDERR_TAIL:]}")
        expected = sum(slot.frames for slot in slots)
        try:
            written = probe(partial).frames
        except EncoderError as exc:
            raise DubError(f"could not read the voiced video: {exc}") from exc
        if written != expected:
            raise DubError(f"the voiced video has {written} frames, expected {expected}")
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)
    return out
