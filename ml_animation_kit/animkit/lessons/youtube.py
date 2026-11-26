"""YouTube chapters and subtitles for a voiced lesson, from its ``<key>_lesson.narration.json``.

The narration file already says when each chapter starts and when its voice speaks, so both are exact:

- ``youtube_chapters``: the chapter list YouTube turns into a seekable progress bar. YouTube shows chapters
  only when the first starts at 0:00, there are at least 3, and each lasts at least 10 seconds, so a
  shorter chapter joins the one before it ("3-D view & More features").
- ``subtitles``: an SRT file of exactly what the voice says, in cues short enough to read, timed by
  sharing each chapter's voice among its cues by their length.
"""

import math
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from animkit.lessons.stage import TRACKER

MIN_CHAPTER_S = 10.0  # YouTube's shortest chapter
MIN_CHAPTERS = 3  # YouTube's fewest chapters
SUBTITLE_CHARS = 84  # two lines of 42 characters, the usual limit for a readable subtitle
LABELS = {"title": "Introduction", **dict(TRACKER)}  # the lesson's own chapter names, as its header shows them
JOIN = " & "
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
MS_PER_S = 1000
SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")

Row = Mapping[str, Any]  # one chapter of a narration file


@dataclass(frozen=True)
class Chapter:
    """One YouTube chapter.

    Attributes:
        start_s: When it starts, in seconds.
        label: Its name.
    """

    start_s: float
    label: str


def timestamp(seconds: float) -> str:
    """A time as YouTube writes it in a description.

    Args:
        seconds: Seconds from the start, 0 or more.

    Returns:
        "1:05" under an hour, "1:02:05" from an hour on; seconds are rounded down.
    """
    whole = math.floor(seconds)
    hours, rest = divmod(whole, SECONDS_PER_HOUR)
    minutes, secs = divmod(rest, SECONDS_PER_MINUTE)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _spans(rows: Sequence[Row], total_s: float, labels: Mapping[str, str]) -> tuple[tuple[float, float, str], ...]:
    """Each chapter as (start, end, label), an end being the next chapter's start or the video's end."""
    starts = [float(row["start_s"]) for row in rows]
    return tuple((start, end, labels.get(row["chapter"], str(row["chapter"]).capitalize()))
                 for row, start, end in zip(rows, starts, [*starts[1:], total_s]))


def _joined(first: tuple[float, float, str], second: tuple[float, float, str]) -> tuple[float, float, str]:
    """Two neighbouring spans as one."""
    return first[0], second[1], f"{first[2]}{JOIN}{second[2]}"


def youtube_chapters(rows: Sequence[Row], total_s: float,
                     labels: Mapping[str, str] = LABELS) -> tuple[Chapter, ...]:
    """The lesson's chapters as YouTube will show them, a short one joined to the one before it.

    Args:
        rows: The narration file's ``chapters``, in order, each with ``chapter`` and ``start_s``.
        total_s: The voiced video's length.
        labels: The name of each chapter key.

    Returns:
        The chapters, the first at 0:00, each lasting at least ``MIN_CHAPTER_S``.

    Raises:
        ValueError: If fewer than ``MIN_CHAPTERS`` chapters are long enough.
    """
    kept: tuple[tuple[float, float, str], ...] = ()
    waiting = None  # a first chapter too short to stand alone, joined to the next
    for span in _spans(rows, total_s, labels):
        span = _joined(waiting, span) if waiting else span
        waiting = None
        if span[1] - span[0] >= MIN_CHAPTER_S:
            kept = (*kept, span)
        elif kept:
            kept = (*kept[:-1], _joined(kept[-1], span))
        else:
            waiting = span
    if len(kept) < MIN_CHAPTERS:
        raise ValueError(f"YouTube needs at least {MIN_CHAPTERS} chapters of {MIN_CHAPTER_S:g} s or more; "
                         f"this video has {len(kept)}")
    return tuple(Chapter(0.0 if index == 0 else start, label) for index, (start, _, label) in enumerate(kept))


def chapter_list(chapters: Sequence[Chapter]) -> str:
    """The chapters as a description lists them: one "0:00 Introduction" line each."""
    return "\n".join(f"{timestamp(chapter.start_s)} {chapter.label}" for chapter in chapters)


def _srt_time(seconds: float) -> str:
    """A time as an SRT file writes it: 00:01:02,350."""
    ms = round(seconds * MS_PER_S)
    hours, rest = divmod(ms, SECONDS_PER_HOUR * MS_PER_S)
    minutes, rest = divmod(rest, SECONDS_PER_MINUTE * MS_PER_S)
    secs, ms = divmod(rest, MS_PER_S)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _pieces(text: str) -> tuple[str, ...]:
    """``text`` cut into subtitle cues: at sentence ends, then between words, none over ``SUBTITLE_CHARS``."""
    return tuple(piece for sentence in SENTENCE_END.split(text.strip())
                 for piece in textwrap.wrap(sentence, SUBTITLE_CHARS, break_long_words=False))


def _timed(row: Row) -> tuple[tuple[float, float, str], ...]:
    """One chapter's cues as (start, end, text), its voice shared among them by their length."""
    pieces = _pieces(str(row["spoken"]))
    start, length = float(row["voice_at_s"]), float(row["voice_s"])
    weights = [len(piece) for piece in pieces]
    total = sum(weights)
    edges = [start + length * sum(weights[:i]) / total for i in range(len(pieces) + 1)]
    return tuple((edges[i], edges[i + 1], piece) for i, piece in enumerate(pieces))


def subtitles(rows: Sequence[Row]) -> str:
    """An SRT file of everything the voice says, each cue on screen while it is spoken.

    Args:
        rows: The narration file's ``chapters``, each with ``spoken``, ``voice_at_s`` and ``voice_s``.

    Returns:
        The SRT text; a chapter with no voice has no cue.
    """
    cues = [cue for row in rows if float(row["voice_s"]) > 0 and str(row["spoken"]).strip() for cue in _timed(row)]
    return "\n".join(f"{n}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n"
                     for n, (start, end, text) in enumerate(cues, start=1))
