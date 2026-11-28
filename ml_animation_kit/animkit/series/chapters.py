"""A series video as a list of card chapters: each one's length, picture and words.

A ``SeriesChapter`` is the series' counterpart of a lesson chapter: it lasts ``seconds``, its painter
draws the stage at a progress t from 0 to 1 (moving for the first ``moving`` share, then holding), and
its ``narration`` is what the voice will say over it, written as a caption would be (the voice-over turns
symbols and abbreviations into words). A ``Series`` is the whole video.

The chapters file (``<name>.chapters.json``, written next to the rendered video) records each chapter's
frame count and narration, so the voice-over can be made later without building the series again.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from animkit.lessons.stage import Stage
from animkit.lessons.timing import MIN_CHAPTER_FRAMES, Timing

SLUG = re.compile(r"^[a-z][a-z0-9_]*$")  # chapter keys and file names: lower case, digits, underscores
CHAPTER_FIELDS = ("chapter", "label", "frames", "narration")

Paint = Callable[[Stage, float], None]


@dataclass(frozen=True)
class SeriesChapter:
    """One card of a series video.

    Attributes:
        key: Its name in files, for example "audience".
        label: Its name for people, for example "Who it's for" (YouTube chapters use it).
        seconds: How long it lasts at a duration scale of 1.
        narration: What the voice says over it, as a caption would write it.
        paint: Draws the stage at progress t, 0 to 1.
        moving: The share of the chapter that moves; the rest holds the last picture.
    """

    key: str
    label: str
    seconds: float
    narration: str
    paint: Paint
    moving: float = 0.6

    def __post_init__(self) -> None:
        """Refuses a chapter that cannot be played or said.

        Raises:
            ValueError: If the key is not a slug, the label or narration is blank, the length is not
                positive, or ``moving`` is not in (0, 1].
        """
        if not SLUG.match(self.key):
            raise ValueError(f"a chapter key is lower case with underscores, got {self.key!r}")
        if not self.label.strip() or not self.narration.strip():
            raise ValueError(f"chapter {self.key!r} needs a label and narration")
        if self.seconds <= 0:
            raise ValueError(f"chapter {self.key!r} must last more than 0 s, got {self.seconds}")
        if not 0 < self.moving <= 1:
            raise ValueError(f"chapter {self.key!r}: moving must be in (0, 1], got {self.moving}")

    def frames(self, timing: Timing) -> int:
        """How many frames it lasts: its seconds × scale × fps, rounded, and at least ``MIN_CHAPTER_FRAMES``.

        Args:
            timing: The video's frame rate and duration scale.

        Returns:
            The chapter's length in frames.
        """
        return max(MIN_CHAPTER_FRAMES, round(self.seconds * timing.scale * timing.fps))


@dataclass(frozen=True)
class Series:
    """A whole series video.

    Attributes:
        name: Its file name without the suffix, for example "introduction".
        title: Its title, for example "Wine Quality: Introduction".
        source_note: The data credit in the footer's bottom-right corner.
        chapters: Its chapters, in order.
    """

    name: str
    title: str
    source_note: str
    chapters: tuple[SeriesChapter, ...]

    def __post_init__(self) -> None:
        """Refuses a series without a usable name, without chapters, or with two chapters of one key.

        Raises:
            ValueError: If any of those is the case.
        """
        if not SLUG.match(self.name):
            raise ValueError(f"a series name is lower case with underscores, got {self.name!r}")
        if not self.chapters:
            raise ValueError("a series needs at least one chapter")
        keys = [chapter.key for chapter in self.chapters]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise ValueError(f"chapter keys must be unique; repeated: {', '.join(repeated)}")

    def frames(self, timing: Timing) -> tuple[tuple[str, int], ...]:
        """Each chapter's key and frame count, in order (what the voice-over plans with)."""
        return tuple((chapter.key, chapter.frames(timing)) for chapter in self.chapters)

    def total_frames(self, timing: Timing) -> int:
        """The whole video's length in frames."""
        return sum(frames for _, frames in self.frames(timing))


def card(key: str, label: str, seconds: float, narration: str, painter: Callable[..., None], *,
         moving: float = 0.6, **content: Any) -> SeriesChapter:
    """A chapter that paints one card: ``painter(stage, t, **content)``.

    Args:
        key: The chapter's name in files.
        label: Its name for people.
        seconds: How long it lasts.
        narration: What the voice says over it.
        painter: A card painter from ``cards`` or ``charts``.
        moving: The share of the chapter that moves.
        **content: The painter's keyword arguments (kicker, headline, lines …).

    Returns:
        The chapter.
    """
    return SeriesChapter(key, label, seconds, narration, partial(painter, **content), moving)


def chapters_record(series: Series, timing: Timing) -> dict[str, Any]:
    """What the chapters file holds: the series, its frame rate, and each chapter's frames and words.

    Args:
        series: The series.
        timing: The timing it was rendered with.

    Returns:
        A JSON-ready record.
    """
    return {"series": series.name, "title": series.title, "fps": timing.fps,
            "chapters": [{"chapter": chapter.key, "label": chapter.label, "frames": chapter.frames(timing),
                          "narration": chapter.narration} for chapter in series.chapters]}


def write_chapters(series: Series, timing: Timing, path: Path) -> None:
    """Writes the chapters file, as readable UTF-8 JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(chapters_record(series, timing), indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def read_chapters(path: Path) -> dict[str, Any]:
    """Reads a chapters file and checks every chapter has what the voice-over needs.

    Args:
        path: The ``<name>.chapters.json`` file.

    Returns:
        The record, as ``chapters_record`` made it.

    Raises:
        ValueError: If the file has no frame rate or chapters, or a chapter misses a field.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record.get("fps"), int) or not record.get("chapters"):
        raise ValueError(f"{path} has no frame rate or no chapters")
    for row in record["chapters"]:
        missing = [field for field in CHAPTER_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}: chapter {row.get('chapter', '?')!r} has no {', '.join(missing)}")
    return record
