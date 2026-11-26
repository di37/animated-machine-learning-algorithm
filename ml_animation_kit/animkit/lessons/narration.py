"""The narration script: one line for the voice per chapter, taken from what that chapter shows.

The voice says each chapter's main text, the very words on screen, so it can never claim anything the
video does not. The fine print stays on screen only: the subtitle's training method, the comparison's
confidence interval and what was held equal, and the takeaway card's extra facts. ``speech_text.spoken``
turns the symbols into words.
"""

from dataclasses import dataclass
from typing import Any

from animkit.lessons.speech_text import spoken
from animkit.lessons.timing import CHAPTERS


@dataclass(frozen=True)
class Cue:
    """What the voice says in one chapter.

    Attributes:
        chapter: The chapter's key, one of ``CHAPTERS``.
        screen: The text as the chapter shows it.
        spoken: The same text in words a voice says correctly.
    """

    chapter: str
    screen: str
    spoken: str


def chapter_texts(lesson: Any, scene: Any, extras: Any) -> dict[str, str]:
    """Each chapter's main on-screen text.

    Args:
        lesson: The lesson; its ``story`` holds the captions.
        scene: The project's scene: it writes the data and exam captions.
        extras: The exam, comparison and 3-D view, as ``director.prepare_extras`` makes them.

    Returns:
        Chapter key: the text it shows, for every chapter.
    """
    story = lesson.story
    return {
        "title": f"{story.title}. {story.objective}",
        "data": scene.data_caption(lesson),
        "initial": story.initial,
        "training": story.training,
        "final": story.final,
        "inference": story.inference,
        "exam": scene.exam_caption(lesson, extras.exam),
        "surface": extras.surface.caption,
        "features": extras.comparison.headline,
        "takeaway": f"{story.takeaway} Limitation: {story.limitation}",
    }


def narration_script(lesson: Any, scene: Any, extras: Any) -> tuple[Cue, ...]:
    """The lesson's narration: one cue per chapter, in the order the chapters play.

    Args:
        lesson: The lesson.
        scene: The project's scene.
        extras: The exam, comparison and 3-D view.

    Returns:
        A cue for each of ``CHAPTERS``. A scene with a ``spoken_phrases`` attribute ((written, spoken) pairs)
        has its own notation said its way first.
    """
    texts = chapter_texts(lesson, scene, extras)
    phrases = getattr(scene, "spoken_phrases", ())
    return tuple(Cue(chapter=chapter, screen=texts[chapter], spoken=spoken(texts[chapter], phrases))
                 for chapter in CHAPTERS)
