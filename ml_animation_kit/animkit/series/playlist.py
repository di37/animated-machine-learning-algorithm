"""The chapters every playlist's introduction and conclusion share, whatever the dataset.

A dataset's project supplies the words that name its data (what one row is); these chapters need
none of them:

- ``welcome_chapter``: the introduction's opening slide.
- ``audience_chapter``: who the series is for. It is not a first course in machine learning, and says so.
- ``real_chapter``: what the viewer is watching: real training and real inference, not just animation.
- ``lesson_plan_chapter``: the ten chapters every lesson runs, as a road map.
- ``first_lesson_chapter``: the hand-over to lesson 1.
- ``links_chapter``: where to go next: the site, the code and the other playlist.
- ``thank_you_chapter``: the conclusion's closing slide.
"""

from collections.abc import Mapping

from animkit.lessons import theme
from animkit.lessons.config import FOOTER
from animkit.lessons.stage import TRACKER
from animkit.series.cards import Line, Step, paint_banner, paint_roadmap, paint_text
from animkit.series.chapters import SeriesChapter, card

SITE = "di37.github.io/animated-machine-learning-algorithm"
REPO = "github.com/di37/animated-machine-learning-algorithm"
BRAND = "Machine Learning Algorithms, Animated"
AUTHOR = FOOTER.split(" · ", 1)[1]  # the footer without its initials: the name, role and motto
TITLE_NOTE = "the algorithm and its goal"  # the road map's first pill, the lesson's title card

AUDIENCE_HEADLINE = "Not a first course in machine learning"
AUDIENCE_LINES = (
    Line("You already know training, validation and test data, what an error measures, and what fitting a "
         "model means.", marker="✓"),
    Line("You want to see what the algorithms you already use actually do while they learn: real data, real "
         "models, every number computed.", marker="✓"),
    Line("New to machine learning? Start with an introductory course first, then come back.", marker="✗",
         colour=theme.ERROR),
)
AUDIENCE_NARRATION = (
    "This series is not a first course in machine learning. It is for you if you already know what training, "
    "validation and test data are, what an error measures, and what fitting a model means, and you want to see "
    "what the algorithms you already use actually do while they learn. If you are new to machine learning, "
    "start with an introductory course first, then come back.")


def audience_chapter(seconds: float = 14.0) -> SeriesChapter:
    """Who the series is for: people who know the basics, not beginners.

    Args:
        seconds: How long it lasts.

    Returns:
        The chapter.
    """
    return card("audience", "Who it's for", seconds, AUDIENCE_NARRATION, paint_text, kicker="WHO THIS SERIES IS FOR",
                headline=AUDIENCE_HEADLINE, lines=AUDIENCE_LINES)


def lesson_plan_chapter(notes: Mapping[str, str], narration: str, seconds: float = 14.0) -> SeriesChapter:
    """The ten chapters every lesson runs, as pills lighting up in order.

    Args:
        notes: One short note per lesson chapter key (data, initial, …, takeaway), in the dataset's words.
        narration: What the voice says over it.
        seconds: How long it lasts.

    Returns:
        The chapter.

    Raises:
        KeyError: If a chapter of the lessons has no note.
    """
    steps = (Step("Title", TITLE_NOTE), *(Step(label, notes[key]) for key, label in TRACKER))
    return card("lesson_plan", "How a lesson runs", seconds, narration, paint_roadmap, moving=0.8,
                kicker="HOW EVERY LESSON RUNS", headline="Ten chapters, always in the same order", steps=steps,
                note="Training shows only real snapshots of the model: nothing between two of them is invented.")


def first_lesson_chapter(title: str, how: str, count: int, seconds: float = 6.0) -> SeriesChapter:
    """The hand-over to the playlist's first lesson.

    Args:
        title: The first lesson's title.
        how: How its algorithm learns, in a few words.
        count: How many lessons the playlist has.
        seconds: How long it lasts.

    Returns:
        The chapter.
    """
    return card("first_lesson", "First lesson", seconds, f"Let's start with {title}.", paint_text,
                kicker=f"LESSON 1 OF {count}", headline=f"First up: {title}", subhead=how)


def links_chapter(other_playlist: str, narration: str, seconds: float = 9.0) -> SeriesChapter:
    """Where to go next: the site with every video, the code, and the other playlist.

    Args:
        other_playlist: The other playlist, named with its task, for example "wine quality (classification)".
        narration: What the voice says over it.
        seconds: How long it lasts.

    Returns:
        The chapter.
    """
    return card("keep_going", "Keep going", seconds, narration, paint_text, kicker="KEEP GOING",
                headline="All 76 videos, and the code behind them", lines=(
                    Line(f"Live training dashboards and algorithm races: {SITE}", marker="▶"),
                    Line(f"Open-source code: {REPO}", marker="▶"),
                    Line(f"The other playlist: {other_playlist}", marker="▶")))


def welcome_chapter(series_title: str, seconds: float = 6.0) -> SeriesChapter:
    """The introduction's opening slide: "Welcome", the series' name and who made it.

    Args:
        series_title: The playlist's name.
        seconds: How long it lasts.

    Returns:
        The chapter.
    """
    return card("welcome", "Welcome", seconds, f"Welcome to {BRAND}: {series_title}.", paint_banner, moving=0.5,
                kicker=BRAND.upper(), headline="Welcome", subhead=series_title, note=AUTHOR)


def thank_you_chapter(series_title: str, seconds: float = 8.0) -> SeriesChapter:
    """The conclusion's closing slide: "Thank you", and an ask to share.

    Args:
        series_title: The playlist's name.
        seconds: How long it lasts.

    Returns:
        The chapter.
    """
    return card("thank_you", "Thank you", seconds, f"Thank you for watching {series_title}. If an algorithm "
                "clicked for you here, share it with someone who would enjoy it too.", paint_banner, moving=0.5,
                kicker=BRAND.upper(), headline="Thank you", subhead=f"for watching {series_title}", note=AUTHOR)


REAL_HEADLINE = "Real training and real inference, not just animation"
REAL_LINES = (
    Line("Training: every frame is a real state of a real model, recorded while it trained.", marker="✓",
         detail="The lessons replay those states at a teaching pace. Nothing between two of them is invented."),
    Line("Inference: every prediction is computed by the trained model for an example it has never seen.",
         marker="✓", detail="The numbers on screen are the model's own outputs, never typed in by hand."),
    Line("Live dashboards: on the project site, each frame is drawn straight after a real piece of training.",
         marker="✓", detail=f"{SITE}"),
)
REAL_NARRATION = (
    "What you are watching is not just a beautiful animation. Every frame of training is a real state of a real "
    "model, recorded while it trained, and every prediction is computed by that trained model for an example it "
    "has never seen. Nothing is drawn to look right. On the project site, the live dashboards go further: they "
    "draw each frame as the training happens.")


def real_chapter(seconds: float = 14.0) -> SeriesChapter:
    """What the viewer is watching: real training and real inference, not an animation drawn to look right.

    The lessons replay recorded training states at a teaching pace, so the chapter does not call them live;
    it keeps that word for the dashboards, which draw each frame straight after a real piece of training.

    Args:
        seconds: How long it lasts.

    Returns:
        The chapter.
    """
    return card("real", "Real, not staged", seconds, REAL_NARRATION, paint_text, moving=0.75,
                kicker="WHAT YOU ARE WATCHING", headline=REAL_HEADLINE, lines=REAL_LINES)
