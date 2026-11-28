"""The chapters every playlist's introduction and conclusion share, whatever the dataset."""

import pytest

from animkit.lessons.speech_text import spoken, unexpanded_acronyms
from animkit.lessons.stage import TRACKER
from animkit.lessons.config import FOOTER
from animkit.series.playlist import (SITE, audience_chapter, first_lesson_chapter, lesson_plan_chapter,
                                     links_chapter, real_chapter, thank_you_chapter, welcome_chapter)

NOTES = {key: f"note for {key}" for key, _ in TRACKER}


def test_the_audience_chapter_says_it_is_not_for_beginners():
    chapter = audience_chapter()
    assert chapter.narration.startswith("This series is not a first course in machine learning.")
    assert "introductory course" in chapter.narration
    lines = chapter.paint.keywords["lines"]
    assert [line.marker for line in lines] == ["✓", "✓", "✗"]
    assert unexpanded_acronyms(spoken(chapter.narration)) == ()


def test_the_lesson_plan_has_the_title_and_every_lesson_chapter_in_order():
    steps = lesson_plan_chapter(NOTES, "Ten chapters.").paint.keywords["steps"]
    assert [step.label for step in steps] == ["Title", *(label for _, label in TRACKER)]
    assert steps[1].note == "note for data" and len(steps) == 10


def test_a_lesson_chapter_without_a_note_is_refused():
    with pytest.raises(KeyError):
        lesson_plan_chapter({"data": "x"}, "Ten chapters.")


def test_the_first_lesson_chapter_hands_over_by_name():
    chapter = first_lesson_chapter("Logistic Regression", "trained by gradient descent", 9)
    assert chapter.narration == "Let's start with Logistic Regression."
    assert chapter.paint.keywords["kicker"] == "LESSON 1 OF 9"


def test_the_links_chapter_points_to_the_site_and_the_other_playlist():
    lines = links_chapter("the other one", "Thanks.").paint.keywords["lines"]
    assert SITE in lines[0].text and lines[2].text.endswith("the other one")


def test_the_welcome_slide_welcomes_viewers_to_the_series_by_name():
    chapter = welcome_chapter("Wine Quality")
    assert chapter.key == "welcome" and chapter.paint.keywords["headline"] == "Welcome"
    assert chapter.narration == "Welcome to Machine Learning Algorithms, Animated: Wine Quality."
    assert chapter.paint.keywords["subhead"] == "Wine Quality" and chapter.paint.keywords["note"] in FOOTER


def test_the_thank_you_slide_thanks_viewers_for_watching_the_series():
    chapter = thank_you_chapter("Wine Quality")
    assert chapter.key == "thank_you" and chapter.paint.keywords["headline"] == "Thank you"
    assert chapter.narration.startswith("Thank you for watching Wine Quality.")
    assert chapter.paint.keywords["subhead"] == "for watching Wine Quality"


def test_the_real_chapter_says_the_training_and_inference_are_real_not_just_animated():
    chapter = real_chapter()
    assert chapter.key == "real" and "not just" in chapter.paint.keywords["headline"]
    assert "not just a beautiful animation" in chapter.narration
    assert "real state of a real model" in chapter.narration and "never seen" in chapter.narration
    lines = " ".join(f"{line.text} {line.detail}" for line in chapter.paint.keywords["lines"])
    assert "Training" in lines and "Inference" in lines and "Live dashboards" in lines


def test_the_real_chapter_does_not_call_the_replayed_lessons_live():
    chapter = real_chapter()
    training = next(line for line in chapter.paint.keywords["lines"] if line.text.startswith("Training"))
    assert "replay" in training.detail and "live" not in training.text.lower()
