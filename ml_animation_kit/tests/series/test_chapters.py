"""A series video is a list of card chapters, each with its length, its picture and the words it says."""

import json

import pytest

from animkit.lessons.timing import MIN_CHAPTER_FRAMES, Timing
from animkit.series.chapters import Series, SeriesChapter, chapters_record, read_chapters, write_chapters


def blank(stage, t):
    """A painter that draws nothing."""


def chapter(key="title", seconds=5.0, **changes):
    fields = {"key": key, "label": key.capitalize(), "seconds": seconds, "narration": f"The {key} chapter.",
              "paint": blank, **changes}
    return SeriesChapter(**fields)


def series(*chapters):
    return Series(name="introduction", title="A Series: Introduction", source_note="Data: somewhere",
                  chapters=chapters or (chapter("title"), chapter("audience", 10.0)))


def test_a_chapter_lasts_its_seconds_at_the_videos_frame_rate():
    assert chapter(seconds=5.0).frames(Timing(fps=30)) == 150
    assert chapter(seconds=5.0).frames(Timing(fps=30, scale=0.1)) == 15


def test_a_very_short_preview_still_has_a_few_frames():
    assert chapter(seconds=0.01).frames(Timing(fps=30)) == MIN_CHAPTER_FRAMES


@pytest.mark.parametrize("bad", [{"key": "Title"}, {"key": "two words"}, {"label": " "}, {"seconds": 0},
                                 {"narration": "  "}, {"moving": 0}, {"moving": 1.5}])
def test_a_chapter_that_cannot_be_played_or_said_is_refused(bad):
    with pytest.raises(ValueError):
        chapter(**bad)


def test_the_series_counts_every_chapters_frames():
    video = series()
    assert video.frames(Timing(fps=10)) == (("title", 50), ("audience", 100))
    assert video.total_frames(Timing(fps=10)) == 150


@pytest.mark.parametrize("bad", [{"name": "My Intro"}, {"chapters": ()}])
def test_a_series_needs_a_file_name_and_chapters(bad):
    fields = {"name": "introduction", "title": "T", "source_note": "", "chapters": (chapter(),), **bad}
    with pytest.raises(ValueError):
        Series(**fields)


def test_two_chapters_with_one_key_are_refused():
    with pytest.raises(ValueError, match="title"):
        series(chapter("title"), chapter("title"))


def test_the_chapters_file_records_what_the_voice_needs(tmp_path):
    path = tmp_path / "introduction.chapters.json"
    write_chapters(series(), Timing(fps=10), path)
    record = read_chapters(path)
    assert record == chapters_record(series(), Timing(fps=10))
    assert record["fps"] == 10 and record["title"] == "A Series: Introduction"
    assert record["chapters"][1] == {"chapter": "audience", "label": "Audience", "frames": 100,
                                     "narration": "The audience chapter."}


def test_a_chapters_file_missing_a_field_is_refused(tmp_path):
    path = tmp_path / "broken.chapters.json"
    path.write_text(json.dumps({"series": "x", "fps": 30, "chapters": [{"chapter": "title"}]}))
    with pytest.raises(ValueError, match="frames"):
        read_chapters(path)
