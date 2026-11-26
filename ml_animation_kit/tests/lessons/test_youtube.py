"""YouTube chapters and subtitles for a voiced lesson, from its narration file."""

import re

import pytest

from animkit.lessons.youtube import (MIN_CHAPTER_S, SUBTITLE_CHARS, Chapter, chapter_list, subtitles, timestamp,
                                     youtube_chapters)

# A voiced lesson's chapters (start, voice start, voice length), like a real narration file.
ROWS = (
    {"chapter": "title", "start_s": 0.0, "voice_at_s": 0.35, "voice_s": 15.7,
     "spoken": "Neural Network (multi-layer perceptron). A small neural network learns a curve."},
    {"chapter": "data", "start_s": 16.7, "voice_at_s": 17.05, "voice_s": 9.7, "spoken": "Each dot is one district."},
    {"chapter": "initial", "start_s": 27.4, "voice_at_s": 27.75, "voice_s": 7.4, "spoken": "A first guess."},
    {"chapter": "training", "start_s": 35.8, "voice_at_s": 36.15, "voice_s": 8.8, "spoken": "It learns."},
    {"chapter": "final", "start_s": 61.8, "voice_at_s": 62.15, "voice_s": 10.5, "spoken": "Trained."},
    {"chapter": "inference", "start_s": 73.2, "voice_at_s": 73.55, "voice_s": 11.2, "spoken": "A new district."},
    {"chapter": "exam", "start_s": 86.2, "voice_at_s": 86.55, "voice_s": 11.2, "spoken": "The final exam."},
    {"chapter": "surface", "start_s": 101.2, "voice_at_s": 101.55, "voice_s": 12.7, "spoken": "A folded sheet."},
    {"chapter": "features", "start_s": 114.9, "voice_at_s": 115.25, "voice_s": 5.4, "spoken": "More inputs help."},
    {"chapter": "takeaway", "start_s": 122.9, "voice_at_s": 123.25, "voice_s": 18.1, "spoken": "It works."},
)
TOTAL_S = 142.0


@pytest.mark.parametrize(("seconds", "shown"), [(0, "0:00"), (9.99, "0:09"), (65.9, "1:05"), (3725, "1:02:05")])
def test_timestamps_are_youtube_style(seconds, shown):
    assert timestamp(seconds) == shown


def test_chapters_start_at_zero_and_each_lasts_long_enough_for_youtube():
    chapters = youtube_chapters(ROWS, TOTAL_S)
    assert chapters[0] == Chapter(0.0, "Introduction")
    ends = [*(chapter.start_s for chapter in chapters[1:]), TOTAL_S]
    assert all(end - chapter.start_s >= MIN_CHAPTER_S for chapter, end in zip(chapters, ends))


def test_a_short_chapter_joins_the_one_before_it():
    labels = [chapter.label for chapter in youtube_chapters(ROWS, TOTAL_S)]
    assert labels == ["Introduction", "Data & First guess", "Training", "Final model", "Prediction", "Final exam",
                      "3-D view & More features", "Takeaway"]


def test_a_short_first_chapter_joins_the_one_after_it():
    rows = ({**ROWS[0], "start_s": 0.0}, {**ROWS[1], "start_s": 4.0}, *ROWS[2:])
    chapters = youtube_chapters(rows, TOTAL_S)
    assert chapters[0] == Chapter(0.0, "Introduction & Data & First guess")  # the 8 s first guess joins it too
    assert chapters[1] == Chapter(35.8, "Training")


def test_too_few_chapters_for_youtube_is_an_error():
    with pytest.raises(ValueError, match="3 chapters"):
        youtube_chapters(ROWS[:2], 30.0)


def test_the_chapter_list_is_one_timestamped_line_each():
    assert chapter_list(youtube_chapters(ROWS, TOTAL_S)).splitlines()[:3] == [
        "0:00 Introduction", "0:16 Data & First guess", "0:35 Training"]


def srt_cues(text):
    blocks = text.strip().split("\n\n")
    return [(block.splitlines()[1], " ".join(block.splitlines()[2:])) for block in blocks]


def seconds(stamp):
    h, m, s = stamp.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_subtitles_say_every_word_in_order_while_it_is_spoken():
    cues = srt_cues(subtitles(ROWS))
    assert " ".join(text for _, text in cues) == " ".join(row["spoken"] for row in ROWS)
    first_start, _ = cues[0][0].split(" --> ")
    assert first_start == "00:00:00,350"
    _, last_end = cues[-1][0].split(" --> ")
    assert seconds(last_end) == pytest.approx(123.25 + 18.1, abs=0.002)


def test_subtitles_are_numbered_and_short_enough_to_read():
    long_line = {**ROWS[0], "spoken": " ".join(["word"] * 80) + "."}
    text = subtitles((long_line, *ROWS[1:]))
    numbers = re.findall(r"(?m)^(\d+)$", text)
    assert numbers == [str(n) for n in range(1, len(numbers) + 1)]
    assert all(len(cue) <= SUBTITLE_CHARS for _, cue in srt_cues(text))


def test_subtitle_cues_never_overlap():
    times = [tuple(map(seconds, timing.split(" --> "))) for timing, _ in srt_cues(subtitles(ROWS))]
    assert all(start < end for start, end in times)
    assert all(end <= next_start + 1e-9 for (_, end), (next_start, _) in zip(times, times[1:]))


def test_a_silent_chapter_has_no_subtitle():
    rows = ({**ROWS[0], "voice_s": 0.0}, *ROWS[1:])
    assert "Neural Network" not in subtitles(rows)
