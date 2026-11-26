"""Planning where each chapter's voice goes, holding frames where it runs long, and dubbing the video."""

import json
import shutil
import subprocess

import pytest

from animkit.lessons.encoder import probe
from animkit.lessons.voiceover import DubError, Slot, dub, plan_voiceover

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")
CHAPTERS = (("title", 10), ("data", 20), ("takeaway", 10))


def test_a_chapter_long_enough_for_its_voice_is_not_held():
    slots = plan_voiceover(CHAPTERS, {"data": 1.0}, fps=10, lead_s=0.2, tail_s=0.3)
    data = slots[1]
    assert (data.source_start, data.source_frames, data.hold) == (10, 20, 0)
    assert data.voice_at == pytest.approx(1.0 + 0.2)


def test_a_voice_longer_than_its_chapter_holds_the_last_frame_until_it_ends():
    slots = plan_voiceover(CHAPTERS, {"title": 1.5}, fps=10, lead_s=0.2, tail_s=0.3)
    assert slots[0].hold == 10  # 0.2 + 1.5 + 0.3 = 2.0 s = 20 frames, against 10
    assert slots[0].frames == 20


def test_later_chapters_move_by_the_frames_held_before_them():
    slots = plan_voiceover(CHAPTERS, {"title": 1.5, "takeaway": 2.0}, fps=10, lead_s=0.2, tail_s=0.3)
    assert [slot.start for slot in slots] == [0, 20, 40]
    assert slots[2].voice_at == pytest.approx(4.0 + 0.2)
    assert slots[2].hold == 15
    assert sum(slot.frames for slot in slots) == 65


def test_a_chapter_without_a_voice_keeps_its_length():
    slots = plan_voiceover(CHAPTERS, {}, fps=10)
    assert [slot.hold for slot in slots] == [0, 0, 0]
    assert all(slot.voice_seconds == 0.0 for slot in slots)


def test_holds_round_up_to_whole_frames():
    slots = plan_voiceover((("title", 10),), {"title": 1.01}, fps=10, lead_s=0.0, tail_s=0.0)
    assert slots[0].hold == 1  # 10.1 frames needs 11


@pytest.mark.parametrize("bad", [{"nope": 1.0}, {"title": -1.0}, {"title": float("nan")}])
def test_unknown_chapters_and_impossible_lengths_are_refused(bad):
    with pytest.raises(ValueError):
        plan_voiceover(CHAPTERS, bad, fps=10)


def test_a_bad_frame_rate_is_refused():
    with pytest.raises(ValueError):
        plan_voiceover(CHAPTERS, {}, fps=0)


def test_slots_are_frozen():
    slot = plan_voiceover(CHAPTERS, {}, fps=10)[0]
    with pytest.raises(AttributeError):
        slot.hold = 3  # type: ignore[misc]
    assert isinstance(slot, Slot)


# ---- dubbing a real (tiny) video --------------------------------------------------------------------

def make_video(path, frames: int, fps: int = 10) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=64x36:rate={fps}", "-frames:v", str(frames), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def make_tone(path, seconds: float) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:sample_rate=24000:duration={seconds}", str(path)], check=True)


def audio_stream(path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=codec_name,sample_rate,channels:format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True)
    info = json.loads(out.stdout)
    return {**info["streams"][0], "duration": float(info["format"]["duration"])}


@needs_ffmpeg
def test_dub_holds_frames_and_lays_the_voice_on_its_chapter(tmp_path):
    source, out = tmp_path / "silent.mp4", tmp_path / "voiced" / "lesson.mp4"
    make_video(source, 40)
    title, takeaway = tmp_path / "title.flac", tmp_path / "takeaway.flac"
    make_tone(title, 1.5)
    make_tone(takeaway, 2.0)
    slots = plan_voiceover(CHAPTERS, {"title": 1.5, "takeaway": 2.0}, fps=10, lead_s=0.2, tail_s=0.3)
    dub(source, slots, {"title": title, "takeaway": takeaway}, out, fps=10)
    info = probe(out)
    assert info.frames == 65
    audio = audio_stream(out)
    assert audio["codec_name"] == "aac"
    assert audio["duration"] == pytest.approx(6.5, abs=0.1)
    assert not list(out.parent.glob("*.partial*"))


@needs_ffmpeg
def test_dub_refuses_a_source_whose_length_does_not_match_the_plan(tmp_path):
    source = tmp_path / "silent.mp4"
    make_video(source, 30)
    slots = plan_voiceover(CHAPTERS, {}, fps=10)
    with pytest.raises(DubError, match="40"):
        dub(source, slots, {}, tmp_path / "out.mp4", fps=10)
    assert not (tmp_path / "out.mp4").exists()


@needs_ffmpeg
def test_dub_refuses_a_missing_clip(tmp_path):
    source = tmp_path / "silent.mp4"
    make_video(source, 40)
    slots = plan_voiceover(CHAPTERS, {"title": 1.0}, fps=10)
    with pytest.raises(DubError, match="title"):
        dub(source, slots, {}, tmp_path / "out.mp4", fps=10)


def test_dub_refuses_a_missing_source(tmp_path):
    slots = plan_voiceover(CHAPTERS, {}, fps=10)
    with pytest.raises(DubError, match="not found"):
        dub(tmp_path / "nothing.mp4", slots, {}, tmp_path / "out.mp4", fps=10)


@needs_ffmpeg
def test_dub_never_writes_over_its_source(tmp_path):
    source = tmp_path / "lesson.mp4"
    make_video(source, 40)
    before = source.read_bytes()
    with pytest.raises(DubError, match="source"):
        dub(source, plan_voiceover(CHAPTERS, {}, fps=10), {}, tmp_path / "." / "lesson.mp4", fps=10)
    assert source.read_bytes() == before
