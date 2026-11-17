"""The parity tools: tile differences, the 3-D box, and the stand-ins for animkit's encoder and recorder."""

import numpy as np

from animkit.lessons.stage import Stage, lesson_style
from manimkit import parity


def frame(value: int = 255) -> np.ndarray:
    return np.full((720, 1280, 3), value, dtype=np.uint8)


def test_matching_frames_differ_by_nothing_and_one_dark_tile_shows():
    assert parity.worst_tile(frame(), frame()) == 0.0
    marked = frame()
    marked[0:60, 0:80] = 0  # one whole tile black
    assert parity.worst_tile(marked, frame()) > 100


def test_the_3d_box_is_the_main_plots_place_and_can_be_left_out():
    rows, columns = parity.box_3d(frame())
    assert 0 < rows.start < rows.stop < 720 and 0 < columns.start < columns.stop < 1280
    blanked = parity.without_3d(frame())
    assert blanked[rows, columns].max() == 0 and blanked[0, 0].min() == 255


def test_each_chapter_is_held_to_its_own_limits():
    drawn = expected = frame()
    assert [limit for _, _, limit in parity.chapter_differences("surface", drawn, expected)] == [
        parity.MOST_TILE_DIFFERENCE, parity.SHEET_TILE_DIFFERENCE]
    assert parity.chapter_differences("title", drawn, expected)[0][2] == parity.CARD_TILE_DIFFERENCE
    assert parity.chapter_differences("training", drawn, expected)[0][2] == parity.MOST_TILE_DIFFERENCE


def test_the_stand_in_encoder_counts_frames_and_keeps_the_last():
    video = parity.LastFrames()
    video.write(b"a")
    video.write(b"b", repeat=3)
    assert (video.frames_written, video.last) == (4, b"b")


def test_the_spy_notes_the_frames_animkit_paints_on():
    with lesson_style():
        stage = Stage(dpi=10)
        video = parity.LastFrames()
        spy = parity.Spy(stage, video)
        seen = []
        spy.animate(10, seen.append, moving=0.5)
    assert spy.painted == [0, 2, 4, 5] and video.frames_written == 10  # every second frame, then t = 1
    assert seen == [0.0, 0.5, 1.0, 1.0]
