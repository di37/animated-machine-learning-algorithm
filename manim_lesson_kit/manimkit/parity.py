"""Checks that a Manim lesson ends every chapter on the matplotlib lesson's picture, frame for frame as long.

The reference is animkit itself: its own chapter players run with its own ``Recorder``, every frame
rasterized by Agg; ``animkit_run`` keeps each chapter's last frame, the frames each moving chapter was
painted on, and the frame count. ``manim_run`` renders the Manim lesson from the same prepared extras and
keeps each chapter's last frame and the player's ``mismatches``.

Pictures are compared blurred and cut into tiles (``worst_tile``): matching frames differ by 1.2 to 2.8 grey
levels in the worst 80 x 60 pixel tile at 1280 x 720 (Agg's hinted glyphs against Manim's unhinted
outlines, and one-pixel outlines such as a random forest's region maps, which the two rasterize a
little differently); the cards' large headlines and the 3-D sheet (Agg leaves seams along its tiles'
slanted edges) get looser limits of their own.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from animkit.lessons import director as kit
from animkit.lessons.director import Extras, Scene
from animkit.lessons.stage import MAIN_RECT, Stage, lesson_style
from animkit.lessons.timing import CHAPTERS, Timing
from manimkit.director import render_lesson

BLUR = 3
TILES = (12, 16)  # rows, columns: 80 x 60 pixel tiles at 1280 x 720
MOST_TILE_DIFFERENCE = 3.0  # matching frames reach 2.8; another chapter's picture or a moved curve goes far past
CARD_TILE_DIFFERENCE = 5.5  # matching cards reach 4.1 on their headlines; hiding any one card line gives 7+
SHEET_TILE_DIFFERENCE = 20.0  # matching 3-D boxes reach 15 (Agg's seams); another chapter's picture is near 90
CARDS = ("title", "features", "takeaway")


class LastFrames:
    """Stands in for animkit's encoder: counts the frames and keeps the last one.

    Attributes:
        frames_written: How many frames have been written, repeats included.
        last: The last frame's RGBA bytes, or None before the first.
    """

    def __init__(self):
        """Creates an encoder that has written nothing."""
        self.frames_written = 0
        self.last: bytes | None = None

    def write(self, frame: bytes, repeat: int = 1) -> None:
        """Takes a frame, as animkit's encoder does.

        Args:
            frame: The frame's RGBA bytes.
            repeat: How many times it is written.
        """
        self.last = frame
        self.frames_written += repeat


class Spy(kit.Recorder):
    """animkit's own Recorder, noting the frame of the chapter each painter runs on.

    Attributes:
        painted: The chapter frames painted so far.
        chapter_start: How many frames had been written when the chapter began.
    """

    def __init__(self, stage: Stage, video: LastFrames):
        """Creates the recorder.

        Args:
            stage: The lesson stage.
            video: The stand-in encoder.
        """
        super().__init__(stage, video)
        self.painted: list[int] = []
        self.chapter_start = 0

    def animate(self, total, paint, moving=1.0):
        """Animates as animkit's Recorder does, noting the frame before each paint.

        Args:
            total: How many frames the chapter has.
            paint: Paints the stage at a time t from 0 to 1.
            moving: The share of the chapter that moves.
        """
        def noted(t):
            """Notes the frame, then paints at ``t``.

            Args:
                t: The time, from 0 to 1.
            """
            self.painted.append(self._video.frames_written - self.chapter_start)
            paint(t)

        super().animate(total, noted, moving)


@dataclass(frozen=True)
class Run:
    """What one lesson run showed.

    Attributes:
        ends: Chapter: its last frame, shape (height, width, 3).
        frames: How many frames the video has.
        painted: For animkit, the frames each moving chapter was painted on.
        unsettled: For Manim, what was on screen but not the chapter's last picture.
    """

    ends: dict[str, np.ndarray]  # chapter: its last frame, (height, width, 3)
    frames: int
    painted: dict[str, list[int]] | None = None  # animkit: the frames each moving chapter was painted on
    unsettled: dict[str, list[str]] | None = None  # Manim: what was on screen but not the last picture


def animkit_run(lesson: Any, scene: Scene, extras: Extras, timing: Timing, dpi: float) -> Run:
    """The matplotlib lesson: animkit's chapters, played as its ``render_lesson`` plays them.

    Args:
        lesson: The lesson.
        scene: The project's ``Scene``.
        extras: What was computed once before the first frame (shared with ``manim_run``).
        timing: The video's timing.
        dpi: The resolution.

    Returns:
        Each chapter's last frame, the frame count, and the frames each chapter was painted on.
    """
    with lesson_style():
        stage = Stage(dpi=dpi, source_note=scene.source_note)
        width, height = stage.size_px
        video = LastFrames()
        out = Spy(stage, video)
        ends, painted = {}, {}
        for name, play in zip(CHAPTERS, kit.CHAPTER_PLAYERS):
            out.painted, out.chapter_start = [], video.frames_written
            play(out, lesson, timing, scene, extras)
            ends[name] = np.frombuffer(video.last, dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()
            shown = video.frames_written - out.chapter_start
            # a closing paint after the last frame (a chapter that moves to its end) is never seen
            painted[name] = [frame for frame in out.painted if frame < shown]
    return Run(ends, video.frames_written, painted=painted)


def manim_run(lesson: Any, scene: Scene, extras: Extras, timing: Timing, dpi: float, path: Path) -> Run:
    """The Manim lesson, rendered to ``path``.

    Args:
        lesson: The lesson.
        scene: The project's ``Scene``.
        extras: What was computed once before the first frame (shared with ``animkit_run``).
        timing: The video's timing.
        dpi: The resolution.
        path: The MP4 to write.

    Returns:
        Each chapter's last frame, the frame count, and each chapter's ``Player.mismatches``.
    """
    ends, unsettled = {}, {}

    def chapter_end(name, player, chapter):
        """Keeps a chapter's last frame and what on screen is not its last picture.

        Args:
            name: The chapter's name.
            player: The player, on the chapter's last frame.
            chapter: What the tracker saw across the chapter.
        """
        ends[name] = player.last_frame().copy()
        unsettled[name] = player.mismatches(chapter)

    report = render_lesson(lesson, path, scene=scene, timing=timing, dpi=dpi, extras=extras, on_chapter=chapter_end)
    return Run(ends, report.frames, unsettled=unsettled)


def worst_tile(drawn: np.ndarray, expected: np.ndarray) -> float:
    """The largest mean grey-level difference of any tile, after a blur.

    Args:
        drawn: The Manim frame, shape (height, width, 3).
        expected: The matplotlib frame, the same shape.

    Returns:
        Grey levels (0 to 255): the mean difference in the worst of the ``TILES``.
    """
    a, b = (np.asarray(Image.fromarray(np.asarray(im, dtype=np.uint8)).convert("L").filter(
        ImageFilter.GaussianBlur(BLUR)), dtype=float) for im in (drawn, expected))
    rows, columns = TILES
    return max(tile.mean() for band in np.array_split(np.abs(a - b), rows, axis=0)
               for tile in np.array_split(band, columns, axis=1))


def box_3d(frame: np.ndarray) -> tuple[slice, slice]:
    """The rows and columns of a frame the 3-D view takes (the main plot's place).

    Args:
        frame: A lesson frame, shape (height, width, 3).

    Returns:
        ``(rows, columns)`` slices.
    """
    height, width = frame.shape[:2]
    left, bottom, box_width, box_height = MAIN_RECT
    return (slice(round((1 - bottom - box_height) * height), round((1 - bottom) * height)),
            slice(round(left * width), round((left + box_width) * width)))


def without_3d(frame: np.ndarray) -> np.ndarray:
    """A frame with the 3-D view blacked out.

    Args:
        frame: A lesson frame, shape (height, width, 3).

    Returns:
        A copy with the 3-D view's box set to black.
    """
    blank = frame.copy()
    blank[box_3d(frame)] = 0
    return blank


def chapter_differences(chapter: str, drawn: np.ndarray, expected: np.ndarray) -> list[tuple[str, float, float]]:
    """Each comparison a chapter's end is held to.

    The 3-D view's chapter is compared twice: the box (with its own, looser limit) and everything else.

    Args:
        chapter: The chapter's name.
        drawn: The Manim frame.
        expected: The matplotlib frame.

    Returns:
        ``(what, worst tile, limit)`` for each comparison.
    """
    if chapter == "surface":
        rows, columns = box_3d(drawn)
        return [("outside the 3-D box", worst_tile(without_3d(drawn), without_3d(expected)), MOST_TILE_DIFFERENCE),
                ("the 3-D box", worst_tile(drawn[rows, columns], expected[rows, columns]), SHEET_TILE_DIFFERENCE)]
    limit = CARD_TILE_DIFFERENCE if chapter in CARDS else MOST_TILE_DIFFERENCE
    return [("the frame", worst_tile(drawn, expected), limit)]
