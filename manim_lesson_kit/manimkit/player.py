"""Plays a chapter's tracks on a Manim scene and writes every frame to the encoder.

The scene holds one Manim shape per life on screen, in the drawing order the tracker saw on the
matplotlib frame, with anything fading out on top. Each track is a Manim animation run frame by
frame (``begin``, ``interpolate``, ``finish``, as ``Scene.play`` does inside). A frame where nothing
moves is written again rather than drawn again.

Manim's own media folders go to a temporary folder, so nothing but the encoded video is written.
Manim seeds Python's and numpy's random generators when it makes a scene; the rest of the program
keeps its own.
"""

import random
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
from manim import Create, FadeIn, FadeOut, Scene, Transform, VMobject, Write, tempconfig

from manimkit.camera import LessonCamera
from manimkit.frame import Frame
from manimkit.motion import APPEAR, CREATE, CUT, FADE_OUT, MORPH, REMOVE, WRITE, Track
from manimkit.timeline import Chapter

REVEALS = {CREATE: Create, WRITE: Write}

FrameWriter = Callable[[bytes, int], None]  # takes a frame's RGBA bytes and how many times to write it


def _same_drawing(shape, expected, tolerance: float = 1e-6) -> bool:
    """Whether two shapes draw the same: the same pieces, points, fills and strokes."""
    ours, theirs = shape.get_family(), expected.get_family()
    if len(ours) != len(theirs):
        return False
    for a, b in zip(ours, theirs):
        if isinstance(a, VMobject) != isinstance(b, VMobject):
            return False
        if not isinstance(a, VMobject):
            if not np.allclose(a.get_center(), b.get_center(), atol=tolerance):
                return False
            continue
        if a.points.shape != b.points.shape or not np.allclose(a.points, b.points, atol=tolerance):
            return False
        if not (np.allclose(a.get_fill_rgbas(), b.get_fill_rgbas(), atol=1e-4) and
                np.allclose(a.get_stroke_rgbas(), b.get_stroke_rgbas(), atol=1e-4) and
                abs(a.get_stroke_width() - b.get_stroke_width()) < 1e-6):
            return False
    return True


@contextmanager
def manim_stage(frame: Frame, background: str, fps: int) -> Iterator["Player"]:
    """A Manim scene drawing ``frame`` at its pixel size, with Manim's settings and seeding undone after.

    Args:
        frame: The frame to draw.
        background: The background colour, as a hex string.
        fps: Frames per second.

    Yields:
        A ``Player`` on a fresh scene.

    Example:
        >>> with manim_stage(frame, "#0f1117", 30) as player:  # doctest: +SKIP
        ...     player.play(chapter, Planner(chapter).tracks(), encoder.write)
    """
    states = random.getstate(), np.random.get_state()
    try:
        with tempfile.TemporaryDirectory() as media, tempconfig(frame.manim_config(media, background, fps)):
            yield Player(frame, Scene(camera_class=LessonCamera))
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])


class Player:
    """Plays chapters on one Manim scene, keeping a shape for every life on screen between them.

    Attributes:
        frame: The frame being drawn.
        scene: The Manim scene drawing it.
    """

    def __init__(self, frame: Frame, scene: Scene):
        """Creates a player with nothing on screen. ``manim_stage`` makes one.

        Args:
            frame: The frame being drawn.
            scene: The Manim scene drawing it (with a ``LessonCamera``).
        """
        self.frame = frame
        self.scene = scene
        self._shapes: dict[int, object] = {}  # life id: its shape on screen
        self._fading: dict[int, object] = {}  # life id: a shape on its way out, drawn on top
        self._running: dict[int, tuple] = {}  # life id: (animation, track)
        self._order = np.empty(0, dtype=np.int64)
        self._last: bytes | None = None

    def play(self, chapter: Chapter, tracks: list[Track], write: FrameWriter) -> int:
        """Plays ``chapter`` frame by frame, writing a still frame once with a repeat count.

        Args:
            chapter: What the tracker saw across the chapter.
            tracks: The chapter's tracks, from a ``Planner``.
            write: Takes a frame's RGBA bytes and how many times to write it.

        Returns:
            How many frames were written (the chapter's length).

        Raises:
            RuntimeError: If an animation is still running after the chapter's last frame.
        """
        starts: dict[int, list[Track]] = {}
        for track in tracks:
            starts.setdefault(track.start, []).append(track)
        held = 0
        for number in range(chapter.frames):
            changed = number in chapter.orders
            if changed:
                self._order = chapter.orders[number]
            for track in starts.get(number, ()):
                self._begin(track)
                changed = True
            changed = self._advance(number) or changed
            if changed or self._last is None:
                if held:
                    write(self._last, held)
                    held = 0
                self._last = self._draw()
                write(self._last, 1)
            else:
                held += 1
        if held:
            write(self._last, held)
        if self._running or self._fading:
            raise RuntimeError(f"{len(self._running)} animations were still running at the chapter's last frame")
        return chapter.frames

    # -- tracks -------------------------------------------------------------------------------
    def _settle(self, life: int) -> None:
        """Finishes whatever this life was doing, so a new track starts from its end state."""
        running = self._running.pop(life, None)
        if running is not None:
            animation, track = running
            animation.finish()
            if track.kind == FADE_OUT:
                self._fading.pop(life, None)

    def _begin(self, track: Track) -> None:
        """Starts a track: builds, removes or hands a shape to its Manim animation, and draws its first step."""
        self._settle(track.life)
        if track.kind == CUT:
            self._shapes[track.life] = track.leaf.build(self.frame)
            return
        if track.kind == REMOVE:
            self._shapes.pop(track.life, None)
            return
        if track.kind == FADE_OUT:
            shape = self._shapes.pop(track.life, None)
            if shape is None:
                return
            self._fading[track.life] = shape
            animation = FadeOut(shape)
        elif track.kind == APPEAR:
            shape = track.leaf.build(self.frame)
            self._shapes[track.life] = shape
            animation = REVEALS.get(track.style, FadeIn)(shape)
        elif track.kind == MORPH:
            shape = self._shapes.get(track.life)
            if shape is None:
                return
            animation = Transform(shape, track.leaf.build(self.frame))
        else:
            raise ValueError(f"unknown track kind {track.kind!r}")
        animation.begin()
        self._running[track.life] = (animation, track)
        self._step(animation, track, track.start)

    @staticmethod
    def _step(animation, track: Track, number: int) -> None:
        """Moves an animation to frame ``number`` of its track (it reaches 1 on the track's last frame)."""
        animation.interpolate((number - track.start + 1) / (track.end - track.start + 1))

    def _advance(self, number: int) -> bool:
        """Moves every running animation to this frame and finishes those that end on it."""
        moved = False
        for life, (animation, track) in list(self._running.items()):
            if track.start < number:
                self._step(animation, track, number)
            moved = True
            if number >= track.end:
                animation.finish()
                del self._running[life]
                if track.kind == FADE_OUT:
                    self._fading.pop(life, None)
        return moved

    # -- drawing ------------------------------------------------------------------------------
    def _draw(self) -> bytes:
        """Draws the scene: shapes in the tracker's drawing order, fading ones on top; returns its RGBA bytes."""
        shapes = self._shapes
        self.scene.mobjects = [shapes[i] for i in self._order if i in shapes] + list(self._fading.values())
        renderer = self.scene.renderer
        renderer.update_frame(self.scene)
        return renderer.get_frame().tobytes()

    def on_screen(self) -> list:
        """The shapes drawn on the last frame.

        Returns:
            The Manim shapes, in drawing order.
        """
        return list(self.scene.mobjects)

    def shape(self, life: int):
        """The shape on screen for a life.

        Args:
            life: The life's id.

        Returns:
            Its Manim shape, or None if it has none.
        """
        return self._shapes.get(life)

    def mismatches(self, chapter: Chapter) -> list[str]:
        """What is on screen that is not the chapter's last recorded picture.

        Args:
            chapter: The chapter just played.

        Returns:
            One message per problem (an animation still running, a shape left over, a shape missing, or
            one in another state than its life's last); empty when the screen is the last picture.
        """
        problems = [f"life {life} is still animating" for life in self._running]
        problems += [f"life {life} is still fading" for life in self._fading]
        alive = {life.id: life for life in chapter.lives if life.end is None}
        problems += [f"life {life} has a shape but has gone" for life in self._shapes if life not in alive]
        for ident, life in alive.items():
            shape = self._shapes.get(ident)
            if shape is None:
                problems.append(f"life {ident} has no shape")
            elif not _same_drawing(shape, life.leaf.build(self.frame)):
                problems.append(f"life {ident} is not drawn as its last state")
        return problems

    def last_frame(self) -> np.ndarray:
        """The last frame written.

        Returns:
            Its RGB values, shape (height, width, 3).

        Raises:
            ValueError: If no frame has been drawn yet.
        """
        if self._last is None:
            raise ValueError("no frame has been drawn yet")
        width, height = self.frame.size_px
        return np.frombuffer(self._last, dtype=np.uint8).reshape(height, width, 4)[..., :3]
