"""Follows each recorded shape from frame to frame: when it appears, every change, and when it goes.

A ``Life`` is one thing on screen. On each sampled frame a leaf with the key of a living thing either
draws the same (nothing happens), is the same thing moved or recoloured (a change: same kind of shape,
moved by at most ``max_step_px``), or is something else in its place (the old life ends, a new one
begins). A life whose key is missing from a frame has gone. Things in a live region (a 3-D view whose
every tile moves with the camera) begin a new life on every frame they change.

The tracker keeps its lives from one chapter to the next, so whatever the next chapter keeps on screen
carries straight over.
"""

from dataclasses import dataclass, field

import numpy as np

from manimkit.leaves import Leaf, displacement_px

MAX_STEP_PX = 80.0  # further than this a frame, a leaf is something else in the same place
CARRIED = -1  # the frame a life carried in from the last chapter started on


@dataclass
class Life:
    """One thing on screen, from the frame it appears to the frame it is gone.

    Attributes:
        id: A number no other life has.
        key: Its leaves' key.
        start: The chapter frame it appeared on, or ``CARRIED``.
        states: ``(frame, leaf)`` pairs: its first state, then each change.
        end: The frame it is gone from; None while it is still on screen.
        replaced: Whether it went because something else took its place (it does not fade away).
        live: Whether it is in a live region (every change starts a new life).
    """

    id: int
    key: tuple
    start: int  # the chapter frame it appeared on, or CARRIED
    states: list[tuple[int, Leaf]] = field(default_factory=list)  # (frame, leaf): its first state, then each change
    end: int | None = None  # the frame it is gone from; None while it is still on screen
    replaced: bool = False  # it went because something else took its place (it does not fade away)
    live: bool = False

    @property
    def leaf(self) -> Leaf:
        """Its latest state.

        Returns:
            The leaf it last changed to.
        """
        return self.states[-1][1]


@dataclass
class Chapter:
    """What the tracker saw across one chapter's sampled frames.

    Attributes:
        frames: How many frames the chapter has.
        lives: Every life on screen at some point of the chapter, in order of their first frame.
        orders: Sampled frame: the life ids on screen, in drawing order (kept only when it changes).
    """

    frames: int
    lives: list[Life]  # every life on screen at some point of the chapter, in order of their first frame
    orders: dict[int, np.ndarray]  # sampled frame: the life ids on screen, in drawing order


class Tracker:
    """Keeps every life on screen, chapter after chapter. Start each chapter with ``begin``.

    Attributes:
        max_step_px: How far a leaf may move in one frame and still be the same thing.

    Example:
        >>> log = Tracker().begin(frames=48, step=2)  # doctest: +SKIP
        >>> for number in range(0, 48, 2):  # doctest: +SKIP
        ...     log.frame(number, leaves_on(number))
        >>> chapter = log.finish()  # doctest: +SKIP
    """

    def __init__(self, max_step_px: float = MAX_STEP_PX):
        """Creates a tracker with nothing on screen.

        Args:
            max_step_px: How far a leaf may move in one frame and still be the same thing.
        """
        self.max_step_px = max_step_px
        self._alive: dict[tuple, Life] = {}
        self._next_id = 0

    def begin(self, frames: int, live=lambda key: False, step: int = 1) -> "ChapterLog":
        """Starts a chapter sampled every ``step`` frames: everything still on screen is carried in.

        Args:
            frames: How many frames the chapter has.
            live: Says whether a key is in a live region.
            step: How many frames apart its samples are (a leaf may move ``step`` times as far).

        Returns:
            The log to feed the chapter's sampled frames to.
        """
        for life in self._alive.values():
            life.start, life.states = CARRIED, [(CARRIED, life.leaf)]
        return ChapterLog(self, frames, live, self.max_step_px * step)

    def _new(self, key: tuple, frame: int, leaf: Leaf, live: bool) -> Life:
        """Begins a new life for ``key`` on ``frame``."""
        life = Life(self._next_id, key, frame, [(frame, leaf)], live=live)
        self._next_id += 1
        self._alive[key] = life
        return life


class ChapterLog:
    """Feeds one chapter's sampled frames to a ``Tracker``; ``finish`` returns the ``Chapter``."""

    def __init__(self, tracker: Tracker, frames: int, live, max_step_px: float):
        """Opens the log. ``Tracker.begin`` makes it.

        Args:
            tracker: The tracker whose lives it follows.
            frames: How many frames the chapter has.
            live: Says whether a key is in a live region.
            max_step_px: How far a leaf may move between two samples and still be the same thing.
        """
        self._tracker = tracker
        self._max_step_px = max_step_px
        self._frames = frames
        self._live = live
        self._lives: dict[int, Life] = {life.id: life for life in tracker._alive.values()}
        self._orders: dict[int, np.ndarray] = {}
        self._previous_order: np.ndarray | None = None

    def frame(self, number: int, leaves: list[Leaf]) -> None:
        """Takes one sampled frame: each leaf continues a life, changes it, or begins a new one.

        Args:
            number: The frame's number in the chapter (frames come in order).
            leaves: The frame's leaves, in drawing order.
        """
        tracker, alive = self._tracker, self._tracker._alive
        seen: set[tuple] = set()
        order = []
        for leaf in leaves:
            key = leaf.key
            if key in seen:  # two leaves with one key: the second is told apart by its order
                key = (*key, "again", len(order))
            seen.add(key)
            life = alive.get(key)
            live = self._live(key)
            if life is not None and life.leaf.signature == leaf.signature:
                pass
            elif life is not None and not live and not life.live and life.leaf.matches(leaf) and \
                    displacement_px(life.leaf, leaf) <= self._max_step_px:
                life.states.append((number, leaf))
            else:
                if life is not None:
                    life.end, life.replaced = number, True
                life = tracker._new(key, number, leaf, live)
                self._lives[life.id] = life
            order.append(life.id)
        for key in [key for key in alive if key not in seen]:
            gone = alive.pop(key)
            gone.end = number
        current = np.asarray(order, dtype=np.int64)
        if self._previous_order is None or not np.array_equal(current, self._previous_order):
            self._orders[number] = current
            self._previous_order = current

    def finish(self) -> Chapter:
        """Ends the chapter.

        Returns:
            What the tracker saw, its lives in order of their first frame.
        """
        lives = sorted(self._lives.values(), key=lambda life: (life.start, life.id))
        return Chapter(self._frames, lives, self._orders)
