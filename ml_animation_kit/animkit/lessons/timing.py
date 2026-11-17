"""Chapter lengths in frames, frame budgets per training snapshot, and easing curves.

Each chapter's length is set in seconds (``CHAPTER_SECONDS``) and turned into frames here, scaled for
previews. Within a chapter, painters take a progress ``t`` from 0 to 1; ``ease`` and ``phase`` turn
it into motion that starts and stops gently, or that happens during only part of the chapter.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from animkit.lessons.config import CHAPTER_SECONDS, FPS

CHAPTERS = ("title", "data", "initial", "training", "final", "inference", "exam", "surface", "features",
            "takeaway")
MIN_CHAPTER_FRAMES = 3


@dataclass(frozen=True)
class Timing:
    """The frame rate and the duration scale a lesson video is rendered with.

    Attributes:
        fps: Frames per second.
        scale: Stretches (>1) or shrinks (<1) every chapter, for example 0.1 for a quick preview.
    """

    fps: int = FPS
    scale: float = 1.0  # stretch (>1) or shrink (<1) every chapter, e.g. for a quick preview

    def __post_init__(self):
        """Refuses a frame rate below 1 or a scale that is not positive.

        Raises:
            ValueError: If ``fps`` is below 1 or ``scale`` is not positive.
        """
        if self.fps < 1:
            raise ValueError(f"fps must be at least 1, got {self.fps}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    def frames(self, chapter: str) -> int:
        """How many frames a chapter lasts: its seconds × scale × fps, rounded, and at least ``MIN_CHAPTER_FRAMES``.

        Args:
            chapter: The chapter's key, one of ``CHAPTERS``.

        Returns:
            The chapter's length in frames.
        """
        seconds = CHAPTER_SECONDS[chapter]
        return max(MIN_CHAPTER_FRAMES, round(seconds * self.scale * self.fps))

    @property
    def total_frames(self) -> int:
        """The whole video's length in frames: every chapter's, added up.

        Returns:
            The total frame count.
        """
        return sum(self.frames(chapter) for chapter in CHAPTERS)


def allocate_frames(weights: Sequence[float], total: int) -> tuple[int, ...]:
    """Split ``total`` frames in proportion to ``weights``.

    Every entry gets at least one frame (each snapshot must be seen), so when ``total`` is smaller
    than the number of weights the result sums to ``len(weights)`` instead.

    The split uses largest remainders: each entry gets the whole part of its exact share, and the frames
    left over go to the entries with the biggest fractions (the earlier one on a tie). An entry still left
    with no frame borrows one from the longest.

    Args:
        weights: How long each entry should last, relative to the others.
        total: The frames to share out.

    Returns:
        The frames for each entry, in order.

    Raises:
        ValueError: If the weights are empty, not finite, negative, or all zero.

    Example:
        >>> allocate_frames([1, 1, 2], 10)
        (3, 2, 5)
    """
    values = np.asarray(weights, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, non-empty and not all zero")
    if total <= values.size:
        return (1,) * values.size
    exact = values / values.sum() * total
    frames = np.floor(exact).astype(int)
    leftover = total - int(frames.sum())
    frames[np.argsort(-(exact - frames), kind="stable")[:leftover]] += 1  # largest remainders
    for empty in np.flatnonzero(frames == 0):  # lend one frame from the longest entry
        frames[np.argmax(frames)] -= 1
        frames[empty] = 1
    return tuple(int(f) for f in frames)


def ease(t: float) -> float:
    """Smoothstep on [0, 1], clamped outside.

    Args:
        t: Progress, 0 to 1.

    Returns:
        3t² − 2t³: 0 at 0 and 1 at 1, with zero speed at both ends.
    """
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def phase(t: float, start: float, end: float) -> float:
    """Eased progress of ``t`` through the window [start, end].

    Before ``start`` it is 0 and after ``end`` it is 1, so a painter can make something happen during
    just one part of a chapter.

    Args:
        t: The chapter's progress, 0 to 1.
        start: Where the window opens, as chapter progress.
        end: Where it closes.

    Returns:
        The eased progress through the window, 0 to 1.

    Raises:
        ValueError: If ``end`` is not after ``start``.

    Example:
        >>> phase(0.5, 0.25, 0.75)
        0.5
    """
    if end <= start:
        raise ValueError("phase window must have end > start")
    return ease((t - start) / (end - start))
