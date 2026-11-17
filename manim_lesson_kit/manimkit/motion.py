"""Turns each life into Manim animations on the chapter's frames.

- A thing that appears is drawn in (``Create`` for a line, ``Write`` for a large text, ``FadeIn`` for
  the rest). If matplotlib grows or fades it in over several frames, the reveal lasts as long and ends
  on its last state there; otherwise it takes ``REVEAL_SECONDS``.
- A thing that moves or changes colour on consecutive samples is transformed over those frames; a
  single jump is smoothed over ``REVEAL_SECONDS``.
- A thing that goes fades out over ``FADE_SECONDS``, unless something else took its place, which is a
  cut. Live things are cuts too, and so is every change after the first frame of a chapter that holds
  real states (training snapshots): they show exactly what matplotlib showed, on the frame it showed
  it, and nothing between two of them is invented.
- At a chapter's first frame the old picture fades out before anything new is drawn in, so the two
  are never on screen together. A thing that changed while it waited is drawn in as it is by then.

Every animation ends by the chapter's last frame, so a chapter ends on matplotlib's last picture; a
plan that would not is refused before a frame is drawn.
"""

from dataclasses import dataclass

import numpy as np

from manimkit.leaves import Leaf, PathLeaf, TextLeaf
from manimkit.timeline import CARRIED, Chapter, Life

REVEAL_SECONDS = 0.4  # a thing takes this long to be drawn in
FADE_SECONDS = 0.2  # and this long to fade out
WRITE_PT = 17.0  # text this large is written in; smaller text fades in
GROWN = 0.5  # a first state under half the width, height or opacity of a later one is still growing in

APPEAR, MORPH, CUT, FADE_OUT, REMOVE = "appear", "morph", "cut", "fade_out", "remove"
CREATE, WRITE, FADE_IN = "create", "write", "fade_in"


class PlanError(ValueError):
    """A chapter's tracks would not end on its last picture."""


@dataclass(frozen=True)
class Track:
    """One animation of one life: from frame ``start`` to ``end`` (both drawn), ending on ``leaf``.

    Attributes:
        life: The life's id.
        kind: ``APPEAR``, ``MORPH``, ``CUT``, ``FADE_OUT`` or ``REMOVE``.
        start: Its first frame.
        end: Its last frame.
        leaf: The state it ends on; None for a track that takes the thing away.
        style: For ``APPEAR``, how it is drawn in: ``CREATE``, ``WRITE`` or ``FADE_IN``.
    """

    life: int
    kind: str
    start: int
    end: int
    leaf: Leaf | None = None
    style: str = ""


def reveal_style(leaf: Leaf) -> str:
    """How a new thing is drawn in.

    Args:
        leaf: Its state when it is drawn in.

    Returns:
        ``WRITE`` for large text, ``CREATE`` for an unfilled line, ``FADE_IN`` for the rest.
    """
    if isinstance(leaf, TextLeaf):
        return WRITE if leaf.prop.get_size_in_points() >= WRITE_PT else FADE_IN
    if isinstance(leaf, PathLeaf) and not leaf.filled:
        return CREATE
    return FADE_IN


def _presence(leaf: Leaf) -> tuple[float, float, float]:
    """How wide and tall (pixels) and how opaque a leaf is."""
    if isinstance(leaf, PathLeaf):
        drawn = leaf.drawn_vertices()
        width, height = (float(v) for v in np.ptp(drawn, axis=0)) if len(drawn) else (0.0, 0.0)
        alpha = max(leaf.fill[3] if leaf.fill else 0.0, leaf.stroke[3] if leaf.stroke else 0.0)
        return width, height, alpha
    if isinstance(leaf, TextLeaf):
        return leaf.width_px, leaf.prop.get_size_in_points(), leaf.color[3]
    return float(leaf.pixels.shape[1]), float(leaf.pixels.shape[0]), 1.0


def grows_in(first: Leaf, later: Leaf) -> bool:
    """Whether ``first`` is ``later`` still being drawn in, rather than ``later`` somewhere else.

    Args:
        first: A thing's first state.
        later: A later state of it.

    Returns:
        True if ``first`` is much narrower, shorter or fainter than ``later``.
    """
    return any(now < GROWN * then for now, then in zip(_presence(first), _presence(later)))


def _runs(frames: list[int], step: int = 1) -> list[tuple[int, int]]:
    """Frame numbers grouped into (first, last) runs of changes on consecutive samples, ``step`` frames apart."""
    runs: list[tuple[int, int]] = []
    for frame in frames:
        if runs and frame - runs[-1][1] <= step:
            runs[-1] = (runs[-1][0], frame)
        else:
            runs.append((frame, frame))
    return runs


class Planner:
    """Plans one chapter's tracks: how each life appears, changes and goes.

    Attributes:
        chapter: What the tracker saw across the chapter.
        transition: Whether the old picture fades out before anything new is drawn in.
        cuts_after_start: Whether every change after the first frame is a cut (held real states).
        step: Frames between two sampled frames of the chapter's motion.
        last: The chapter's last frame.
        fade: How many frames a fade-out takes.
        reveal: How many frames a reveal takes.

    Example:
        >>> tracks = Planner(chapter, fps=30).tracks()  # doctest: +SKIP
    """

    def __init__(self, chapter: Chapter, *, fps: int = 30, transition: bool = True, cuts_after_start: bool = False,
                 step: int = 1):
        """Creates the planner; fades and reveals shrink to fit a short chapter.

        Args:
            chapter: What the tracker saw across the chapter.
            fps: The video's frame rate.
            transition: Whether the old picture fades out first.
            cuts_after_start: Whether every change after the first frame is a cut.
            step: Frames between two sampled frames of the chapter's motion.
        """
        self.chapter = chapter
        self.transition = transition
        self.cuts_after_start = cuts_after_start
        self.step = step
        self.last = chapter.frames - 1
        # A short chapter (a quick preview) shortens its fades and reveals so they still fit inside it.
        self.fade = min(max(1, round(FADE_SECONDS * fps)), max(1, chapter.frames // 3))
        self.reveal = min(max(1, round(REVEAL_SECONDS * fps)), max(1, chapter.frames // 2))

    def tracks(self) -> list[Track]:
        """Every life's tracks, checked.

        Returns:
            The tracks in order of their first frame.

        Raises:
            PlanError: If a life's tracks leave the chapter, overlap, or end on an earlier state than
                matplotlib last drew.
        """
        tracks: list[Track] = []
        for life in self.chapter.lives:
            planned = self._life(life)
            self._check(life, planned)
            tracks.extend(planned)
        return sorted(tracks, key=lambda track: (track.start, track.life))

    def _check(self, life: Life, tracks: list[Track]) -> None:
        """Refuses tracks outside the chapter, overlapping ones, and an end on anything but the last state."""
        previous_end = -1
        for track in tracks:
            if not 0 <= track.start <= track.end <= self.last or track.start <= previous_end:
                raise PlanError(f"life {life.id}: {track.kind} on frames {track.start}-{track.end} does not fit "
                                f"after frame {previous_end} in a {self.chapter.frames}-frame chapter")
            previous_end = track.end
        shown = [track for track in tracks if track.leaf is not None]
        if life.end is None and shown and shown[-1].leaf is not life.leaf:
            raise PlanError(f"life {life.id} would end on an earlier state than the one matplotlib last drew")

    def _cut(self, frame: int) -> bool:
        """Whether a change on ``frame`` is a cut because the chapter holds real states."""
        return self.cuts_after_start and frame > 0

    def _life(self, life: Life) -> list[Track]:
        """One life's tracks: its appearance, its changes and its ending, in order."""
        out: list[Track] = []
        changes = [frame for frame, _ in life.states[1:]]
        busy_until = -1  # the last frame an earlier track of this life uses
        start = life.start
        if start != CARRIED:
            if life.live or self._cut(start):
                out.append(Track(life.id, CUT, start, start, life.states[0][1]))
                busy_until = start
            else:
                first = max(start, self.fade) if self.transition else start  # after the old picture has faded
                if life.end is not None and first >= life.end:  # gone before it could be drawn in
                    return []
                appear, changes = self._appearance(life, first, changes)
                out.append(appear)
                busy_until = appear.end
        out.extend(self._changes(life, changes, busy_until))
        out.extend(self._ending(life))
        return out

    def _appearance(self, life: Life, first: int, changes: list[int]) -> tuple[Track, list[int]]:
        """How a new thing is drawn in from frame ``first``, and the changes left after it.

        It is drawn in as it is when its reveal starts. If matplotlib then grows or fades it in, the reveal
        lasts as long and ends on its last state there; if it moves instead, it is drawn in where it has
        got to after a reveal's length, and moves on from there.

        Args:
            life: The new thing.
            first: The frame its reveal starts on.
            changes: The frames it changes on.

        Returns:
            The ``APPEAR`` track, and the changes still to animate after it.
        """
        states = dict(life.states)
        settled = max(frame for frame in states if frame <= first)
        later = [frame for frame in changes if frame > settled]
        target = settled
        runs = _runs(later, self.step)
        if not self.cuts_after_start and runs and runs[0][0] <= settled + self.step:
            if grows_in(states[settled], states[runs[0][1]]):
                target = runs[0][1]
            else:
                target = max(frame for frame in states if frame <= max(first + self.reveal - 1, settled))
            later = [frame for frame in later if frame > target]
        end = max(first, self._before_next(max(target, first + self.reveal - 1), later, life))
        leaf = states[target]
        return Track(life.id, APPEAR, first, end, leaf, reveal_style(leaf)), later

    def _before_next(self, end: int, later_changes: list[int], life: Life) -> int:
        """``end`` pulled back so it finishes before the life's next change, its end and the chapter's."""
        limits = [self.last]
        if later_changes:
            limits.append(later_changes[0] - 1)
        if life.end is not None:
            limits.append(life.end - 1)
        return min(end, *limits)

    def _runs_of(self, changes: list[int]) -> list[tuple[int, int]]:
        """Runs of changes, each to become one move.

        With held states, the first frame's change is a run of its own and each later change is a cut,
        so no move ever spans two real states.
        """
        if not self.cuts_after_start:
            return _runs(changes, self.step)
        return [(frame, frame) for frame in changes]

    def _changes(self, life: Life, changes: list[int], busy_until: int) -> list[Track]:
        """A life's moves and cuts, each starting after ``busy_until`` and ending before the next change."""
        states = dict(life.states)
        out = []
        runs = self._runs_of(changes)
        for index, (first, last) in enumerate(runs):
            if life.live or self._cut(first):
                out.extend(Track(life.id, CUT, frame, frame, states[frame]) for frame in range(first, last + 1)
                           if frame in states)
                busy_until = last
                continue
            # it moves from the frame after the sample it was last seen still on
            start = min(max(first - self.step + 1, busy_until + 1, 0), self.last)
            end = last if last > first else first + self.reveal - 1
            following = [runs[index + 1][0]] if index + 1 < len(runs) else []
            end = max(start, self._before_next(end, following, life))
            out.append(Track(life.id, MORPH, start, end, states[last]))
            busy_until = end
        return out

    def _ending(self, life: Life) -> list[Track]:
        """How a life goes: a fade-out, or a cut if something took its place (none while it stays)."""
        if life.end is None:
            return []
        end = life.end
        if end == 0 and self.transition:
            return [Track(life.id, FADE_OUT, 0, self.fade - 1)]
        if life.replaced or life.live or self._cut(end):
            return [Track(life.id, REMOVE, end, end)]
        return [Track(life.id, FADE_OUT, end, min(end + self.fade - 1, self.last))]
