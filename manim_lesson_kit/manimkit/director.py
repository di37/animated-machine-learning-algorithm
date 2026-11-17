"""Plays a lesson's ten chapters into an MP4 with Manim, frame for frame as long as animkit's video.

Each chapter is animkit's own painter (``animkit.lessons.director``) run on a matplotlib ``Stage`` that
is never rasterized: on the frames animkit renders while the chapter moves (every second one) the stage
is painted, recorded and followed (``timeline``), then Manim plays the changes on every frame
(``motion``, ``player``). Still stretches are one picture written repeatedly, as in animkit. Training
holds each real snapshot and cuts to the next, as the matplotlib video does: nothing between two
snapshots is invented. The 3-D view is live: each sample shows the camera where matplotlib has it.

Everything a lesson shows is still computed by animkit and the project's ``Scene``; the test-set
predictions (with their measured speed), the 3-D view and the all-features comparison are prepared
once, before the first frame, exactly as animkit prepares them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from animkit.lessons import director as kit
from animkit.lessons import theme
from animkit.lessons.config import DEFAULT_RESOLUTION, RESOLUTIONS
from animkit.lessons.director import ANIMATION_STEP, Extras, RenderReport, Scene
from animkit.lessons.encoder import Mp4Encoder
from animkit.lessons.stage import Stage, lesson_style
from animkit.lessons.timing import CHAPTERS, Timing, allocate_frames, phase
from manimkit.frame import Frame
from manimkit.motion import Planner
from manimkit.player import FrameWriter, Player, manim_stage
from manimkit.recorder import record
from manimkit.timeline import Chapter, Tracker

LIVE_3D = ("axes", "main3d")  # the 3-D view's region: every frame is drawn as matplotlib has it
# mplot3d draws the box's panes, grid, ticks and axis labels in groups of their own, outside the 3-D axes
LIVE_3D_GROUPS = frozenset({"pane3d", "grid3d", "axis3d"})

Paint = Callable[[], None]


@dataclass(frozen=True)
class Plan:
    """One chapter: its length, and the frames on which the stage is painted.

    Attributes:
        name: The chapter's name (one of animkit's ``CHAPTERS``).
        frames: How many frames it has.
        samples: ``(frame, painter)`` pairs: on that frame, the painter paints the stage.
        cuts_after_start: Whether it holds real states (training snapshots), so each change is a cut.
        live_3d: Whether it shows the 3-D view, which is drawn as matplotlib has it on every sample.
        step: Frames between two samples.
    """

    name: str
    frames: int
    samples: tuple[tuple[int, Paint], ...]
    cuts_after_start: bool = False
    live_3d: bool = False
    step: int = 1  # frames between two samples


def name_regions(stage: Stage) -> None:
    """Gives the stage's axes and figure texts gids, so their shapes keep their keys whichever are shown.

    Args:
        stage: The lesson stage, after a painter has laid it out.
    """
    regions = {"main": stage.main, "panel": stage.panel, "metric": stage.metric, "diagram": stage.diagram,
               "card": stage.card, "caption": stage._caption_ax}
    regions.update({f"facet{i}": ax for i, ax in enumerate(stage._facets)})
    if stage._main_3d is not None:
        regions.update(main3d=stage._main_3d, backdrop3d=stage._backdrop_3d)
    for name, ax in regions.items():
        ax.set_gid(name)
    for index, text in enumerate(stage.fig.texts):
        text.set_gid(f"figtext{index}")


def animated(name: str, frames: int, moving: float, paint: Callable[[float], None], step: int = ANIMATION_STEP,
             **options) -> Plan:
    """A chapter painted as animkit's ``Recorder.animate`` paints it.

    The stage is painted at t = frame / (motion - 1) on every ``step``-th frame of the chapter's first
    ``moving`` share, then at t = 1 for the frames after (if any). Those are the samples; Manim moves
    things on every frame in between.

    Args:
        name: The chapter's name.
        frames: How many frames it has.
        moving: The share of it (0 to 1) that moves; the rest holds the last picture.
        paint: Paints the stage at a time t from 0 to 1.
        step: Frames between two samples.
        **options: Further ``Plan`` fields (``live_3d``).

    Returns:
        The chapter's plan.
    """
    motion = max(1, round(frames * moving))
    samples = [(index, (lambda t=index / max(motion - 1, 1): paint(t))) for index in range(0, motion, step)]
    if motion < frames:
        samples.append((motion, lambda: paint(1.0)))
    return Plan(name, frames, tuple(samples), step=step, **options)


def training_holds(lesson: Any, timing: Timing) -> tuple[int, ...]:
    """Frames each training snapshot is held, as animkit allots them.

    Every snapshot gets at least one, so a lesson with more snapshots than training frames runs that
    much longer, in animkit as here.

    Args:
        lesson: The lesson, with its snapshots and their weights.
        timing: The video's timing.

    Returns:
        One frame count per snapshot.
    """
    weights = lesson.frame_weights or (1.0,) * len(lesson.snapshots)
    return allocate_frames(weights, timing.frames("training"))


def lesson_frames(lesson: Any, timing: Timing) -> int:
    """How many frames animkit's video of ``lesson`` has, and so this one.

    Args:
        lesson: The lesson.
        timing: The video's timing.

    Returns:
        The video's length, in frames.
    """
    return timing.total_frames - timing.frames("training") + sum(training_holds(lesson, timing))


def plans(stage: Stage, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> list[Plan]:
    """The ten chapters, as animkit's ``CHAPTER_PLAYERS`` play them.

    Args:
        stage: The lesson stage the painters paint.
        lesson: The lesson.
        timing: The video's timing.
        scene: The project's ``Scene``: the task-specific drawing.
        extras: What was computed once before the first frame.

    Returns:
        One plan per chapter, in order.
    """
    story = lesson.story
    f = timing.frames
    holds = training_holds(lesson, timing)
    starts = [sum(holds[:index]) for index in range(len(holds))]
    training = tuple((start, (lambda i=index: kit.paint_training_state(stage, lesson, i, "training", story.training,
                                                                       scene=scene)))
                     for index, start in enumerate(starts))
    return [
        animated("title", f("title"), 0.2, lambda t: kit.paint_title(stage, lesson, t)),
        animated("data", f("data"), 0.7, lambda t: kit.paint_data(stage, lesson, t, scene)),
        animated("initial", f("initial"), 0.35, lambda t: kit.paint_training_state(
            stage, lesson, 0, "initial", story.initial, residual_grow=phase(t, 0.0, 1.0), scene=scene)),
        Plan("training", sum(holds), training, cuts_after_start=True),
        Plan("final", f("final"), ((0, lambda: kit.paint_training_state(stage, lesson, lesson.final_index, "final",
                                                                          story.final, scene=scene)),)),
        animated("inference", f("inference"), 0.9, lambda t: kit.paint_inference(stage, lesson, t, scene)),
        animated("exam", f("exam"), 0.85, lambda t: kit.paint_exam(stage, lesson, extras.exam, t, scene)),
        animated("surface", f("surface"), 1.0, lambda t: kit.paint_surface(stage, lesson, extras.surface, t, scene),
                 live_3d=True),
        animated("features", f("features"), 0.6, lambda t: kit.paint_features(stage, lesson, extras.comparison, t)),
        animated("takeaway", f("takeaway"), 0.15,
                 lambda t: kit.paint_takeaway(stage, lesson, t, scene, extras.comparison)),
    ]


def _live_3d(key: tuple) -> bool:
    """Whether a key is in the 3-D view or one of the groups mplot3d draws its box in."""
    groups = key[:-1]
    return LIVE_3D in groups or any(group[0] in LIVE_3D_GROUPS for group in groups)


def _never_live(key: tuple) -> bool:
    """No key is live (a chapter without the 3-D view)."""
    return False


def play_chapter(plan: Plan, stage: Stage, tracker: Tracker, player: Player, write: FrameWriter,
                 fps: int) -> Chapter:
    """Paints, records and follows the chapter's samples, then plays it.

    Args:
        plan: The chapter's plan.
        stage: The lesson stage.
        tracker: The tracker following the lesson's lives from chapter to chapter.
        player: The player drawing the frames.
        write: Takes a frame's RGBA bytes and how many times to write it.
        fps: Frames per second.

    Returns:
        What the tracker saw across the chapter.
    """
    log = tracker.begin(plan.frames, live=_live_3d if plan.live_3d else _never_live, step=plan.step)
    for number, paint in plan.samples:
        paint()
        name_regions(stage)
        leaves, _ = record(stage.fig)
        log.frame(number, leaves)
    chapter = log.finish()
    planner = Planner(chapter, fps=fps, cuts_after_start=plan.cuts_after_start, step=plan.step)
    player.play(chapter, planner.tracks(), write)
    return chapter


def prepare_extras(lesson: Any, scene: Scene) -> Extras:
    """What animkit computes once per video before any frame.

    Args:
        lesson: The lesson.
        scene: The project's ``Scene``.

    Returns:
        The measured exam, the all-features comparison and the 3-D view.
    """
    return Extras(exam=scene.prepare_exam(lesson), comparison=scene.compare(lesson),
                  surface=scene.prepare_surface(lesson))


def render_lesson(lesson: Any, path: Path, *, scene: Scene, timing: Timing = Timing(),
                  dpi: float = RESOLUTIONS[DEFAULT_RESOLUTION], chapters: tuple[str, ...] | None = None,
                  on_chapter: Callable[[str, Player, Chapter], None] | None = None,
                  extras: Extras | None = None) -> RenderReport:
    """Renders ``lesson`` to ``path`` with Manim, frame for frame as long as animkit's video.

    Args:
        lesson: The lesson to render.
        path: The MP4 to write.
        scene: The project's ``Scene``: it draws the task-specific parts.
        timing: The video's timing.
        dpi: The resolution (animkit's ``RESOLUTIONS``).
        chapters: Plays only these chapters; all ten by default.
        on_chapter: Called as ``on_chapter(name, player, chapter)`` after each chapter's last frame, for
            tests that look at the screen.
        extras: Prepared here unless given (a test passes the ones its matplotlib reference uses, so
            both show the same measured speed).

    Returns:
        Where the video went, and how many frames and seconds it has.

    Raises:
        ValueError: If ``chapters`` names a chapter there is not.

    Example:
        >>> render_lesson(lesson, Path("linear.mp4"), scene=scene, chapters=("title",))  # doctest: +SKIP
    """
    unknown = set(chapters or ()) - set(CHAPTERS)
    if unknown:
        raise ValueError(f"unknown chapters {sorted(unknown)}; there are {', '.join(CHAPTERS)}")
    extras = extras or prepare_extras(lesson, scene)
    with lesson_style():
        stage = Stage(dpi=dpi, source_note=scene.source_note)
        frame = Frame.of(stage.fig)
        width, height = stage.size_px
        with manim_stage(frame, theme.CANVAS, timing.fps) as player, \
                Mp4Encoder(Path(path), width, height, timing.fps) as video:
            tracker = Tracker()
            for plan in plans(stage, lesson, timing, scene, extras):
                if chapters is not None and plan.name not in chapters:
                    continue
                chapter = play_chapter(plan, stage, tracker, player, video.write, timing.fps)
                if on_chapter is not None:
                    on_chapter(plan.name, player, chapter)
    return RenderReport(path=Path(path), frames=video.frames_written, seconds=video.frames_written / timing.fps)
