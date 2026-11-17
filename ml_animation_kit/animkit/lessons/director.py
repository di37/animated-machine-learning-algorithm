"""Plays a lesson's ten chapters into an MP4, one chapter after another.

The chapters: title → data → first guess → training → final model → prediction on an unseen
example → final exam on every test example → the model in 3-D over two inputs → more features →
takeaway.

Motion (dots appearing, the new example travelling, lines drawing) is rendered every
``ANIMATION_STEP`` frames; still moments are one render written repeatedly. Everything specific to
a task or dataset is drawn by the ``Scene`` each project passes in.

Each chapter has a painter (``paint_*``), which draws the stage for one moment ``t`` from 0 to 1,
and a player (``play_*``), which decides how many frames the chapter lasts and which of them move.
``render_lesson`` runs the players in order into one encoder.
"""

import math
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from animkit.lessons import plots, theme
from animkit.lessons.config import DEFAULT_RESOLUTION, FIGSIZE, RESOLUTIONS
from animkit.lessons.encoder import Mp4Encoder
from animkit.lessons.lesson import Comparison
from animkit.lessons.stage import TRACKER, Stage, lesson_style, reset_axes
from animkit.lessons.surface import Surface, draw_notes, draw_surface
from animkit.lessons.timing import Timing, allocate_frames, phase

ROADMAP = "  →  ".join(label for _, label in TRACKER)  # title card: the chapters to come
ANIMATION_STEP = 2  # render every 2nd frame while something moves (15 distinct frames/s at 30 fps)
GHOSTS = 3
CARD_DESIGN_HEIGHT_IN = 10.8  # the full-screen cards were laid out on a canvas this tall
CARD_SCALE = FIGSIZE[1] / CARD_DESIGN_HEIGHT_IN  # their text grows with the canvas, so a card looks the same


class Scene(Protocol):
    """Task-specific drawing (regression curve, classification maps) the chapters delegate to.

    A project implements every method. ``lesson`` is always the project's own lesson object, and
    ``axes`` the main plots (``facets`` of them, side by side).

    Attributes:
        subtitle: The start of every workspace subtitle, before the lesson's method.
        source_note: The data credit in the footer.
        facets: How many side-by-side main plots.
    """

    subtitle: str
    source_note: str  # dataset credit in the footer
    facets: int  # how many side-by-side main plots

    def data_label(self, lesson: Any) -> str:
        """The stage label for the data chapter.

        Args:
            lesson: The lesson.

        Returns:
            The label, at the right of the subtitle line.
        """
        ...

    def data_caption(self, lesson: Any) -> str:
        """The caption for the data chapter.

        Args:
            lesson: The lesson.

        Returns:
            The caption.
        """
        ...

    def facts(self, lesson: Any) -> tuple[str, ...]:
        """Short facts about the data, for the data chapter's side panel.

        Args:
            lesson: The lesson.

        Returns:
            One fact per line, top to bottom.
        """
        ...

    def draw_data(self, axes: tuple[Axes, ...], lesson: Any, t: float) -> None:
        """Draws the data on the main plots.

        Args:
            axes: The main plots.
            lesson: The lesson.
            t: The chapter's progress, 0 to 1.
        """
        ...

    def draw_state(self, axes: tuple[Axes, ...], lesson: Any, index: int, *, ghosts: int,
                   grow: float, note: str | None) -> None:
        """Draws the model at one training snapshot over the data.

        Args:
            axes: The main plots.
            lesson: The lesson.
            index: The snapshot's index in ``lesson.snapshots``.
            ghosts: How many earlier snapshots to draw faintly behind it (only during training).
            grow: How far the error marks have grown in, 0 to 1 (the first-guess chapter animates them).
            note: A callout for the final model; ``None`` in the other chapters.
        """
        ...

    def draw_metric(self, ax: Axes, lesson: Any, upto: int | None, mark_final: bool = False) -> None:
        """Draws the metric strip: the training metric, snapshot by snapshot.

        Args:
            ax: The metric strip.
            lesson: The lesson.
            upto: The last snapshot to include; ``None`` in the data chapter, before any training.
            mark_final: Whether to mark the final model.
        """
        ...

    def final_note(self, lesson: Any) -> str:
        """The callout on the final model's plot.

        Args:
            lesson: The lesson.

        Returns:
            The note.
        """
        ...

    def inference_label(self, lesson: Any) -> str:
        """The stage label for the prediction chapter.

        Args:
            lesson: The lesson.

        Returns:
            The label.
        """
        ...

    def draw_inference(self, axes: tuple[Axes, ...], lesson: Any, t: float) -> None:
        """Draws the prediction on the unseen example on the main plots.

        Args:
            axes: The main plots.
            lesson: The lesson.
            t: The chapter's progress, 0 to 1.
        """
        ...

    def explain(self, lesson: Any):
        """The worked prediction for the unseen example.

        Args:
            lesson: The lesson.

        Returns:
            An ``Explanation``: the worked lines, the last one the model check.
        """
        ...

    def takeaway_facts(self, lesson: Any) -> tuple[str, ...]:
        """Facts for the takeaway card.

        Args:
            lesson: The lesson.

        Returns:
            One fact per line.
        """
        ...

    def prepare_exam(self, lesson: Any) -> Any:
        """Scores every test example, once, before any frame is drawn.

        Args:
            lesson: The lesson.

        Returns:
            Whatever the exam chapter needs; it is passed back to ``exam_label``, ``exam_caption`` and
            ``draw_exam``.
        """
        ...

    def exam_label(self, lesson: Any, exam: Any) -> str:
        """The stage label for the final exam.

        Args:
            lesson: The lesson.
            exam: What ``prepare_exam`` returned.

        Returns:
            The label.
        """
        ...

    def exam_caption(self, lesson: Any, exam: Any) -> str:
        """The caption for the final exam.

        Args:
            lesson: The lesson.
            exam: What ``prepare_exam`` returned.

        Returns:
            The caption.
        """
        ...

    def draw_exam(self, axes: tuple[Axes, ...], panel: Axes, board: Axes, lesson: Any, exam: Any,
                  t: float) -> None:
        """Draws the final exam: the test examples scored as ``t`` runs from 0 to 1.

        Args:
            axes: The main plots.
            panel: The side panel.
            board: The metric strip, used as the score board.
            lesson: The lesson.
            exam: What ``prepare_exam`` returned.
            t: The chapter's progress, 0 to 1.
        """
        ...

    def compare(self, lesson: Any) -> Comparison:
        """Scores the lesson's model against the same algorithm given every feature.

        Args:
            lesson: The lesson.

        Returns:
            The comparison, for the more-features card.
        """
        ...

    def prepare_surface(self, lesson: Any) -> Surface:
        """Builds the 3-D view of the trained model over two of its inputs.

        Args:
            lesson: The lesson.

        Returns:
            The surface, already checked.
        """
        ...


class Diagram(Protocol):
    """How the algorithm computes, drawn from the model's current parameters (optional per lesson)."""

    def draw_training(self, ax: Axes, snapshot: Any) -> None:
        """Draws the algorithm as it stands at one training snapshot.

        Args:
            ax: The diagram's axes.
            snapshot: The training snapshot.
        """
        ...

    def draw_inference(self, ax: Axes, query: Any, t: float) -> None:
        """Shows the algorithm working on the unseen example.

        Args:
            ax: The diagram's axes.
            query: The unseen example, as the project holds it.
            t: Progress through the reveal, 0 to 1.
        """
        ...


def _make_room_for_diagram(stage: Stage, lesson: Any) -> "Diagram | None":
    """Gives the lesson's diagram, if it has one, the top of the side column; returns it, or None."""
    diagram = getattr(lesson, "diagram", None)
    stage.share_panel(diagram is not None)
    return diagram


@dataclass(frozen=True)
class Extras:
    """What is computed once per video, before any frame.

    The test-set predictions (with measured speed), the 3-D view over two inputs and the all-features
    comparison; any extra model they need is trained here.

    Attributes:
        exam: The scored test set, as the scene's ``prepare_exam`` returns it.
        comparison: The more-features comparison.
        surface: The 3-D view.
    """

    exam: Any
    comparison: Comparison
    surface: Surface


@dataclass(frozen=True)
class RenderReport:
    """What a rendered lesson video came to.

    Attributes:
        path: The video file.
        frames: How many frames it holds.
        seconds: Its length, in seconds.
    """

    path: Path
    frames: int
    seconds: float


class Recorder:
    """Renders the stage and writes the frame to the video, held for as many frames as asked.

    Attributes:
        stage: The canvas the painters draw on.
    """

    def __init__(self, stage: Stage, video: Mp4Encoder):
        """Joins a stage to an open encoder.

        Args:
            stage: The canvas.
            video: The open MP4 encoder.
        """
        self.stage = stage
        self._video = video

    def shoot(self, frames: int = 1) -> None:
        """Renders the stage once and writes it for ``frames`` frames; does nothing for 0 or fewer.

        Args:
            frames: How many frames the current picture lasts.
        """
        if frames > 0:
            self._video.write(self.stage.render(), repeat=frames)

    def animate(self, total: int, paint: Callable[[float], None], moving: float = 1.0) -> None:
        """Paint t in [0, 1] across the first ``moving`` share of ``total`` frames, then hold t = 1.

        While moving, the stage is repainted every ``ANIMATION_STEP`` frames, each render held for the
        frames in between.

        Args:
            total: The chapter's length, in frames.
            paint: Draws the stage for one value of t.
            moving: The share of the chapter that moves; the rest holds the final picture.
        """
        motion = max(1, round(total * moving))
        written = 0
        while written < motion:
            paint(written / max(motion - 1, 1))
            count = min(ANIMATION_STEP, motion - written)
            self.shoot(count)
            written += count
        paint(1.0)
        self.shoot(total - written)


def _header(stage: Stage, lesson: Any, scene: Scene, chapter: str, label: str = "") -> None:
    """Sets a workspace chapter's header: the lesson's title, the scene's subtitle and method, the tracker."""
    stage.set_header(lesson.story.title, f"{scene.subtitle} · {lesson.story.method}", chapter, label)


# ---- title and takeaway cards -------------------------------------------------------------------

def _card_pt(size: float) -> float:
    """A font size from the cards' design canvas, scaled to the real one."""
    return size * CARD_SCALE


def paint_title(stage: Stage, lesson: Any, t: float) -> None:
    """Paints the title card: the lesson's name, method and objective, and the chapters to come.

    Args:
        stage: The canvas.
        lesson: The lesson.
        t: The chapter's progress, 0 to 1; the card fades in.
    """
    card = stage.show_card()
    fade = phase(t, 0.0, 1.0)
    story = lesson.story
    card.add_patch(_accent_bar(fade))
    card.text(0.08, 0.74, "MACHINE LEARNING, ANIMATED", fontsize=_card_pt(18), weight="bold",
              color=theme.HIGHLIGHT, alpha=fade)
    card.text(0.08, 0.64, story.title, fontsize=_card_pt(64), weight="bold", color=theme.HEADING, alpha=fade)
    card.text(0.08, 0.555, story.method[:1].upper() + story.method[1:], fontsize=_card_pt(28),
              color=theme.NAVY, alpha=fade)
    card.text(0.08, 0.48, "\n".join(textwrap.wrap(story.objective, 80)), fontsize=_card_pt(23),
              color=theme.INK_SECONDARY, alpha=fade, va="top", linespacing=1.45)
    card.text(0.08, 0.22, ROADMAP,
              fontsize=_card_pt(18), color=theme.INK_MUTED, alpha=fade)


def _accent_bar(alpha: float) -> Rectangle:
    """The thin vertical bar beside a card's text."""
    return Rectangle((0.055, 0.2), 0.006, 0.58, facecolor=theme.HEADING, alpha=alpha, edgecolor="none")


def paint_takeaway(stage: Stage, lesson: Any, t: float, scene: Scene,
                   comparison: Comparison | None = None) -> None:
    """Paints the takeaway card: the main lesson, the limitation and the scene's facts.

    Args:
        stage: The canvas.
        lesson: The lesson.
        t: The chapter's progress, 0 to 1; the card fades in.
        scene: The project's scene, for the facts.
        comparison: The more-features comparison, whose takeaway line is added; ``None`` for none.
    """
    card = stage.show_card()
    fade = phase(t, 0.0, 1.0)
    story = lesson.story
    card.add_patch(_accent_bar(fade))
    card.text(0.08, 0.76, f"TAKEAWAY · {story.title.upper()}", fontsize=_card_pt(18), weight="bold",
              color=theme.HIGHLIGHT, alpha=fade)
    card.text(0.08, 0.7, "\n".join(textwrap.wrap(story.takeaway, 58)), fontsize=_card_pt(34), weight="bold",
              color=theme.NAVY, alpha=fade, va="top", linespacing=1.3)
    card.text(0.08, 0.43, "\n".join(textwrap.wrap(f"Limitation: {story.limitation}", 95)), fontsize=_card_pt(20),
              color=theme.INK_SECONDARY, alpha=fade, va="top", linespacing=1.4)
    facts = (*scene.takeaway_facts(lesson), *((comparison.takeaway,) if comparison else ()))
    for row, fact in enumerate(facts):
        card.text(0.08, 0.27 - row * 0.05, fact, fontsize=_card_pt(19), color=theme.NAVY, alpha=fade)


BAR_COLOURS = {"baseline": theme.INK_MUTED, "lesson": theme.MODEL, "richer": theme.HIGHLIGHT}
DETAIL_CHARS = 44  # keeps each row's detail left of its bar


def paint_features(stage: Stage, lesson: Any, comparison: Comparison, t: float) -> None:
    """The same algorithm with every feature, against the lesson's model: one bar per model.

    The bars are revealed in turn, and a bar's length is the real test score.

    Args:
        stage: The canvas.
        lesson: The lesson.
        comparison: The comparison to draw.
        t: The chapter's progress, 0 to 1.
    """
    card = stage.show_card()
    card.add_patch(_accent_bar(1.0))
    card.text(0.08, 0.76, f"MORE FEATURES · {lesson.story.title.upper()}", fontsize=_card_pt(18), weight="bold",
              color=theme.HIGHLIGHT)
    card.text(0.08, 0.7, "\n".join(textwrap.wrap(comparison.headline, 62)), fontsize=_card_pt(30), weight="bold",
              color=theme.NAVY, va="top", linespacing=1.3)
    card.text(0.08, 0.53, comparison.metric, fontsize=_card_pt(16), color=theme.INK_SECONDARY)
    top = max(row.value for row in comparison.rows) or 1.0
    shown = max(1, math.ceil(len(comparison.rows) * phase(t, 0.0, 0.6)))
    for index, row in enumerate(comparison.rows[:shown]):
        y = 0.46 - index * 0.085
        width = 0.42 * row.value / top
        card.text(0.08, y, row.label, fontsize=_card_pt(17), weight="bold", color=theme.NAVY, va="center")
        card.text(0.08, y - 0.022, "\n".join(textwrap.wrap(row.detail, DETAIL_CHARS)), fontsize=_card_pt(13),
                  color=theme.INK_SECONDARY, va="top", linespacing=1.2)
        card.add_patch(Rectangle((0.36, y - 0.022), width, 0.044, facecolor=BAR_COLOURS[row.role],
                                 edgecolor="none"))
        card.text(0.37 + width, y, row.text, fontsize=_card_pt(18), weight="bold", color=theme.NAVY, va="center")
    card.text(0.08, 0.135, "\n".join(textwrap.wrap(comparison.note, 120)), fontsize=_card_pt(14),
              color=theme.INK_SECONDARY, va="top", alpha=phase(t, 0.6, 0.8))


# ---- workspace chapters ---------------------------------------------------------------------------

def paint_data(stage: Stage, lesson: Any, t: float, scene: Scene) -> None:
    """Paints the data chapter: the data on the main plots and a few facts about it in the side panel.

    Args:
        stage: The canvas.
        lesson: The lesson.
        t: The chapter's progress, 0 to 1; the facts appear one by one.
        scene: The project's scene.
    """
    stage.show_workspace()
    _header(stage, lesson, scene, "data", scene.data_label(lesson))
    stage.share_panel(False)
    scene.draw_data(stage.main_facets(scene.facets), lesson, t)
    reset_axes(stage.panel)
    stage.panel.text(0.04, 0.88, "The dataset", fontsize=theme.PANEL_TITLE_SIZE, weight="bold",
                     color=theme.NAVY, va="center")
    for row, fact in enumerate(scene.facts(lesson)):
        stage.panel.text(0.04, 0.72 - row * 0.125, fact, fontsize=14, color=theme.INK_SECONDARY,
                         va="center", alpha=phase(t, 0.1 * row, 0.1 * row + 0.3))
    scene.draw_metric(stage.metric, lesson, None)
    stage.set_caption(scene.data_caption(lesson))


def paint_training_state(stage: Stage, lesson: Any, index: int, chapter: str, caption: str,
                         residual_grow: float = 1.0, *, scene: Scene) -> None:
    """Paints the model at one training snapshot: main plots, diagram, panel, metric strip and caption.

    The first-guess, training and final-model chapters all paint with this.

    Args:
        stage: The canvas.
        lesson: The lesson.
        index: The snapshot's index in ``lesson.snapshots``.
        chapter: "initial", "training" or "final". Earlier snapshots are ghosted only in "training";
            "final" adds the scene's note and marks the final model on the metric strip.
        caption: The caption.
        residual_grow: How far the error marks have grown in, 0 to 1.
        scene: The project's scene.
    """
    stage.show_workspace()
    snap = lesson.snapshots[index]
    _header(stage, lesson, scene, chapter, snap.label)
    scene.draw_state(stage.main_facets(scene.facets), lesson, index,
                     ghosts=GHOSTS if chapter == "training" else 0, grow=residual_grow,
                     note=scene.final_note(lesson) if chapter == "final" else None)
    diagram = _make_room_for_diagram(stage, lesson)
    if diagram is not None:
        diagram.draw_training(stage.diagram, snap)
    stage.panel.clear()
    lesson.panel.draw_training(stage.panel, snap)
    scene.draw_metric(stage.metric, lesson, index, mark_final=chapter == "final")
    stage.set_caption(caption)


def paint_exam(stage: Stage, lesson: Any, exam: Any, t: float, scene: Scene) -> None:
    """Paints the final exam: every test example scored in turn.

    Args:
        stage: The canvas.
        lesson: The lesson.
        exam: The scored test set, from ``Extras``.
        t: The chapter's progress, 0 to 1.
        scene: The project's scene.
    """
    stage.show_workspace()
    _header(stage, lesson, scene, "exam", scene.exam_label(lesson, exam))
    stage.share_panel(False)
    stage.panel.clear()
    scene.draw_exam(stage.main_facets(scene.facets), stage.panel, stage.metric, lesson, exam, t)
    stage.set_caption(scene.exam_caption(lesson, exam))


def paint_surface(stage: Stage, lesson: Any, surface: Surface, t: float, scene: Scene) -> None:
    """The model as a surface over two inputs, the camera swinging as t runs from 0 to 1.

    Args:
        stage: The canvas.
        lesson: The lesson.
        surface: The 3-D view, from ``Extras``.
        t: The chapter's progress, 0 to 1.
        scene: The project's scene.
    """
    stage.show_workspace()
    _header(stage, lesson, scene, "surface", surface.title)
    stage.share_panel(False)
    draw_surface(stage.main_3d(), surface, t)
    draw_notes(stage.panel, "Reading the 3-D view", surface.reading, t)
    draw_notes(stage.metric, "How this model scores", surface.scores, t)
    stage.set_caption(surface.caption)


def paint_inference(stage: Stage, lesson: Any, t: float, scene: Scene) -> None:
    """Paints the prediction on the unseen example, with the worked prediction appearing line by line.

    The main plots, the diagram and the panel show the model at work; the metric strip holds the worked
    prediction.

    Args:
        stage: The canvas.
        lesson: The lesson.
        t: The chapter's progress, 0 to 1.
        scene: The project's scene.
    """
    stage.show_workspace()
    _header(stage, lesson, scene, "inference", scene.inference_label(lesson))
    scene.draw_inference(stage.main_facets(scene.facets), lesson, t)
    diagram = _make_room_for_diagram(stage, lesson)
    if diagram is not None:
        diagram.draw_inference(stage.diagram, lesson.data.query, phase(t, 0.3, 0.85))
    stage.panel.clear()
    lesson.panel.draw_inference(stage.panel, lesson.data.query, phase(t, 0.3, 0.85))
    explanation = scene.explain(lesson)
    shown = math.ceil(len(explanation.lines) * phase(t, 0.35, 0.85))
    plots.draw_explanation(stage.metric, explanation, shown, "The prediction, worked through")
    stage.set_caption(lesson.story.inference)


# ---- chapter players ------------------------------------------------------------------------------

def play_title(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the title card: it fades in over the first fifth, then holds.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    out.animate(timing.frames("title"), lambda t: paint_title(out.stage, lesson, t), moving=0.2)


def play_data(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the data chapter: it moves for the first 70%, then holds.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    out.animate(timing.frames("data"), lambda t: paint_data(out.stage, lesson, t, scene), moving=0.7)


def play_initial(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the first guess: the untrained model, its error marks growing in over the first 35%.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    out.animate(timing.frames("initial"), lambda t: paint_training_state(
        out.stage, lesson, 0, "initial", lesson.story.initial, residual_grow=phase(t, 0.0, 1.0),
        scene=scene), moving=0.35)


def play_training(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays training: each snapshot held for its share of the chapter.

    The shares come from ``lesson.frame_weights``, or are equal when it has none.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    weights = lesson.frame_weights or (1.0,) * len(lesson.snapshots)
    for index, frames in enumerate(allocate_frames(weights, timing.frames("training"))):
        paint_training_state(out.stage, lesson, index, "training", lesson.story.training, scene=scene)
        out.shoot(frames)


def play_final(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the final model: one still picture, held for the whole chapter.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    paint_training_state(out.stage, lesson, lesson.final_index, "final", lesson.story.final, scene=scene)
    out.shoot(timing.frames("final"))


def play_inference(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the prediction on the unseen example: it moves for 90% of the chapter.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    out.animate(timing.frames("inference"), lambda t: paint_inference(out.stage, lesson, t, scene),
                moving=0.9)


def play_exam(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the final exam: it moves for the first 85%, then holds.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    out.animate(timing.frames("exam"), lambda t: paint_exam(out.stage, lesson, extras.exam, t, scene),
                moving=0.85)


def play_surface(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the 3-D view: the camera swings for the whole chapter.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    out.animate(timing.frames("surface"), lambda t: paint_surface(out.stage, lesson, extras.surface, t, scene))


def play_features(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the more-features card: it moves for the first 60%, then holds.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    out.animate(timing.frames("features"), lambda t: paint_features(out.stage, lesson, extras.comparison, t),
                moving=0.6)


def play_takeaway(out: Recorder, lesson: Any, timing: Timing, scene: Scene, extras: Extras) -> None:
    """Plays the takeaway card: it fades in over the first 15%, then holds.

    Args:
        out: The recorder writing the video.
        lesson: The lesson.
        timing: The video's frame rate and duration scale.
        scene: The project's scene.
        extras: What was computed before the first frame.
    """
    out.animate(timing.frames("takeaway"),
                lambda t: paint_takeaway(out.stage, lesson, t, scene, extras.comparison), moving=0.15)


CHAPTER_PLAYERS = (play_title, play_data, play_initial, play_training, play_final, play_inference,
                   play_exam, play_surface, play_features, play_takeaway)


def render_lesson(lesson: Any, path: Path, *, scene: Scene, timing: Timing = Timing(),
                  dpi: float = RESOLUTIONS[DEFAULT_RESOLUTION]) -> RenderReport:
    """Render every chapter of ``lesson`` to ``path``; ``scene`` draws the task-specific parts.

    The extras (exam, comparison, 3-D view) are computed first, so a lesson that cannot be shown fails
    before ffmpeg starts.

    Args:
        lesson: The project's lesson.
        path: The MP4 file to write.
        scene: The project's scene.
        timing: The frame rate and duration scale.
        dpi: The resolution.

    Returns:
        The video's path, frame count and length.
    """
    extras = Extras(exam=scene.prepare_exam(lesson), comparison=scene.compare(lesson),
                    surface=scene.prepare_surface(lesson))
    with lesson_style():
        stage = Stage(dpi=dpi, source_note=scene.source_note)
        width, height = stage.size_px
        with Mp4Encoder(Path(path), width, height, timing.fps) as video:
            recorder = Recorder(stage, video)
            for play in CHAPTER_PLAYERS:
                play(recorder, lesson, timing, scene, extras)
    return RenderReport(path=Path(path), frames=video.frames_written,
                        seconds=video.frames_written / timing.fps)
