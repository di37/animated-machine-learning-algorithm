"""animkit's lesson director with the regression scene as the default, so house code can call it bare.

The chapters, the Scene protocol and the MP4 pipeline live in animkit.lessons.director. A director plays
a lesson's chapters (title, data, training, prediction, exam, ...) into frames; everything specific to
the task, here a regression curve over house prices, it asks a ``Scene`` to draw. Each function below
wraps the animkit function of the same name and passes ``REGRESSION``, this project's scene, unless the
caller gives another.
"""

from pathlib import Path
from typing import Any

from animkit.lessons import director as kit_director
from animkit.lessons.config import DEFAULT_RESOLUTION, RESOLUTIONS
from animkit.lessons.director import RenderReport, Scene
from animkit.lessons.stage import Stage
from animkit.lessons.timing import Timing
from house_anim.lessons.regression_scene import REGRESSION

__all__ = ["RenderReport", "Scene", "paint_data", "paint_inference", "paint_takeaway", "paint_training_state",
           "render_lesson"]


def paint_data(stage: Stage, lesson: Any, t: float, scene: Scene = REGRESSION) -> None:
    """Paints one frame of the data chapter, the training districts appearing as dots.

    Args:
        stage: The figure and its axes, to paint on.
        lesson: The lesson being played.
        t: How far through the chapter's motion, from 0 to 1.
        scene: Draws the task-specific parts; the regression scene by default.
    """
    kit_director.paint_data(stage, lesson, t, scene)


def paint_training_state(stage: Stage, lesson: Any, index: int, chapter: str, caption: str,
                         residual_grow: float = 1.0, scene: Scene = REGRESSION) -> None:
    """Paints one training snapshot: the model's curve, its residuals, the metric chart and the panel.

    Args:
        stage: The figure and its axes, to paint on.
        lesson: The lesson being played.
        index: Which of the lesson's snapshots to paint.
        chapter: The chapter's name, for the header.
        caption: The caption under the plots.
        residual_grow: How far the residual lines have grown, from 0 (none) to 1 (full length).
        scene: Draws the task-specific parts; the regression scene by default.
    """
    kit_director.paint_training_state(stage, lesson, index, chapter, caption, residual_grow, scene=scene)


def paint_inference(stage: Stage, lesson: Any, t: float, scene: Scene = REGRESSION) -> None:
    """Paints one frame of the prediction chapter, the model pricing the unseen district.

    Args:
        stage: The figure and its axes, to paint on.
        lesson: The lesson being played.
        t: How far through the chapter's motion, from 0 to 1.
        scene: Draws the task-specific parts; the regression scene by default.
    """
    kit_director.paint_inference(stage, lesson, t, scene)


def paint_takeaway(stage: Stage, lesson: Any, t: float, scene: Scene = REGRESSION) -> None:
    """Paints one frame of the closing takeaway card.

    Args:
        stage: The figure and its axes, to paint on.
        lesson: The lesson being played.
        t: How far through the card's fade-in, from 0 to 1.
        scene: Draws the task-specific parts; the regression scene by default.
    """
    kit_director.paint_takeaway(stage, lesson, t, scene)


def render_lesson(lesson: Any, path: Path, *, timing: Timing = Timing(),
                  dpi: float = RESOLUTIONS[DEFAULT_RESOLUTION], scene: Scene = REGRESSION) -> RenderReport:
    """Render every chapter of ``lesson`` to ``path``; ``scene`` draws the task-specific parts.

    Args:
        lesson: The lesson to render.
        path: The MP4 file to write.
        timing: How long each chapter lasts.
        dpi: Pixels per inch of the lesson canvas, which sets the video's resolution.
        scene: Draws the task-specific parts; the regression scene by default.

    Returns:
        The written file, its frame count and its length in seconds.
    """
    return kit_director.render_lesson(lesson, path, scene=scene, timing=timing, dpi=dpi)
