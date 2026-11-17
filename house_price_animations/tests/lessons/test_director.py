import dataclasses
import shutil

import numpy as np
import pytest
from matplotlib.text import Text

from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.encoder import probe
from animkit.lessons.stage import Stage
from house_anim.lessons.director import paint_inference, render_lesson
from house_anim.lessons.stories import build_lesson, lesson_keys
from tests.lessons.conftest import TINY_DPI, TINY_TIMING

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def figure_texts(stage: Stage) -> str:
    return "\n".join(text.get_text() for text in stage.fig.findobj(match=Text))


def test_stage_renders_1440p_by_default():
    stage = Stage()
    assert stage.size_px == (2560, 1440)
    assert len(stage.render()) == 2560 * 1440 * 4


@pytest.mark.parametrize(("name", "size"), [("1080p", (1920, 1080)), ("720p", (1280, 720)),
                                            ("preview", (384, 216))])
def test_stage_renders_every_named_resolution_at_16_by_9(name, size):
    assert Stage(dpi=RESOLUTIONS[name]).size_px == size


@needs_ffmpeg
@pytest.mark.parametrize("key", lesson_keys())
def test_every_lesson_renders_a_playable_video(tmp_path, income_data, key):
    lesson = build_lesson(key, income_data)
    report = render_lesson(lesson, tmp_path / f"{key}.mp4", timing=TINY_TIMING, dpi=TINY_DPI)
    info = probe(report.path)
    assert info.codec == "h264" and info.pix_fmt == "yuv420p"
    assert (info.width, info.height) == (384, 216)
    assert info.frames == report.frames >= TINY_TIMING.total_frames


@pytest.mark.parametrize("key", ["linear_regression", "decision_tree", "knn"])
def test_finished_inference_frame_shows_prediction_and_actual(income_data, key):
    lesson = build_lesson(key, income_data)
    stage = Stage(dpi=TINY_DPI)
    paint_inference(stage, lesson, t=1.0)
    texts = figure_texts(stage)
    query = lesson.data.query
    predicted = float(lesson.predict(np.array([query.income]))[0]) * 100.0
    assert f"${predicted:,.0f}k" in texts
    assert f"${query.price * 100.0:,.0f}k" in texts
    assert "✓" in texts


@pytest.mark.parametrize("key", ["ridge", "knn"])
def test_log_scale_metric_ticks_are_plain_numbers(income_data, key):
    from animkit.lessons.stage import lesson_style
    from house_anim.lessons.director import paint_training_state

    lesson = build_lesson(key, income_data)
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_training_state(stage, lesson, 2, "training", lesson.story.training)
        stage.render()
        labels = [tick.get_text() for tick in stage.metric.get_xticklabels()]
    assert labels and not any("mathdefault" in label or "$" in label for label in labels)


class SpyDiagram:
    """Records what the director hands a lesson's diagram."""

    def __init__(self):
        self.snapshots, self.reveals, self.visible = [], [], set()

    def draw_training(self, ax, snapshot):
        self.snapshots.append(snapshot)
        self.visible.add(ax.get_visible())

    def draw_inference(self, ax, query, t):
        self.reveals.append(t)
        self.visible.add(ax.get_visible())


@needs_ffmpeg
def test_director_feeds_the_diagram_each_state_and_a_growing_reveal(tmp_path, income_data):
    lesson = build_lesson("linear_regression", income_data)
    spy = SpyDiagram()
    render_lesson(dataclasses.replace(lesson, diagram=spy), tmp_path / "spy.mp4", timing=TINY_TIMING, dpi=TINY_DPI)
    assert spy.visible == {True}
    assert all(any(seen is snapshot for snapshot in lesson.snapshots) for seen in spy.snapshots)
    assert any(seen is lesson.final for seen in spy.snapshots)
    assert spy.reveals[0] == 0.0 and spy.reveals[-1] == 1.0 and spy.reveals == sorted(spy.reveals)
