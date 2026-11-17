import shutil

import pytest
from matplotlib.text import Text

from animkit.lessons.director import paint_inference, paint_training_state, render_lesson
from animkit.lessons.encoder import probe
from animkit.lessons.stage import Stage, lesson_style
from tests.lessons.conftest import TINY_DPI, TINY_TIMING
from titanic_anim.lessons.scene import CLASSIFICATION
from titanic_anim.lessons.stories import build_lesson, lesson_keys

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def figure_texts(stage: Stage) -> str:
    return "\n".join(text.get_text() for text in stage.fig.findobj(match=Text))


@needs_ffmpeg
@pytest.mark.parametrize("key", lesson_keys())
def test_every_classifier_lesson_renders_a_playable_video(tmp_path, passenger_data, key):
    lesson = build_lesson(key, passenger_data)
    report = render_lesson(lesson, tmp_path / f"{key}.mp4", timing=TINY_TIMING, dpi=TINY_DPI,
                           scene=CLASSIFICATION)
    info = probe(report.path)
    assert info.codec == "h264" and (info.width, info.height) == (384, 216)
    assert info.frames == report.frames >= TINY_TIMING.total_frames


@pytest.mark.parametrize("key", ["logistic_regression", "linear_svm", "knn"])
def test_finished_inference_frame_shows_the_score_and_the_truth(passenger_data, key):
    lesson = build_lesson(key, passenger_data)
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_inference(stage, lesson, 1.0, CLASSIFICATION)
    texts = figure_texts(stage)
    score = float(lesson.score(lesson.data.query.features[None, :])[0])
    assert CLASSIFICATION.describe_score(score, lesson.kind) in texts
    assert ("actually survived" in texts) or ("actually did not survive" in texts)
    assert "✓" in texts


def test_training_frames_draw_one_map_per_sex(passenger_data):
    lesson = build_lesson("decision_tree", passenger_data)
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_training_state(stage, lesson, 2, "training", lesson.story.training, scene=CLASSIFICATION)
        stage.render()
    titles = [ax.get_title(loc="left") for ax in stage.main_facets(2)]  # lesson style puts titles left
    assert titles[0].startswith("Women") and titles[1].startswith("Men")


def test_score_maps_are_labelled_honestly(passenger_data):
    assert "P(survive)" in CLASSIFICATION.describe_score(0.73, "probability")
    assert "score" in CLASSIFICATION.describe_score(0.73, "margin")
    assert "P(" not in CLASSIFICATION.describe_score(0.73, "margin")


def test_draw_boundary_reports_when_no_boundary_crosses_a_map(passenger_data):
    import matplotlib.pyplot as plt
    import numpy as np

    from titanic_anim.lessons import maps

    _, ax = plt.subplots()
    flat = np.full(passenger_data.surface_shape[1:], 0.3)
    assert maps.draw_boundary(ax, passenger_data, flat, "probability") is False
    split = np.tile(np.linspace(0.0, 1.0, passenger_data.surface_shape[2]), (passenger_data.surface_shape[1], 1))
    assert maps.draw_boundary(ax, passenger_data, split, "probability") is True


def test_a_map_without_a_boundary_says_so(passenger_data):
    lesson = build_lesson("decision_tree", passenger_data)  # frame 0: one leaf, the same score everywhere
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_training_state(stage, lesson, 0, "initial", lesson.story.initial, scene=CLASSIFICATION)
    assert figure_texts(stage).count("no boundary on this map") == 2


def test_metric_axis_never_clips_an_accuracy(passenger_data):
    from titanic_anim.lessons.scene import METRIC_FLOOR, metric_floor

    for key in lesson_keys():
        lesson = build_lesson(key, passenger_data)
        lowest = min(min(s.train_accuracy, s.val_accuracy) for s in lesson.snapshots)
        assert metric_floor(lesson) <= min(lowest, METRIC_FLOOR)
