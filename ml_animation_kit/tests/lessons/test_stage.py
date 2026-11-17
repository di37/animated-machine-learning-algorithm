"""The canvas layout: every box on the canvas between header and caption, right edges lined up, and a
header whose chapter tracker never runs into the title or past the margin."""

import numpy as np
import pytest

from animkit.lessons import stage as layout
from animkit.lessons.config import HEADER
from animkit.lessons.stage import TRACKER, Stage, lesson_style

WORKSPACE = (layout.MAIN_RECT, layout.PANEL_RECT, layout.METRIC_RECT, layout.DIAGRAM_RECT,
             layout.SHARED_PANEL_RECT, layout.SHARED_METRIC_RECT)
SIDE_COLUMN = WORKSPACE[1:]
RIGHT = layout.RIGHT_IN / layout.WIDTH_IN


def test_every_box_sits_on_the_canvas_between_the_header_and_the_caption():
    caption_top = layout.CAPTION_RECT[1] + layout.CAPTION_RECT[3]
    subtitle = layout.SUBTITLE_Y_IN / layout.HEIGHT_IN
    for left, bottom, width, height in (*WORKSPACE, layout.CAPTION_RECT):
        assert 0.0 < left < left + width < 1.0 and 0.0 < bottom < bottom + height < 1.0
    for _, bottom, _, height in WORKSPACE:
        assert caption_top < bottom and bottom + height < subtitle


def test_the_main_plot_stays_left_of_the_side_column_and_the_right_edges_line_up():
    main_left, _, main_width, _ = layout.MAIN_RECT
    assert all(main_left + main_width < left for left, *_ in SIDE_COLUMN)
    assert {round(left, 9) for left, *_ in SIDE_COLUMN} == {round(layout.PANEL_RECT[0], 9)}
    for left, _, width, _ in (*SIDE_COLUMN, layout.CAPTION_RECT):
        assert left + width == pytest.approx(RIGHT)


def test_the_tracker_ends_at_the_right_margin_clear_of_the_title_and_of_itself():
    with lesson_style():
        stage = Stage(dpi=100.0)
        stage.set_header("Logistic Regression", "subtitle", "training", "Step 1 of 40")
        stage.fig.draw_without_rendering()
    renderer = stage.fig.canvas.get_renderer()
    labels = {label for _, label in TRACKER}
    tracker = [text.get_window_extent(renderer) for text in stage.fig.texts if text.get_text() in labels]
    title = next(text for text in stage.fig.texts if text.get_text() == "Logistic Regression")
    width_px = stage.size_px[0]
    assert len(tracker) == len(TRACKER)
    assert title.get_window_extent(renderer).x1 < tracker[0].x0
    assert all(left.x1 < right.x0 for left, right in zip(tracker, tracker[1:]))
    assert tracker[-1].x1 == pytest.approx(RIGHT * width_px, abs=0.01 * width_px)  # the label-width estimate


def test_the_header_line_sits_top_right_above_the_tracker_on_every_frame():
    with lesson_style():
        stage = Stage(dpi=100.0)
        stage.set_header("Logistic Regression", "subtitle", "training", "Step 1 of 40")
        stage.fig.draw_without_rendering()
        renderer = stage.fig.canvas.get_renderer()
        header = next(text for text in stage.fig.texts if text.get_text() == HEADER)
        box = header.get_window_extent(renderer)
        labels = {label for _, label in TRACKER}
        tracker_top = max(text.get_window_extent(renderer).y1 for text in stage.fig.texts if text.get_text() in labels)
        width_px, height_px = stage.size_px
        assert box.x1 == pytest.approx(RIGHT * width_px, abs=1.0) and tracker_top < box.y0 < box.y1 < height_px
        assert header.get_zorder() > stage.card.get_zorder()  # drawn over the title and takeaway cards too
        stage.show_card()
        pixels = np.frombuffer(stage.render(), dtype=np.uint8).reshape(height_px, width_px, 4)
        rows = slice(height_px - int(box.y1), height_px - int(box.y0))
        assert pixels[rows, int(box.x0):int(box.x1), :3].min() < 100  # its dark ink shows on the card
