"""The camera draws hard-edged shapes without antialiasing and keeps each still shape's Cairo path."""

import numpy as np
from manim import Circle, Square

from manimkit.camera import ALIASED, LessonCamera
from tests.test_leaves import square


def frame_of(player, *shapes) -> np.ndarray:
    player.scene.mobjects = list(shapes)
    player.scene.renderer.update_frame(player.scene)
    return player.scene.renderer.get_frame()[..., :3].astype(int)


def test_the_scene_uses_the_lesson_camera(player):
    assert isinstance(player.scene.renderer.camera, LessonCamera)


def test_a_hard_edged_shape_has_no_partly_covered_pixels(player, small_frame):
    shape = Circle(radius=0.5).set_fill("#000000", opacity=1).set_stroke(width=0)
    smooth = frame_of(player, shape)
    setattr(shape, ALIASED, True)
    hard = frame_of(player, shape)
    assert len(np.unique(hard[..., 0])) == 2  # black and the white background, nothing between
    assert len(np.unique(smooth[..., 0])) > 2


def test_a_shapes_path_is_kept_until_its_points_are_replaced(player, small_frame):
    shape = square(10, 10).build(small_frame)
    first = frame_of(player, shape)
    camera = player.scene.renderer.camera
    cached = camera._paths[shape][1]
    assert np.array_equal(frame_of(player, shape), first) and camera._paths[shape][1] is cached
    shape.points = shape.points + np.array([0.5, 0.0, 0.0])  # a new array, as Manim's animations assign
    moved = frame_of(player, shape)
    assert camera._paths[shape][1] is not cached and not np.array_equal(moved, first)


def test_shapes_with_gradients_go_through_manims_own_drawing(player):
    shape = Square(side_length=1).set_fill(["#ff0000", "#0000ff"], opacity=1).set_stroke(width=0)
    pixels = frame_of(player, shape)
    row = pixels[pixels.shape[0] // 2]
    painted = row[row.sum(axis=1) < 3 * 255]
    assert painted[0, 0] != painted[-1, 0]  # red on one side, blue on the other
    assert shape not in player.scene.renderer.camera._paths


def test_colours_come_out_in_the_right_channels(player, small_frame):
    shape = square(70, 40, fill=(1.0, 0.0, 0.0, 1.0)).build(small_frame)  # pixels 70-80 across, 40-50 up
    pixels = frame_of(player, shape)
    assert tuple(pixels[90 - 45, 75]) == (255, 0, 0)
