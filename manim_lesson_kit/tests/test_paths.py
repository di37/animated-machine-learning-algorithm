"""matplotlib paths to Manim's cubic points, and clipping to the box Agg clips to."""

import numpy as np
import pytest
from matplotlib.path import Path

from manimkit.paths import (
    CLOSEPOLY,
    CURVE3,
    CURVE4,
    LINETO,
    MOVETO,
    STOP,
    bounds,
    cubic_points,
    inside,
    outside,
)


def curves(points: np.ndarray) -> np.ndarray:
    return points.reshape(-1, 4, 2)


def test_a_line_is_one_cubic_with_handles_at_thirds():
    quads = curves(cubic_points(np.array([[0.0, 0.0], [3.0, 6.0]]), np.array([MOVETO, LINETO])))
    assert quads.shape == (1, 4, 2)
    np.testing.assert_allclose(quads[0], [[0, 0], [1, 2], [2, 4], [3, 6]])


def test_a_closed_square_ends_where_it_started():
    square = Path([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], closed=True)
    quads = curves(cubic_points(square.vertices, square.codes))
    assert len(quads) == 4
    np.testing.assert_allclose(quads[-1, -1], [0, 0])
    np.testing.assert_allclose(quads[:, 0], [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_an_open_triangle_closes_with_a_line_back_to_its_start():
    codes = np.array([MOVETO, LINETO, LINETO, CLOSEPOLY, STOP])
    vertices = np.array([[0, 0], [2, 0], [1, 2], [9, 9], [0, 0]], dtype=float)  # the close's vertex is a dummy
    quads = curves(cubic_points(vertices, codes))
    assert len(quads) == 3
    np.testing.assert_allclose(quads[-1, [0, -1]], [[1, 2], [0, 0]])


def loop_points(vertices, codes):
    """The same conversion, one segment at a time, to check the quick numpy pass against."""
    out, start, current, i = [], None, None, 0
    while i < len(codes):
        code = codes[i]
        if code == MOVETO:
            start = current = vertices[i]
            i += 1
        elif code == LINETO:
            out.append([current, current + (vertices[i] - current) / 3, current + 2 * (vertices[i] - current) / 3,
                        vertices[i]])
            current = vertices[i]
            i += 1
        elif code == CURVE4:
            out.append([current, vertices[i], vertices[i + 1], vertices[i + 2]])
            current = vertices[i + 2]
            i += 3
        elif code == CLOSEPOLY:
            if not np.array_equal(current, start):
                out.append([current, current + (start - current) / 3, current + 2 * (start - current) / 3, start])
            current = start
            i += 1
        else:
            break
    return np.array(out, dtype=float).reshape(-1, 2)


@pytest.mark.parametrize("path", [Path.circle((3, 4), 2), Path.unit_rectangle(), Path.unit_regular_star(5),
                                  Path([[0, 0], [1, 2], [3, 1], [4, 4]])], ids=["circle", "rectangle", "star", "polyline"])
def test_the_quick_pass_matches_the_segment_by_segment_conversion(path):
    np.testing.assert_allclose(cubic_points(path.vertices, path.codes if path.codes is not None else
                                            np.array([MOVETO] + [LINETO] * (len(path.vertices) - 1))),
                               loop_points(path.vertices, path.codes if path.codes is not None else
                                           np.array([MOVETO] + [LINETO] * (len(path.vertices) - 1))))


def test_quadratic_curves_and_several_pieces_take_the_general_conversion():
    vertices = np.array([[0, 0], [1, 2], [2, 0], [5, 5], [6, 5]], dtype=float)
    codes = np.array([MOVETO, CURVE3, CURVE3, MOVETO, LINETO])
    quads = curves(cubic_points(vertices, codes))
    assert len(quads) == 2
    np.testing.assert_allclose(quads[0], [[0, 0], [2 / 3, 4 / 3], [4 / 3, 4 / 3], [2, 0]])  # quadratic as cubic
    np.testing.assert_allclose(quads[1, [0, -1]], [[5, 5], [6, 5]])


def test_a_move_alone_draws_nothing():
    assert cubic_points(np.array([[1.0, 1.0]]), np.array([MOVETO])).shape == (0, 2)


def test_bounds_leave_out_the_dummy_vertices_of_a_close_and_a_stop():
    vertices = np.array([[1, 1], [3, 1], [3, 2], [99, 99], [0, 0]], dtype=float)
    codes = np.array([MOVETO, LINETO, LINETO, CLOSEPOLY, STOP])
    assert bounds(vertices, codes) == (1.0, 1.0, 3.0, 2.0)
    assert bounds(np.empty((0, 2)), np.empty(0)) == (0.0, 0.0, 0.0, 0.0)


def test_inside_and_outside_count_half_a_strokes_width():
    clip = (0, 0, 3, 3)
    assert inside((1, 1, 2, 2), clip) and not inside((1, 1, 4, 2), clip)
    assert not inside((1, 1, 2.9, 2), clip, reach=0.5)  # its stroke pokes out past the right edge
    assert outside((4, 4, 5, 5), clip) and not outside((3.2, 1, 4, 2), clip, reach=0.5)
