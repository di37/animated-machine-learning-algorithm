"""matplotlib paths as Manim's cubic Bézier points, and the boxes they are clipped to.

A matplotlib path is a list of vertices with a code for each: move to, line to, a quadratic curve
(``CURVE3``, one control point), a cubic curve (``CURVE4``, two control points), close, or stop.
Manim draws one flat list of cubic curves, four points each (anchor, handle, handle, anchor), and
starts a new piece wherever one curve does not end where the next begins. A straight line becomes a
cubic with its handles at one and two thirds of the way along; a quadratic becomes the cubic that
traces the same curve.

A path that reaches past its clip box keeps the box, and the camera clips it there, as Agg's
rasterizer does.
"""

import numpy as np
from matplotlib.path import Path

STOP, MOVETO, LINETO, CURVE3, CURVE4, CLOSEPOLY = (Path.STOP, Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE4,
                                                   Path.CLOSEPOLY)
THIRD = 1.0 / 3.0


def _line(start: np.ndarray, end: np.ndarray) -> list[np.ndarray]:
    """A straight line as a cubic: handles a third and two thirds of the way along."""
    return [start, start + (end - start) * THIRD, start + (end - start) * 2 * THIRD, end]


def _quadratic(start: np.ndarray, control: np.ndarray, end: np.ndarray) -> list[np.ndarray]:
    """A quadratic Bézier as the cubic tracing the same curve (degree elevation)."""
    return [start, start + 2 * THIRD * (control - start), end + 2 * THIRD * (control - end), end]


def _single_piece(vertices: np.ndarray, codes: np.ndarray) -> np.ndarray | None:
    """Manim points for one move then lines and cubics (perhaps closed), in one pass of numpy.

    Polygons, polylines and circles are all like this, and they are most of what a frame draws, so
    they skip the segment-by-segment loop.

    Args:
        vertices: The path's vertices, shape (n, 2).
        codes: The path's codes, one per vertex.

    Returns:
        Manim points, shape (4k, 2); or None for any other path (several pieces, quadratic curves).
    """
    count = len(codes)
    while count and codes[count - 1] == STOP:
        count -= 1
    closed = count > 1 and codes[count - 1] == CLOSEPOLY
    body = codes[1:count - 1] if closed else codes[1:count]
    if count < 2 or codes[0] != MOVETO or len(body) == 0 or not np.all((body == LINETO) | (body == CURVE4)):
        return None
    curve = body == CURVE4
    in_curve = np.cumsum(curve)
    ends = np.flatnonzero((body == LINETO) | (curve & (in_curve % 3 == 0))) + 1  # vertex index ending each piece
    if curve.sum() % 3:
        return None
    previous = np.concatenate([[0], ends[:-1]])
    starts, finishes = vertices[previous], vertices[ends]
    is_line = codes[ends] == LINETO
    first = np.where(is_line[:, None], starts + (finishes - starts) * THIRD, vertices[np.maximum(ends - 2, 0)])
    second = np.where(is_line[:, None], starts + (finishes - starts) * 2 * THIRD, vertices[np.maximum(ends - 1, 0)])
    quads = np.stack([starts, first, second, finishes], axis=1)
    if closed and not np.array_equal(finishes[-1], vertices[0]):
        back = np.array(_line(finishes[-1], vertices[0]))
        quads = np.concatenate([quads, back[None]], axis=0)
    return quads.reshape(-1, 2)


def cubic_points(vertices: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """A cleaned matplotlib path as Manim's cubic Bézier points.

    A subpath with no segments draws nothing and is dropped; a close adds the line back to its start.

    Args:
        vertices: The path's vertices, shape (n, 2).
        codes: The path's codes, one per vertex (``Path.MOVETO``, ``Path.LINETO`` and so on).

    Returns:
        Manim points, shape (4k, 2): four per cubic curve.

    Example:
        >>> cubic_points(np.array([[0.0, 0.0], [3.0, 6.0]]), np.array([MOVETO, LINETO])).tolist()
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    """
    vertices = np.asarray(vertices, dtype=float)
    codes = np.asarray(codes)
    quick = _single_piece(vertices, codes)
    if quick is not None:
        return quick
    curves: list[np.ndarray] = []
    start = current = None
    i, count = 0, len(codes)
    while i < count:
        code = codes[i]
        if code == MOVETO:
            start = current = vertices[i]
            i += 1
        elif code == LINETO:
            curves.extend(_line(current, vertices[i]))
            current = vertices[i]
            i += 1
        elif code == CURVE3:
            curves.extend(_quadratic(current, vertices[i], vertices[i + 1]))
            current = vertices[i + 1]
            i += 2
        elif code == CURVE4:
            curves.extend([current, vertices[i], vertices[i + 1], vertices[i + 2]])
            current = vertices[i + 2]
            i += 3
        elif code == CLOSEPOLY:
            if start is not None and current is not None and not np.array_equal(current, start):
                curves.extend(_line(current, start))
            current = start
            i += 1
        else:  # STOP
            break
    return np.array(curves, dtype=float).reshape(-1, 2)


def bounds(vertices: np.ndarray, codes: np.ndarray) -> tuple[float, float, float, float]:
    """The box around every vertex a path draws through.

    A close's or a stop's vertex is a placeholder that nothing is drawn to, so it is left out.

    Args:
        vertices: The path's vertices, shape (n, 2).
        codes: The path's codes, one per vertex.

    Returns:
        ``(x0, y0, x1, y1)``, or all zeros for a path that draws nothing.
    """
    drawn = vertices[(codes != STOP) & (codes != CLOSEPOLY)] if len(codes) else vertices
    if len(drawn) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(drawn[:, 0].min()), float(drawn[:, 1].min()), float(drawn[:, 0].max()), float(drawn[:, 1].max())


def inside(box: tuple[float, float, float, float], clip: tuple[float, float, float, float], reach: float = 0.0) -> bool:
    """Whether a box lies within a clip box, once grown by ``reach`` on every side.

    Args:
        box: ``(x0, y0, x1, y1)`` of what is drawn.
        clip: ``(x0, y0, x1, y1)`` of the clip box.
        reach: How far the drawing spills past the box (half a stroke's width).

    Returns:
        True if nothing drawn reaches past the clip box.
    """
    return (box[0] - reach >= clip[0] and box[1] - reach >= clip[1] and box[2] + reach <= clip[2]
            and box[3] + reach <= clip[3])


def outside(box: tuple[float, float, float, float], clip: tuple[float, float, float, float], reach: float = 0.0) -> bool:
    """Whether a box, grown by ``reach`` on every side, misses a clip box altogether.

    Args:
        box: ``(x0, y0, x1, y1)`` of what is drawn.
        clip: ``(x0, y0, x1, y1)`` of the clip box.
        reach: How far the drawing spills past the box (half a stroke's width).

    Returns:
        True if nothing drawn falls inside the clip box, so nothing would show.
    """
    return (box[2] + reach < clip[0] or box[3] + reach < clip[1] or box[0] - reach > clip[2]
            or box[1] - reach > clip[3])
