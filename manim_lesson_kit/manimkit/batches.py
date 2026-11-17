"""A matplotlib collection's items as leaves in one pass of numpy, instead of one draw call per item.

matplotlib's generic ``draw_path_collection`` walks a scatter's thousands of markers (or a 3-D sheet's
tiles) one Python call at a time. This does what it does with arrays: every item's path, transform,
offset, colours, line width, dashes and antialiasing are cycled exactly as matplotlib cycles them
(item ``i`` takes entry ``i mod n`` of each list), an item outside the clip box is dropped as Agg's
clipping would drop it, and one that reaches past it keeps the box, for the camera to clip it to.

A collection this cannot take (a non-affine transform, paths of different shapes) returns None and
goes through the generic path.
"""

import hashlib

import numpy as np
from matplotlib.colors import to_rgba_array
from matplotlib.path import Path

from manimkit.leaves import PRECISION, PathLeaf
from manimkit.paths import inside

INVISIBLE = 1.0 / 512


def _cycled(values, count: int, index: np.ndarray):
    """Entry ``index mod count`` of ``values`` for every index, as matplotlib cycles them (None if empty)."""
    return None if count == 0 else values[index % count]


def _colours(values, alpha: float | None) -> np.ndarray:
    """Colours as an (n, 4) RGBA array, with a forced alpha applied, rounded for stable signatures."""
    colours = to_rgba_array(values) if len(values) else np.empty((0, 4))
    if alpha is not None and len(colours):
        colours = colours.copy()
        colours[:, 3] = alpha
    return np.round(colours, 4)


def _same_shape(paths) -> bool:
    """Whether every path has the same number of vertices and the same codes, so they stack as one array."""
    first = paths[0]
    return all(len(p.vertices) == len(first.vertices) and (
        (p.codes is None and first.codes is None) or
        (p.codes is not None and first.codes is not None and np.array_equal(p.codes, first.codes))) for p in paths)


def _codes(path: Path) -> np.ndarray:
    """A path's codes; a path without them is a move followed by lines."""
    if path.codes is not None:
        return np.asarray(path.codes, dtype=np.uint8)
    codes = np.full(len(path.vertices), Path.LINETO, dtype=np.uint8)
    if len(codes):
        codes[0] = Path.MOVETO
    return codes


def _digest(array: np.ndarray) -> bytes:
    """A short hash of an array rounded to ``PRECISION`` decimals."""
    return hashlib.blake2b(np.round(array, PRECISION).tobytes(), digest_size=12).digest()


def collection_leaves(gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors,
                      linewidths, linestyles, antialiaseds, prefix: tuple, first: int, px_per_pt: float):
    """A collection's items as leaves, computed for all items at once.

    The arguments are the ones matplotlib passes to ``RendererBase.draw_path_collection``; each item's
    vertices are ``master_transform`` after its own transform, moved by its transformed offset.

    Args:
        gc: The graphics context (clip box, forced alpha, cap and join style).
        master_transform: The transform applied to every item.
        paths: The collection's paths; item ``i`` uses path ``i mod len(paths)``.
        all_transforms: Per-item 3 × 3 transforms (for example a scatter's sizes), cycled likewise.
        offsets: Per-item offsets, cycled likewise.
        offset_trans: The transform taking offsets to display pixels.
        facecolors: Per-item fill colours.
        edgecolors: Per-item stroke colours.
        linewidths: Per-item line widths, in points.
        linestyles: Per-item ``(offset, dashes)`` pairs.
        antialiaseds: Per-item antialiasing flags.
        prefix: The key of the group the items are drawn in.
        first: The index the first item's key takes in its group.
        px_per_pt: Pixels per point at the figure's dpi.

    Returns:
        ``(leaves, items)``: the leaves, keyed ``prefix + (first + item,)``, and how many items the
        collection has, kept or not; or None when the collection needs matplotlib's generic walk.
    """
    paths = list(paths)
    if not paths or not master_transform.is_affine or not _same_shape(paths):
        return None
    if gc.get_clip_path()[0] is not None or gc.get_hatch() is not None:
        return None
    alpha = gc.get_alpha() if gc.get_forced_alpha() else None
    faces, edges = _colours(facecolors, alpha), _colours(edgecolors, alpha)
    if len(faces) == 0 and len(edges) == 0:
        return [], 0
    n_paths, n_transforms, n_offsets = len(paths), len(all_transforms), len(offsets)
    n_ids = max(n_paths, n_transforms)
    count = max(n_ids, n_offsets)
    item = np.arange(count)
    path_index = (item % n_ids) % n_paths
    stacked = np.stack([np.asarray(p.vertices, dtype=float) for p in paths])  # (paths, m, 2)
    if not np.all(np.isfinite(stacked)):
        return None
    homogeneous = np.concatenate([stacked, np.ones((*stacked.shape[:2], 1))], axis=2)  # (paths, m, 3)
    master = master_transform.get_matrix()
    if n_transforms:
        per_item = np.asarray(all_transforms, dtype=float)[(item % n_ids) % n_transforms]  # (count, 3, 3)
        matrices = np.einsum("ij,njk->nik", master, per_item)
    else:
        matrices = np.broadcast_to(master, (count, 3, 3))
    vertices = np.einsum("nmk,njk->nmj", homogeneous[path_index], matrices)[:, :, :2]
    moved = offset_trans.transform(np.asarray(offsets, dtype=float)) if n_offsets else np.zeros((1, 2))
    shift = moved[item % n_offsets] if n_offsets else np.zeros((count, 2))
    vertices = vertices + shift[:, None, :]
    leaves = _leaves(gc, prefix, first, paths, path_index, vertices, faces, edges, linewidths, linestyles,
                     antialiaseds, matrices if n_transforms else None, master, px_per_pt)
    return leaves, count


def marker_leaves(gc, marker_path, marker_trans, path, trans, rgbFace, prefix: tuple, first: int, px_per_pt: float,
                  height: float):
    """One leaf per marker of a line, or of a uniform scatter (matplotlib sends those as markers).

    The arguments are the ones matplotlib passes to ``RendererBase.draw_markers``.

    Args:
        gc: The graphics context (stroke colour and width, clip box, forced alpha).
        marker_path: The marker's shape.
        marker_trans: The transform sizing the marker, in pixels about its centre.
        path: The positions to draw a marker at (its vertices).
        trans: The transform taking those positions to display pixels.
        rgbFace: The markers' fill colour, or None.
        prefix: The key of the group the markers are drawn in.
        first: The index the first marker's key takes in its group.
        px_per_pt: Pixels per point at the figure's dpi.
        height: The canvas's height, in pixels (Agg centres markers counting down from the top).

    Returns:
        ``(leaves, items)``: the leaves, centred where Agg centres them, and how many markers there are;
        or None when the markers need matplotlib's generic walk.
    """
    if not marker_trans.is_affine or gc.get_clip_path()[0] is not None or gc.get_hatch() is not None:
        return None
    positions = trans.transform(np.asarray(path.vertices, dtype=float)) if len(path.vertices) else np.empty((0, 2))
    if path.codes is not None:  # a close or stop carries no marker
        positions = positions[(path.codes != Path.CLOSEPOLY) & (path.codes != Path.STOP)]
    positions = agg_marker_centres(positions, height)
    alpha = gc.get_alpha() if gc.get_forced_alpha() else None
    faces = _colours([rgbFace], alpha) if rgbFace is not None else np.empty((0, 4))
    edges = _colours([gc.get_rgb()], alpha) if gc.get_linewidth() > 0 else np.empty((0, 4))
    marker = marker_trans.transform(np.asarray(marker_path.vertices, dtype=float))
    vertices = marker[None, :, :] + positions[:, None, :]
    dashes = gc.get_dashes()
    leaves = _leaves(gc, prefix, first, [Path(marker, marker_path.codes)], np.zeros(len(positions), dtype=int),
                     vertices, faces, edges, [gc.get_linewidth()], [dashes] if dashes[1] else [],
                     [gc.get_antialiased()], None, np.eye(3), px_per_pt)
    return leaves, len(positions)


def agg_marker_centres(positions: np.ndarray, height: float) -> np.ndarray:
    """Where Agg centres markers: on a pixel centre, counting down from the top, floor(v + 0.5) + 0.5.

    Args:
        positions: Marker positions in display pixels (counted up from the bottom), shape (n, 2).
        height: The canvas's height, in pixels.

    Returns:
        The positions Agg draws them at, in the same coordinates, shape (n, 2).
    """
    if len(positions) == 0:
        return positions
    from_top = np.column_stack([positions[:, 0], height - positions[:, 1]])
    centred = np.floor(from_top + 0.5) + 0.5
    return np.column_stack([centred[:, 0], height - centred[:, 1]])


def _leaves(gc, prefix, first, paths, path_index, vertices, faces, edges, linewidths, linestyles, antialiaseds,
            matrices, master, px_per_pt: float) -> list[PathLeaf]:
    """The leaves of a batch whose vertices are worked out: dropped if outside, clipped if past the box.

    Args:
        gc: The graphics context.
        prefix: The key of the group the items are drawn in.
        first: The index the first item's key takes.
        paths: The distinct paths (items index into them through ``path_index``).
        path_index: Each item's path.
        vertices: Each item's vertices in display pixels, shape (count, m, 2).
        faces: Fill colours, cycled per item.
        edges: Stroke colours, cycled per item.
        linewidths: Line widths in points, cycled per item.
        linestyles: ``(offset, dashes)`` pairs, cycled per item.
        antialiaseds: Antialiasing flags, cycled per item.
        matrices: Each item's full transform, or None when every item shares ``master``.
        master: The collection's transform.
        px_per_pt: Pixels per point at the figure's dpi.

    Returns:
        One ``PathLeaf`` per visible item, in drawing order.
    """
    count = len(vertices)
    item = np.arange(count)
    codes = _codes(paths[0])
    box = gc.get_clip_rectangle()
    clip = None if box is None else tuple(float(v) for v in box.extents)
    placed = vertices[:, (codes != Path.CLOSEPOLY) & (codes != Path.STOP), :]  # a close's vertex is a dummy
    lows, highs = placed.min(axis=1), placed.max(axis=1)
    finite = np.all(np.isfinite(lows), axis=1)
    if clip is not None:
        outside = (highs[:, 0] < clip[0]) | (highs[:, 1] < clip[1]) | (lows[:, 0] > clip[2]) | (lows[:, 1] > clip[3])
    else:
        outside = np.zeros(count, dtype=bool)
    face = _cycled(faces, len(faces), item)
    edge = _cycled(edges, len(edges), item)
    widths = np.asarray(linewidths, dtype=float) if len(linewidths) else np.array([gc.get_linewidth()])
    width = widths[item % len(widths)] if len(edges) else np.zeros(count)
    shape_ids = _shape_ids(paths, path_index, vertices, matrices, master)
    cap, join = gc.get_capstyle(), gc.get_joinstyle()
    leaves = []
    for i in np.flatnonzero(finite & ~outside):
        fill = None if face is None or face[i, 3] < INVISIBLE else tuple(face[i])
        lw = float(width[i])
        stroke = None if edge is None or edge[i, 3] < INVISIBLE or lw <= 0 else tuple(edge[i])
        if fill is None and stroke is None:
            continue
        dashes = linestyles[i % len(linestyles)][1] if len(linestyles) else None
        reach = lw * px_per_pt / 2 if stroke is not None else 0.0
        box = (lows[i, 0], lows[i, 1], highs[i, 0], highs[i, 1])
        leaves.append(PathLeaf(
            (*prefix, first + int(i)), vertices[i], codes, fill=fill, stroke=stroke, linewidth=lw,
            dashes=tuple(float(d) for d in dashes) if dashes else None, cap=cap, join=join,
            clip=clip if clip is not None and not inside(box, clip, reach) else None,
            antialiased=bool(antialiaseds[i % len(antialiaseds)]) if len(antialiaseds) else True,
            shape_id=shape_ids[i]))
    return leaves


def _shape_ids(paths, path_index, vertices, matrices, master) -> list:
    """Each item's geometry in a few bytes, so equal geometry is found without hashing every vertex.

    An id is the item's path, the collection's transform and the item's, and where the item is put.

    Args:
        paths: The distinct paths.
        path_index: Each item's path.
        vertices: Each item's vertices in display pixels.
        matrices: Each item's transform, or None when they all share ``master``.
        master: The collection's transform.

    Returns:
        One hashable id per item.
    """
    master_digest = _digest(np.asarray(master, dtype=float))
    path_digests = [_digest(np.asarray(p.vertices, dtype=float)) + master_digest for p in paths]
    corners = np.round(vertices[:, 0, :], PRECISION)
    scales = (np.round(matrices[:, :2, :2].reshape(len(vertices), -1), 4) if matrices is not None
              else np.zeros((len(vertices), 1)))
    return [(path_digests[p], corner.tobytes(), scale.tobytes())
            for p, corner, scale in zip(path_index, corners, scales)]
