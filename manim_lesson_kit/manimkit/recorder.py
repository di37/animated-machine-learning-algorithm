"""Records what a matplotlib figure draws, as leaves, instead of rasterizing it.

``LeafRenderer`` is a matplotlib renderer. matplotlib lays the frame out exactly as it does for Agg
(text is measured by the figure's own Agg renderer), then hands this renderer every path, marker,
collection item, text line and image; each becomes a ``Leaf`` in drawing order. Paths are transformed,
snapped and clipped as Agg does it, so a leaf's pixels are the ones Agg would draw.

Each leaf's key is the trail of artist groups matplotlib opened around it: a group with a gid (the
lesson stage names its axes and figure texts) is known by that gid, any other by its place among its
siblings of the same kind, then the leaf by its place in its group. Artists that redraw from scratch
on every frame keep their keys, because they are drawn in the same order.

Example:
    >>> leaves, frame = record(figure)  # doctest: +SKIP
"""

import warnings
from collections import Counter

import numpy as np
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import get_hinting_flag
from matplotlib.colors import to_rgba
from matplotlib.transforms import Affine2D

from manimkit.batches import collection_leaves, marker_leaves
from manimkit.frame import Frame
from manimkit.leaves import ImageLeaf, Leaf, PathLeaf, TextLeaf
from manimkit.paths import bounds, inside, outside

Rgba = tuple[float, float, float, float]
INVISIBLE = 1.0 / 512  # a colour this transparent draws nothing
_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    """A warning about something the recorder cannot draw as Agg does, given once per run."""
    if message not in _WARNED:
        _WARNED.add(message)
        warnings.warn(f"manimkit: {message}", RuntimeWarning, stacklevel=3)


def _alpha(gc, rgba) -> Rgba | None:
    """The colour a graphics context really draws with (its forced alpha wins), or None if invisible."""
    if rgba is None:
        return None
    r, g, b, a = to_rgba(rgba)
    if gc.get_forced_alpha():
        a = gc.get_alpha()
    return None if a < INVISIBLE else (round(r, 4), round(g, 4), round(b, 4), round(a, 4))


class _Group:
    """One artist group matplotlib has opened: what it is called and how much has been drawn in it."""

    __slots__ = ("name", "label", "children", "leaves")

    def __init__(self, name: str, label):
        """Opens the group.

        Args:
            name: The kind of artist it was opened for.
            label: Its gid, or its place among its siblings of the same kind.
        """
        self.name, self.label = name, label
        self.children: Counter = Counter()
        self.leaves = 0


class LeafRenderer(RendererBase):
    """Turns one draw of a figure into leaves. Use ``record(figure)``.

    The methods without their own description are the ones matplotlib asks of every renderer
    (``RendererBase`` documents them); here they answer as the figure's Agg renderer would.

    Attributes:
        width: The canvas's width, in pixels.
        height: The canvas's height, in pixels.
        dpi: The figure's resolution.
        leaves: The leaves recorded so far, in drawing order.
    """

    def __init__(self, figure):
        """Creates a renderer the size of ``figure``'s canvas.

        Args:
            figure: The matplotlib ``Figure`` to record; its Agg renderer measures text.
        """
        super().__init__()
        self._metrics = figure.canvas.get_renderer()  # Agg, at the figure's size and dpi: text measured as on a frame
        self.width, self.height = self._metrics.width, self._metrics.height
        self.dpi = figure.dpi
        self.leaves: list[Leaf] = []
        self._stack = [_Group("root", "root")]

    # -- what matplotlib asks of a renderer ------------------------------------------------------
    def get_canvas_width_height(self):
        """See base class."""
        return self.width, self.height

    def get_text_width_height_descent(self, s, prop, ismath):
        """See base class."""
        return self._metrics.get_text_width_height_descent(s, prop, ismath)

    def points_to_pixels(self, points):
        """See base class."""
        return points * self.dpi / 72.0

    def flipy(self):
        """See base class."""
        return False

    def option_image_nocomposite(self):
        """See base class."""
        return True  # every image is its own leaf

    def option_scale_image(self):
        """See base class."""
        return False

    def open_group(self, s, gid=None):
        """Opens an artist group: leaves drawn until it closes have it in their key.

        Args:
            s: The kind of artist (``"axes"``, ``"line2d"`` and so on).
            gid: The artist's gid, if it has one; otherwise the group is known by its place among
                its siblings of the same kind.
        """
        parent = self._stack[-1]
        place = parent.children[s]
        parent.children[s] += 1
        self._stack.append(_Group(s, gid if gid else place))

    def close_group(self, s):
        """Closes the innermost artist group.

        Args:
            s: The kind of artist the group was opened for.
        """
        if len(self._stack) > 1:
            self._stack.pop()

    # -- drawing --------------------------------------------------------------------------------
    def _prefix(self) -> tuple:
        """The key of the innermost group: ``(kind, label)`` for every open group but the root."""
        return tuple((g.name, g.label) for g in self._stack[1:])

    def _key(self) -> tuple:
        """The next leaf's key in the innermost group (and moves the group's count on)."""
        group = self._stack[-1]
        index = group.leaves
        group.leaves += 1
        return (*self._prefix(), index)

    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors,
                             edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
        """Records a collection's items, all at once when numpy can take them, otherwise one by one.

        See base class for the arguments; ``batches.collection_leaves`` describes each of them.
        """
        batch = collection_leaves(gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors,
                                  edgecolors, linewidths, linestyles, antialiaseds, self._prefix(),
                                  self._stack[-1].leaves, self.points_to_pixels(1.0))
        if batch is None:  # a collection numpy cannot take in one pass: matplotlib's item-by-item walk
            super().draw_path_collection(gc, master_transform, paths, all_transforms, offsets, offset_trans,
                                         facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls,
                                         offset_position)
            return
        self._take(*batch)

    def _take(self, leaves: list[Leaf], items: int) -> None:
        """Adds a batch's leaves; the group's count moves past every item, kept or not, so keys stay put."""
        self._stack[-1].leaves += items
        self.leaves.extend(leaves)

    def _clip_box(self, gc) -> tuple[float, float, float, float] | None:
        """The graphics context's clip box in display pixels, warning once about clipping it cannot do."""
        box = gc.get_clip_rectangle()
        if gc.get_clip_path()[0] is not None:
            _warn_once("a clip path that is not a rectangle is drawn unclipped")
        if gc.get_hatch() is not None:
            _warn_once("hatching is not drawn")
        return None if box is None else tuple(float(v) for v in box.extents)

    def draw_path(self, gc, path, transform, rgbFace=None):
        """Records one path, snapped, simplified and clipped as Agg would draw it.

        A path that draws nothing (invisible, or wholly outside its clip box) is not recorded.

        Args:
            gc: The graphics context: stroke colour and width, dashes, clip box, snapping.
            path: The matplotlib ``Path``.
            transform: The transform taking it to display pixels.
            rgbFace: The fill colour, or None for an unfilled path.
        """
        fill = _alpha(gc, rgbFace)
        linewidth = gc.get_linewidth()
        stroke = _alpha(gc, gc.get_rgb()) if linewidth > 0 else None
        if fill is None and stroke is None:
            return
        clip = self._clip_box(gc)
        # Agg snaps and clips counting down from the top; so does this, then turns the result the right way up
        flip = Affine2D().scale(1.0, -1.0).translate(0.0, self.height)
        from_top = None if clip is None else (clip[0], self.height - clip[3], clip[2], self.height - clip[1])
        cleaned = path.cleaned(transform + flip, remove_nans=True, clip=from_top if fill is None else None,
                               simplify=path.should_simplify and fill is None, curves=True,
                               stroke_width=self.points_to_pixels(linewidth), snap=gc.get_snap())
        vertices, codes = cleaned.vertices.copy(), cleaned.codes
        vertices[:, 1] = self.height - vertices[:, 1]
        if len(codes) == 0:
            return
        reach = self.points_to_pixels(linewidth) / 2 if stroke is not None else 0.0
        drawn = bounds(vertices, codes)
        if clip is not None and outside(drawn, clip, reach):
            return
        offset, dashes = gc.get_dashes()
        self.leaves.append(PathLeaf(
            self._key(), vertices.astype(np.float32), codes.astype(np.uint8), fill=fill, stroke=stroke,
            linewidth=float(linewidth), dashes=tuple(float(d) for d in dashes) if dashes else None,
            cap=gc.get_capstyle(), join=gc.get_joinstyle(),
            clip=clip if clip is not None and not inside(drawn, clip, reach) else None,
            antialiased=bool(gc.get_antialiased())))

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        """Records one leaf per marker, so markers can come and go one by one.

        Args:
            gc: The graphics context.
            marker_path: The marker's shape.
            marker_trans: The transform sizing the marker.
            path: The positions to draw a marker at.
            trans: The transform taking the positions to display pixels.
            rgbFace: The markers' fill colour, or None.
        """
        batch = marker_leaves(gc, marker_path, marker_trans, path, trans, rgbFace, self._prefix(),
                              self._stack[-1].leaves, self.points_to_pixels(1.0), self.height)
        if batch is not None:
            self._take(*batch)
            return
        for vertices, code in path.iter_segments(trans, simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                self.draw_path(gc, marker_path, marker_trans + Affine2D().translate(x, y), rgbFace)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """Records one line of text, its origin where Agg puts it.

        Args:
            gc: The graphics context (its colour is the text's).
            x: The baseline origin's x, in display pixels.
            y: The baseline origin's y, in display pixels.
            s: The line's text.
            prop: Its font properties.
            angle: Its rotation, in degrees.
            ismath: Whether it is mathtext.
            mtext: The ``Text`` artist it comes from (unused).
        """
        color = _alpha(gc, gc.get_rgb())
        if color is None or not s.strip():
            return
        width = self._metrics.get_text_width_height_descent(s, prop, ismath)[0]
        origin = (float(x), float(y)) if ismath or angle else self._agg_origin(x, y, s, prop)
        self.leaves.append(TextLeaf(self._key(), s, prop.copy(), origin, float(angle), color, float(width), ismath))

    def _agg_origin(self, x: float, y: float, s: str, prop) -> tuple[float, float]:
        """Where Agg puts an unrotated line's baseline origin.

        Agg rounds the corner of the line's bitmap (the baseline plus the hinted descent, counted down
        from the top) to a whole pixel; this does the same.

        Args:
            x: The origin matplotlib asked for, in display pixels.
            y: The origin matplotlib asked for, in display pixels.
            s: The line's text.
            prop: Its font properties.

        Returns:
            The origin Agg draws at, in display pixels.
        """
        font = self._metrics._prepare_font(prop)  # the font Agg draws with, hinted as Agg hints it
        font.set_text(s, 0, flags=get_hinting_flag())
        descent = font.get_descent() / 64.0
        x_offset, y_offset = (value / 64.0 for value in font.get_bitmap_offset())
        from_top = self.height - y
        corner = round(from_top + y_offset + descent)
        return float(round(x + x_offset) - x_offset), float(self.height - (corner - y_offset - descent))

    def draw_image(self, gc, x, y, im, transform=None):
        """Records an image, its top row first as Manim wants it.

        The image arrives with its artist's alpha already in its alpha channel (Agg does not apply the
        graphics context's alpha to it again), and its bottom row first. A wholly transparent image
        is not recorded.

        Args:
            gc: The graphics context.
            x: The image's left edge, in display pixels.
            y: The image's bottom edge, in display pixels.
            im: The RGBA pixels, shape (height, width, 4), bottom row first.
            transform: Unused (``option_scale_image`` is False, so matplotlib never sends one).
        """
        pixels = np.array(im[::-1], dtype=np.uint8, copy=True)
        if pixels.size and pixels[..., 3].max() > 0:
            self.leaves.append(ImageLeaf(self._key(), pixels, (float(x), float(y))))


def record(figure) -> tuple[list[Leaf], Frame]:
    """Draws ``figure`` once through a ``LeafRenderer``.

    Args:
        figure: A matplotlib ``Figure``.

    Returns:
        Its leaves in drawing order, and its frame.
    """
    renderer = LeafRenderer(figure)
    figure.draw(renderer)
    return renderer.leaves, Frame.of(figure)
