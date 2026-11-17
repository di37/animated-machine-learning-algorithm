"""The smallest things a matplotlib frame draws, as recorded: a path, a line of text, an image.

A ``Leaf`` keeps what its renderer was told (in display pixels) and can build the Manim shape that
draws it. Two ideas make frame-to-frame animation possible:

- its ``key`` says where it sits among the frame's artists, so the same thing on the next frame has
  the same key;
- its ``signature`` is equal exactly when two leaves would draw the same, so an unchanged thing is
  recognised without comparing pixels.
"""

import hashlib

import numpy as np
from manim import DashedVMobject, ImageMobject, Mobject, VMobject
from manim.constants import CapStyleType, LineJointType

from manimkit import glyphs
from manimkit.camera import ALIASED, CLIP
from manimkit.frame import Frame
from manimkit.paths import CLOSEPOLY, STOP, cubic_points

PATH, TEXT, IMAGE = "path", "text", "image"
STOP_CODE, CLOSE_CODE = int(STOP), int(CLOSEPOLY)
PRECISION = 2  # decimals of a pixel kept when comparing geometry
CAPS = {"butt": CapStyleType.BUTT, "round": CapStyleType.ROUND, "projecting": CapStyleType.SQUARE}
JOINS = {"miter": LineJointType.MITER, "round": LineJointType.ROUND, "bevel": LineJointType.BEVEL}

Rgba = tuple[float, float, float, float]


def _rounded(values: np.ndarray) -> bytes:
    """Geometry rounded to ``PRECISION`` decimals of a pixel, as bytes for hashing."""
    return np.round(np.asarray(values, dtype=float), PRECISION).tobytes()


_PROTOTYPE: list[VMobject] = []  # a blank shape's attributes, copied instead of making each shape afresh


def _blank(points: np.ndarray) -> VMobject:
    """An unfilled, unstroked shape through Manim ``points``.

    It is the blank prototype's attributes with every array, list and dict copied: what Manim's
    deep ``copy`` gives for a shape without submobjects, many times quicker.

    Args:
        points: Manim points, shape (4k, 3).

    Returns:
        A new ``VMobject`` through those points, drawing nothing until it is filled or stroked.
    """
    if not _PROTOTYPE:
        _PROTOTYPE.append(VMobject().set_fill(opacity=0.0).set_stroke(width=0, opacity=0.0))
    shape = object.__new__(VMobject)
    shape.__dict__.update({name: value.copy() if isinstance(value, (np.ndarray, list, dict)) else value
                           for name, value in _PROTOTYPE[0].__dict__.items()})
    shape.points = points
    return shape


class Leaf:
    """One recorded draw call. Subclasses say how it compares and how it is drawn.

    Attributes:
        key: Where it sits among the frame's artists: the trail of artist groups it was drawn in,
            then its place in its group.
        signature: Equal for two leaves exactly when they would draw the same.
        kind: ``"path"``, ``"text"`` or ``"image"``.
    """

    __slots__ = ("key", "signature")
    kind = ""

    def __init__(self, key: tuple, signature: int):
        """Creates the leaf.

        Args:
            key: Where it sits among the frame's artists.
            signature: A hash of everything that decides how it draws.
        """
        self.key = key
        self.signature = signature

    def build(self, frame: Frame) -> Mobject:
        """A fresh Manim shape drawing this leaf.

        Args:
            frame: The frame the leaf's pixels are mapped onto.

        Returns:
            A new Manim shape.
        """
        raise NotImplementedError

    def anchor(self) -> np.ndarray:
        """A point that moves with the leaf.

        Returns:
            The point in display pixels, shape (2,).
        """
        raise NotImplementedError

    def matches(self, other: "Leaf") -> bool:
        """Whether one can turn into the other smoothly: the same kind of shape, only moved or recoloured.

        Args:
            other: Another leaf, usually the same key on another frame.

        Returns:
            True if a Manim transform from one to the other is meaningful.
        """
        raise NotImplementedError

    def area_px(self) -> float:
        """Roughly how much of the frame it covers.

        Returns:
            Square pixels.
        """
        return 0.0


class PathLeaf(Leaf):
    """A filled and/or stroked path, already transformed and snapped as Agg would draw it.

    Attributes:
        vertices: The path's vertices in display pixels, shape (n, 2).
        codes: The path's codes, one per vertex.
        clip: The box (display pixels) Agg clips it to, kept only when the path reaches past it; the
            camera clips the shape there, as Agg's rasterizer does.
        fill: The fill colour (RGBA), or None for an unfilled path.
        stroke: The stroke colour (RGBA), or None for an unstroked path.
        linewidth: The stroke's width, in points.
        dashes: The on/off dash lengths in points, or None for a solid line.
        cap: The stroke's cap style, as matplotlib names it.
        join: The stroke's join style, as matplotlib names it.
        antialiased: False for a shape drawn hard-edged (a 3-D sheet's tiles).
    """

    __slots__ = ("vertices", "codes", "clip", "fill", "stroke", "linewidth", "dashes", "cap", "join", "antialiased")
    kind = PATH

    def __init__(self, key: tuple, vertices: np.ndarray, codes: np.ndarray, *, fill: Rgba | None,
                 stroke: Rgba | None, linewidth: float, dashes: tuple | None, cap: str, join: str,
                 clip: tuple[float, float, float, float] | None = None, antialiased: bool = True,
                 shape_id: bytes | None = None):
        """Records the path and works out its signature.

        Args:
            key: Where it sits among the frame's artists.
            vertices: The path's vertices in display pixels.
            codes: The path's codes.
            fill: The fill colour, or None.
            stroke: The stroke colour, or None.
            linewidth: The stroke width, in points.
            dashes: The dash pattern in points, or None.
            cap: The cap style.
            join: The join style.
            clip: The clip box in display pixels, or None when the path stays inside it.
            antialiased: Whether to draw it with antialiasing.
            shape_id: Stands for the geometry when the caller knows it already (a collection's marker
                and offset); otherwise the vertices are hashed.
        """
        self.vertices = vertices
        self.codes = codes
        self.clip = clip
        self.fill, self.stroke = fill, stroke
        self.linewidth, self.dashes, self.cap, self.join = linewidth, dashes, cap, join
        self.antialiased = antialiased
        if shape_id is None:
            shape_id = hashlib.blake2b(_rounded(vertices) + codes.tobytes(), digest_size=16).digest()
        style = (fill, stroke, round(linewidth, 3) if stroke else 0.0, dashes, cap, join, antialiased, clip)
        super().__init__(key, hash((PATH, shape_id, style)))

    @property
    def filled(self) -> bool:
        """Whether the path has a fill.

        Returns:
            True if it has a fill colour.
        """
        return self.fill is not None

    def drawn_vertices(self) -> np.ndarray:
        """The vertices that place the path.

        Returns:
            Its vertices without the placeholders of a close or a stop, shape (m, 2).
        """
        return self.vertices[(self.codes != STOP_CODE) & (self.codes != CLOSE_CODE)]

    def anchor(self) -> np.ndarray:
        """The mean of its drawn vertices.

        Returns:
            The point in display pixels, shape (2,).
        """
        drawn = self.drawn_vertices()
        return drawn.mean(axis=0) if len(drawn) else np.zeros(2)

    def area_px(self) -> float:
        """The area of the box around its drawn vertices.

        Returns:
            Square pixels.
        """
        drawn = self.drawn_vertices()
        if len(drawn) == 0:
            return 0.0
        span = drawn.max(axis=0) - drawn.min(axis=0)
        return float(span[0] * span[1])

    def matches(self, other: Leaf) -> bool:
        """Whether ``other`` is the same kind of path: same codes, fill, stroke, dashes and clip.

        Args:
            other: Another leaf.

        Returns:
            True if it differs only in where its vertices are and what colour it is.
        """
        if not isinstance(other, PathLeaf):
            return False
        same_shape = np.array_equal(self.codes, other.codes) and self.vertices.shape == other.vertices.shape
        return (same_shape and (self.fill is None) == (other.fill is None) and self.clip == other.clip
                and (self.stroke is None) == (other.stroke is None) and self.dashes == other.dashes)

    def build(self, frame: Frame) -> Mobject:
        """The path as a Manim shape: filled, stroked, dashed, hard-edged and clipped as recorded.

        Args:
            frame: The frame the path's pixels are mapped onto.

        Returns:
            A ``VMobject``, or a ``DashedVMobject`` for a dashed line.
        """
        points = cubic_points(self.vertices, self.codes)
        shape = _blank(frame.points(points) if len(points) else np.zeros((0, 3)))
        shape.joint_type = JOINS.get(self.join, LineJointType.AUTO)
        shape.cap_style = CAPS.get(self.cap, CapStyleType.AUTO)
        if self.fill is not None:
            shape.fill_rgbas = np.array([self.fill], dtype=float)
        if self.stroke is not None and self.linewidth > 0:
            shape.stroke_rgbas = np.array([self.stroke], dtype=float)
            shape.stroke_width = frame.stroke_width(self.linewidth)
        if not self.antialiased:
            setattr(shape, ALIASED, True)
        if self.dashes and self.stroke is not None and len(points):
            shape = _dashed(shape, self.dashes, frame)
        if self.clip is not None:
            corners = frame.points(np.array([self.clip[:2], self.clip[2:]], dtype=float))
            box = (*corners[0][:2], *corners[1][:2])
            for part in shape.get_family():
                setattr(part, CLIP, box)
        return shape


def _dashed(shape: VMobject, dashes: tuple, frame: Frame) -> Mobject:
    """A stroked shape cut into its on/off dash pattern.

    Manim spaces dashes evenly along the line, so the count is the line's length over one pattern's
    length, rounded; a pattern of several lengths keeps its first pair.

    Args:
        shape: The stroked shape.
        dashes: The on/off lengths, in points.
        frame: The frame (units are inches, so points convert by 72).

    Returns:
        A ``DashedVMobject``.
    """
    on, off = dashes[0], dashes[1] if len(dashes) > 1 else dashes[0]
    period = (on + off) / glyphs.POINTS_PER_INCH  # inches = units
    count = max(1, round(shape.get_arc_length() / period)) if period > 0 else 1
    return DashedVMobject(shape, num_dashes=count, dashed_ratio=on / (on + off), equal_lengths=True)


class TextLeaf(Leaf):
    """One line of text, as matplotlib laid it out.

    Attributes:
        string: The line's text.
        prop: Its font properties.
        origin: Where its baseline starts, in display pixels.
        angle: Its rotation, in degrees.
        color: Its colour (RGBA).
        width_px: The width Agg's hinted text takes, in pixels.
        ismath: Whether it is mathtext.
    """

    __slots__ = ("string", "prop", "origin", "angle", "color", "width_px", "ismath")
    kind = TEXT

    def __init__(self, key: tuple, string: str, prop, origin: tuple[float, float], angle: float, color: Rgba,
                 width_px: float, ismath: bool | str = False):
        """Records the line and works out its signature.

        Args:
            key: Where it sits among the frame's artists.
            string: The line's text.
            prop: Its font properties.
            origin: Where its baseline starts, in display pixels.
            angle: Its rotation, in degrees.
            color: Its colour.
            width_px: The width Agg's hinted text takes, in pixels.
            ismath: Whether it is mathtext.
        """
        self.string, self.prop, self.origin, self.angle = string, prop, origin, angle
        self.color, self.width_px, self.ismath = color, width_px, ismath
        super().__init__(key, hash((TEXT, string, glyphs.font_key(prop), round(origin[0], PRECISION),
                                    round(origin[1], PRECISION), round(angle, 3), color, ismath)))

    def anchor(self) -> np.ndarray:
        """Its baseline origin.

        Returns:
            The point in display pixels, shape (2,).
        """
        return np.asarray(self.origin, dtype=float)

    def area_px(self) -> float:
        """Its width times its font size.

        Returns:
            Roughly square pixels.
        """
        return self.width_px * self.prop.get_size_in_points()

    def matches(self, other: Leaf) -> bool:
        """Whether ``other`` is the same text in the same font and angle (only moved or recoloured).

        Args:
            other: Another leaf.

        Returns:
            True if a Manim transform from one to the other is meaningful.
        """
        return (isinstance(other, TextLeaf) and self.string == other.string and self.angle == other.angle
                and glyphs.font_key(self.prop) == glyphs.font_key(other.prop))

    def build(self, frame: Frame) -> Mobject:
        """The line as one filled Manim shape of glyph outlines.

        Args:
            frame: The frame the line's pixels are mapped onto.

        Returns:
            A ``VMobject``.
        """
        inches = glyphs.line_points(self.string, self.prop, self.origin, self.angle, dpi=frame.dpi,
                                    hinted_width_px=self.width_px, ismath=self.ismath)
        shape = _blank(frame.points(inches * frame.dpi) if len(inches) else np.zeros((0, 3)))
        shape.fill_rgbas = np.array([self.color], dtype=float)
        return shape


class ImageLeaf(Leaf):
    """An RGBA image drawn one pixel per pixel.

    Attributes:
        pixels: The image, shape (height, width, 4), top row first.
        origin: Its bottom-left corner, in display pixels.
    """

    __slots__ = ("pixels", "origin")
    kind = IMAGE

    def __init__(self, key: tuple, pixels: np.ndarray, origin: tuple[float, float]):
        """Records the image and works out its signature.

        Args:
            key: Where it sits among the frame's artists.
            pixels: The image, top row first.
            origin: Its bottom-left corner, in display pixels.
        """
        self.pixels, self.origin = pixels, origin
        digest = hashlib.blake2b(pixels.tobytes(), digest_size=16).digest()
        super().__init__(key, hash((IMAGE, digest, pixels.shape, round(origin[0], PRECISION),
                                    round(origin[1], PRECISION))))

    def anchor(self) -> np.ndarray:
        """Its bottom-left corner.

        Returns:
            The point in display pixels, shape (2,).
        """
        return np.asarray(self.origin, dtype=float)

    def area_px(self) -> float:
        """Its pixel count.

        Returns:
            Square pixels.
        """
        return float(self.pixels.shape[0] * self.pixels.shape[1])

    def matches(self, other: Leaf) -> bool:
        """Whether ``other`` is the same image (it may have moved).

        Args:
            other: Another leaf.

        Returns:
            True if both images have the same pixels.
        """
        return isinstance(other, ImageLeaf) and other.pixels.shape == self.pixels.shape and np.array_equal(
            other.pixels, self.pixels)

    def build(self, frame: Frame) -> Mobject:
        """The image as a Manim image, its pixels on the frame's pixels.

        Args:
            frame: The frame the image's pixels are mapped onto.

        Returns:
            An ``ImageMobject``.
        """
        height_px, width_px = self.pixels.shape[:2]
        image = ImageMobject(self.pixels, scale_to_resolution=frame.dpi)
        image.stretch_to_fit_width(width_px / frame.dpi).stretch_to_fit_height(height_px / frame.dpi)
        corner = frame.point(*self.origin)
        return image.move_to(corner + np.array([width_px, height_px, 0.0]) / frame.dpi / 2)


def displacement_px(before: Leaf, after: Leaf) -> float:
    """How far a leaf moved between two frames.

    Args:
        before: Its state on one frame.
        after: Its state on a later frame.

    Returns:
        The furthest any vertex moved, in pixels (for a path); otherwise how far its anchor moved.
    """
    if isinstance(before, PathLeaf) and isinstance(after, PathLeaf) and before.vertices.shape == after.vertices.shape:
        return float(np.abs(after.vertices - before.vertices).max()) if len(before.vertices) else 0.0
    return float(np.abs(after.anchor() - before.anchor()).max())
