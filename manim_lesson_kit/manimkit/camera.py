"""Manim's Cairo camera, made to draw thousands of still shapes a frame, some hard-edged or clipped.

- A 3-D sheet is a mesh of tiles that matplotlib draws hard-edged: each pixel belongs to one tile.
  Antialiased, every shared edge would be half-covered twice and show as a light seam, so a shape
  marked ``aliased`` is drawn without antialiasing.
- A line or dot reaching past its axes is cut off at the axes edge, line width and all, by a shape's
  ``clip_box``, as Agg's rasterizer clips it.
- A lesson frame holds thousands of dots that do not move. Manim turns every shape's points into a
  Cairo path on every frame; here a shape keeps its Cairo path until its points are replaced (Manim's
  animations always assign new points rather than editing them in place). A shape with one solid
  fill and one solid stroke is then drawn directly; anything else (gradients, sheen, a background
  stroke) goes through Manim's own drawing.
"""

import weakref

import cairo
from manim.camera.camera import CAP_STYLE_MAP, LINE_JOIN_MAP, Camera

ALIASED = "aliased"  # the attribute a shape carries when it is drawn hard-edged
CLIP = "clip_box"  # (x0, y0, x1, y1) in Manim units: the box a shape is clipped to, as matplotlib clips it


class LessonCamera(Camera):
    """Manim's Cairo camera, drawing ``aliased`` and ``clip_box`` shapes and caching still shapes' paths.

    Use it as a scene's ``camera_class``; it draws everything else exactly as Manim's camera does.
    """

    def __init__(self, *args, **kwargs):
        """Creates the camera with an empty path cache.

        Args:
            *args: Passed on to Manim's ``Camera``.
            **kwargs: Passed on to Manim's ``Camera``.
        """
        super().__init__(*args, **kwargs)
        # shape: (the points array its path was made from, the Cairo path); kept off the shape, which
        # Manim deep-copies, and dropped with it
        self._paths: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def display_vectorized(self, vmobject, ctx: cairo.Context):
        """Draws one shape: hard-edged if it is marked ``aliased``, clipped if it has a ``clip_box``.

        Args:
            vmobject: The Manim shape.
            ctx: The Cairo context drawing the frame.

        Returns:
            The camera, as Manim's ``display_vectorized`` returns it.
        """
        aliased = getattr(vmobject, ALIASED, False)
        clip = getattr(vmobject, CLIP, None)
        if aliased or clip is not None:
            ctx.save()
            if aliased:
                ctx.set_antialias(cairo.ANTIALIAS_NONE)
            if clip is not None:
                x0, y0, x1, y1 = clip
                ctx.new_path()
                ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
                ctx.clip()
        try:
            if self._simple(vmobject):
                self._draw_simple(vmobject, ctx)
            else:
                super().display_vectorized(vmobject, ctx)
        finally:
            if aliased or clip is not None:
                ctx.restore()
        return self

    @staticmethod
    def _simple(vmobject) -> bool:
        """Whether a shape has one solid fill and one solid stroke, and nothing else to draw."""
        return (len(vmobject.get_fill_rgbas()) == 1 and len(vmobject.get_stroke_rgbas()) == 1
                and vmobject.get_stroke_width(background=True) == 0 and not vmobject.get_sheen_factor())

    def _path(self, vmobject, ctx: cairo.Context):
        """A shape's Cairo path: the cached one while its points are the same array, otherwise made anew."""
        cached = self._paths.get(vmobject)
        if cached is not None and cached[0] is vmobject.points:
            return cached[1]
        self.set_cairo_context_path(ctx, vmobject)
        path = ctx.copy_path()
        self._paths[vmobject] = (vmobject.points, path)
        return path

    def _draw_simple(self, vmobject, ctx: cairo.Context) -> None:
        """Fills and strokes a simple shape straight from its cached path, as Manim's camera would."""
        if len(vmobject.points) == 0:
            return
        path = self._path(vmobject, ctx)
        ctx.new_path()
        ctx.append_path(path)
        red, green, blue, alpha = vmobject.get_fill_rgbas()[0]
        if alpha > 0:
            ctx.set_source_rgba(blue, green, red, alpha)  # the pixel array is BGRA to Cairo, as in Manim
            ctx.fill_preserve()
        width = vmobject.get_stroke_width()
        red, green, blue, alpha = vmobject.get_stroke_rgbas()[0]
        if width > 0 and alpha > 0:
            ctx.set_source_rgba(blue, green, red, alpha)
            ctx.set_line_width(width * self.cairo_line_width_multiple)
            ctx.set_line_join(LINE_JOIN_MAP[vmobject.joint_type])
            ctx.set_line_cap(CAP_STYLE_MAP[vmobject.cap_style])
            ctx.stroke_preserve()
        ctx.new_path()
