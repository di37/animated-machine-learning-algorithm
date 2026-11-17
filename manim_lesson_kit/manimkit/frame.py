"""Where a matplotlib figure's pixels land on a Manim frame, and the Manim settings for that frame.

One Manim unit is one inch of the figure, with the origin at its centre, so a 25.6 × 14.4 inch
lesson canvas is a 25.6 × 14.4 unit frame whatever its resolution. matplotlib's display pixels
start at the bottom left and grow up and to the right, as Manim's units do, so mapping one onto
the other is a scale (divide by the dpi) and a shift (move the origin to the centre).
"""

from dataclasses import dataclass

import numpy as np

POINTS_PER_INCH = 72.0
CAIRO_LINE_WIDTH = 0.01  # Manim's Cairo camera draws a stroke_width of 1 as 0.01 units wide


@dataclass(frozen=True)
class Frame:
    """A figure's size and resolution: the frame its pixels are mapped onto.

    Attributes:
        width_in: The figure's width, in inches (and so in Manim units).
        height_in: The figure's height, in inches.
        dpi: Pixels per inch: the figure's resolution, and the frame's.
    """

    width_in: float
    height_in: float
    dpi: float

    def __post_init__(self):
        """Refuses a frame without a positive size and resolution.

        Raises:
            ValueError: If the width, height or dpi is not positive.
        """
        if self.width_in <= 0 or self.height_in <= 0 or self.dpi <= 0:
            raise ValueError(f"a frame needs a positive size and dpi, got {self}")

    @classmethod
    def of(cls, figure) -> "Frame":
        """The frame of a matplotlib figure.

        Args:
            figure: A matplotlib ``Figure``.

        Returns:
            Its size in inches and its dpi, as a ``Frame``.
        """
        width, height = figure.get_size_inches()
        return cls(float(width), float(height), float(figure.dpi))

    @property
    def size_px(self) -> tuple[int, int]:
        """The pixel size matplotlib's canvas gives the figure.

        matplotlib truncates rather than rounds, so 25.6 inches at 96 dpi is 2457 pixels.

        Returns:
            ``(width, height)`` in whole pixels.
        """
        return int(self.width_in * self.dpi), int(self.height_in * self.dpi)  # truncated, as matplotlib does

    def points(self, xy_px: np.ndarray) -> np.ndarray:
        """Display pixels as Manim points.

        Args:
            xy_px: Pixel positions, shape (n, 2), counted from the figure's bottom left.

        Returns:
            Manim points, shape (n, 3), in units from the frame's centre (z is 0).
        """
        xy = np.asarray(xy_px, dtype=float).reshape(-1, 2) / self.dpi
        out = np.zeros((len(xy), 3))
        out[:, 0] = xy[:, 0] - self.width_in / 2
        out[:, 1] = xy[:, 1] - self.height_in / 2
        return out

    def point(self, x_px: float, y_px: float) -> np.ndarray:
        """One display pixel as a Manim point.

        Args:
            x_px: Pixels from the figure's left edge.
            y_px: Pixels from the figure's bottom edge.

        Returns:
            The Manim point, shape (3,).
        """
        return self.points(np.array([[x_px, y_px]]))[0]

    def stroke_width(self, linewidth_pt: float) -> float:
        """Manim's stroke width for a matplotlib line width.

        Args:
            linewidth_pt: The line width in points (1/72 inch), as matplotlib gives it.

        Returns:
            The ``stroke_width`` Manim's Cairo camera draws that wide.
        """
        return linewidth_pt / POINTS_PER_INCH / CAIRO_LINE_WIDTH

    def manim_config(self, media_dir: str, background: str, fps: int) -> dict:
        """The Manim settings that draw this frame at its pixel size, writing nothing but frames.

        Args:
            media_dir: A folder for Manim's own media (a temporary one: nothing is kept there).
            background: The background colour, as a hex string.
            fps: Frames per second.

        Returns:
            Settings for ``manim.tempconfig``.
        """
        width, height = self.size_px
        return {"media_dir": media_dir, "write_to_movie": False, "save_last_frame": False,
                "disable_caching": True, "log_to_file": False, "pixel_width": width, "pixel_height": height,
                "frame_width": self.width_in, "frame_height": self.height_in, "frame_rate": fps,
                "background_color": background, "progress_bar": "none", "verbosity": "WARNING"}
