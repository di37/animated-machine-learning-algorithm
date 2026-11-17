"""The 3-D chapter's picture: a fitted model's output drawn as height over two of its inputs.

A project builds a ``Surface`` from real model outputs (one sheet per output map, say one per group);
this module only draws it. Everything must sit inside the box, so a project that clips heights to the
z axis says so in its notes. matplotlib clips 3-D artists to the axes' square frame, and the box is
zoomed to spread past that frame across the main plot, so every artist here is drawn unclipped.

Painting order follows depth where it matters: the sheets and the threshold plane are one set of tiles
that matplotlib sorts by depth, and dots below any sheet are painted before them, the others after.
Notes that would not fit their side card are refused when the Surface is built, before a frame is encoded.
The camera swings either side of its starting angle and back, so the depth reads in a flat video and
the last frame matches the first.
"""

import math
import textwrap
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from animkit.data import read_only
from animkit.lessons import theme
from animkit.lessons.lesson import LessonError
from animkit.lessons.stage import HEIGHT_IN, METRIC_RECT, PANEL_RECT, reset_axes
from animkit.lessons.timing import phase

ELEVATION, AZIMUTH = 24.0, -52.0
SWING_DEGREES = 18.0  # azimuth stays within -70..-34: never near side-on to an axis, where depth collapses
BOX_ASPECT = (1.75, 1.2, 0.9)
BOX_ZOOM = 1.16  # spreads past the square 3-D frame, across the main plot's width, at every camera angle
SHEET_ALPHA = 0.82
PLANE_ALPHA = 0.32
UNDER_Z, TILES_Z, OVER_Z, QUERY_Z, LABEL_Z = 0.5, 1.0, 3.0, 5.0, 7.0  # painting order
DOT_SIZE = 18
DOT_ALPHA = 0.75
STICK_ALPHA = 0.5
STICK_WIDTH = 1.0
QUERY_SIZE = 240
QUERY_STICK_WIDTH = 2.2
QUERY_EDGE_WIDTH = 1.8
LABEL_PAD = 16  # axis titles clear the tick labels
LABEL_LIFT = 0.07  # the unseen example's label floats this share of the z range above it
SHEET_LABEL_AT = 0.8  # sheet names sit this far along the first input, clear of the z-axis tick labels
LABEL_BOX = dict(boxstyle="round,pad=0.3", facecolor=theme.PANEL, edgecolor="none", alpha=0.9)
POINTS_PER_INCH = 72.0
SIDE_NOTE_SIZE = 15.0  # the side notes' largest font; they shrink only to fit
MIN_SIDE_NOTE_SIZE = 9.0
NOTE_CHARS = 76  # wrap width for the side notes at SIDE_NOTE_SIZE
NOTE_LEFT, NOTE_TITLE_Y = 0.04, 0.9  # axes fractions
NOTE_TOP, NOTE_BOTTOM, NOTE_GAP = 0.78, 0.05, 0.035  # axes fractions: first and last note, the gap between
NOTE_LINE = 1.35  # line pitch / font size
NOTE_STAGGER = 0.08  # each note starts appearing this much later than the one before
NOTE_FADE = 0.15
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Axis:
    """One axis of the 3-D box: its title, range and ticks.

    Attributes:
        label: The axis title.
        limits: ``(low, high)``, increasing.
        ticks: Tick positions, inside the limits.
        tick_labels: One label per tick.
    """

    label: str
    limits: tuple[float, float]
    ticks: tuple[float, ...]
    tick_labels: tuple[str, ...]

    def __post_init__(self):
        """Refuses limits that do not increase, ticks without one label each, and ticks outside the axis.

        Raises:
            ValueError: If any of those checks fails.
        """
        low, high = self.limits
        if not low < high:
            raise ValueError(f"axis limits must increase, got {self.limits}")
        if len(self.ticks) != len(self.tick_labels):
            raise ValueError("every tick needs exactly one label")
        if not self.holds(np.asarray(self.ticks, dtype=float)):
            raise ValueError(f"ticks {self.ticks} fall outside the axis {self.limits}")

    def holds(self, values: np.ndarray) -> bool:
        """Every value is inside the axis (non-finite values never are).

        Args:
            values: Values on this axis.

        Returns:
            True when every value is within the limits, give or take ``TOLERANCE``.
        """
        values = np.asarray(values, dtype=float)
        low, high = self.limits
        return bool(np.all(values >= low - TOLERANCE) and np.all(values <= high + TOLERANCE))


@dataclass(frozen=True)
class Sheet:
    """One output map: heights over the (y, x) grid and a face colour for each grid point.

    A name labels the sheet when several sheets share the box. The arrays are stored as read-only copies.

    Attributes:
        heights: The model's outputs, shape (len(y), len(x)).
        colours: An RGBA colour per grid point, shape (len(y), len(x), 4).
        label: The sheet's name inside the box; empty for none.
    """

    heights: np.ndarray  # (len(y), len(x))
    colours: np.ndarray  # (len(y), len(x), 4) RGBA
    label: str = ""

    def __post_init__(self):
        """Stores read-only copies of the arrays, refusing heights that are not 2-D or colours that do not match.

        Raises:
            ValueError: If the heights are not 2-D, or there is not one RGBA colour per height.
        """
        heights, colours = read_only(self.heights), read_only(self.colours)
        if heights.ndim != 2 or colours.shape != (*heights.shape, 4):
            raise ValueError(f"a sheet needs 2-D heights and one RGBA colour each, got {heights.shape} "
                             f"and {colours.shape}")
        object.__setattr__(self, "heights", heights)
        object.__setattr__(self, "colours", colours)


@dataclass(frozen=True)
class Dots:
    """Training rows, drawn as dots in the box.

    ``fitted`` holds the model's height under the first few dots; each of those gets a stick joining it
    to the surface: its error.

    Attributes:
        x: Each dot's first input.
        y: Each dot's second input.
        z: Each dot's height.
        colours: Each dot's colour.
        fitted: The model's height under each of the first ``len(fitted)`` dots; ``None`` for no sticks.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    colours: tuple[str, ...]
    fitted: np.ndarray | None = None

    def __post_init__(self):
        """Stores read-only copies, refusing coordinates of different lengths or more fitted heights than dots.

        Raises:
            ValueError: If the coordinates and colours differ in length, or there are more fitted heights
                than dots.
        """
        parts = [read_only(part) for part in (self.x, self.y, self.z)]
        fitted = None if self.fitted is None else read_only(self.fitted)
        if len({len(part) for part in parts} | {len(self.colours)}) != 1:
            raise ValueError("dot coordinates and colours must have the same length")
        if fitted is not None and len(fitted) > len(parts[0]):
            raise ValueError("there are more fitted heights than dots")
        for name, part in zip(("x", "y", "z"), parts):
            object.__setattr__(self, name, part)
        object.__setattr__(self, "colours", tuple(self.colours))
        object.__setattr__(self, "fitted", fitted)


@dataclass(frozen=True)
class Marker:
    """The unseen example, at the height the model gives it.

    Attributes:
        x: Its first input.
        y: Its second input.
        z: The model's output for it.
        label: The text floating above it.
    """

    x: float
    y: float
    z: float
    label: str


@dataclass(frozen=True)
class Surface:
    """Everything the 3-D chapter draws, checked when it is built.

    Building one refuses a height that is not finite or falls outside its axis, and side notes that would
    not fit their cards, so a bad surface fails before a frame is encoded.

    Attributes:
        x: The grid along the first input.
        y: The grid along the second input.
        x_axis: The first input's axis.
        y_axis: The second input's axis.
        z_axis: The height axis: the model's output.
        sheets: One or more output maps over the grid.
        dots: The training rows.
        query: The unseen example.
        title: The stage label in the header.
        caption: The caption under the plot.
        reading: Notes for the side panel: how to read the picture.
        scores: Notes for the metric strip: how the model drawn here scores.
        plane: A flat level, such as the decision threshold; ``None`` for none.
        plane_label: The plane's name.
    """

    x: np.ndarray  # grid along the first input
    y: np.ndarray  # grid along the second input
    x_axis: Axis
    y_axis: Axis
    z_axis: Axis
    sheets: tuple[Sheet, ...]
    dots: Dots
    query: Marker
    title: str  # stage label in the header
    caption: str
    reading: tuple[str, ...]  # side panel: how to read the picture
    scores: tuple[str, ...]  # metric strip: how the model drawn here scores
    plane: float | None = None  # a flat level, e.g. the decision threshold
    plane_label: str = ""

    def __post_init__(self):
        """Stores read-only grids, then checks the sheets, the values against their axes, and the notes.

        Raises:
            ValueError: If there is no sheet, a sheet does not match the grid, a value falls outside its
                axis, or the notes do not fit their cards.
            LessonError: If any value is not a finite number.
        """
        object.__setattr__(self, "x", read_only(self.x))
        object.__setattr__(self, "y", read_only(self.y))
        object.__setattr__(self, "sheets", tuple(self.sheets))
        if not self.sheets:
            raise ValueError("a surface needs at least one sheet")
        if any(sheet.heights.shape != (len(self.y), len(self.x)) for sheet in self.sheets):
            raise ValueError("every sheet must have one height per (y, x) grid point")
        self._check_inside()
        note_layout(self.reading, PANEL_RECT[3] * HEIGHT_IN * POINTS_PER_INCH, "reading")
        note_layout(self.scores, METRIC_RECT[3] * HEIGHT_IN * POINTS_PER_INCH, "scores")

    def _check_inside(self) -> None:
        """Refuses any coordinate or height that is not finite, or that falls outside its axis."""
        dots, query = self.dots, self.query
        checks = (
            (self.x_axis, (self.x, dots.x, [query.x]), "first input"),
            (self.y_axis, (self.y, dots.y, [query.y]), "second input"),
            (self.z_axis, (*(sheet.heights for sheet in self.sheets), dots.z, [query.z],
                           *(() if dots.fitted is None else (dots.fitted,)),
                           *(() if self.plane is None else ([self.plane],))), "height"),
        )
        for axis, parts, name in checks:
            if not all(np.all(np.isfinite(np.asarray(part, dtype=float))) for part in parts):
                raise LessonError(f"a {name} is not a finite number: the model produced NaN or infinity")
            if not all(axis.holds(part) for part in parts):
                raise ValueError(f"a {name} falls outside its axis {axis.limits}: clip it and say so")


def camera(t: float) -> tuple[float, float]:
    """(elevation, azimuth) at chapter time t: one full sine swing, so the last frame matches the first.

    Args:
        t: The chapter's progress, 0 to 1.

    Returns:
        The camera's elevation and azimuth, in degrees.
    """
    return ELEVATION, AZIMUTH + SWING_DEGREES * math.sin(2.0 * math.pi * min(max(t, 0.0), 1.0))


def draw_surface(ax: Axes3D, surface: Surface, t: float) -> None:
    """Draws the whole 3-D picture for chapter time ``t``: the box, tiles, labels, dots and unseen example.

    Dots under a sheet are painted before the tiles and the others after, and every artist is drawn
    unclipped so the zoomed box can spread past matplotlib's square frame.

    Args:
        ax: The 3-D axes; it is cleared first.
        surface: What to draw.
        t: The chapter's progress, 0 to 1, which sets the camera angle.
    """
    ax.clear()
    _frame(ax, surface)
    _tiles(ax, surface)
    _labels(ax, surface)
    under = _under_the_sheets(surface)
    _dots(ax, surface.dots, under, UNDER_Z)
    _dots(ax, surface.dots, ~under, OVER_Z)
    _query(ax, surface)
    for artist in (*ax.collections, *ax.get_lines()):
        artist.set_clip_on(False)  # the zoomed box spreads past the square frame matplotlib would clip to
    ax.view_init(*camera(t))


def _frame(ax: Axes3D, surface: Surface) -> None:
    """Sets the box's aspect and zoom, and each axis's limits, ticks and title, over transparent panes."""
    ax.set_box_aspect(BOX_ASPECT, zoom=BOX_ZOOM)
    for spec, axis, set_limits, set_ticks, set_label in (
            (surface.x_axis, ax.xaxis, ax.set_xlim, ax.set_xticks, ax.set_xlabel),
            (surface.y_axis, ax.yaxis, ax.set_ylim, ax.set_yticks, ax.set_ylabel),
            (surface.z_axis, ax.zaxis, ax.set_zlim, ax.set_zticks, ax.set_zlabel)):
        set_limits(*spec.limits)
        set_ticks(spec.ticks, spec.tick_labels)
        set_label(spec.label, labelpad=LABEL_PAD, fontsize=theme.AXIS_TITLE_SIZE, color=theme.INK_SECONDARY)
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(labelsize=theme.TICK_SIZE, colors=theme.INK_MUTED)


def _quads(surface: Surface, heights: np.ndarray) -> np.ndarray:
    """(tiles, 4 corners, xyz): one tile per grid cell, corners in order around it."""
    grid_x, grid_y = np.meshgrid(surface.x, surface.y)
    cells = ((slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
             (slice(1, None), slice(1, None)), (slice(1, None), slice(None, -1)))
    corners = [np.stack([grid_x[rows, cols], grid_y[rows, cols], heights[rows, cols]], axis=-1)
               for rows, cols in cells]
    return np.stack(corners, axis=2).reshape(-1, 4, 3)


def _tiles(ax: Axes3D, surface: Surface) -> None:
    """Every sheet, and the plane, as one collection so its tiles are painted far to near."""
    layers = [(sheet.heights, sheet.colours[:-1, :-1].reshape(-1, 4) * (1.0, 1.0, 1.0, SHEET_ALPHA))
              for sheet in surface.sheets]
    if surface.plane is not None:
        flat = np.full((len(surface.y), len(surface.x)), surface.plane)
        tiles = (len(surface.y) - 1) * (len(surface.x) - 1)
        layers.append((flat, np.tile(to_rgba(theme.INACTIVE, PLANE_ALPHA), (tiles, 1))))
    verts = np.concatenate([_quads(surface, heights) for heights, _ in layers])
    faces = np.concatenate([colours for _, colours in layers])
    ax.add_collection3d(Poly3DCollection(verts, facecolors=faces, edgecolors="none", linewidths=0.0,
                                         antialiased=False, zorder=TILES_Z))


def _labels(ax: Axes3D, surface: Surface) -> None:
    """Writes the sheet names and the plane's name.

    Sheet names sit inside the box, spread across the second input so two sheets do not share a spot;
    the plane's name sits at its back corner.
    """
    column = round((len(surface.x) - 1) * SHEET_LABEL_AT)
    count = len(surface.sheets)
    for index, sheet in enumerate(surface.sheets):
        if sheet.label:
            row = round((len(surface.y) - 1) * (1.0 - index / count))
            ax.text(surface.x[column], surface.y[row], sheet.heights[row, column], sheet.label,
                    fontsize=theme.NOTE_SIZE, weight="bold", color=theme.NAVY, ha="center", va="center",
                    zorder=LABEL_Z, bbox=LABEL_BOX)
    if surface.plane is not None and surface.plane_label:
        ax.text(surface.x_axis.limits[0], surface.y_axis.limits[0], surface.plane, surface.plane_label,
                fontsize=theme.NOTE_SIZE, color=theme.INK_SECONDARY, ha="left", va="bottom", zorder=LABEL_Z,
                bbox=LABEL_BOX)


def _under_the_sheets(surface: Surface) -> np.ndarray:
    """Which dots sit below any sheet at their grid cell; from an elevated camera that sheet hides them."""
    dots = surface.dots
    columns = np.abs(surface.x[None, :] - dots.x[:, None]).argmin(axis=1)
    rows = np.abs(surface.y[None, :] - dots.y[:, None]).argmin(axis=1)
    highest = np.max([sheet.heights[rows, columns] for sheet in surface.sheets], axis=0)
    return dots.z < highest


def _dots(ax: Axes3D, dots: Dots, chosen: np.ndarray, zorder: float) -> None:
    """Draws the chosen dots at one painting depth, with an error stick from each fitted dot to the surface."""
    fitted = np.empty(0) if dots.fitted is None else dots.fitted
    for i in np.flatnonzero(chosen[:len(fitted)]):  # sticks belong to the first len(fitted) dots
        ax.plot([dots.x[i]] * 2, [dots.y[i]] * 2, [dots.z[i], fitted[i]], color=theme.ERROR, alpha=STICK_ALPHA,
                linewidth=STICK_WIDTH, zorder=zorder - 0.1)
    if chosen.any():
        ax.scatter(dots.x[chosen], dots.y[chosen], dots.z[chosen], s=DOT_SIZE,
                   c=[colour for colour, keep in zip(dots.colours, chosen) if keep], alpha=DOT_ALPHA,
                   depthshade=False, linewidths=0, zorder=zorder)


def _query(ax: Axes3D, surface: Surface) -> None:
    """Draws the unseen example: a stick up from the floor, a diamond at its height, and its label above."""
    query, floor = surface.query, surface.z_axis.limits[0]
    lift = LABEL_LIFT * (surface.z_axis.limits[1] - floor)
    ax.plot([query.x, query.x], [query.y, query.y], [floor, query.z], color=theme.UNSEEN,
            linewidth=QUERY_STICK_WIDTH, zorder=QUERY_Z - 0.1)
    ax.scatter([query.x], [query.y], [query.z], s=QUERY_SIZE, marker="D", color=theme.UNSEEN,
               edgecolors=theme.PANEL, linewidths=QUERY_EDGE_WIDTH, depthshade=False, zorder=QUERY_Z)
    ax.text(query.x, query.y, query.z + lift, query.label, fontsize=theme.NOTE_SIZE, weight="bold",
            color=theme.NAVY, ha="center", va="bottom", zorder=LABEL_Z, bbox=LABEL_BOX)


def note_layout(notes: tuple[str, ...], height_pt: float, name: str) -> tuple[list[str], float]:
    """The wrapped notes and the font size that fits them in a card ``height_pt`` tall.

    Args:
        notes: The notes, one paragraph each.
        height_pt: The card's height, in points.
        name: What the notes are, for the error message.

    Returns:
        Each note wrapped into lines, and the font size: ``SIDE_NOTE_SIZE``, or smaller to fit.

    Raises:
        ValueError: If the notes would need a font smaller than ``MIN_SIDE_NOTE_SIZE``.
    """
    blocks = ["\n".join(textwrap.wrap(note, NOTE_CHARS)) for note in notes]
    lines = sum(block.count("\n") + 1 for block in blocks)
    room = NOTE_TOP - NOTE_BOTTOM - NOTE_GAP * (len(blocks) - 1)
    size = min(SIDE_NOTE_SIZE, room * height_pt / (max(lines, 1) * NOTE_LINE))
    if size < MIN_SIDE_NOTE_SIZE:
        raise ValueError(f"{len(notes)} {name} notes ({lines} lines) do not fit their {height_pt:.0f} pt card")
    return blocks, size


def draw_notes(ax: Axes, title: str, notes: tuple[str, ...], t: float) -> None:
    """A titled card of short notes that appear one after another.

    The font shrinks only when the notes would not otherwise fit the card's real height.

    Args:
        ax: The card's axes; it is cleared first.
        title: The card's title.
        notes: The notes, top to bottom.
        t: The chapter's progress, 0 to 1; each note fades in a little after the one before.
    """
    reset_axes(ax)
    ax.text(NOTE_LEFT, NOTE_TITLE_Y, title, fontsize=theme.PANEL_TITLE_SIZE, weight="bold", color=theme.NAVY,
            va="center")
    height_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * POINTS_PER_INCH
    blocks, size = note_layout(notes, height_pt, title)
    y = NOTE_TOP
    for index, block in enumerate(blocks):
        start = NOTE_STAGGER * index
        ax.text(NOTE_LEFT, y, block, fontsize=size, color=theme.INK_SECONDARY, va="top", linespacing=NOTE_LINE,
                alpha=phase(t, start, start + NOTE_FADE))
        y -= (block.count("\n") + 1) * size * NOTE_LINE / height_pt + NOTE_GAP
