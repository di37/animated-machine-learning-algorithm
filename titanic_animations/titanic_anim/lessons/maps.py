"""Age × fare maps, one per sex: the model's score as shading, its decision boundary, the passengers.

Shading is diverging (red = did not survive, grey = the threshold, blue = survived), on a 0–1 scale
for probabilities and a signed scale for score-only models, which are never labelled P(survive).

A model reads age, fare and sex. Fix the sex and what is left is a function of two inputs, which can be
drawn as a picture: every cell of the map grid shaded by the model's score, the decision boundary where
that score crosses the threshold, and the passengers on top. Two maps, women's and men's, show the whole
model. Fare runs up the y axis as log(1 + fare), labelled in dollars.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from animkit.lessons import theme
from titanic_anim.lessons.config import AGE, AGE_RANGE, FARE_TICKS, FEMALE, LOG_FARE, SEX_TITLES
from titanic_anim.lessons.data import PassengerData
from titanic_anim.lessons.lesson import PROBABILITY, threshold

MIDPOINT = "#eef1f4"
SCORE_CMAP = LinearSegmentedColormap.from_list("survival", (theme.ERROR, MIDPOINT, theme.MODEL))
MARGIN_LIMIT = 2.0  # |score| drawn fully saturated for score-only models
SURFACE_ALPHA = 0.62
RING = dict(edgecolors=theme.PANEL, linewidths=0.9)
PERCENT = FuncFormatter(lambda value, _: f"{value:.0%}")


def score_norm(kind: str) -> Normalize:
    """How a score maps onto the colour scale.

    Args:
        kind: ``PROBABILITY`` or ``MARGIN``.

    Returns:
        0 to 1 for a probability; for a signed score, a scale centred on 0 that is fully saturated at
        ±``MARGIN_LIMIT``.
    """
    if kind == PROBABILITY:
        return Normalize(0.0, 1.0)
    return TwoSlopeNorm(vcenter=0.0, vmin=-MARGIN_LIMIT, vmax=MARGIN_LIMIT)


def on_map(features: np.ndarray, sex: int) -> np.ndarray:
    """Which rows of ``features`` belong on map ``sex`` (0 = women, 1 = men).

    Args:
        features: (age, log fare, female) rows, or a single row.
        sex: The map: 0 for women, 1 for men.

    Returns:
        One flag per row, True for the rows on that map.
    """
    return np.atleast_2d(features)[:, FEMALE] == (1.0 if sex == 0 else 0.0)


def map_of(features: np.ndarray) -> int:
    """The map a single passenger appears on.

    Args:
        features: One (age, log fare, female) row.

    Returns:
        0 (the women's map) or 1 (the men's).
    """
    return 0 if features[FEMALE] == 1.0 else 1


def setup_map(ax: Axes, data: PassengerData, sex: int, fare_labels: bool | None = None,
              fare_ticks: tuple[int, ...] = FARE_TICKS) -> None:
    """Clears and styles the axes for one map: age across, log fare up, titled with its training count.

    Fare labels default to the left (women's) map only. A small map passes fewer ``fare_ticks`` so its
    labels do not crowd.

    Args:
        ax: The map's axes.
        data: The passengers.
        sex: The map: 0 for women, 1 for men.
        fare_labels: Whether to label the fare ticks; None labels them on the women's map only.
        fare_ticks: The fares to mark, in dollars.
    """
    ax.clear()
    theme.style_axes(ax, grid_axis=None)
    ax.set_xlim(*AGE_RANGE)
    ax.set_ylim(data.log_fares[0], data.log_fares[-1])
    ax.yaxis.set_major_locator(FixedLocator(np.log1p(fare_ticks)))
    ax.yaxis.set_minor_locator(NullLocator())
    labelled = sex == 0 if fare_labels is None else fare_labels
    ax.set_yticklabels([f"${fare:,}" for fare in fare_ticks] if labelled else [])
    ax.set_xlabel("Age (years)")
    if sex == 0:
        ax.set_ylabel("Fare paid (log scale)")
    count = int(np.sum(on_map(data.x_train, sex)))
    ax.set_title(f"{SEX_TITLES[sex]} · {count:,} in training", fontsize=theme.PANEL_TITLE_SIZE)


def _extent(data: PassengerData) -> tuple[float, float, float, float]:
    """The map grid's edges in data units, (left, right, bottom, top), as ``imshow`` takes them."""
    return data.ages[0], data.ages[-1], data.log_fares[0], data.log_fares[-1]


def draw_surface(ax: Axes, data: PassengerData, surface: np.ndarray, kind: str,
                 alpha: float = SURFACE_ALPHA) -> None:
    """Shades one map by the model's score: red below the threshold, grey at it, blue above.

    Args:
        ax: The map's axes.
        data: The passengers, for the grid's edges.
        surface: The model's score on this map's cells, shape (fares, ages).
        kind: ``PROBABILITY`` or ``MARGIN``; it sets the colour scale.
        alpha: The shading's opacity.
    """
    ax.imshow(surface, origin="lower", extent=_extent(data), aspect="auto", cmap=SCORE_CMAP,
              norm=score_norm(kind), alpha=alpha, interpolation="nearest", zorder=0)


def draw_boundary(ax: Axes, data: PassengerData, surface: np.ndarray, kind: str, *,
                  level: float | None = None, color: str = theme.NAVY, width: float = 2.6,
                  alpha: float = 1.0, dashed: bool = False) -> bool:
    """Draws the contour where the score crosses the threshold (or ``level``): the decision boundary.

    Args:
        ax: The map's axes.
        data: The passengers, for the grid's ages and fares.
        surface: The model's score on this map's cells, shape (fares, ages).
        kind: ``PROBABILITY`` or ``MARGIN``; it sets the threshold.
        level: A score to trace instead of the threshold (the SVM's ±1 margin lines).
        color: The line's colour.
        width: The line's width, in points.
        alpha: The line's opacity.
        dashed: Whether to dash the line.

    Returns:
        True if it was drawn; False, drawing nothing, when the score never crosses it on this map.
    """
    value = threshold(kind) if level is None else level
    if not surface.min() < value < surface.max():
        return False
    ax.contour(data.ages, data.log_fares, surface, levels=[value], colors=[color], linewidths=width,
               alpha=alpha, linestyles="dashed" if dashed else "solid", zorder=2)
    return True


def draw_passengers(ax: Axes, data: PassengerData, sex: int, *, count: int | None = None,
                    alpha: float = 0.9, wrong: np.ndarray | None = None, ring: float = 1.0) -> None:
    """Survivors as blue dots, others as red crosses; misclassified ones circled in navy.

    Args:
        ax: The map's axes.
        data: The passengers; the training ones are drawn.
        sex: The map: 0 for women, 1 for men.
        count: Draw only the first ``count`` passengers of the display order; None draws them all.
        alpha: The marks' opacity.
        wrong: For each training passenger in display order, whether the model gets them wrong; None
            circles nobody.
        ring: The circles' opacity; 0 hides them.
    """
    shown = data.display if count is None else data.display[:count]
    keep = on_map(data.x_train[shown], sex)
    rows = shown[keep]
    x, y = data.x_train[rows, AGE], data.x_train[rows, LOG_FARE]
    survived = data.y_train[rows] == 1
    ax.scatter(x[survived], y[survived], s=26, marker="o", color=theme.MODEL, alpha=alpha, zorder=4, **RING)
    ax.scatter(x[~survived], y[~survived], s=30, marker="X", color=theme.ERROR, alpha=alpha, zorder=4, **RING)
    if wrong is not None and ring > 0:
        missed = wrong[:len(shown)][keep]
        ax.scatter(x[missed], y[missed], s=95, facecolors="none", edgecolors=theme.NAVY, linewidths=1.3,
                   alpha=ring, zorder=5)


def mini_maps(ax: Axes, data: PassengerData, surfaces: np.ndarray, title: str, *, kind: str = PROBABILITY,
              norm: Normalize | None = None, boundary: bool = True) -> None:
    """Both maps side by side inside one panel (women left, men right), for side-panel comparisons.

    Args:
        ax: The panel's axes.
        data: The passengers (the small maps draw none).
        surfaces: The values to shade, shape (2, fares, ages), the women's map first.
        title: The panel's title.
        kind: ``PROBABILITY`` or ``MARGIN``; it sets the threshold, and the colour scale unless ``norm``
            is given.
        norm: A colour scale to use instead, for values that are not scores (boosting's log-odds steps).
        boundary: Whether to draw the threshold contour.
    """
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    for sex, left in ((0, 0.0), (1, 0.52)):
        box = (left, 0.08, 0.48, 0.84)
        inset = ax.inset_axes(box)
        inset.set_axis_off()
        inset.imshow(surfaces[sex], origin="lower", aspect="auto", cmap=SCORE_CMAP,
                     norm=norm or score_norm(kind), interpolation="nearest")
        if boundary:
            value = threshold(kind)
            if surfaces[sex].min() < value < surfaces[sex].max():
                inset.contour(surfaces[sex], levels=[value], colors=[theme.NAVY], linewidths=1.6)
        ax.text(left + 0.24, 0.0, SEX_TITLES[sex], ha="center", va="bottom", fontsize=11,
                color=theme.INK_SECONDARY)
