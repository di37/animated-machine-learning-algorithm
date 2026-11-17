"""The passenger list: every test passenger is a square, grouped by class and by women / men.

The list is a picture of the test set. Classes run top to bottom (1st first), with the women on the
left and the men on the right, eight squares to a row. As the model classifies passengers, their
squares fill with the colour of its score, a ✗ marks every wrong prediction, and an outline rings
the latest batch. Because survival depended so much on class and sex, the blocks show at a glance
where the model is right and where it is guessing.
"""

import numpy as np
from matplotlib.axes import Axes

from animkit import theme as base
from titanic_anim import theme
from titanic_anim.data import FEATURE_COLUMNS
from titanic_anim.frames import InferenceFrame
from titanic_anim.models import SCORE_MARGIN
from titanic_anim.passengers import ordinal

SQUARES_PER_ROW = 8  # per women / men block
BLOCK_GAP = 1.6  # empty columns between the women and the men
CLASS_GAP = 1.4  # empty rows between classes
SQUARE_FILL = 0.8  # share of a cell the square covers
SEX_GROUPS = ("women", "men")


def manifest_layout(raw: np.ndarray) -> tuple[np.ndarray, list[tuple[float, str]], float, float]:
    """Where each test passenger's square sits on the list.

    Args:
        raw: The test passengers' encoded, unscaled features, in the order they are classified.

    Returns:
        ``(positions, class_labels, width, height)``: the (x, y) cell of each passenger, in that order;
        each class's label as ``(y, text)``; and the grid's width and height, in cells.
    """
    pclass = raw[:, FEATURE_COLUMNS.index("pclass")].astype(int)
    group = np.where(raw[:, FEATURE_COLUMNS.index("female")] >= 0.5, 0, 1)
    positions = np.zeros((len(raw), 2))
    labels, top = [], 0.0
    for travel_class in sorted(set(pclass)):
        rows_used = 0
        for sex in (0, 1):
            members = np.flatnonzero((pclass == travel_class) & (group == sex))
            slots = np.arange(len(members))
            positions[members, 0] = sex * (SQUARES_PER_ROW + BLOCK_GAP) + slots % SQUARES_PER_ROW
            positions[members, 1] = top + slots // SQUARES_PER_ROW
            rows_used = max(rows_used, int(np.ceil(len(members) / SQUARES_PER_ROW)))
        labels.append((top, f"{ordinal(travel_class)}\nclass"))
        top += rows_used + CLASS_GAP
    return positions, labels, 2 * SQUARES_PER_ROW + BLOCK_GAP, top - CLASS_GAP


class PassengerManifest:
    """The passenger list panel of the inference dashboard.

    Attributes:
        ax: The axes the list is drawn on.
        positions: The (x, y) cell of every test passenger.
        filled: The squares of the passengers classified so far, coloured by score.
        wrong: The ✗ marks on wrong predictions.
        latest: The outlines around the latest batch.
    """

    def __init__(self, ax: Axes, raw_test: np.ndarray, score_kind: str):
        """Lays out the list: an empty outline per passenger, the group labels and a colour bar.

        Args:
            ax: The axes to draw on.
            raw_test: The test passengers' encoded, unscaled features.
            score_kind: ``SCORE_PROBABILITY`` or ``SCORE_MARGIN``: how the squares are coloured.
        """
        self.ax = ax
        self._kind = score_kind
        self.positions, class_labels, width, height = manifest_layout(raw_test)
        ax.set_xlim(-2.6, width)
        ax.set_ylim(height + 0.2, -1.6)  # first class on top
        ax.set_aspect("equal")
        ax.set_axis_off()
        size = self._marker_size(width + 2.6, height + 1.8)
        ax.scatter(*self.positions.T, s=size, marker="s", facecolors="none",
                   edgecolors=base.BASELINE, linewidths=0.6)
        self.filled = ax.scatter([], [], s=size, marker="s", c=[], cmap=theme.SURVIVAL_CMAP,
                                 norm=theme.score_norm(score_kind), linewidths=0)
        self.wrong = ax.scatter([], [], s=size * 0.45, marker="x", color=base.INK_PRIMARY, linewidths=1.1)
        self.latest = ax.scatter([], [], s=size * 1.9, marker="s", facecolors="none",
                                 edgecolors=base.INK_PRIMARY, linewidths=1.3)
        for index, name in enumerate(SEX_GROUPS):
            ax.text(index * (SQUARES_PER_ROW + BLOCK_GAP) + (SQUARES_PER_ROW - 1) / 2, -1.0, name,
                    ha="center", fontsize=base.FONT_SUBTITLE, color=base.INK_SECONDARY)
        for y, text in class_labels:
            ax.text(-0.9, y, text, ha="right", va="top", fontsize=base.FONT_LABEL, color=base.INK_SECONDARY)
        ax.set_title("The passenger list · ✗ = wrong prediction")
        self._colorbar()

    def _marker_size(self, columns: float, rows: float) -> float:
        """The square marker's area (points²) that fills ``SQUARE_FILL`` of a grid cell."""
        box = self.ax.get_position()
        width_in, height_in = self.ax.figure.get_size_inches()
        cell_pt = min(box.width * width_in / columns, box.height * height_in / rows) * 72.0
        return (cell_pt * SQUARE_FILL) ** 2

    def _colorbar(self) -> None:
        """Adds the score's colour bar under the list, labelled for probabilities or margins."""
        colorbar = self.ax.figure.colorbar(self.filled, ax=self.ax, orientation="horizontal",
                                           fraction=0.035, pad=0.02)
        label = ("decision value f(x): red = did not survive, blue = survived" if self._kind == SCORE_MARGIN
                 else "predicted P(survive): red = did not survive, blue = survived")
        colorbar.set_label(label, color=base.INK_SECONDARY)
        colorbar.outline.set_visible(False)

    def update(self, frame: InferenceFrame) -> None:
        """Fills in the passengers classified so far.

        Args:
            frame: The latest inference frame.
        """
        done = self.positions[:frame.n_done]
        self.filled.set_offsets(done)
        self.filled.set_array(frame.y_score)
        self.wrong.set_offsets(done[frame.y_pred != frame.y_true].reshape(-1, 2))
        self.latest.set_offsets(done[frame.batch_start:])
