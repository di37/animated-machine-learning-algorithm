"""Pieces shared by the classifier stories: input scaling, wording, and the weight-bar panel.

The models that weigh or compare inputs (logistic regression, the perceptron and SVM, kNN and the MLP)
see standardised inputs: each column minus its training mean, divided by its training SD, so age in
years, log fare and the 0/1 sex flag are measured in comparable units. The trees and naive Bayes read
the raw columns.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes

from animkit.data import read_only
from animkit.lessons import theme
from titanic_anim.lessons.config import AGE, FEMALE, LOG_FARE
from titanic_anim.lessons.data import Passenger
from titanic_anim.lessons.lesson import signed_small

WEIGHT_NAMES = ("bias", "age", "fare", "woman")


@dataclass(frozen=True)
class Standardiser:
    """Training-set mean / SD scaling; the same numbers are reused for every later input.

    Attributes:
        mean: Each column's mean over the training passengers.
        scale: Each column's SD over the training passengers (1 for a column that never varies).

    Example:
        >>> standardise = Standardiser.fit(data.x_train)  # doctest: +SKIP
        >>> z = standardise(data.query.features)  # doctest: +SKIP
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardiser":
        """Learns the scaling from the training passengers.

        Args:
            x: The training passengers' (age, log fare, female) rows.

        Returns:
            The standardiser, with read-only means and SDs.
        """
        return cls(read_only(x.mean(axis=0)), read_only(np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Scales feature rows with the training numbers.

        Args:
            x: (age, log fare, female) rows, or a single row.

        Returns:
            The scaled inputs, the same shape: 0 is the training mean, ±1 one SD from it.
        """
        return (np.asarray(x, dtype=float) - self.mean) / self.scale


def describe_passenger(features: np.ndarray) -> str:
    """A passenger in words.

    Args:
        features: One (age, log fare, female) row.

    Returns:
        For example '30-year-old woman, $15 fare'.
    """
    sex = "woman" if features[FEMALE] == 1.0 else "man"
    return f"{features[AGE]:.0f}-year-old {sex}, ${np.expm1(features[LOG_FARE]):,.0f} fare"


def z_line(features: np.ndarray, z: np.ndarray) -> str:
    """The worked prediction's first line: each raw input and the scaled value the model sees.

    Args:
        features: One passenger's (age, log fare, female) row.
        z: The same row, scaled.

    Returns:
        For example 'scaled inputs: age 30 → +0.05, fare $15 → -0.32, woman → +1.35'.
    """
    return (f"scaled inputs: age {features[AGE]:.0f} → {z[AGE]:+.2f}, fare ${np.expm1(features[LOG_FARE]):,.0f} "
            f"→ {z[LOG_FARE]:+.2f}, {'woman' if features[FEMALE] == 1.0 else 'man'} → {z[FEMALE]:+.2f}")


def sigmoid(value):
    """The logistic sigmoid σ(v) = 1 / (1 + e^-v): squashes any score into a probability between 0 and 1.

    Args:
        value: A score or an array of scores (log-odds).

    Returns:
        σ of each, as floats; σ(0) = 0.5.
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


class WeightsPanel:
    """Bias and one weight per scaled input; at inference, each input's push on the score.

    A ``ClassPanel`` for the linear lessons. A bar's height is a weight; at inference it is that weight
    times this passenger's scaled input, so the bars add up to the score.
    """

    def __init__(self, limit: float, weights_of: Callable[[object], np.ndarray], title: str,
                 final: np.ndarray, standardise: Standardiser, note: Callable[[object], str] | None = None):
        """Sets up the panel for one lesson.

        Args:
            limit: The bars' axis runs from -limit to +limit during training, fixed so the bars visibly grow.
            weights_of: Reads the (bias, weights...) vector from a snapshot's ``detail``.
            title: The panel's title during training.
            final: The final model's (bias, weights...), used at inference.
            standardise: The lesson's input scaling.
            note: Writes a line under the bars from a snapshot's ``detail``; None writes none.
        """
        self._limit = limit
        self._weights_of = weights_of
        self._title = title
        self._final = final
        self._standardise = standardise
        self._note = note

    def _bars(self, ax: Axes, values: np.ndarray, title: str, limit: float) -> None:
        """Draws one labelled bar per value: blue up for positive, red down for negative."""
        theme.style_axes(ax, grid_axis="y")
        ax.set_ylim(-limit, limit)
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_xticks(range(len(values)), WEIGHT_NAMES[:len(values)])
        ax.axhline(0.0, color=theme.INK_MUTED, linewidth=1.0)
        ax.bar(range(len(values)), values, width=0.55, zorder=3,
               color=[theme.MODEL if v >= 0 else theme.ERROR for v in values])
        for i, value in enumerate(values):
            ax.text(i, value + (0.04 if value >= 0 else -0.04) * limit, signed_small(value), ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=12, color=theme.NAVY)
        ax.set_title(title)

    def draw_training(self, ax: Axes, snapshot) -> None:
        """Draws the bias and weights at one moment of training, with the note if there is one.

        Args:
            ax: The side panel's axes.
            snapshot: The moment to draw.
        """
        self._bars(ax, self._weights_of(snapshot.detail), self._title, self._limit)
        if self._note is not None:
            ax.text(0.99, 0.03, self._note(snapshot.detail), transform=ax.transAxes, ha="right",
                    fontsize=12, color=theme.INK_SECONDARY)

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the bias and each weight × input for the unseen passenger, bar by bar as ``t`` grows.

        Args:
            ax: The side panel's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        z = self._standardise(query.features)
        pushes = np.concatenate([[self._final[0]], self._final[1:] * z])
        shown = np.where(np.arange(len(pushes)) < np.ceil(len(pushes) * min(max(t, 0.0), 1.0)), pushes, 0.0)
        limit = max(float(np.abs(pushes).max()) * 1.35, 0.5)
        self._bars(ax, shown, "This passenger: bias + each weight × input", limit)

    def overlay_training(self, axes, snapshot) -> None:
        """No extras on the maps.

        Args:
            axes: The two maps, women's first.
            snapshot: The moment being drawn.
        """

    def overlay_inference(self, axes, query, t: float) -> None:
        """No extras on the maps.

        Args:
            axes: The two maps, women's first.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
