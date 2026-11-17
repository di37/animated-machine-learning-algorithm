"""Gaussian naive Bayes: count passengers, and the class priors and per-class bell curves update.

Each frame calls ``partial_fit`` with the next batch of (shuffled) training passengers, so the
estimates on screen are exactly what the model has seen so far. Sex is a 0/1 input; its "bell
curve" mean is simply the share of women in each class.

Bayes' rule says P(survived | passenger) is proportional to P(survived) × P(passenger | survived). The
"naive" step treats the inputs as independent within each class, so P(passenger | class) is a product of
one bell curve (a Gaussian) per input. Training is counting: each class's share of the passengers (its
prior), and each input's mean and variance within the class. The work is done in logs, where the
product becomes a sum.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
from matplotlib.axes import Axes
from scipy.special import logsumexp
from sklearn.naive_bayes import GaussianNB

from animkit.data import read_only
from animkit.lessons import theme
from animkit.lessons.lesson import Story
from titanic_anim.config import RANDOM_STATE
from titanic_anim.lessons.config import AGE, AGE_RANGE, LOG_FARE
from titanic_anim.lessons.data import Passenger, PassengerData
from titanic_anim.lessons.lesson import (
    PROBABILITY,
    ClassLesson,
    ClassSnapshot,
    make_snapshot,
    percent,
    verified_score,
)
from titanic_anim.lessons.lesson_diagrams import BayesDiagram
from titanic_anim.lessons.stories.common import describe_passenger

COUNTS = (10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480)
INSET_ROW = (0.32, 0.47, 0.62)  # bottom, width, height of each curve inset: room below for ticks and label
CURVE_POINTS = 200


@dataclass(frozen=True)
class NbState:
    """Naive Bayes' estimates after counting some of the passengers.

    Attributes:
        seen: How many training passengers have been counted.
        prior: Each class's share of them, (did not survive, survived).
        mean: Each class's mean of each input, shape (2, 3).
        var: Each class's variance of each input, shape (2, 3).
    """

    seen: int
    prior: np.ndarray  # (2,) did not survive, survived
    mean: np.ndarray  # (2, 3)
    var: np.ndarray  # (2, 3)


def _capture(model: GaussianNB, seen: int) -> NbState:
    """Copies the model's estimates after it has counted ``seen`` passengers."""
    return NbState(seen, read_only(model.class_prior_), read_only(model.theta_), read_only(model.var_))


def joint_log_likelihood(state: NbState, features: np.ndarray) -> np.ndarray:
    """(n, 2): log P(class) + Σ log N(x_j; mean, var), exactly as GaussianNB computes it.

    Args:
        state: The estimates to use.
        features: (age, log fare, female) rows, or a single row.

    Returns:
        Each row's log prior plus log likelihood, per class: column 0 'did not survive', column 1 'survived'.
    """
    x = np.atleast_2d(np.asarray(features, dtype=float))[:, None, :]
    terms = np.log(2 * np.pi * state.var) + (x - state.mean) ** 2 / state.var
    return np.log(state.prior) - 0.5 * terms.sum(axis=2)


def posterior(state: NbState, features: np.ndarray) -> np.ndarray:
    """P(survive) by Bayes' rule: normalise the two joint likelihoods.

    Args:
        state: The estimates to use.
        features: (age, log fare, female) rows, or a single row.

    Returns:
        Each row's P(survive). ``logsumexp`` adds the two likelihoods without leaving logs, so tiny ones
        do not round to 0.
    """
    jll = joint_log_likelihood(state, features)
    return np.exp(jll[:, 1] - logsumexp(jll, axis=1))


def _pdf(state: NbState, column: int, xs: np.ndarray) -> np.ndarray:
    """Each class's bell curve for one input, evaluated at ``xs``: shape (2, len(xs))."""
    mean, var = state.mean[:, column][:, None], state.var[:, column][:, None]
    return np.exp(-(xs - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


class BellCurvePanel:
    """The naive Bayes side panel (a ``ClassPanel``): each class's bell curve for age and for log fare.

    At inference a line marks where the unseen passenger falls on each curve.
    """

    def __init__(self, final: NbState, data: PassengerData):
        """Sets up the panel.

        Args:
            final: The final estimates, used at inference.
            data: The passengers, for the fare axis's range.
        """
        self._final = final
        self._ages = np.linspace(*AGE_RANGE, CURVE_POINTS)
        self._fares = np.linspace(data.log_fares[0], data.log_fares[-1], CURVE_POINTS)

    def _curves(self, ax: Axes, state: NbState, title: str, marks: np.ndarray | None = None) -> None:
        """Draws the age and fare insets: survivors' curve blue, others' red, ``marks`` as lines if given."""
        ax.clear()
        ax.set_axis_off()
        ax.set_title(title)
        for column, xs, box, name in ((AGE, self._ages, (0.0, *INSET_ROW), "age"),
                                      (LOG_FARE, self._fares, (0.53, *INSET_ROW), "fare (log)")):
            inset = ax.inset_axes(box)
            theme.style_axes(inset, grid_axis=None)
            inset.set_yticks([])
            curves = _pdf(state, column, xs)
            inset.plot(xs, curves[1], color=theme.MODEL, linewidth=2.4, label="survived")
            inset.plot(xs, curves[0], color=theme.ERROR, linewidth=2.4, label="did not")
            if marks is not None:
                inset.axvline(marks[column], color=theme.UNSEEN, linewidth=2.0)
            inset.set_xlabel(name, fontsize=12)
            if column == LOG_FARE:  # the fare curves leave its top-left corner empty
                inset.legend(loc="upper left", fontsize=10, frameon=False, handlelength=1.2, borderaxespad=0.1)

    def draw_training(self, ax: Axes, snapshot: ClassSnapshot) -> None:
        """Draws the bell curves as counted so far.

        Args:
            ax: The side panel's axes.
            snapshot: The moment to draw.
        """
        state: NbState = snapshot.detail
        self._curves(ax, state, f"Bell curve per class, from {state.seen} passengers")

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the final bell curves, marking the unseen passenger's age and fare once ``t`` is past 0.

        Args:
            ax: The side panel's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        self._curves(ax, self._final, "Where this passenger falls on each curve",
                     marks=query.features if t > 0 else None)

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


def _schedule(data: PassengerData) -> tuple[np.ndarray, list[int]]:
    """A shuffled order whose first batch already contains both classes, and the running counts."""
    order = np.random.default_rng(RANDOM_STATE).permutation(data.n_train)
    labels = data.y_train[order]
    first = max(int(np.argmax(labels == 0)), int(np.argmax(labels == 1))) + 1
    counts = sorted({max(first, c) for c in COUNTS if c < data.n_train} | {data.n_train})
    return order, counts


def _story(data: PassengerData, last: ClassSnapshot) -> Story:
    """The lesson's narration; ``last`` is the final snapshot, for its accuracies."""
    return Story(
        key="naive_bayes", title="Naive Bayes",
        method="Gaussian naive Bayes: priors and bell curves, counted from the data",
        objective=("Naive Bayes never searches for a boundary. It counts: how common survival is, and "
                   "what ages and fares look like in each group. Watch its estimates, and its map, firm up "
                   "as it sees more passengers."),
        initial=("After only a handful of passengers the bell curves are guesses, and so is the map. "
                 "Shading is the model's P(survive)."),
        training=("Each frame counts more passengers (partial_fit). The panel shows the per-class bell "
                  "curves for age and fare; 'naive' means each input is treated as independent."),
        final=(f"All {data.n_train} passengers counted: train accuracy {last.train_accuracy:.0%}, validation "
               f"{last.val_accuracy:.0%}. Being a woman is by far the strongest evidence for survival."),
        inference=("For an unseen passenger, multiply the prior by how likely their age, fare and sex are "
                   "under each class, then normalise the two numbers into a probability."),
        takeaway="Naive Bayes is counting made principled: priors times per-feature likelihoods, normalised by Bayes' rule.",
        limitation=("it assumes the inputs are independent within each class, which real ages and fares "
                    "are not quite, and bell curves fit skewed fares and 0/1 sex only roughly."),
        progress_label="passengers counted")


def build(data: PassengerData) -> ClassLesson:
    """Builds the naive Bayes lesson: one snapshot after each batch of passengers is counted.

    Args:
        data: The passengers.

    Returns:
        The lesson, with the model that has counted every training passenger as the final one.
    """
    order, counts = _schedule(data)
    model = GaussianNB()
    states: list[NbState] = []
    snapshots: list[ClassSnapshot] = []
    seen = 0
    for count in counts:
        rows = order[seen:count]
        model.partial_fit(data.x_train[rows], data.y_train[rows], classes=np.array([0, 1]))
        seen = count
        state = _capture(model, count)
        states.append(state)
        snapshots.append(make_snapshot(partial(posterior, state), data, PROBABILITY, label=f"{count} passengers", progress=count,
                                       note=f"{count} passengers counted\nprior: {state.prior[1]:.0%} survived",
                                       detail=state))
    final = states[-1]

    def predict(x: np.ndarray) -> np.ndarray:
        """The final model's P(survive), from scikit-learn.

        Args:
            x: (age, log fare, female) rows.

        Returns:
            One P(survive) per row.
        """
        return model.predict_proba(np.asarray(x, dtype=float))[:, 1]

    def explain(features: np.ndarray):
        """Works one passenger's prediction through: the two log scores, then Bayes' rule.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            The verified explanation, checked against scikit-learn's ``predict_proba``.
        """
        jll = joint_log_likelihood(final, features)[0]
        worked = float(np.exp(jll[1] - logsumexp(jll)))
        lines = [f"passenger: {describe_passenger(features)}",
                 f"log prior + Σ log likelihood: survived {jll[1]:+.2f}, did not {jll[0]:+.2f}",
                 f"P(survive) = e^{jll[1]:.2f} / (e^{jll[1]:.2f} + e^{jll[0]:.2f}) = {percent(worked)}",
                 f"{percent(worked)} {'>' if worked > 0.5 else '≤'} 50% → predicted: "
                 f"{'survived' if worked > 0.5 else 'did not survive'}"]
        return verified_score(lines, worked, float(predict(features[None, :])[0]), PROBABILITY)

    snaps = tuple(snapshots)
    return ClassLesson(
        story=_story(data, snaps[-1]), data=data, snapshots=snaps, final_index=len(snaps) - 1, score=predict,
        kind=PROBABILITY, explain=explain, panel=BellCurvePanel(final, data),
        diagram=BayesDiagram(final, posterior),
        config={"model": "GaussianNB", "inputs": "age, log fare, woman (unscaled)",
                "counts": ", ".join(map(str, counts)), "order_seed": str(RANDOM_STATE)})
