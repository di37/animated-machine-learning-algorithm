"""Algorithm diagrams for the classifier lessons: how each model turns age, fare and sex into an answer.

A diagram sits above the side panel and is drawn from the model's real parameters at every frame. The
tree and neural-network lessons need none: their side panel already is the model's structure.

``LinearDiagram`` and ``BayesDiagram`` are this project's own; both implement animkit's ``Diagram``
protocol (``draw_training`` and ``draw_inference``) and draw with animkit's ``diagrams`` helpers. The
forest, boosting and kNN diagrams are animkit's shared ones (``animkit.lessons.model_diagrams``), set up
here with ``PASSENGER``, a ``Reading`` of how a Titanic passenger and a prediction read on screen.
"""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from animkit.lessons import diagrams
from animkit.lessons.model_diagrams import BoostingDiagram, ForestDiagram, NeighboursDiagram, Reading
from animkit.lessons.lesson import plural
from titanic_anim.lessons.config import AGE, FEMALE, LOG_FARE
from titanic_anim.lessons.data import Passenger, PassengerData
from titanic_anim.lessons.lesson import ClassSnapshot, outcome, percent, signed, signed_small
from titanic_anim.lessons.maps import SCORE_CMAP

INPUT_NAMES = ("age", "fare", "woman")
LINEAR_OUTPUTS = {
    "logistic": ("σ", "P(survive)"),
    "perceptron": ("sign", "survived?"),
    "svm": ("score", "sign"),
}
LINEAR_TITLES = {
    "logistic": "The model: P(survive) = σ(bias + Σ weight × input)",
    "perceptron": "The model: survived if bias + Σ weight × input > 0",
    "svm": "The model: a score with a margin, survived if score > 0",
}
CLASSES = (("survived", 1), ("did not", 0))
Standardise = Callable[[np.ndarray], np.ndarray]  # raw passenger features → the scaled inputs a model sees
LOG_TERMS = ("log prior", "age", "log fare", "sex")


def survival_colour(value: np.ndarray):
    """Tree nodes coloured by the share of their passengers who survived.

    Args:
        value: A tree node's value as scikit-learn stores it: per-class counts or shares, (did not survive,
            survived).

    Returns:
        The node's colour on the maps' red-grey-blue scale.
    """
    counts = np.ravel(value)
    return SCORE_CMAP(float(counts[1] / counts.sum()))


def weight_scale(thetas: Iterable[np.ndarray]) -> float:
    """The diagram's edge scale for a lesson: the largest input weight any of its states reaches.

    Args:
        thetas: Every state's (bias, weights...) vector; the bias is left out.

    Returns:
        The largest absolute weight, or 1 if every weight is 0.
    """
    return max((float(np.max(np.abs(np.asarray(theta)[1:]))) for theta in thetas), default=0.0) or 1.0


def _inputs(features: np.ndarray, z: np.ndarray) -> list[str]:
    """The input labels at inference: each raw value → the scaled value the model sees."""
    sex = "woman" if features[FEMALE] == 1.0 else "man"
    return [f"{features[AGE]:.0f} y → {z[AGE]:+.2f}", f"${np.expm1(features[LOG_FARE]):,.0f} → {z[LOG_FARE]:+.2f}",
            f"{sex} → {z[FEMALE]:+.2f}"]


class LinearDiagram:
    """Scaled inputs → weighted edges → Σ (+ bias) → σ for logistic, sign for perceptron and SVM.

    Implements animkit's ``Diagram`` protocol. Training edges share the lesson's largest weight as their
    scale, so they visibly grow; at inference each edge shows its weight × this passenger's input.

    Example:
        >>> LinearDiagram("logistic", standardise, lambda s: s.theta, final.theta, 2.0,
        ...               probability=sigmoid)  # doctest: +SKIP
    """

    def __init__(self, kind: str, standardise: Standardise, theta_of: Callable[[Any], np.ndarray],
                 final_theta: np.ndarray, scale: float, probability: Callable[[float], float] | None = None):
        """Sets up the diagram for one linear lesson.

        Args:
            kind: ``"logistic"``, ``"perceptron"`` or ``"svm"``: which title and output chain to draw.
            standardise: Turns a passenger's raw features into the scaled inputs the model sees.
            theta_of: Reads the (bias, weights...) vector from a snapshot's ``detail``.
            final_theta: The final model's (bias, weights...), used at inference.
            scale: The edge scale: the largest weight the lesson reaches (``weight_scale``).
            probability: Turns the score into P(survive), for the logistic diagram only.

        Raises:
            ValueError: If ``probability`` is given for a perceptron or SVM, or missing for logistic.
        """
        if (kind == "logistic") != (probability is not None):
            raise ValueError("only the logistic diagram turns its score into a probability")
        self._kind = kind
        self._probability = probability
        self._standardise = standardise
        self._theta_of = theta_of
        self._final = final_theta
        self._scale = scale

    def _draw(self, ax: Axes, theta: np.ndarray, **inference: Any) -> None:
        """Draws the weighted sum for one θ; ``inference`` passes a passenger's values on to animkit."""
        diagrams.weighted_sum(ax, INPUT_NAMES, theta[1:], f"+ bias {theta[0]:+.2f}", LINEAR_OUTPUTS[self._kind],
                              title=LINEAR_TITLES[self._kind], fmt=signed_small, scale=self._scale,
                              **inference)

    def draw_training(self, ax: Axes, snapshot: ClassSnapshot) -> None:
        """Draws the weights at one moment of training.

        Args:
            ax: The diagram's axes.
            snapshot: The moment to draw.
        """
        self._draw(ax, self._theta_of(snapshot.detail))

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the final model's sum for the unseen passenger, input by input, then its result.

        Args:
            ax: The diagram's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        z = self._standardise(query.features)
        score = float(self._final[0] + z @ self._final[1:])
        result = (percent(float(self._probability(score))) if self._probability is not None else
                  f"{signed(score)} → {outcome(int(score > 0))}")
        self._draw(ax, self._final, values=_inputs(query.features, z), pushes=self._final[1:] * z,
                   result=result, reveal=t)


def log_terms(state, features: np.ndarray) -> np.ndarray:
    """The log terms that add up to naive Bayes' score for each outcome.

    Args:
        state: A naive Bayes state: the class priors and each class's means and variances.
        features: One passenger's (age, log fare, female) row.

    Returns:
        Shape (class, 4): the log prior, then the log bell-curve density of age, log fare and sex. Row 0
        is 'did not survive', row 1 'survived'; each row sums to GaussianNB's joint log-likelihood.
    """
    densities = -0.5 * (np.log(2 * np.pi * state.var) + (features - state.mean) ** 2 / state.var)
    return np.column_stack([np.log(state.prior), densities[:, [AGE, LOG_FARE, FEMALE]]])


class BayesDiagram:
    """Per outcome: prior × a bell-curve likelihood per input, normalised into P(survive) by Bayes' rule.

    Implements animkit's ``Diagram`` protocol. During training each factor shows the estimate counted so
    far; at inference the factors' logs add up per outcome, and the two totals are normalised.
    """

    def __init__(self, final, posterior: Callable[[Any, np.ndarray], np.ndarray]):
        """Sets up the diagram with the final model, for inference.

        Args:
            final: The final naive Bayes state.
            posterior: Turns a state and feature rows into P(survive) for each row.
        """
        self._final = final
        self._posterior = posterior

    def draw_training(self, ax: Axes, snapshot: ClassSnapshot) -> None:
        """Draws each outcome's prior and bell curves as counted so far.

        Args:
            ax: The diagram's axes.
            snapshot: The moment to draw.
        """
        state = snapshot.detail
        sd = np.sqrt(state.var)
        rows = [(name, [f"prior\n{state.prior[c]:.0%}",
                        f"age\nmean {state.mean[c, AGE]:.0f}\nsd {sd[c, AGE]:.0f}",
                        f"log fare\nmean {state.mean[c, LOG_FARE]:.1f}\nsd {sd[c, LOG_FARE]:.1f}",
                        f"women\n{state.mean[c, FEMALE]:.0%}"]) for name, c in CLASSES]
        diagrams.factor_rows(ax, rows, title=f"Prior × bell-curve likelihoods, from {plural(state.seen, 'passenger')}",
                             merge="normalise")

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the unseen passenger's log terms per outcome, and P(survive) once ``t`` reaches 1.

        Args:
            ax: The diagram's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        terms = log_terms(self._final, query.features)
        rows = [(name, [f"{label}\n{signed_small(terms[c, column])}" for column, label in enumerate(LOG_TERMS)])
                for name, c in CLASSES]
        result = f"P(survive)\n{percent(float(self._posterior(self._final, query.features[None, :])[0]))}"
        diagrams.factor_rows(ax, rows, title="This passenger: logs add up per outcome, then normalise",
                             merge="normalise", joiner="+", result=result if t >= 1.0 else None)


PASSENGER = Reading(subject="passenger", row=lambda query: query.features[None, :],
                    predict=lambda model, row: float(model.predict_proba(row)[0, 1]), show=percent)


def forest_diagram(forest) -> ForestDiagram:
    """The forest's first trees feeding an average P(survive), each voting at inference.

    Args:
        forest: The fitted ``RandomForestClassifier`` (grown by warm start, so earlier frames are its first trees).

    Returns:
        animkit's forest diagram, set up for a passenger.
    """
    return ForestDiagram(forest, PASSENGER, colour=survival_colour, summary="average\nP(survive)",
                         show_vote="{:.0%}".format)


def boosting_diagram(booster, data: PassengerData) -> BoostingDiagram:
    """F₀ (log-odds of the base rate) plus each small tree's log-odds correction, then σ.

    Args:
        booster: The boosting lesson's ``Booster``, which reads the log-odds after any number of trees.
        data: The passengers; F₀ is the same for every passenger, so one grid row gives it.

    Returns:
        animkit's boosting diagram, set up for a passenger.
    """
    return BoostingDiagram(booster, PASSENGER, f0=float(booster.base(data.grid[:1])[0]), show_score=signed_small,
                           show_step=signed_small, link="σ", step_noun="log-odds correction")


def neighbours_diagram(data: PassengerData, final, k: int, standardise: Standardise) -> NeighboursDiagram:
    """A kNN has no weights: store, find the k most similar passengers, take the share who survived.

    Args:
        data: The passengers.
        final: The final fitted ``KNeighborsClassifier``, used at inference.
        k: The final number of neighbours.
        standardise: Turns a passenger's raw features into the scaled inputs the model compares.

    Returns:
        animkit's neighbours diagram, set up for a passenger.
    """
    def steps(count: int) -> list[str]:
        """The diagram's four steps, for a model with ``count`` neighbours.

        Args:
            count: The model's k.

        Returns:
            One label per step: store, find the nearest, take the share, P(survive).
        """
        return [f"store all\n{data.n_train:,} passengers", f"find the k = {count}\nmost similar",
                "share who\nsurvived", "P(survive)"]

    def values(query: Passenger) -> list[str]:
        """What each step gives for the unseen passenger, from the final model's real neighbours.

        Args:
            query: The unseen passenger.

        Returns:
            One value per step: how many are stored, how many of the neighbours survived, the share,
            and P(survive).
        """
        rows = final.kneighbors(standardise(query.features)[None, :], return_distance=False)[0]
        survived = int(data.y_train[rows].sum())
        return [f"{data.n_train:,} stored", f"{survived} survived, {len(rows) - survived} not",
                f"{survived}/{len(rows)}", percent(survived / len(rows))]

    return NeighboursDiagram(PASSENGER, k, steps=steps, values=values, verb="decides")
