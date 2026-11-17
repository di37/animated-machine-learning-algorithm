"""k-nearest neighbours: "training" stores the passengers; each frame refits with a larger k.

P(survive) is the share of the k most similar passengers (scaled age, fare and sex) who survived.
Similar means close: the Euclidean distance between standardised (age, log fare, female) rows, so the
three inputs are measured in comparable units. Every k in ``K_VALUES`` (up to the number of training
passengers) is a frame; the one with the best validation accuracy is kept.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from sklearn.neighbors import KNeighborsClassifier

from animkit.lessons import theme
from animkit.lessons.data import read_only_indices
from animkit.lessons.lesson import Story, plural
from titanic_anim.lessons import maps
from titanic_anim.lessons.config import AGE, LOG_FARE
from titanic_anim.lessons.data import Passenger, PassengerData
from titanic_anim.lessons.lesson import (
    PROBABILITY,
    ClassLesson,
    ClassSnapshot,
    best_validation,
    make_snapshot,
    percent,
    verified_score,
)
from titanic_anim.lessons.lesson_diagrams import neighbours_diagram
from titanic_anim.lessons.stories.common import Standardiser, describe_passenger

K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PROBE = np.array([30.0, np.log1p(10.0), 0.0])  # the side panel follows a 30-year-old man on a $10 ticket
PANEL_FARE_TICKS = (0, 10, 50, 250)  # the side map is small: fewer fare labels


@dataclass(frozen=True)
class KnnState:
    """The kNN model at one frame.

    Attributes:
        k: How many neighbours vote.
        probe_neighbours: The training rows nearest ``PROBE``, the passenger the side panel follows.
    """

    k: int
    probe_neighbours: np.ndarray  # training rows nearest PROBE


class NeighbourPanel:
    """kNN's side panel (a ``ClassPanel``): one map with a passenger's k nearest neighbours ringed.

    During training it follows ``PROBE`` as k grows; at inference it shows the unseen passenger's
    neighbours, and rings them on the main maps too.
    """

    def __init__(self, data: PassengerData, standardise: Standardiser, final: KNeighborsClassifier):
        """Sets up the panel.

        Args:
            data: The passengers.
            standardise: The input scaling the distances are measured in.
            final: The kept model, used at inference.
        """
        self._data = data
        self._standardise = standardise
        self._final = final

    def neighbours(self, model: KNeighborsClassifier, features: np.ndarray) -> np.ndarray:
        """A passenger's nearest training passengers under a model.

        Args:
            model: A fitted kNN; its k sets how many.
            features: One passenger's (age, log fare, female) row, unscaled.

        Returns:
            The neighbours' training rows, nearest first.
        """
        return model.kneighbors(self._standardise(features)[None, :], return_distance=False)[0]

    def _map(self, ax: Axes, features: np.ndarray, rows: np.ndarray, title: str) -> None:
        """Draws the passenger's own map with their neighbours ringed and the share who survived."""
        data = self._data
        sex = maps.map_of(features)
        maps.setup_map(ax, data, sex, fare_labels=True, fare_ticks=PANEL_FARE_TICKS)
        maps.draw_passengers(ax, data, sex, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.scatter(data.x_train[rows, AGE], data.x_train[rows, LOG_FARE], s=90, facecolors="none",
                   edgecolors=theme.HIGHLIGHT, linewidths=1.8, zorder=6)
        ax.scatter([features[AGE]], [features[LOG_FARE]], s=160, marker="D", color=theme.UNSEEN, zorder=7,
                   edgecolors=theme.PANEL, linewidths=1.5)
        survived = int(data.y_train[rows].sum())
        ax.text(0.98, 0.95, f"{survived} of {len(rows)} survived → {survived / len(rows):.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=13, weight="bold", color=theme.NAVY,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=theme.PANEL, edgecolor="none", alpha=0.9))

    def draw_training(self, ax: Axes, snapshot: ClassSnapshot) -> None:
        """Draws the ``PROBE`` passenger's k nearest neighbours at this frame's k.

        Args:
            ax: The side panel's axes.
            snapshot: The moment to draw.
        """
        state: KnnState = snapshot.detail
        self._map(ax, PROBE, state.probe_neighbours,
                  f"The {state.k} nearest to a {describe_passenger(PROBE)}")

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the unseen passenger's nearest neighbours under the kept model.

        Args:
            ax: The side panel's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1 (the panel does not change with it).
        """
        rows = self.neighbours(self._final, query.features)
        self._map(ax, query.features, rows, f"The new passenger's {plural(len(rows), 'nearest neighbour')}")

    def overlay_training(self, axes, snapshot) -> None:
        """The panel shows the neighbourhood.

        Args:
            axes: The two maps, women's first.
            snapshot: The moment being drawn.
        """

    def overlay_inference(self, axes, query: Passenger, t: float) -> None:
        """Rings the unseen passenger's neighbours on the main maps, fading in with ``t``.

        Large neighbourhoods can reach across both sexes, so each map rings the neighbours on it.

        Args:
            axes: The two maps, women's first.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        if t <= 0:
            return
        data = self._data
        rows = self.neighbours(self._final, query.features)
        for sex, ax in enumerate(axes):
            mine = rows[maps.on_map(data.x_train[rows], sex)]
            ax.scatter(data.x_train[mine, AGE], data.x_train[mine, LOG_FARE], s=95, facecolors="none",
                       edgecolors=theme.HIGHLIGHT, linewidths=1.8, alpha=t, zorder=6)


def _story(data: PassengerData, chosen: ClassSnapshot, k_best: int) -> Story:
    """The lesson's narration; ``chosen`` is the kept snapshot and ``k_best`` its k."""
    return Story(
        key="knn", title="k-Nearest Neighbours",
        method="no training, just memory: ask the k most similar passengers",
        objective=("k-nearest neighbours learns no formula. It stores every passenger and, for a new one, "
                   "looks up the k most similar: their survival share is the prediction. The only real "
                   "choice is k; watch what too few and too many neighbours do to the maps."),
        initial=(f"Training is just storing {data.n_train} passengers. With k = 1 every point copies the "
                 "single most similar passenger, so the maps are a speckled 0% / 100% patchwork."),
        training=("Each frame refits with a larger k (no weights, no gradients). Similarity uses scaled "
                  "age, fare and sex: small neighbourhoods stay within one sex, very large ones must mix both."),
        final=(f"Validation chose k = {k_best}: train accuracy {chosen.train_accuracy:.0%}, validation "
               f"{chosen.val_accuracy:.0%}. Small k chases noise; huge k drifts towards the overall rate."),
        inference=(f"The unseen passenger finds the {k_best} most similar passengers (teal rings); the share "
                   "of them who survived is the probability."),
        takeaway="kNN predicts by analogy: similar passengers probably shared a fate, and k sets how many to trust.",
        limitation=("every prediction searches the stored passengers, and the answer depends on how the "
                    "inputs are scaled: 'similar' is a choice, not a fact."),
        progress_label="k, the number of neighbours (log scale)", log_progress=True)


def build(data: PassengerData) -> ClassLesson:
    """Builds the kNN lesson: one snapshot per k, the best on validation kept.

    Args:
        data: The passengers.

    Returns:
        The lesson.
    """
    standardise = Standardiser.fit(data.x_train)
    z = standardise(data.x_train)
    ks = tuple(k for k in K_VALUES if k <= data.n_train)
    models = [KNeighborsClassifier(n_neighbors=k).fit(z, data.y_train) for k in ks]

    def scorer(model: KNeighborsClassifier):
        """One k's model as a scorer.

        Args:
            model: A fitted kNN.

        Returns:
            A function from unscaled feature rows to P(survive).
        """
        return lambda x: model.predict_proba(standardise(x))[:, 1]

    def note(k: int) -> str:
        """The training callout for one k.

        Args:
            k: The number of neighbours.

        Returns:
            What that k does to the maps.
        """
        if k == 1:
            return "k = 1: copy the single most similar passenger\nevery map cell is 0% or 100%"
        return f"k = {k}: survival share of the {k} most similar\nmore neighbours, smoother maps"

    probe_rows = [read_only_indices(m.kneighbors(standardise(PROBE)[None, :], return_distance=False)[0])
                  for m in models]
    snapshots = tuple(make_snapshot(scorer(m), data, PROBABILITY, label=f"k = {k}", progress=k, note=note(k),
                                    detail=KnnState(k, rows))
                      for k, m, rows in zip(ks, models, probe_rows))
    best = best_validation(snapshots)
    final, k_best = models[best], ks[best]
    panel = NeighbourPanel(data, standardise, final)
    score = scorer(final)

    def explain(features: np.ndarray):
        """Works one passenger's prediction through: find the k nearest, count the survivors.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            The verified explanation, checked against the kept model's ``predict_proba``.
        """
        rows = panel.neighbours(final, features)
        survived = int(data.y_train[rows].sum())
        worked = survived / len(rows)
        lines = ["similarity: distance in scaled age, fare and sex",
                 f"k = {k_best}: the {k_best} most similar training passengers",
                 f"{survived} survived, {len(rows) - survived} did not",
                 f"P(survive) = {survived}/{len(rows)} = {percent(worked)}"]
        return verified_score(lines, worked, float(score(features[None, :])[0]), PROBABILITY)

    return ClassLesson(
        story=_story(data, snapshots[best], k_best), data=data, snapshots=snapshots, final_index=best,
        score=score, kind=PROBABILITY, explain=explain, panel=panel,
        diagram=neighbours_diagram(data, final, k_best, standardise),
        config={"model": "KNeighborsClassifier", "k_values": ", ".join(map(str, ks)), "weights": "uniform",
                "metric": "euclidean on standardised age, log fare, woman",
                "selection": "highest validation accuracy"})
