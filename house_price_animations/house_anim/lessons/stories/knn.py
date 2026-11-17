"""k-nearest neighbours: "training" only stores the districts; the lesson sweeps k, a genuine refit each frame.

k runs from 1, which copies single districts, to very large k, which blurs everything into the average.

To price a district, kNN finds the k training districts with the closest incomes and averages their
prices. Nothing is fitted: the stored data is the model, and k is the one choice. k trades variance
for bias. With k = 1 the curve jumps with every noisy district; with very large k it flattens towards
the overall average. The k values grow by doubling, so the metric chart uses a log axis, and the
validation districts choose the final k. The side panel zooms in on the neighbourhood of one fixed
income (``PROBE_INCOME``) during the sweep, and on the unseen district's at inference.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from sklearn.neighbors import KNeighborsRegressor

from animkit.lessons import theme
from animkit.lessons.lesson import plural
from animkit.lessons.timing import phase
from house_anim.lessons.config import INCOME_TO_K, PRICE_TO_K, Y_RANGE_K
from house_anim.lessons.data import IncomeData, Query, read_only_indices
from house_anim.lessons.lesson import (
    Explanation,
    Lesson,
    Panel,
    Snapshot,
    Story,
    best_validation,
    income_k,
    make_snapshot,
    money_k,
    predict_one,
    verified,
)
from house_anim.lessons.lesson_diagrams import neighbours_diagram
from house_anim.lessons.plots import K_FORMAT

K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
PROBE_INCOME = 3.5  # the $35k income whose neighbourhood the side panel shows during the sweep
CHUNK = 2_000  # predict in chunks so k = 1024 never builds a huge neighbour matrix
MAX_PANEL_DOTS = 3_000


@dataclass(frozen=True)
class KnnState:
    """One k of the sweep, for the side panel.

    Attributes:
        k: The number of neighbours.
        probe_neighbours: The training rows nearest ``PROBE_INCOME`` at this k.
    """

    k: int
    probe_neighbours: np.ndarray  # training rows nearest PROBE_INCOME


def _model(data: IncomeData, k: int) -> KNeighborsRegressor:
    """A kNN model with ``k`` neighbours over the training incomes: fitting only stores them."""
    return KNeighborsRegressor(n_neighbors=k).fit(data.x_train.reshape(-1, 1), data.y_train)


def _predictor(model: KNeighborsRegressor) -> Callable[[np.ndarray], np.ndarray]:
    """The model as a predictor that works in chunks of ``CHUNK`` incomes, to bound memory."""
    def predict(x: np.ndarray) -> np.ndarray:
        """The model's prices.

        Args:
            x: Raw incomes ($10k).

        Returns:
            The predicted prices ($100k): each the mean of its k neighbours' prices.
        """
        column = np.asarray(x, dtype=float).reshape(-1, 1)
        return np.concatenate([model.predict(column[i:i + CHUNK]) for i in range(0, len(column), CHUNK)])
    return predict


def neighbours(model: KNeighborsRegressor, income: float) -> np.ndarray:
    """The training districts nearest one income.

    Args:
        model: A fitted kNN model; its k sets how many are found.
        income: The income ($10k).

    Returns:
        The k nearest training rows, nearest first.
    """
    return model.kneighbors(np.array([[income]]), return_distance=False)[0]


class NeighbourPanel(Panel):
    """A zoom on one income's neighbourhood: the k nearest districts lit, and the average of their prices."""

    def __init__(self, data: IncomeData, final: KNeighborsRegressor):
        """Keeps the data and the final model.

        Args:
            data: The lesson's data.
            final: The kNN model with the k validation chose.
        """
        self._data = data
        self._final = final

    def _zoom(self, ax: Axes, centre: float, rows: np.ndarray, title: str, reveal: float = 1.0) -> None:
        """Zooms on ``centre``'s neighbours ``rows``: they fade in with ``reveal``; at 1 their average appears."""
        data = self._data
        nearest = data.x_train[rows]
        half = max(np.abs(nearest - centre).max() * 1.6, 0.25)
        theme.style_axes(ax)
        ax.set_xlim((centre - half) * INCOME_TO_K, (centre + half) * INCOME_TO_K)
        ax.set_ylim(*Y_RANGE_K)
        ax.xaxis.set_major_formatter(K_FORMAT)
        ax.yaxis.set_major_formatter(K_FORMAT)
        ax.set_title(title)
        window = np.flatnonzero(np.abs(data.x_train - centre) <= half)[:MAX_PANEL_DOTS]
        ax.scatter(data.x_train[window] * INCOME_TO_K, data.y_train[window] * PRICE_TO_K, s=8,
                   color=theme.DOTS, alpha=0.5, linewidths=0)
        ax.axvline(centre * INCOME_TO_K, color=theme.NAVY, linewidth=1.4)
        if reveal > 0:
            ax.scatter(nearest * INCOME_TO_K, data.y_train[rows] * PRICE_TO_K, s=22, color=theme.HIGHLIGHT,
                       alpha=reveal, zorder=3, linewidths=0)
        if reveal >= 1:
            mean = float(data.y_train[rows].mean()) * PRICE_TO_K
            ax.plot([nearest.min() * INCOME_TO_K, nearest.max() * INCOME_TO_K], [mean, mean],
                    color=theme.MODEL, linewidth=3.2, zorder=4)
            ax.text(0.98, 0.94, f"average of the {len(rows)}: ${mean:,.0f}k", transform=ax.transAxes,
                    ha="right", va="top", fontsize=13, weight="bold", color=theme.NAVY)

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws the k nearest districts to ``PROBE_INCOME`` and their average, for this snapshot's k.

        Args:
            ax: The side panel.
            snapshot: The k being shown; its ``detail`` is a ``KnnState``.
        """
        state: KnnState = snapshot.detail
        self._zoom(ax, PROBE_INCOME, state.probe_neighbours,
                   f"The {plural(state.k, 'nearest district')} to a {income_k(PROBE_INCOME)} income")

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Lights the unseen district's neighbours (in the first half of ``t``), then their average.

        Args:
            ax: The side panel.
            query: The unseen district.
            t: How far through the animation, from 0 to 1.
        """
        rows = neighbours(self._final, query.income)
        self._zoom(ax, query.income, rows, f"The new house's {plural(len(rows), 'nearest neighbour')}",
                   reveal=phase(t, 0.0, 0.5))

    def overlay_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Shades the neighbours' income range on the main plot and lights them, fading in with ``t``.

        Args:
            ax: The main plot.
            query: The unseen district.
            t: How far the overlay has faded in, from 0 (not drawn) to 1.
        """
        rows = neighbours(self._final, query.income)
        nearest = self._data.x_train[rows]
        if t <= 0:
            return
        ax.axvspan(nearest.min() * INCOME_TO_K, nearest.max() * INCOME_TO_K, color=theme.HIGHLIGHT,
                   alpha=0.14 * t, zorder=0)
        ax.scatter(nearest * INCOME_TO_K, self._data.y_train[rows] * PRICE_TO_K, s=20,
                   color=theme.HIGHLIGHT, alpha=t, zorder=4, linewidths=0)


def _story(data: IncomeData, chosen: Snapshot, k_best: int) -> Story:
    """Every word of the kNN video, with the k validation chose and its errors."""
    return Story(
        key="knn", title="k-Nearest Neighbours",
        method="no training, just memory: average the k closest districts",
        objective=("k-nearest neighbours never fits a formula. It stores every district and, when "
                   "asked about a new one, averages the prices of the k most similar. The only "
                   "real choice is k: watch what too few and too many neighbours do."),
        initial=(f"Training is just storing all {data.n_train:,} districts. With k = 1 each "
                 "prediction copies the single closest district, noise and all."),
        training=("Each frame refits with a larger k (no gradients, no weights). The right panel "
                  "shows the neighbourhood around a $35k income growing."),
        final=(f"Validation chose k = {k_best}: train RMSE ${chosen.train_rmse_k:,.0f}k, validation "
               f"${chosen.val_rmse_k:,.0f}k. Small k chases noise; huge k flattens towards the average."),
        inference=(f"The unseen district finds its {k_best} nearest neighbours by income (teal); its "
                   "prediction is simply their average price."),
        takeaway="kNN predicts by analogy: similar inputs should have similar outputs, and k sets how many to trust.",
        limitation=("every prediction searches the stored data, so it gets slow and memory-hungry, "
                    "and with many features 'nearest' stops meaning much."),
        progress_label="k, the number of neighbours (log scale)", log_progress=True)


def build(data: IncomeData) -> Lesson:
    """Builds the kNN lesson: one model per k in ``K_VALUES`` (up to the number of training districts).

    Args:
        data: The income view to learn from.

    Returns:
        The lesson, its final model the k with the lowest validation RMSE.
    """
    ks = tuple(k for k in K_VALUES if k <= data.n_train)
    models = [_model(data, k) for k in ks]

    def note(k: int) -> str:
        """The callout for one k.

        Args:
            k: The number of neighbours.

        Returns:
            Two lines of callout text: what k does to the curve.
        """
        if k == 1:
            return "k = 1: copy the single closest district\nevery quirk in the data becomes a jump in the curve"
        return f"k = {k}: average the {k} closest districts\nmore neighbours, smoother curve"

    snapshots = tuple(
        make_snapshot(_predictor(model), data, label=f"k = {k}", progress=k, note=note(k),
                      detail=KnnState(k, read_only_indices(neighbours(model, PROBE_INCOME))))
        for k, model in zip(ks, models))
    best = best_validation(snapshots)
    final, k_best = models[best], ks[best]
    predict = _predictor(final)

    def explain(income: float) -> Explanation:
        """Works kNN's price for one income: its k nearest districts, then the average of their prices.

        Args:
            income: The income ($10k).

        Returns:
            The worked lines, checked against the model's own prediction.
        """
        rows = neighbours(final, income)
        prices = data.y_train[rows]
        worked = float(prices.mean())
        lines = [f"stored: all {data.n_train:,} training districts (nothing else is learned)",
                 f"k = {k_best}: nearest incomes {income_k(data.x_train[rows].min())} to "
                 f"{income_k(data.x_train[rows].max())}",
                 f"their prices: {money_k(prices.min())} to {money_k(prices.max())}",
                 f"ŷ = average of the {k_best} prices = {money_k(worked)}"]
        return verified(lines, worked, predict_one(predict, income))

    story = _story(data, snapshots[best], k_best)
    return Lesson(
        story=story, data=data, snapshots=snapshots, final_index=best, predict=predict,
        explain=explain, panel=NeighbourPanel(data, final),
        diagram=neighbours_diagram(data, final, k_best, predict),
        config={"model": "KNeighborsRegressor", "k_values": ", ".join(map(str, ks)),
                "weights": "uniform", "metric": "euclidean on raw MedInc",
                "selection": "lowest validation RMSE"})
