"""Algorithm diagrams for the regression lessons: how each model turns an income into a price.

Each diagram is drawn from the model's real parameters at every frame. The tree and neural-network
lessons need none: their side panel already is the model's structure.

A diagram sits above a lesson's side panel and implements animkit's ``Diagram`` protocol: a
``draw_training`` for each snapshot and a ``draw_inference`` for the unseen district. The linear and
polynomial diagrams are drawn here with animkit's ``diagrams.weighted_sum`` (inputs, weighted edges,
a Σ hub). The forest, boosting and kNN diagrams are animkit's own (``animkit.lessons.model_diagrams``),
given ``HOUSE``, this project's ``Reading`` of a query as a house price, and its money formats.
"""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from sklearn.pipeline import Pipeline

from animkit.lessons import diagrams, theme
from animkit.lessons.model_diagrams import BoostingDiagram, ForestDiagram, NeighboursDiagram, Reading
from house_anim.lessons.config import PRICE_CAP, PRICE_TO_K
from house_anim.lessons.data import IncomeData, Query
from house_anim.lessons.lesson import Snapshot, income_k, money_k, predict_one

PRICE_NORM = Normalize(0.0, PRICE_CAP)
FINE_BELOW_K = 10.0  # corrections smaller than $10k keep one decimal, so they never read as "$0k"
DOLLARS_BELOW_K = 1.0  # and those under $1k are shown in whole dollars
DOLLARS_PER_K = 1000.0
WIDTH_DECADES = 3.0  # polynomial weights span orders of magnitude: edge widths on a log scale, like the bars


def price_colour(value: np.ndarray):
    """Tree nodes coloured by the mean price they predict (light = cheap, dark = expensive).

    Args:
        value: A tree node's value array; its first entry is the node's mean price ($100k).

    Returns:
        The node's RGBA colour, on a scale from $0 to the $500k cap.
    """
    return theme.value_colormap()(PRICE_NORM(float(np.ravel(value)[0])))


def signed_money(value: float) -> str:
    """An amount with its sign, for an edge's weight or push: '+$12k' or '−$12k'.

    Args:
        value: The amount ($100k).

    Returns:
        The signed amount in whole $k.
    """
    return money_k(value, signed=True)


def correction(value: float) -> str:
    """A boosting step: '+$12k', '−$4.2k', '+$37' (late trees add only dollars), or '$0'.

    Args:
        value: The step ($100k).

    Returns:
        The signed step: whole $k from $10k up, $k to one decimal from $1k, whole dollars below that.
    """
    amount = value * PRICE_TO_K
    sign = "+" if amount >= 0 else "−"
    if abs(amount) >= FINE_BELOW_K:
        return signed_money(value)
    if abs(amount) >= DOLLARS_BELOW_K:
        return f"{sign}${abs(amount):.1f}k"
    dollars = round(abs(amount) * DOLLARS_PER_K)
    return f"{sign}${dollars:,}" if dollars else "$0"


class LinearDiagram:
    """ŷ = θ₀ + θ₁·z, z = scaled income: one weighted input into Σ, animkit's ``Diagram`` for linear regression.

    Edge widths share one scale for the whole lesson (the largest θ₁ it reaches), so the weight visibly grows.
    """

    def __init__(self, standardise, final_theta: tuple[float, float], scale: float):
        """Keeps what the diagram needs to draw any snapshot and the final model's prediction.

        Args:
            standardise: Turns raw incomes ($10k) into the scaled z the model sees.
            final_theta: The final model's (θ₀, θ₁), in $100k.
            scale: The edge-width scale: the largest θ₁ the lesson reaches.
        """
        self._standardise = standardise
        self._final = final_theta
        self._scale = scale

    def _draw(self, ax: Axes, theta, **inference: Any) -> None:
        """Draws the weighted sum for (θ₀, θ₁); ``inference`` passes the query's values, pushes and result."""
        diagrams.weighted_sum(ax, ["income"], np.array([theta[1]]), f"+ θ₀ {money_k(theta[0])}", ["ŷ price"],
                              title="The model: ŷ = θ₀ + θ₁ · z", fmt=signed_money, scale=self._scale, **inference)

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws the model with one training step's weights.

        Args:
            ax: The diagram's axes.
            snapshot: The step; its ``detail.theta`` holds (θ₀, θ₁).
        """
        self._draw(ax, snapshot.detail.theta)

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Draws the final model pricing the unseen district: income → z, the push θ₁·z, then ŷ.

        Args:
            ax: The diagram's axes.
            query: The unseen district.
            t: How far the reveal has got, from 0 to 1; the price appears at 1.
        """
        z = float(self._standardise(np.array([query.income]))[0])
        price = self._final[0] + self._final[1] * z
        self._draw(ax, self._final, values=[f"{income_k(query.income)} → z {z:+.2f}"],
                   pushes=np.array([self._final[1] * z]), result=money_k(price), reveal=t)


class PolynomialDiagram:
    """x, x², …: one weighted edge per term into Σ; grey terms are not in the model, dashed ones are 0.

    animkit's ``Diagram`` for the polynomial, Ridge and Lasso lessons. Training edges share the largest
    weight any snapshot reaches as their scale; edge widths follow a log scale, as the weights span
    orders of magnitude.
    """

    def __init__(self, final: Pipeline, names: tuple[str, ...], title: str, scale: float):
        """Keeps the final model and how to label and scale its terms.

        Args:
            final: The final fitted pipeline (scalers, polynomial features, then the regressor).
            names: Every term's name, up to the highest degree any snapshot uses.
            title: The diagram's title.
            scale: The edge-width scale for training: the largest weight any snapshot reaches ($100k).
        """
        self._final = final
        self._names = names
        self._title = title
        self._scale = scale

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws the weighted sum with one fit's weights and intercept.

        Args:
            ax: The diagram's axes.
            snapshot: The fit; its ``detail`` holds ``coef`` (NaN for absent terms) and ``intercept``.
        """
        state = snapshot.detail
        diagrams.weighted_sum(ax, self._names, state.coef, f"+ b {money_k(state.intercept)}", ["ŷ price"],
                              title=self._title, fmt=signed_money, scale=self._scale, decades=WIDTH_DECADES)

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Draws the final model pricing the unseen district: each term's push lights up, then ŷ appears.

        Args:
            ax: The diagram's axes.
            query: The unseen district.
            t: How far the reveal has got, from 0 to 1; the price appears at 1.
        """
        features = self._final[:-1].transform(np.array([[query.income]]))[0]
        weights = self._final[-1].coef_
        coef = np.full(len(self._names), np.nan)
        coef[:len(weights)] = weights
        price = float(self._final.predict(np.array([[query.income]]))[0])
        diagrams.weighted_sum(ax, self._names, coef, f"+ b {money_k(float(self._final[-1].intercept_))}",
                              ["ŷ price"], title=self._title, fmt=signed_money, decades=WIDTH_DECADES,
                              values=[""] * len(self._names),
                              pushes=np.r_[weights * features, np.zeros(len(self._names) - len(weights))],
                              result=money_k(price), reveal=t)


HOUSE = Reading(subject="house", row=lambda query: np.array([[query.income]]),
                predict=lambda model, row: float(model.predict(row)[0]), show=money_k)


def forest_diagram(forest) -> ForestDiagram:
    """The forest's first trees feeding an average, each voting a price at inference.

    Args:
        forest: The fitted random forest.

    Returns:
        animkit's ``ForestDiagram`` for it, reading the query as a house and colouring nodes by price.
    """
    return ForestDiagram(forest, HOUSE, colour=price_colour, summary="average\nof all trees")


def boosting_diagram(booster, data: IncomeData) -> BoostingDiagram:
    """F₀ (the average training price) plus each small tree's correction, in dollars.

    Args:
        booster: The boosting lesson's model wrapper, with ``.model``, ``.base`` and ``.trees``.
        data: The lesson's data; F₀ is read off the booster's base at its first grid income.

    Returns:
        animkit's ``BoostingDiagram`` for it, with prices and corrections shown in money.
    """
    return BoostingDiagram(booster, HOUSE, f0=float(booster.base(data.grid[:1])[0]), show_score=money_k,
                           show_step=correction)


def neighbours_diagram(data: IncomeData, final, k: int, predict) -> NeighboursDiagram:
    """A kNN model has no weights: store, look up the k nearest incomes, average their prices.

    Args:
        data: The lesson's data, whose training districts are stored.
        final: The final fitted ``KNeighborsRegressor``, used to find the query's neighbours.
        k: The final number of neighbours.
        predict: The final model, raw income ($10k) in, price ($100k) out.

    Returns:
        animkit's ``NeighboursDiagram``: four steps, labelled for each k and for the query.
    """
    def steps(count: int) -> list[str]:
        """The four steps' labels.

        Args:
            count: The number of neighbours, k, at this moment of the lesson.

        Returns:
            One label per step: store, find the nearest, average, price.
        """
        return [f"store all\n{data.n_train:,} districts", f"find the k = {count}\nnearest incomes",
                "average\ntheir prices", "ŷ price"]

    def values(query: Query) -> list[str]:
        """What each step gives for the unseen district.

        Args:
            query: The unseen district.

        Returns:
            The stored count, the neighbours' income range, how many are averaged, and the price.
        """
        rows = final.kneighbors(HOUSE.row(query), return_distance=False)[0]
        incomes = data.x_train[rows]
        return [f"{data.n_train:,} stored", f"{income_k(incomes.min())}–{income_k(incomes.max())}",
                f"mean of {len(rows)}", money_k(predict_one(predict, query.income))]

    return NeighboursDiagram(HOUSE, k, steps=steps, values=values)
