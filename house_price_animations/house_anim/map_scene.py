"""Watch the model learn California: each model prices the same typical house across the state.

Four models train in lockstep (``MAP_MODEL_KEYS``). After each tick, every model predicts the
typical house at every cell of the ``GeoGrid``, so the map shows the shape of what it has learned:
a straight line can only tilt one flat plane, a tree cuts the state into boxes, boosting sharpens
those boxes, and a neural network blends smooth gradients. ``MAP_CAPTIONS`` says this in a phrase
per model.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from house_anim.config import TARGET_TO_THOUSANDS
from house_anim.data import HousingSplits, read_only
from house_anim.geo_grid import GeoGrid
from house_anim.lockstep import Lane, lockstep
from house_anim.models import ModelSpec
from house_anim.panels import format_step
from house_anim.trainers import checked_predict

MAP_MODEL_KEYS = ("linear_regression", "decision_tree", "gradient_boosting", "mlp")
MAP_CAPTIONS = {
    "linear_regression": "can only tilt one flat plane",
    "ridge": "a flat plane, gently shrunk",
    "lasso": "a flat plane, with fewer features",
    "sgd": "a flat plane, found step by step",
    "decision_tree": "cuts the state into boxes",
    "random_forest": "averages many box-cutting trees",
    "gradient_boosting": "sharpens the map tree by tree",
    "knn": "copies the nearest neighbourhoods",
    "mlp": "blends smooth gradients",
}


@dataclass(frozen=True)
class MapPanel:
    """One model's map at one tick.

    Attributes:
        key: The model's key.
        short_title: Its name for the panel title.
        step_text: How far it has trained, such as ``"depth 7"``.
        val_rmse_k: Its validation RMSE, in $k.
        prices_k: One predicted price per grid cell inside California, in $k.
    """

    key: str
    short_title: str
    step_text: str
    val_rmse_k: float
    prices_k: np.ndarray  # one predicted price per grid cell inside California


@dataclass(frozen=True)
class MapFrame:
    """Every model's map at one lockstep tick.

    Attributes:
        tick: This tick, from 1.
        ticks: How many ticks there are.
        panels: One panel per model, in the order the models were given.
    """

    tick: int
    ticks: int
    panels: tuple[MapPanel, ...]

    @property
    def is_final(self) -> bool:
        """Whether this is the last tick.

        Returns:
            True on the last tick, when every model has finished its schedule.
        """
        return self.tick == self.ticks


def _panel(lane: Lane, grid: GeoGrid) -> MapPanel:
    """Predicts one lane's model on every inside cell and packs the result as a panel."""
    prices = checked_predict(lane.model, grid.features) * TARGET_TO_THOUSANDS
    return MapPanel(
        key=lane.spec.key,
        short_title=lane.spec.short_title,
        step_text=lane.spec.step_format.format(format_step(lane.frame.step)),
        val_rmse_k=lane.frame.val_scores.rmse * TARGET_TO_THOUSANDS,
        prices_k=read_only(prices),
    )


def map_frames(
    specs: Sequence[ModelSpec], splits: HousingSplits, grid: GeoGrid, ticks: int
) -> Iterator[MapFrame]:
    """One frame per lockstep tick; a model's map is only re-predicted when it trained further.

    Args:
        specs: The models to train and map.
        splits: The data they train on.
        grid: The cells to price.
        ticks: How many frames to make.

    Yields:
        One ``MapFrame`` per tick.
    """
    previous: dict[str, tuple[int, MapPanel]] = {}
    for snapshot in lockstep(specs, splits, ticks):
        panels = []
        for lane in snapshot.lanes:
            cached = previous.get(lane.spec.key)
            fresh = cached is None or cached[0] != lane.frame.step_index
            panels.append(_panel(lane, grid) if fresh else cached[1])
        previous = {p.key: (lane.frame.step_index, p) for p, lane in zip(panels, snapshot.lanes)}
        yield MapFrame(snapshot.tick, snapshot.ticks, tuple(panels))
