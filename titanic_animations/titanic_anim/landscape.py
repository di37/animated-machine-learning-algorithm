"""Who survives? Each model's P(survive) for every age, class and sex, repainted as it trains.

The grid holds one typical passenger per cell: the class's median fare, Southampton, and the
median family size of training passengers near that age (children travel with parents).

Its rows are the six groups (three classes, women and men) and its columns the ages 0 to 80. Each
model's map is its score for every cell, so the maps show the shape each algorithm gives its
answer: one smooth S-curve, boxes, bumps. The models train in lockstep, as in the race, and a map
is predicted again only when its model has trained further.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from animkit.data import read_only
from animkit.lockstep import Lane, lockstep
from animkit.panels import format_step
from titanic_anim.data import FEATURE_COLUMNS, TitanicSplits
from titanic_anim.models import ClassifierSpec
from titanic_anim.passengers import COLUMN, ordinal
from titanic_anim.trainers import survival_scores, training_steps

LANDSCAPE_MODEL_KEYS = ("logistic_regression", "decision_tree", "gradient_boosting", "mlp")
LANDSCAPE_CAPTIONS = {
    "logistic_regression": "one smooth S-curve, the same tilt at every age",
    "sgd": "one smooth S-curve, found step by step",
    "svm": "smooth bumps around the support vectors",
    "naive_bayes": "multiplies one bell curve per feature",
    "decision_tree": "cuts passengers into boxes",
    "random_forest": "averages many box-cutting trees",
    "gradient_boosting": "sharpens the boxes tree by tree",
    "knn": "copies the most similar passengers",
    "mlp": "bends smooth curves through the data",
}
AGES = tuple(range(0, 81))
NEIGHBOURHOOD_YEARS = 5


@dataclass(frozen=True)
class SurvivalGrid:
    """The typical passengers, one per (group, age) cell, ready for a model.

    Attributes:
        ages: The ages along the columns, in years.
        groups: One label per row, e.g. "1st class women".
        features: The cells' standardised features, rows in (group, age) order.
    """

    ages: np.ndarray
    groups: tuple[str, ...]  # one label per row, e.g. "1st class women"
    features: np.ndarray  # scaled model inputs, rows in (group, age) order

    @property
    def shape(self) -> tuple[int, int]:
        """The grid's size.

        Returns:
            ``(groups, ages)``.
        """
        return len(self.groups), len(self.ages)

    def to_image(self, values: np.ndarray) -> np.ndarray:
        """Folds one value per cell into the grid's shape, to draw as an image.

        Args:
            values: One value per cell, in (group, age) order.

        Returns:
            The values, shape ``(groups, ages)``.

        Raises:
            ValueError: If there is not one value per cell.
        """
        if values.size != self.shape[0] * self.shape[1]:
            raise ValueError(f"expected {self.shape[0] * self.shape[1]} cells, got {values.size}")
        return np.asarray(values).reshape(self.shape)


def _family_by_age(raw: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Median (sibsp, parch) of training passengers within a few years of each age."""
    age = raw[:, COLUMN["age"]]
    family = raw[:, [COLUMN["sibsp"], COLUMN["parch"]]]
    fallback = np.median(family, axis=0)
    rows = []
    for value in ages:
        near = np.abs(age - value) <= NEIGHBOURHOOD_YEARS
        rows.append(np.median(family[near], axis=0) if near.any() else fallback)
    return np.array(rows)


def build_survival_grid(splits: TitanicSplits, ages=AGES) -> SurvivalGrid:
    """Builds the typical passenger of every class, sex and age.

    Args:
        splits: The splits; the training passengers give the fares and family sizes, and the scaler.
        ages: The ages along the columns, in years.

    Returns:
        The grid: two rows (women, men) per class found among the training passengers.
    """
    raw = splits.raw_train
    ages = np.asarray(ages, dtype=float)
    family = _family_by_age(raw, ages)
    classes = sorted({int(c) for c in raw[:, COLUMN["pclass"]]})
    rows, groups = [], []
    for travel_class in classes:
        fare = float(np.median(raw[raw[:, COLUMN["pclass"]] == travel_class, COLUMN["fare"]]))
        for female, label in ((1.0, "women"), (0.0, "men")):
            cells = np.zeros((len(ages), len(FEATURE_COLUMNS)))
            cells[:, COLUMN["pclass"]] = travel_class
            cells[:, COLUMN["female"]] = female
            cells[:, COLUMN["age"]] = ages
            cells[:, [COLUMN["sibsp"], COLUMN["parch"]]] = family
            cells[:, COLUMN["fare"]] = fare  # embarked_C = embarked_Q = 0: Southampton
            rows.append(cells)
            groups.append(f"{ordinal(travel_class)} class {label}")
    scaled = (np.vstack(rows) - splits.feature_mean) / splits.feature_scale
    return SurvivalGrid(ages=read_only(ages), groups=tuple(groups), features=read_only(scaled))


@dataclass(frozen=True)
class LandscapePanel:
    """One model's map at one tick.

    Attributes:
        key: The model's key.
        short_title: Its name on the dashboard.
        step_text: Where it is in its schedule, e.g. "depth 4".
        val_accuracy: Its validation accuracy now.
        probabilities: One P(survive) per grid cell.
    """

    key: str
    short_title: str
    step_text: str
    val_accuracy: float
    probabilities: np.ndarray  # one P(survive) per grid cell


@dataclass(frozen=True)
class LandscapeFrame:
    """Every model's map at one tick.

    Attributes:
        tick: This tick, from 1.
        ticks: How many ticks there are.
        panels: One map per model.
    """

    tick: int
    ticks: int
    panels: tuple[LandscapePanel, ...]

    @property
    def is_final(self) -> bool:
        """Whether this is the last tick.

        Returns:
            True on the last tick.
        """
        return self.tick == self.ticks


def _panel(lane: Lane, grid: SurvivalGrid) -> LandscapePanel:
    """One model's map at this tick: the live model's score for every grid cell."""
    return LandscapePanel(
        key=lane.spec.key,
        short_title=lane.spec.short_title,
        step_text=lane.spec.step_format.format(format_step(lane.frame.step)),
        val_accuracy=lane.frame.val_scores.accuracy,
        probabilities=survival_scores(lane.model, grid.features, lane.spec.score_kind),
    )


def landscape_frames(
    specs: Sequence[ClassifierSpec], splits: TitanicSplits, grid: SurvivalGrid, ticks: int
) -> Iterator[LandscapeFrame]:
    """One frame per lockstep tick; a model's landscape is only re-predicted when it trained further.

    Args:
        specs: The models to paint.
        splits: The splits the models train on.
        grid: The typical passengers to score.
        ticks: How many frames there are; at tick k every model is k/ticks through its schedule.

    Yields:
        A ``LandscapeFrame`` per tick.
    """
    previous: dict[str, tuple[int, LandscapePanel]] = {}
    for snapshot in lockstep(specs, splits, ticks, step_fn=training_steps):
        panels = []
        for lane in snapshot.lanes:
            cached = previous.get(lane.spec.key)
            fresh = cached is None or cached[0] != lane.frame.step_index
            panels.append(_panel(lane, grid) if fresh else cached[1])
        previous = {p.key: (lane.frame.step_index, p) for p, lane in zip(panels, snapshot.lanes)}
        yield LandscapeFrame(snapshot.tick, snapshot.ticks, tuple(panels))
