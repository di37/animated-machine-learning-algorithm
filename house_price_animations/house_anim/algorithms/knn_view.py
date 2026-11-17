"""K-nearest neighbours on the California map: houses pile up in memory, then a query finds its k.

KNN's "training" only stores houses, so the training view shows the stored houses filling the map.
At inference the house is linked to its k nearest training houses. "Nearest" is measured over all
8 standardised features, not just location, so a neighbour can be far away on the map; each is
drawn at its own location, coloured by its price, and the prediction is their mean price.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from house_anim import theme
from house_anim.algorithms.common import PRICE_CMAP, PRICE_NORM, AlgorithmView, price_k
from house_anim.data import COORD_COLUMNS, FEATURE_COLUMNS, HousingSplits

COORD_INDEX = tuple(FEATURE_COLUMNS.index(c) for c in COORD_COLUMNS)


class NeighboursView(AlgorithmView):
    """The KNN diagram: a map of the stored training houses.

    Attributes:
        stored: The stored houses' dots.
        links_drawn: How many neighbour links the last prediction drew (the tests read it).
    """

    links_drawn = 0

    def __init__(self, ax: Axes, splits: HousingSplits):
        """Sets up the empty map over the training houses' locations.

        Args:
            ax: The axes to draw in.
            splits: The data; its raw training coordinates place the houses.
        """
        super().__init__(ax, splits)
        self._coords = splits.coords_train
        self._prepare("")

    def _prepare(self, title: str) -> None:
        """Clears the view and redraws the empty map, its limits and aspect fixed to the training houses."""
        self._blank(title)
        lon, lat = self._coords[:, 0], self._coords[:, 1]
        pad = 0.3
        self.ax.set_xlim(lon.min() - pad, lon.max() + pad)
        self.ax.set_ylim(lat.min() - pad, lat.max() + pad)
        self.ax.set_aspect(1.0 / np.cos(np.deg2rad(lat.mean())))
        self.stored = self.ax.scatter([], [], s=4, color=theme.BASELINE, linewidths=0, zorder=1)

    def show_training(self, model, frame) -> None:
        """Shows the houses the model has stored so far.

        A data-growth fit stores the first rows of the training set, so those are the dots drawn.

        Args:
            model: The fitted ``KNeighborsRegressor``.
            frame: The step's ``TrainingFrame`` (unused).
        """
        self._prepare("The algorithm: k-nearest neighbours memorises houses")
        self.stored.set_offsets(self._coords[:model.n_samples_fit_])
        self.caption(f"{model.n_samples_fit_:,} training houses stored  ·  no weights, no loss: "
                     f"a new house will borrow from its {model.n_neighbors} closest", y=-0.04)

    def _raw_location(self, x: np.ndarray) -> tuple[float, float]:
        """A house's (longitude, latitude) in degrees, undoing the scaling of those two features."""
        lon, lat = (x[i] * self.splits.feature_scale[i] + self.splits.feature_mean[i] for i in COORD_INDEX)
        return float(lon), float(lat)

    def show_prediction(self, model, x: np.ndarray, house_number: int) -> None:
        """Links the house (a star) to its k nearest training houses, coloured by their prices.

        Args:
            model: The fitted ``KNeighborsRegressor``.
            x: The house's scaled features.
            house_number: The house's position among the test houses, from 1, for the title.
        """
        self._prepare(f"House #{house_number:,} finds its {model.n_neighbors} nearest neighbours")
        self.stored.set_offsets(self._coords[:model.n_samples_fit_])
        _, indices = model.kneighbors(x[None, :])
        neighbours = self._coords[indices[0]]
        prices = self.splits.y_train[indices[0]]
        home = self._raw_location(x)
        self.ax.add_collection(LineCollection([(home, tuple(n)) for n in neighbours],
                                              colors=theme.INK_PRIMARY, linewidths=1.0, alpha=0.7, zorder=2))
        self.ax.scatter(neighbours[:, 0], neighbours[:, 1], s=70, c=prices, cmap=PRICE_CMAP, norm=PRICE_NORM,
                        edgecolors=theme.INK_PRIMARY, linewidths=0.8, zorder=3)
        self.ax.scatter([home[0]], [home[1]], s=260, marker="*", color=theme.SERIES_2,
                        edgecolors=theme.INK_PRIMARY, linewidths=0.8, zorder=4)
        self.links_drawn = len(neighbours)
        self.caption(f"nearest in all 8 features (drawn at their map location)  ·  "
                     f"average of their prices = {price_k(float(prices.mean()))}", y=-0.04)
