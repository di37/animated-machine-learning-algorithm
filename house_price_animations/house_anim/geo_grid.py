"""A longitude/latitude grid over populated California, each cell holding the same typical house.

To see what a model has learned about location, every cell gets the same house: the median of
every feature (income, age, rooms, occupancy...), with only longitude and latitude set to the
cell's. Predicting every cell then paints a map of price by place alone. Cells far from any
training house (the ocean, empty land) are left out, since no model learned anything there.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KDTree

from house_anim.data import COORD_COLUMNS, FEATURE_COLUMNS, HousingSplits, read_only

GRID_STEP_DEG = 0.06
MASK_RADIUS_DEG = 0.12  # cells farther than this from every training house stay blank


@dataclass(frozen=True)
class GeoGrid:
    """The map's cells, and the model input for each cell near real houses.

    Attributes:
        extent: The grid's (lon_min, lon_max, lat_min, lat_max), in degrees, for ``imshow``.
        shape: (rows, columns): rows are latitudes, columns longitudes.
        inside: Read-only boolean mask of the cells near real houses.
        features: Scaled model inputs, one row per inside cell, in the mask's row-major order.
    """

    extent: tuple[float, float, float, float]  # lon_min, lon_max, lat_min, lat_max
    shape: tuple[int, int]  # (rows = latitudes, columns = longitudes)
    inside: np.ndarray  # bool mask of cells near real houses
    features: np.ndarray  # scaled model inputs, one row per inside cell

    def to_image(self, values: np.ndarray) -> np.ndarray:
        """Spread one value per inside cell back onto the grid; outside cells become NaN.

        NaN cells are drawn blank by ``imshow``.

        Args:
            values: One value per inside cell, in the order of ``features``.

        Returns:
            An image of ``shape``.

        Raises:
            ValueError: If there is not exactly one value per inside cell.
        """
        if len(values) != int(self.inside.sum()):
            raise ValueError(f"expected {int(self.inside.sum())} cells, got {len(values)}")
        image = np.full(self.shape, np.nan)
        image[self.inside] = values
        return image


def build_geo_grid(
    splits: HousingSplits, step_deg: float = GRID_STEP_DEG, radius_deg: float = MASK_RADIUS_DEG
) -> GeoGrid:
    """Builds the grid over the training houses' bounding box, one cell margin wide.

    Args:
        splits: The data: training coordinates set the box and the mask, training features set the
            typical house, and the scaler's mean and scale put each cell's coordinates on the
            models' scaled axes.
        step_deg: The cell size, in degrees.
        radius_deg: A cell is kept if a training house lies within this many degrees of it.

    Returns:
        The grid.

    Raises:
        ValueError: If ``step_deg`` or ``radius_deg`` is not positive.
    """
    if step_deg <= 0 or radius_deg <= 0:
        raise ValueError("step_deg and radius_deg must be positive")
    coords = splits.coords_train
    lon_min, lat_min = coords.min(axis=0) - step_deg
    lon_max, lat_max = coords.max(axis=0) + step_deg
    lons = np.arange(lon_min, lon_max + step_deg / 2, step_deg)
    lats = np.arange(lat_min, lat_max + step_deg / 2, step_deg)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    cells = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])

    distance, _ = KDTree(coords).query(cells, k=1)
    inside = (distance[:, 0] <= radius_deg).reshape(grid_lon.shape)
    inside.setflags(write=False)

    # Median of standardised columns == standardised median, so this is the typical house.
    typical = np.median(splits.X_train, axis=0)
    features = np.tile(typical, (int(inside.sum()), 1))
    for column, values in zip(COORD_COLUMNS, cells[inside.ravel()].T):
        index = FEATURE_COLUMNS.index(column)
        features[:, index] = (values - splits.feature_mean[index]) / splits.feature_scale[index]

    return GeoGrid(
        extent=(float(lons[0]), float(lons[-1]), float(lats[0]), float(lats[-1])),
        shape=grid_lon.shape,
        inside=inside,
        features=read_only(features),
    )
