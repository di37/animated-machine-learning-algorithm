import numpy as np
import pytest

from house_anim.data import FEATURE_COLUMNS
from house_anim.geo_grid import build_geo_grid

LON, LAT = FEATURE_COLUMNS.index("Longitude"), FEATURE_COLUMNS.index("Latitude")


@pytest.fixture(scope="module")
def grid(splits):
    return build_geo_grid(splits, step_deg=0.5, radius_deg=0.6)


def test_grid_spans_the_houses(grid, splits):
    lon_min, lon_max, lat_min, lat_max = grid.extent
    assert lon_min <= splits.coords_train[:, 0].min() and lon_max >= splits.coords_train[:, 0].max()
    assert lat_min <= splits.coords_train[:, 1].min() and lat_max >= splits.coords_train[:, 1].max()
    assert grid.inside.shape == grid.shape


def test_only_cells_near_houses_are_kept(grid):
    assert 0 < grid.inside.sum() <= grid.inside.size
    assert grid.features.shape == (grid.inside.sum(), len(FEATURE_COLUMNS))


def test_far_away_cells_are_dropped(splits):
    tight = build_geo_grid(splits, step_deg=0.25, radius_deg=0.05)
    assert tight.inside.sum() < tight.inside.size


def test_every_cell_holds_the_same_typical_house_except_location(grid, splits):
    others = [i for i in range(len(FEATURE_COLUMNS)) if i not in (LON, LAT)]
    typical = np.median(splits.X_train, axis=0)
    np.testing.assert_allclose(grid.features[:, others], np.broadcast_to(typical[others], (len(grid.features), len(others))))
    assert np.ptp(grid.features[:, LON]) > 0 and np.ptp(grid.features[:, LAT]) > 0


def test_location_columns_are_scaled_like_the_training_data(grid, splits):
    raw_lon = grid.features[:, LON] * splits.feature_scale[LON] + splits.feature_mean[LON]
    lon_min, lon_max, _, _ = grid.extent
    assert lon_min <= raw_lon.min() and raw_lon.max() <= lon_max


def test_to_image_fills_kept_cells_and_blanks_the_rest(grid):
    values = np.arange(grid.inside.sum(), dtype=float)
    image = grid.to_image(values)
    assert image.shape == grid.shape
    np.testing.assert_array_equal(image[grid.inside], values)
    assert np.isnan(image[~grid.inside]).all()


def test_to_image_rejects_wrong_length(grid):
    with pytest.raises(ValueError, match="cells"):
        grid.to_image(np.zeros(3))


def test_invalid_resolution_raises(splits):
    with pytest.raises(ValueError, match="positive"):
        build_geo_grid(splits, step_deg=0.0, radius_deg=0.1)
