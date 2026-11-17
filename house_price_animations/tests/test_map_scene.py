from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from house_anim.geo_grid import build_geo_grid
from house_anim.map_scene import MAP_CAPTIONS, MAP_MODEL_KEYS, map_frames
from house_anim.models import get_spec


@pytest.fixture(scope="module")
def grid(splits):
    return build_geo_grid(splits, step_deg=0.5, radius_deg=0.6)


def map_specs():
    return (
        get_spec("linear_regression"),
        replace(get_spec("decision_tree"), steps=(1, 2, 3)),
    )


def test_default_map_models_each_have_a_caption():
    assert len(MAP_MODEL_KEYS) == 4
    assert set(MAP_MODEL_KEYS) <= set(MAP_CAPTIONS)


def test_one_panel_per_model_per_tick(splits, grid):
    frames = tuple(map_frames(map_specs(), splits, grid, ticks=4))
    assert len(frames) == 4
    assert all([p.key for p in f.panels] == ["linear_regression", "decision_tree"] for f in frames)
    assert frames[-1].is_final


def test_panel_prices_are_finite_thousands_on_kept_cells(splits, grid):
    final = tuple(map_frames(map_specs(), splits, grid, ticks=3))[-1]
    for panel in final.panels:
        assert panel.prices_k.shape == (grid.inside.sum(),)
        assert np.isfinite(panel.prices_k).all()
        assert not panel.prices_k.flags.writeable
        assert panel.val_rmse_k > 0


def test_depth_one_tree_paints_at_most_two_prices(splits, grid):
    first = next(map_frames(map_specs(), splits, grid, ticks=3))
    tree = first.panels[1]
    assert tree.step_text == "depth 1"
    assert len(np.unique(tree.prices_k)) <= 2


def test_map_frames_are_frozen(splits, grid):
    first = next(map_frames(map_specs(), splits, grid, ticks=2))
    with pytest.raises(FrozenInstanceError):
        first.panels[0].key = "x"
