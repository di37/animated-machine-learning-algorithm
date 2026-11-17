"""Shared fixtures: a small synthetic California-like dataset so tests never hit the network."""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from house_anim.data import split_and_scale

N_ROWS = 500


def make_housing_frame(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "MedInc": rng.uniform(0.5, 15.0, n_rows),
            "HouseAge": rng.uniform(1.0, 52.0, n_rows),
            "AveRooms": rng.uniform(2.0, 10.0, n_rows),
            "AveBedrms": rng.uniform(0.8, 2.0, n_rows),
            "Population": rng.uniform(10.0, 5000.0, n_rows),
            "AveOccup": rng.uniform(1.0, 6.0, n_rows),
            "Latitude": rng.uniform(32.5, 42.0, n_rows),
            "Longitude": rng.uniform(-124.3, -114.3, n_rows),
        }
    )
    signal = 0.3 * frame["MedInc"] - 0.05 * (frame["Latitude"] - 36.0)
    noise = rng.normal(0.0, 0.2, n_rows)
    return frame.assign(MedHouseVal=np.clip(signal + noise, 0.15, 5.0))


def fake_fetcher(frame: pd.DataFrame):
    def fetch(as_frame: bool = True):
        return SimpleNamespace(frame=frame.copy())

    return fetch


@pytest.fixture(scope="session")
def housing_frame() -> pd.DataFrame:
    return make_housing_frame()


@pytest.fixture(scope="session")
def splits(housing_frame):
    return split_and_scale(housing_frame)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
