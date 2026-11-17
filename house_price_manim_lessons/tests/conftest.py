"""Shared fixtures: the small synthetic California-like dataset the house project tests with, so no test
needs the network, and the lesson built from it."""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from house_anim.data import split_and_scale  # noqa: E402
from house_anim.lessons.data import income_view  # noqa: E402

N_ROWS = 500


def make_housing_frame(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """The frame ``house_price_animations/tests/conftest.py`` makes: incomes, ages, rooms and places with a
    price that rises with income."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "MedInc": rng.uniform(0.5, 15.0, n_rows), "HouseAge": rng.uniform(1.0, 52.0, n_rows),
        "AveRooms": rng.uniform(2.0, 10.0, n_rows), "AveBedrms": rng.uniform(0.8, 2.0, n_rows),
        "Population": rng.uniform(10.0, 5000.0, n_rows), "AveOccup": rng.uniform(1.0, 6.0, n_rows),
        "Latitude": rng.uniform(32.5, 42.0, n_rows), "Longitude": rng.uniform(-124.3, -114.3, n_rows)})
    signal = 0.3 * frame["MedInc"] - 0.05 * (frame["Latitude"] - 36.0)
    return frame.assign(MedHouseVal=np.clip(signal + rng.normal(0.0, 0.2, n_rows), 0.15, 5.0))


def fake_fetcher(frame: pd.DataFrame):
    def fetch(as_frame: bool = True):
        return SimpleNamespace(frame=frame.copy())

    return fetch


@pytest.fixture(scope="session")
def income_data():
    return income_view(split_and_scale(make_housing_frame()))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
