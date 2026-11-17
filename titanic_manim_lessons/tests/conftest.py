"""Shared fixtures: the small synthetic passenger list the Titanic project tests with, so no test needs the
network, and the age / fare / sex view the lessons are built from."""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from titanic_anim.lessons.data import passenger_view  # noqa: E402

N_ROWS = 400


def make_titanic_frame(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """The frame ``titanic_animations/tests/conftest.py`` makes: OpenML's columns, with survival depending on
    sex, class and age, some missing ages, fares and ports, and the leaking ``boat`` and ``body`` columns."""
    rng = np.random.default_rng(seed)
    pclass = rng.choice([1, 2, 3], n_rows, p=[0.25, 0.2, 0.55])
    female = rng.random(n_rows) < 0.36
    age = rng.uniform(0.5, 75.0, n_rows)
    fare = np.select([pclass == 1, pclass == 2], [rng.uniform(30, 250, n_rows), rng.uniform(10, 40, n_rows)],
                     rng.uniform(5, 20, n_rows))
    logit = 2.4 * female - 0.9 * (pclass - 2) - 0.025 * (age - 30.0) - 0.5
    survived = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    age_with_gaps = np.where(rng.random(n_rows) < 0.15, np.nan, age)
    fare[3] = np.nan
    embarked = pd.Series(rng.choice(["C", "Q", "S"], n_rows, p=[0.2, 0.1, 0.7]), dtype=object)
    embarked.iloc[[5, 9]] = None
    return pd.DataFrame({
        "pclass": pclass, "survived": pd.Categorical(survived.astype(str)),
        "name": [f"Passenger {i}" for i in range(n_rows)], "sex": pd.Categorical(np.where(female, "female", "male")),
        "age": age_with_gaps, "sibsp": rng.poisson(0.5, n_rows), "parch": rng.poisson(0.4, n_rows), "fare": fare,
        "embarked": pd.Categorical(embarked), "boat": np.where(survived == 1, "7", None),
        "body": np.where(survived == 0, 1.0, np.nan)})


def fake_fetcher(frame: pd.DataFrame):
    def fetch(as_frame: bool = True):
        return SimpleNamespace(frame=frame.copy())

    return fetch


@pytest.fixture(scope="session")
def passenger_data():
    return passenger_view(make_titanic_frame())


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
