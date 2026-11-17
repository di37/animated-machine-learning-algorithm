"""Shared fixtures for the kit's own tests: a headless matplotlib backend and no leaked figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
