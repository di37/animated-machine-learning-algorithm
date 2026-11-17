"""Titanic lesson fixtures: the age / fare / sex view of the synthetic passenger list."""

import pytest

from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.timing import Timing
from tests.conftest import make_titanic_frame
from titanic_anim.lessons.data import passenger_view

TINY_TIMING = Timing(fps=6, scale=0.08)
TINY_DPI = RESOLUTIONS["preview"]


@pytest.fixture(scope="session")
def passenger_data():
    return passenger_view(make_titanic_frame())
