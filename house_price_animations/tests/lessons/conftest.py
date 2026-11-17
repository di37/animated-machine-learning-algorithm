"""Lesson fixtures: the one-feature (median income) view of the synthetic housing splits."""

import pytest

from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.timing import Timing
from house_anim.lessons.data import income_view

# Few frames at the 384x216 preview resolution, so every lesson renders end to end in a second or two.
TINY_TIMING = Timing(fps=6, scale=0.08)
TINY_DPI = RESOLUTIONS["preview"]


@pytest.fixture(scope="session")
def income_data(splits):
    return income_view(splits)
