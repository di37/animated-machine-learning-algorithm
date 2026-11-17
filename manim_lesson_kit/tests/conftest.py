"""Shared fixtures: Manim set up for a small frame, and matplotlib without a window."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from manimkit.frame import Frame  # noqa: E402
from manimkit.player import manim_stage  # noqa: E402

SMALL = (3.2, 1.8, 50.0)  # a 160 x 90 pixel figure
BACKGROUND = "#ffffff"


def figure(width_in: float = SMALL[0], height_in: float = SMALL[1], dpi: float = SMALL[2]) -> Figure:
    fig = Figure(figsize=(width_in, height_in), dpi=dpi, facecolor=BACKGROUND)
    FigureCanvasAgg(fig)
    return fig


def agg_pixels(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)


@pytest.fixture
def small_frame() -> Frame:
    return Frame(*SMALL)


@pytest.fixture
def player(small_frame):
    with manim_stage(small_frame, BACKGROUND, fps=30) as stage:
        yield stage


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
