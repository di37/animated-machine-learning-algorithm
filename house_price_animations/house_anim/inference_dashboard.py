"""Live inference dashboard: a California map filling in as test houses are priced.

The trained model predicts the unseen test houses batch by batch, and after every batch the
figure shows, in $k:

- on the left, the algorithm diagram working through the latest house;
- a map of California where each test house turns from grey to its predicted price's colour,
  with the latest batch ringed;
- predicted vs actual, the running RMSE and MAE, and the residual histogram, filling in house by
  house;
- on the right, the equation card working the latest house's prediction through;
- a ticker line: the latest house's prediction against its true value, batch latency and
  throughput.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from animkit.equation_panel import EquationPanel
from house_anim import theme
from house_anim.algorithms import algorithm_view
from house_anim.config import (
    EQUATION_PANEL_RECT,
    INFERENCE_ALGORITHM_RECT,
    INFERENCE_FIGSIZE,
    TARGET_TO_THOUSANDS,
)
from house_anim.data import HousingSplits
from house_anim.equations import EquationCard
from house_anim.frames import InferenceFrame
from house_anim.models import ModelSpec
from house_anim.panels import (
    PredictedVsActual,
    ProgressBar,
    ResidualHistogram,
    TwoLineChart,
    add_header,
    money_k,
    signed_money_k,
)

PRICE_COLOR_MAX_K = 500.0


def format_rate(per_second: float) -> str:
    """A throughput for the status line, such as ``"12,345 houses/s"``.

    Args:
        per_second: Houses predicted per second.

    Returns:
        The text; a dash when the rate is not finite (no measurable predict time yet).
    """
    return "—" if not np.isfinite(per_second) else f"{per_second:,.0f} houses/s"


class PriceMap:
    """Longitude/latitude scatter: grey = waiting, blue ramp = predicted price, rings = latest batch.

    Attributes:
        ax: The map's axes.
        points: The houses predicted so far, coloured by predicted price ($0 to $500k).
        latest: Rings around the latest batch's houses.
    """

    def __init__(self, ax: Axes, all_coords: np.ndarray):
        """Draws every test house in grey, with a colour bar for the prices to come.

        Args:
            ax: The axes to draw in.
            all_coords: Every test house's raw (longitude, latitude).
        """
        self.ax = ax
        theme.style_axes(ax, grid_axis="both")
        ax.scatter(all_coords[:, 0], all_coords[:, 1], s=5, color=theme.NOT_YET, linewidths=0)
        self.points = ax.scatter([], [], s=7, c=[], cmap=theme.price_colormap(),
                                 norm=Normalize(0.0, PRICE_COLOR_MAX_K), linewidths=0)
        self.latest = ax.scatter([], [], s=40, facecolors="none", edgecolors=theme.INK_PRIMARY,
                                 linewidths=0.8)
        mean_latitude = float(np.mean(all_coords[:, 1]))
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))  # degrees -> roughly equal km
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title("Where the predictions land · predicted median value")
        colorbar = ax.figure.colorbar(self.points, ax=ax, orientation="horizontal",
                                      fraction=0.04, pad=0.08)
        colorbar.set_label("predicted median house value (\\$k)", color=theme.INK_SECONDARY)
        colorbar.outline.set_visible(False)

    def update(self, coords: np.ndarray, predicted_k: np.ndarray, batch_start: int) -> None:
        """Colours every house predicted so far and rings the latest batch.

        Args:
            coords: The (longitude, latitude) of the houses predicted so far.
            predicted_k: Their predicted prices, in $k.
            batch_start: The index of the latest batch's first house.
        """
        self.points.set_offsets(coords)
        self.points.set_array(predicted_k)
        self.latest.set_offsets(coords[batch_start:])


class InferenceDashboard:
    """The inference figure for one trained model, redrawn after every predict batch.

    Attributes:
        figure: The matplotlib figure the sinks record.
        progress: The progress bar through the test houses.
        status_text: The line with houses done, batch latency and throughput.
        ticker_text: The line about the latest house.
        algorithm: The model's animated algorithm diagram.
        map: The California map of predictions.
        predictions: Test predicted vs actual.
        running: The running RMSE and MAE.
        residuals: The test residual histogram.
        equations: The equation card's panel.
    """

    def __init__(self, spec: ModelSpec, splits: HousingSplits, figsize=INFERENCE_FIGSIZE):
        """Builds the figure with every panel empty and every test house grey on the map.

        Args:
            spec: The model (its title and its algorithm diagram).
            splits: The data; only the test houses are drawn.
            figsize: The figure size, in inches.
        """
        plt.rcParams.update(theme.RC_PARAMS)
        n_total = len(splits.y_test)
        self.figure = plt.figure(figsize=figsize)
        add_header(self.figure, f"{spec.title} · inference on unseen houses",
                   f"Streaming {n_total:,} held-out test houses through model.predict(); "
                   "each dot appears the moment its price is predicted.")
        self.progress = ProgressBar(self.figure)
        self.status_text = self.figure.text(0.04, 0.855, "", fontsize=theme.FONT_SUBTITLE,
                                            color=theme.INK_PRIMARY)
        self.ticker_text = self.figure.text(0.04, 0.825, "", fontsize=theme.FONT_SUBTITLE,
                                            color=theme.INK_SECONDARY)
        self.algorithm = algorithm_view(spec, self.figure.add_axes(INFERENCE_ALGORITHM_RECT), splits)
        grid = self.figure.add_gridspec(3, 2, left=0.315, right=0.705, top=0.76, bottom=0.07,
                                        width_ratios=(1.0, 1.05), hspace=0.75, wspace=0.26)
        self.map = PriceMap(self.figure.add_subplot(grid[:, 0]), splits.coords_test)
        self.predictions = PredictedVsActual(self.figure.add_subplot(grid[0, 1]))
        self.running = self._build_running(self.figure.add_subplot(grid[1, 1]), n_total)
        self.residuals = ResidualHistogram(self.figure.add_subplot(grid[2, 1]))
        self.equations = EquationPanel(self.figure.add_axes(EQUATION_PANEL_RECT))

    @staticmethod
    def _build_running(ax: Axes, n_total: int) -> TwoLineChart:
        """The running-error chart, its x axis spanning every test house."""
        chart = TwoLineChart(ax, ("RMSE", "MAE"))
        ax.set_xlim(0, n_total)
        ax.set_xlabel("test houses predicted so far")
        ax.set_ylabel("error (\\$k)")
        return chart

    @property
    def map_points(self):
        """The map's predicted houses, for the tests.

        Returns:
            The matplotlib scatter collection.
        """
        return self.map.points

    @property
    def rmse_line(self):
        """The running-RMSE line, for the tests.

        Returns:
            The matplotlib line.
        """
        return self.running.lines[0]

    def render(self, frame: InferenceFrame, card: EquationCard | None = None, model=None,
               house: np.ndarray | None = None) -> None:
        """Draw a frame; ``card`` updates the equations, ``model`` + ``house`` the algorithm diagram.

        Args:
            frame: The snapshot after one predict batch.
            card: The prediction card for the latest house; None leaves the panel as it is.
            model: The model, for the algorithm diagram.
            house: The latest house's scaled features, for the algorithm diagram; the diagram is
                updated only when both ``model`` and ``house`` are given.
        """
        if card is not None:
            self.equations.update(card)
        if model is not None and house is not None:
            self.algorithm.show_prediction(model, house, frame.n_done)
        actual_k = frame.y_true * TARGET_TO_THOUSANDS
        predicted_k = frame.y_pred * TARGET_TO_THOUSANDS
        counts = np.array([p.n_done for p in frame.history])
        rmse_k = np.array([p.rmse for p in frame.history]) * TARGET_TO_THOUSANDS
        mae_k = np.array([p.mae for p in frame.history]) * TARGET_TO_THOUSANDS

        self.progress.update(frame.n_done / frame.n_total)
        self.map.update(frame.coords, predicted_k, frame.batch_start)
        self.predictions.update(actual_k, predicted_k, "Test houses · predicted vs actual")
        self.running.update(counts, (rmse_k, mae_k))
        r2 = frame.scores.r2
        self.running.ax.set_title(f"Running test error · R² {r2:.3f}" if np.isfinite(r2)
                                  else "Running test error")
        self.residuals.update(predicted_k - actual_k)
        self.status_text.set_text(
            f"{frame.n_done:,}/{frame.n_total:,} houses predicted   ·   "
            f"last batch {frame.batch_latency_ms:.2f} ms   ·   "
            f"throughput {format_rate(frame.throughput_per_s)}"
        )
        self.ticker_text.set_text(self._ticker(frame, actual_k[-1], predicted_k[-1]))

    @staticmethod
    def _ticker(frame: InferenceFrame, actual_k: float, predicted_k: float) -> str:
        """The latest house's line: where it is, predicted and actual price, and the error in $k and %."""
        longitude, latitude = frame.coords[-1]
        error_k = predicted_k - actual_k
        percent = error_k / actual_k * 100.0 if actual_k else float("nan")
        return (
            f"Latest: house #{frame.n_done:,} near ({longitude:.2f}, {latitude:.2f})   ·   "
            f"predicted {money_k(predicted_k)}   ·   actual {money_k(actual_k)}   ·   "
            f"off by {signed_money_k(error_k)} ({percent:+.1f}%)"
        )
