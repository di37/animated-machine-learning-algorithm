"""Side-by-side California price maps, one per model, repainted as each model trains.

Each map is an image of the ``GeoGrid``: one pixel per cell, coloured by the price the model gives
the typical house there, on one shared colour scale ($0 to $500k) so the maps can be compared.
Cells away from real houses are NaN and drawn transparent. Away from the equator a degree of
longitude is shorter on the ground than a degree of latitude, so each map's aspect ratio is
1/cos(mean latitude): equal distances on the ground are then roughly equal on screen.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from house_anim import theme
from house_anim.config import MAP_FIGSIZE
from house_anim.geo_grid import GeoGrid
from house_anim.map_scene import MAP_CAPTIONS, MapFrame
from house_anim.models import ModelSpec
from house_anim.panels import ProgressBar, add_header, money_k

PRICE_COLOR_MAX_K = 500.0


class MapDashboard:
    """The map scene's figure: header, progress bar, one map per model and a shared colour bar.

    Attributes:
        figure: The matplotlib figure the sinks record.
        progress: The progress bar through every model's schedule.
        status_text: The line under the header.
        images: One map image per model.
        step_texts: Each map's "how far trained, validation RMSE" readout.

    Example:
        >>> dashboard = MapDashboard(specs, build_geo_grid(splits))  # doctest: +SKIP
        >>> dashboard.render(frame)  # doctest: +SKIP
    """

    def __init__(self, specs: Sequence[ModelSpec], grid: GeoGrid, figsize=MAP_FIGSIZE):
        """Builds the figure with every map blank.

        Args:
            specs: The models, one map each, left to right.
            grid: The cells every map is painted on.
            figsize: The figure size, in inches.
        """
        plt.rcParams.update(theme.RC_PARAMS)
        self._grid = grid
        self.figure = plt.figure(figsize=figsize)
        add_header(self.figure, "Watch the model learn California",
                   "Each map is the predicted price of the same typical house (median income, age, "
                   "rooms, occupancy) moved to every point in the state, repainted as the model trains.")
        self.progress = ProgressBar(self.figure)
        self.status_text = self.figure.text(0.04, 0.855, "", fontsize=theme.FONT_SUBTITLE,
                                            color=theme.INK_PRIMARY)
        cmap = theme.price_colormap().with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
        norm = Normalize(0.0, PRICE_COLOR_MAX_K)
        layout = self.figure.add_gridspec(1, len(specs), left=0.03, right=0.98, top=0.79,
                                          bottom=0.17, wspace=0.06)
        self.images, self.step_texts = [], []
        for index, spec in enumerate(specs):
            self._add_map(self.figure.add_subplot(layout[0, index]), spec, cmap, norm)
        bar_axes = self.figure.add_axes((0.3, 0.075, 0.4, 0.02))
        colorbar = self.figure.colorbar(self.images[0], cax=bar_axes, orientation="horizontal")
        colorbar.set_label("predicted price of the typical house (\\$k)", color=theme.INK_SECONDARY)
        colorbar.outline.set_visible(False)

    def _add_map(self, ax, spec: ModelSpec, cmap, norm) -> None:
        """Adds one blank map with its title, its caption and its readout."""
        image = ax.imshow(np.full(self._grid.shape, np.nan), extent=self._grid.extent,
                          origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        mean_latitude = (self._grid.extent[2] + self._grid.extent[3]) / 2
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_latitude)))  # degrees -> roughly equal km
        ax.set_axis_off()
        ax.set_title(spec.short_title, fontsize=13)
        ax.text(0.5, -0.03, MAP_CAPTIONS.get(spec.key, ""), transform=ax.transAxes, ha="center",
                va="top", fontsize=theme.FONT_SUBTITLE, color=theme.INK_SECONDARY, style="italic")
        # Bottom-left of the extent is open ocean, so the readout never covers the map.
        step_text = ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=theme.FONT_SUBTITLE,
                            color=theme.INK_PRIMARY, linespacing=1.5)
        self.images.append(image)
        self.step_texts.append(step_text)

    def render(self, frame: MapFrame) -> None:
        """Repaints every map from one tick's predictions.

        Args:
            frame: Every model's map at this tick, in the order the maps were made.
        """
        self.progress.update(frame.tick / frame.ticks)
        for image, text, panel in zip(self.images, self.step_texts, frame.panels):
            image.set_data(self._grid.to_image(panel.prices_k))
            text.set_text(f"{panel.step_text}\nval RMSE {money_k(panel.val_rmse_k)}")
        self.status_text.set_text(
            f"{frame.tick / frame.ticks:.0%} through each model's training schedule   ·   "
            "blue = pricier location for the very same house"
        )
