"""Side-by-side survival landscapes, one per model: age across, class and sex down.

A 2x2 grid of maps under a header, a progress bar and a status line, with one colour bar for all
of them: blue for likely to survive, red for likely not. Every map uses the same colours and the
same grid of typical passengers, so the differences between them are the algorithms' own.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from animkit import theme as base
from animkit.config import MAP_FIGSIZE
from animkit.panels import ProgressBar, add_header
from titanic_anim import theme
from titanic_anim.landscape import LANDSCAPE_CAPTIONS, LandscapeFrame, SurvivalGrid
from titanic_anim.models import ClassifierSpec
from titanic_anim.panels import percent

GRID_ROWS, GRID_COLUMNS = 2, 2


class LandscapeDashboard:
    """The survival landscape figure for up to four models.

    Attributes:
        figure: The matplotlib figure, which animkit's sinks show and record.
        progress: The bar showing how far through training the models are.
        status_text: The line under the header.
        images: Each model's map.
        step_texts: Each map's step and validation accuracy.
    """

    def __init__(self, specs: Sequence[ClassifierSpec], grid: SurvivalGrid, figsize=MAP_FIGSIZE):
        """Builds the figure, a map per model and the shared colour bar.

        Args:
            specs: The models, at most four (one per cell of the 2x2 grid).
            grid: The typical passengers the maps show.
            figsize: The figure's size in inches.
        """
        plt.rcParams.update(base.RC_PARAMS)
        self._grid = grid
        self.figure = plt.figure(figsize=figsize)
        add_header(self.figure, "Who survives? Watch the models learn the Titanic",
                   "Each map is the predicted chance of surviving for a typical passenger of every age, "
                   "class and sex, repainted as the model trains.")
        self.progress = ProgressBar(self.figure)
        self.status_text = self.figure.text(0.04, 0.855, "", fontsize=base.FONT_SUBTITLE, color=base.INK_PRIMARY)
        layout = self.figure.add_gridspec(GRID_ROWS, GRID_COLUMNS, left=0.13, right=0.98, top=0.79,
                                          bottom=0.17, hspace=0.55, wspace=0.18)
        self.images, self.step_texts = [], []
        for index, spec in enumerate(specs):
            self._add_panel(self.figure.add_subplot(layout[index // GRID_COLUMNS, index % GRID_COLUMNS]),
                            spec, first_column=index % GRID_COLUMNS == 0)
        bar_axes = self.figure.add_axes((0.3, 0.07, 0.4, 0.02))
        colorbar = self.figure.colorbar(self.images[0], cax=bar_axes, orientation="horizontal")
        colorbar.set_label("predicted P(survive): red = did not survive, blue = survived",
                           color=base.INK_SECONDARY)
        colorbar.outline.set_visible(False)

    def _add_panel(self, ax, spec: ClassifierSpec, first_column: bool) -> None:
        """Adds one model's map, grey until it is first painted; only the left column labels the groups."""
        ages = self._grid.ages
        image = ax.imshow(np.full(self._grid.shape, 0.5), aspect="auto",
                          cmap=theme.SURVIVAL_CMAP, norm=theme.PROBABILITY_NORM, interpolation="nearest",
                          extent=(ages[0] - 0.5, ages[-1] + 0.5, len(self._grid.groups) - 0.5, -0.5))
        ax.set_yticks(range(len(self._grid.groups)), self._grid.groups if first_column else [])
        for row in (1.5, 3.5):  # thin rules between the classes
            ax.axhline(row, color=base.SURFACE, linewidth=2.0)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("age")
        ax.set_title(f"{spec.short_title} · {LANDSCAPE_CAPTIONS.get(spec.key, '')}", fontsize=11)
        # Below the axes, right of the centred "age" label, so it never meets the title.
        step_text = ax.text(1.0, -0.16, "", transform=ax.transAxes, ha="right", va="top",
                            fontsize=base.FONT_LABEL, color=base.INK_SECONDARY)
        self.images.append(image)
        self.step_texts.append(step_text)

    def render(self, frame: LandscapeFrame) -> None:
        """Repaints every map for one tick.

        Args:
            frame: Every model's map at this tick.
        """
        self.progress.update(frame.tick / frame.ticks)
        for image, text, panel in zip(self.images, self.step_texts, frame.panels):
            image.set_data(self._grid.to_image(panel.probabilities))
            text.set_text(f"{panel.step_text}  ·  val accuracy {percent(panel.val_accuracy)}")
        self.status_text.set_text(
            f"{frame.tick / frame.ticks:.0%} through each model's training schedule   ·   "
            "blue = likely to survive, red = likely not"
        )
