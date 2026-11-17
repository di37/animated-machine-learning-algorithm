"""Distance-based classifiers on a 2-D passenger map: KNN's neighbours and the SVM's support vectors.

Both models decide by how close a passenger is to the training passengers, so both are drawn on
the same fixed PCA map (see ``projection``). KNN stores every passenger and lets the k nearest
vote. The RBF-kernel SVM keeps only its support vectors, the hard cases on or inside the margin,
and adds up their pulls α·y·K(x_i, x), which fade with distance, plus a bias: a positive total
means survived. The models measure distance in all eight features, so their nearest passengers
need not look nearest on the map.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from animkit import theme as base
from titanic_anim import theme
from titanic_anim.algorithms.common import AlgorithmView
from titanic_anim.data import TitanicSplits
from titanic_anim.equations.kernels import kernel_terms
from titanic_anim.projection import fit_projection

PAD = 0.4
CAPTION_ROOM = 0.14  # share of the view's height kept free below the map for its caption
MAP_NOTE = "positions: the 8 features squeezed onto 2 axes (PCA)"
SHOWN_SV_LINKS = 10


class ProjectionView(AlgorithmView):
    """Training passengers placed on a fixed PCA map, coloured by what happened to them.

    Attributes:
        ax: The map's axes.
        splits: The splits whose training passengers are drawn.
    """

    def __init__(self, ax: Axes, splits: TitanicSplits):
        """Fits the map and shrinks the axes to leave room for the caption underneath.

        Args:
            ax: The view's axes.
            splits: The splits; the map is fit on the training passengers.
        """
        self._projection = fit_projection(splits)
        self._points = self._projection.points(splits.X_train)
        pos = ax.get_position()
        ax.set_position([pos.x0 + pos.width * 0.06, pos.y0 + pos.height * CAPTION_ROOM,
                         pos.width * 0.94, pos.height * (1.0 - CAPTION_ROOM)])
        super().__init__(ax, splits)

    def _prepare(self, title: str) -> None:
        """Clears the view and sets up the map: axes on, fixed limits, the PCA axes' names."""
        self._blank(title)
        self.ax.set_axis_on()
        base.style_axes(self.ax, grid_axis="both")
        self.ax.tick_params(labelbottom=False, labelleft=False)
        low, high = self._points.min(axis=0) - PAD, self._points.max(axis=0) + PAD
        self.ax.set_xlim(low[0], high[0])
        self.ax.set_ylim(low[1], high[1])
        self.ax.set_xlabel(self._projection.axis_names[0])
        self.ax.set_ylabel(self._projection.axis_names[1])

    def _stored(self, rows: int, alpha: float = 0.45, size: float = 14.0) -> None:
        """Scatters the first ``rows`` training passengers, coloured by what happened to them."""
        colours = [theme.outcome_rgba(y, alpha) for y in self.splits.y_train[:rows]]
        self.ax.scatter(*self._points[:rows].T, s=size, c=colours, linewidths=0, zorder=1)

    def _star(self, x: np.ndarray) -> tuple[float, float]:
        """Marks passenger ``x`` with a star on the map and returns where it sits."""
        home = self._projection.points(x)[0]
        self.ax.scatter([home[0]], [home[1]], s=300, marker="*", color=base.INK_PRIMARY,
                        edgecolors=base.SURFACE, linewidths=0.8, zorder=5)
        return float(home[0]), float(home[1])


class NeighboursView(ProjectionView):
    """KNN's diagram: the passengers it has memorised, and a passenger's k nearest neighbours.

    Attributes:
        links_drawn: How many neighbour links the last prediction drew.
    """

    links_drawn = 0

    def show_training(self, model, frame) -> None:
        """Draws every passenger the model has stored: KNN's training is only memorising.

        Args:
            model: The fitted ``KNeighborsClassifier``.
            frame: The training frame; unused, since everything drawn comes from the model.
        """
        self._prepare("The algorithm: KNN memorises passengers")
        self._stored(model.n_samples_fit_)
        self.caption(f"{model.n_samples_fit_:,} passengers stored (blue survived, red did not)  ·  "
                     f"no weights, no loss  ·  {MAP_NOTE}", y=-0.1)

    def show_prediction(self, model, x: np.ndarray, passenger_number: int) -> None:
        """Links the passenger to its k nearest neighbours; the share of them who survived is P(survive).

        Args:
            model: The fitted ``KNeighborsClassifier``.
            x: The passenger's standardised features.
            passenger_number: The passenger's number, for the title.
        """
        self._prepare(f"Passenger #{passenger_number:,} asks its {model.n_neighbors} nearest neighbours")
        self._stored(model.n_samples_fit_, alpha=0.18)
        _, indices = model.kneighbors(x[None, :])
        neighbours = self._points[indices[0]]
        fates = self.splits.y_train[indices[0]]
        home = self._star(x)
        self.ax.add_collection(LineCollection([(home, tuple(n)) for n in neighbours], colors=base.INK_SECONDARY,
                                              linewidths=1.0, alpha=0.8, zorder=2))
        self.ax.scatter(*neighbours.T, s=80, c=[theme.outcome_rgba(y) for y in fates],
                        edgecolors=base.INK_PRIMARY, linewidths=0.8, zorder=3)
        self.links_drawn = len(neighbours)
        self.caption(f"nearest in all 8 features  ·  {int(fates.sum())} of {len(fates)} survived  →  "
                     f"P(survive) = {fates.mean():.0%}  ·  {MAP_NOTE}", y=-0.1)


class SupportVectorView(ProjectionView):
    """The SVM's diagram: its support vectors, and the strongest pulls on one passenger.

    Attributes:
        vectors_drawn: How many support vectors the last drawing ringed.
        links_drawn: How many pull lines the last prediction drew.
    """

    vectors_drawn = 0
    links_drawn = 0

    def _vectors(self, model) -> np.ndarray:
        """Support vectors drawn as rings sized by |α|; the model was fit on a prefix of the rows."""
        alpha = np.abs(model.dual_coef_[0])
        points = self._points[model.support_]
        colours = [theme.outcome_rgba(int(a > 0)) for a in model.dual_coef_[0]]
        self.ax.scatter(*points.T, s=20 + 120 * alpha / max(alpha.max(), 1e-9), facecolors="none",
                        edgecolors=colours, linewidths=1.3, zorder=3)
        self.vectors_drawn = len(points)
        return points

    def show_training(self, model, frame) -> None:
        """Rings the support vectors among the passengers the model was fit on, each sized by its |α|.

        Args:
            model: The fitted ``SVC``.
            frame: The training frame; unused, since everything drawn comes from the model.
        """
        self._prepare("The algorithm: an SVM keeps only the hard cases")
        rows = int(model.shape_fit_[0])
        self._stored(rows, alpha=0.25, size=10.0)
        self._vectors(model)
        self.caption(f"{self.vectors_drawn:,} of {rows:,} passengers are support vectors (rings, size = α)  ·  "
                     f"the boundary depends on them alone  ·  {MAP_NOTE}", y=-0.1)

    def show_prediction(self, model, x: np.ndarray, passenger_number: int) -> None:
        """Links the passenger to the support vectors that pull on it hardest, and gives the sum f(x).

        Only the ``SHOWN_SV_LINKS`` strongest pulls are drawn, blue towards survived and red away; f(x)
        adds up every pull and the bias.

        Args:
            model: The fitted ``SVC``.
            x: The passenger's standardised features.
            passenger_number: The passenger's number, for the title.
        """
        self._prepare(f"Passenger #{passenger_number:,} vs every support vector")
        self._stored(int(model.shape_fit_[0]), alpha=0.12, size=10.0)
        points = self._vectors(model)
        terms = kernel_terms(model, x)
        home = self._star(x)
        strongest = np.argsort(-np.abs(terms))[:SHOWN_SV_LINKS]
        scale = float(np.abs(terms[strongest]).max()) or 1.0
        self.ax.add_collection(LineCollection(
            [(home, tuple(points[i])) for i in strongest],
            colors=[theme.outcome_rgba(int(terms[i] > 0), 0.85) for i in strongest],
            linewidths=[0.6 + 3.0 * abs(terms[i]) / scale for i in strongest], zorder=2))
        self.links_drawn = len(strongest)
        value = float(terms.sum() + model.intercept_[0])
        self.caption(f"lines = the {len(strongest)} strongest pulls α·y·K (blue towards survived)  ·  "
                     f"f(x) = {value:+.2f}, so {'survived' if value > 0 else 'did not survive'}", y=-0.1)
