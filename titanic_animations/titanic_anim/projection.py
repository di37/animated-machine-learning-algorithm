"""A fixed 2-D map of passengers (PCA of the 8 standardised features) for distance-based diagrams.

KNN and the SVM decide by distances between passengers in all eight features, which cannot be
drawn. Principal component analysis (PCA) finds the two directions along which the training
passengers vary most; projecting onto them gives a flat map that keeps as much of that spread as
two axes can. The map is fit once, on the training passengers, so a point never moves between
frames. Distances on it only approximate the eight-feature distances the models use.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from animkit.data import read_only
from titanic_anim.data import TitanicSplits

LOADINGS_NAMED = 2


@dataclass(frozen=True)
class Projection:
    """A fitted PCA map from standardised features to two axes.

    Attributes:
        mean: The training passengers' mean, subtracted before projecting.
        components: The two directions, shape (2, n_features), one row per map axis.
        axis_names: Each axis's label, naming the features that weigh most in it.
    """

    mean: np.ndarray
    components: np.ndarray  # (2, n_features)
    axis_names: tuple[str, str]  # e.g. "axis 1 ≈ fare, pclass"

    def points(self, X: np.ndarray) -> np.ndarray:
        """Places passengers on the map.

        Args:
            X: Standardised passengers, one per row, or a single passenger.

        Returns:
            Their map positions, shape (n, 2).
        """
        return (np.atleast_2d(X) - self.mean) @ self.components.T


def fit_projection(splits: TitanicSplits) -> Projection:
    """Fits the map on the training passengers.

    Args:
        splits: The splits; only the training passengers are used.

    Returns:
        The read-only map, its axes named after their ``LOADINGS_NAMED`` largest loadings.
    """
    pca = PCA(n_components=2, random_state=0).fit(splits.X_train)
    names = tuple(
        f"axis {axis + 1} ≈ " + ", ".join(splits.feature_names[i]
                                          for i in np.argsort(-np.abs(row))[:LOADINGS_NAMED])
        for axis, row in enumerate(pca.components_)
    )
    return Projection(mean=read_only(pca.mean_), components=read_only(pca.components_), axis_names=names)
