"""Animated diagrams of each classifier, redrawn from the live model during training and inference.

Every view is an animkit ``AlgorithmView``: it owns one region of a dashboard and draws the
algorithm itself, not only its scores. ``show_training(model, frame)`` redraws it from the live
model after each training step: a growing tree, weights on edges, support vectors.
``show_prediction(model, x, number)`` works one passenger through it: the tree's path, the nearest
neighbours, each tree's vote. ``algorithm_view`` picks the view for a spec's estimator; the MLP's
network diagram is animkit's ``NetworkView`` with a sigmoid output.
"""

from matplotlib.axes import Axes
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from animkit.algorithms.network_view import NetworkView
from titanic_anim import theme
from titanic_anim.algorithms.bayes_view import NaiveBayesView
from titanic_anim.algorithms.common import AlgorithmView, sigmoid
from titanic_anim.algorithms.logistic_view import LogisticView
from titanic_anim.algorithms.projection_views import NeighboursView, SupportVectorView
from titanic_anim.algorithms.tree_views import BoostingView, ForestView, TreeView
from titanic_anim.data import TitanicSplits
from titanic_anim.models import ClassifierSpec

__all__ = ["AlgorithmView", "ClassifierNetworkView", "algorithm_view"]


class ClassifierNetworkView(NetworkView):
    """animkit's network diagram with a sigmoid output unit: P(survive) per passenger.

    Attributes:
        output_label: What the output unit means, shown as the output layer's label.
        subject: What one prediction is about, for the title ("Passenger #12 flowing through ...").
    """

    output_label = "P(survive)"
    subject = "Passenger"

    def _output(self, z: float) -> tuple[str, object]:
        """The output unit's label and colour: σ(z) as a percentage, coloured by survival."""
        p = float(sigmoid(z))
        return f"{p:.0%}", theme.survival_rgba(p)


VIEWS = (
    ((LogisticRegression, SGDClassifier), LogisticView),
    (SVC, SupportVectorView),
    (GaussianNB, NaiveBayesView),
    (DecisionTreeClassifier, TreeView),
    (RandomForestClassifier, ForestView),
    (GradientBoostingClassifier, BoostingView),
    (KNeighborsClassifier, NeighboursView),
    (MLPClassifier, ClassifierNetworkView),
)


def algorithm_view(spec: ClassifierSpec, ax: Axes, splits: TitanicSplits) -> AlgorithmView:
    """The diagram for a classifier, chosen by its estimator's type.

    Args:
        spec: The classifier; an estimator is built from it to look at its type.
        ax: The dashboard region to draw in.
        splits: The splits, for the feature names, the scaling and the training passengers.

    Returns:
        The view, blank until its first ``show_training`` or ``show_prediction``.

    Raises:
        ValueError: If no diagram fits the estimator.
    """
    estimator = spec.build()
    for types, view in VIEWS:
        if isinstance(estimator, types):
            return view(ax, splits)
    raise ValueError(f"No algorithm diagram for {type(estimator).__name__}")
