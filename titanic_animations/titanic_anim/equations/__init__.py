"""Live equations: what each classifier computes while it trains, and worked out for one passenger.

Each card is an animkit ``EquationCard`` (a title, its lines, and for a prediction the value the
working arrives at), drawn by animkit's ``EquationPanel`` beside the dashboards. A training card
shows the model's formula with its current numbers, what training minimises, and what this step
changed. A prediction card works one passenger through the same formula and ends with a check line
that compares the result with the model's own ``predict_proba`` or ``decision_function``. The tests
assert the two agree for every classifier, so the arithmetic on screen is the model's own.

``EXPLAINERS`` matches an estimator to its family's two explainers (the first match wins); any
other model gets a generic card.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from titanic_anim.data import TitanicSplits
from titanic_anim.equations import kernels, linear, others, trees
from titanic_anim.equations.cards import EquationCard, validate_passenger
from titanic_anim.frames import TrainingFrame
from titanic_anim.models import STRATEGY_DATA_GROWTH, ClassifierSpec

__all__ = ["EquationCard", "prediction_card", "training_card"]

# (estimator types, training explainer, prediction explainer); first match wins.
EXPLAINERS = (
    ((LogisticRegression, SGDClassifier), linear.training_lines, linear.prediction_lines),
    (SVC, kernels.svm_training, kernels.svm_prediction),
    (GaussianNB, kernels.nb_training, kernels.nb_prediction),
    (DecisionTreeClassifier, trees.tree_training, trees.tree_prediction),
    (RandomForestClassifier, trees.forest_training, trees.forest_prediction),
    (GradientBoostingClassifier, trees.boosting_training, trees.boosting_prediction),
    (KNeighborsClassifier, others.knn_training, others.knn_prediction),
    (MLPClassifier, others.mlp_training, others.mlp_prediction),
)


def _explainers(model):
    """The (training, prediction) explainers of a model's family; a generic pair for any other model."""
    for types, train, predict in EXPLAINERS:
        if isinstance(model, types):
            return train, predict
    return others.generic_training, others.generic_prediction


def training_card(model, frame: TrainingFrame, spec: ClassifierSpec, splits: TitanicSplits) -> EquationCard:
    """The model's formula with current numbers, its objective, and what this step changed.

    Args:
        model: The live model after this step.
        frame: This step's training frame.
        spec: The classifier; a data-growth model is described on the rows it was fit on.
        splits: The splits, for the feature names and the scaling.

    Returns:
        The card, titled "What <model> is computing".
    """
    rows = int(frame.step) if spec.strategy == STRATEGY_DATA_GROWTH else splits.n_train
    lines = _explainers(model)[0](model, frame, splits, rows)
    return EquationCard(f"What {spec.short_title} is computing", tuple(lines))


def prediction_card(model, x: np.ndarray, passenger_number: int, splits: TitanicSplits) -> EquationCard:
    """The equation worked through for one passenger, ending in the model's own answer.

    Args:
        model: The trained model.
        x: The passenger's standardised features.
        passenger_number: The passenger's number, for the title.
        splits: The splits, for the feature names, the scaling and the training labels.

    Returns:
        The card; its ``value`` is the probability, or the SVM's margin, that the working arrives at.

    Raises:
        ValueError: If ``x`` is not one passenger's row of features.
    """
    row = validate_passenger(x, len(splits.feature_names))
    lines, value = _explainers(model)[1](model, row, splits)
    return EquationCard(f"Passenger #{passenger_number:,}: the equation at work", tuple(lines), value)
