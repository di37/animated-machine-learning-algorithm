"""Registry of animated scikit-learn classifiers and how each one is trained step by step.

Each classifier is a ``ClassifierSpec``: animkit's ``ModelSpec`` (how to build the estimator, its
training schedule, and the dashboard's wording) plus how the model scores a passenger. Every
schedule follows one of animkit's four training strategies, and one training frame is one step:

- ``STRATEGY_INCREMENTAL``: ``partial_fit`` on the next chunk of passengers (SGD: an eighth of an
  epoch; the MLP: a whole epoch).
- ``STRATEGY_WARM_START``: keep the fitted ensemble and add members (random forest, boosting).
- ``STRATEGY_PARAM_SWEEP``: refit while one hyper-parameter moves along a schedule (logistic
  regression's C, the decision tree's depth).
- ``STRATEGY_DATA_GROWTH``: refit on more passengers, a learning curve (SVM, Naive Bayes, KNN).

``MODEL_SPECS`` maps each key to its spec and cannot be changed.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from animkit.models import (
    STRATEGY_DATA_GROWTH,
    STRATEGY_INCREMENTAL,
    STRATEGY_PARAM_SWEEP,
    STRATEGY_WARM_START,
    ModelSpec,
)
from titanic_anim.config import RANDOM_STATE, SCORE_MARGIN, SCORE_PROBABILITY

__all__ = ["MODEL_SPECS", "SCORE_MARGIN", "SCORE_PROBABILITY", "STRATEGY_DATA_GROWTH",
           "STRATEGY_INCREMENTAL", "STRATEGY_PARAM_SWEEP", "STRATEGY_WARM_START", "ClassifierSpec",
           "get_spec", "model_keys"]

GROWTH_FRACTIONS = tuple(float(f) for f in np.geomspace(0.03, 1.0, 24))
KNN_K = 11  # odd, so a vote can never tie
SVM_GAMMA = 0.125  # 1 / n_features on standardised data: what gamma="scale" gives on the full set


@dataclass(frozen=True)
class ClassifierSpec(ModelSpec):
    """animkit's ``ModelSpec`` for a classifier, which also says how the model scores a passenger.

    The other fields (``build``, ``strategy``, ``steps``, ``title``, ``note`` ...) are ``ModelSpec``'s.

    Attributes:
        score_kind: ``SCORE_PROBABILITY`` for a model scored by ``predict_proba``, ``SCORE_MARGIN`` for
            one scored by ``decision_function`` (the SVM, which gives no probabilities).

    Example:
        >>> get_spec("svm").score_kind  # doctest: +SKIP
        'margin'
    """

    score_kind: str = SCORE_PROBABILITY  # SCORE_MARGIN for models without predict_proba


def _log10(value: float) -> float:
    """log10 of a step value: logistic regression's learning curve shows log10 C."""
    return math.log10(value)


_SPECS = (
    ClassifierSpec(
        key="logistic_regression",
        title="Logistic Regression (L2-regularised)",
        short_title="Logistic Regression",
        step_format="log10 C = {}",
        strategy=STRATEGY_PARAM_SWEEP,
        step_label="regularisation weakens  (log10 C)",
        note="C grows each frame: the L2 penalty relaxes, the weights grow from zero, "
             "and σ turns the weighted sum into a probability.",
        build=lambda: LogisticRegression(max_iter=1_000),
        steps=tuple(float(c) for c in np.logspace(-4.0, 1.0, 31)),
        param_name="C",
        hold_frames=2,
        step_transform=_log10,
    ),
    ClassifierSpec(
        key="sgd",
        title="SGD Classifier (logistic loss, gradient descent)",
        short_title="SGD (logistic)",
        step_format="epoch {}",
        strategy=STRATEGY_INCREMENTAL,
        step_label="epochs",
        note="partial_fit on 1/8 of the passengers per frame: averaged weights walk downhill on log loss.",
        build=lambda: SGDClassifier(loss="log_loss", learning_rate="constant", eta0=1e-3,
                                    average=True, random_state=RANDOM_STATE),
        steps=tuple(range(1, 11)),
        chunks_per_epoch=8,
    ),
    ClassifierSpec(
        key="svm",
        title="Support Vector Machine (RBF kernel)",
        short_title="SVM (RBF)",
        step_format="{} passengers",
        strategy=STRATEGY_DATA_GROWTH,
        step_label="training passengers seen",
        note="Refit on more passengers each frame: only the support vectors, the passengers on or "
             "inside the margin, shape the boundary.",
        build=lambda: SVC(kernel="rbf", C=1.0, gamma=SVM_GAMMA),
        steps=GROWTH_FRACTIONS,
        hold_frames=3,
        log_x=True,
        score_kind=SCORE_MARGIN,
        no_weights_note=("An RBF-kernel SVM learns no per-feature weights.\n"
                         "It weighs similarity to its support vectors."),
    ),
    ClassifierSpec(
        key="naive_bayes",
        title="Gaussian Naive Bayes",
        short_title="Naive Bayes",
        step_format="{} passengers",
        strategy=STRATEGY_DATA_GROWTH,
        step_label="training passengers counted",
        note="Refit on more passengers each frame: training is counting, a mean and a spread per "
             "feature for survivors and for non-survivors.",
        build=GaussianNB,
        steps=GROWTH_FRACTIONS,
        hold_frames=3,
        log_x=True,
    ),
    ClassifierSpec(
        key="decision_tree",
        title="Decision Tree Classifier",
        short_title="Decision Tree",
        step_format="depth {}",
        strategy=STRATEGY_PARAM_SWEEP,
        step_label="max_depth",
        note="One level deeper each frame: training accuracy keeps climbing, validation accuracy "
             "peaks early and then slips (overfitting).",
        build=lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        steps=tuple(range(1, 15)),
        param_name="max_depth",
        hold_frames=4,
    ),
    ClassifierSpec(
        key="random_forest",
        title="Random Forest Classifier",
        short_title="Random Forest",
        step_format="{} trees",
        strategy=STRATEGY_WARM_START,
        step_label="trees in the forest",
        note="More bootstrapped trees each frame (warm_start): the forest averages their "
             "survival probabilities.",
        build=lambda: RandomForestClassifier(n_estimators=1, warm_start=True, random_state=RANDOM_STATE),
        steps=(*range(1, 11), *range(15, 101, 5)),
        param_name="n_estimators",
        hold_frames=2,
    ),
    ClassifierSpec(
        key="gradient_boosting",
        title="Gradient Boosting Classifier",
        short_title="Gradient Boosting",
        step_format="{} trees",
        strategy=STRATEGY_WARM_START,
        step_label="boosting stages (trees)",
        note="Five more shallow trees each frame, each one correcting the log-odds of the ones before.",
        build=lambda: GradientBoostingClassifier(n_estimators=1, warm_start=True, max_depth=3,
                                                 random_state=RANDOM_STATE),
        steps=tuple(range(5, 205, 5)),
        param_name="n_estimators",
        hold_frames=2,
    ),
    ClassifierSpec(
        key="knn",
        title=f"K-Nearest Neighbours Classifier (k={KNN_K})",
        short_title=f"KNN (k={KNN_K})",
        step_format="{} passengers",
        strategy=STRATEGY_DATA_GROWTH,
        step_label="training passengers memorised",
        note=f"KNN 'training' is memorising passengers; a new one gets the vote of its {KNN_K} "
             "most similar.",
        build=lambda: KNeighborsClassifier(n_neighbors=KNN_K),
        steps=GROWTH_FRACTIONS,
        hold_frames=3,
        log_x=True,
        no_weights_note=("KNN learns no per-feature weights.\n"
                         f"Each passenger gets the vote of the\n{KNN_K} most similar training passengers."),
    ),
    ClassifierSpec(
        key="mlp",
        title="MLP Classifier (neural network 32-16)",
        short_title="MLP 32-16",
        step_format="epoch {}",
        strategy=STRATEGY_INCREMENTAL,
        step_label="epochs",
        note="One Adam epoch of back-propagation per frame through a 32-16 hidden-layer network "
             "with a sigmoid output.",
        build=lambda: MLPClassifier(hidden_layer_sizes=(32, 16), learning_rate_init=1e-3,
                                    batch_size=32, random_state=RANDOM_STATE),
        steps=tuple(range(1, 61)),
        hold_frames=2,
    ),
)

MODEL_SPECS = MappingProxyType({spec.key: spec for spec in _SPECS})


def model_keys() -> tuple[str, ...]:
    """Every classifier's key, in the order they are animated.

    Returns:
        The keys of ``MODEL_SPECS``.
    """
    return tuple(MODEL_SPECS)


def get_spec(key: str) -> ClassifierSpec:
    """The spec of one classifier.

    Args:
        key: Its key, such as ``"random_forest"``.

    Returns:
        Its ``ClassifierSpec``.

    Raises:
        ValueError: If no classifier has that key; the message lists the keys there are.
    """
    try:
        return MODEL_SPECS[key]
    except KeyError:
        raise ValueError(f"Unknown model '{key}'. Choose one of: {', '.join(MODEL_SPECS)}") from None
