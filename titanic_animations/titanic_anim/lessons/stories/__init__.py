"""Registry of classifier lesson builders: key -> (one-line description, build(PassengerData) -> lesson).

Each module in this package is one classifier family's lesson. Its ``build`` trains the model on the
passengers and keeps a real ``ClassSnapshot`` at each moment worth showing. It writes the narration as an
animkit ``Story`` and returns a ``ClassLesson``, with a worked, verified prediction for the unseen
passenger and a side panel (often an animkit ``Diagram`` too) showing how the model works.

The modules, simplest idea first: ``logistic`` (gradient descent, written out), ``linear_models`` (the
perceptron and a linear SVM, which give scores, not probabilities), ``naive_bayes`` (counting), ``tree``
(best-first splits), ``forest`` (averaging bootstrapped trees), ``boosting`` (trees correcting the
log-odds), ``knn`` (memory) and ``mlp`` (a small neural network). ``common`` holds what several share.
The command line lists and builds the lessons through ``lesson_keys``, ``describe`` and ``build_lesson``.
"""

from collections.abc import Callable

from titanic_anim.lessons.data import PassengerData
from titanic_anim.lessons.lesson import ClassLesson
from titanic_anim.lessons.stories import (
    boosting,
    forest,
    knn,
    linear_models,
    logistic,
    mlp,
    naive_bayes,
    tree,
)

Builder = Callable[[PassengerData], ClassLesson]

BUILDERS: dict[str, tuple[str, Builder]] = {
    "logistic_regression": (f"{logistic.STEPS} steps of gradient descent on the log loss; the 50% boundary "
                            "settles", logistic.build),
    "perceptron": ("online updates on mistakes only; a score, not a probability",
                   linear_models.build_perceptron),
    "linear_svm": (f"hinge loss by SGD for {linear_models.SVM_EPOCHS} epochs; boundary and ±1 margin",
                   linear_models.build_linear_svm),
    "naive_bayes": ("priors and per-class bell curves counted from more and more passengers",
                    naive_bayes.build),
    "decision_tree": (f"tree grown one best split at a time to {tree.MAX_LEAVES} leaves; the passenger "
                      "walks root to leaf", tree.build),
    "random_forest": (f"{forest.TREE_COUNTS[-1]} bootstrapped trees; their probabilities averaged",
                      forest.build),
    "gradient_boosting": (f"{boosting.N_TREES} small trees correcting the log-odds at "
                          f"{boosting.LEARNING_RATE:.0%} strength", boosting.build),
    "knn": ("store the passengers, sweep k; the share of the k most similar who survived", knn.build),
    "mlp": (f"3-{'-'.join(map(str, mlp.HIDDEN))}-1 network trained by Adam; forward pass at inference",
            mlp.build),
}


def lesson_keys() -> tuple[str, ...]:
    """Every lesson's key, in the order the lessons are rendered.

    Returns:
        The keys, such as 'logistic_regression' and 'knn'.
    """
    return tuple(BUILDERS)


def describe(key: str) -> str:
    """One line about a lesson, for ``--list``.

    Args:
        key: The lesson's key.

    Returns:
        What the lesson shows.
    """
    return BUILDERS[key][0]


def build_lesson(key: str, data: PassengerData) -> ClassLesson:
    """Trains a lesson's model and builds the lesson.

    Args:
        key: The lesson's key.
        data: The passengers.

    Returns:
        The lesson, ready to render.

    Raises:
        KeyError: If there is no lesson with that key.
    """
    if key not in BUILDERS:
        raise KeyError(f"unknown lesson '{key}'; choose from {', '.join(BUILDERS)}")
    return BUILDERS[key][1](data)
