"""Registry of lesson builders: key -> (one-line description, build(IncomeData) -> Lesson).

Each module in this package tells one algorithm family's story. A builder trains the real model on the
income view, records a ``Snapshot`` of every state worth showing (a gradient step, a degree, a split,
a tree, a k, an epoch), picks the final model, and writes the ``Story``: animkit's record of every word
the video says. It returns a ``Lesson`` with the algorithm's side ``Panel`` and, for most, a diagram.

Read ``linear_gd`` first (the simplest model and the full pattern), then ``polynomial`` (a sweep chosen
on validation, and regularisation), ``tree``, then ``forest`` and ``boosting`` (two ways to combine
trees), ``knn`` and finally ``mlp``.
"""

from collections.abc import Callable

from house_anim.lessons.config import POLY_SAMPLE
from house_anim.lessons.data import IncomeData
from house_anim.lessons.lesson import Lesson
from house_anim.lessons.stories import boosting, forest, knn, linear_gd, mlp, polynomial, tree

Builder = Callable[[IncomeData], Lesson]

BUILDERS: dict[str, tuple[str, Builder]] = {
    "linear_regression": (f"straight line, {linear_gd.STEPS} steps of batch gradient descent across the "
                          "loss bowl", linear_gd.build),
    "polynomial_regression": (f"degree 1 → {polynomial.MAX_DEGREE} on {POLY_SAMPLE} districts: training "
                              "error falls, validation turns", polynomial.build_polynomial),
    "ridge": (f"degree-{polynomial.MAX_DEGREE} polynomial, L2 penalty swept: weights shrink smoothly",
              polynomial.build_ridge),
    "lasso": (f"degree-{polynomial.MAX_DEGREE} polynomial, L1 penalty swept: weights switch off",
              polynomial.build_lasso),
    "decision_tree": (f"tree grown one best split at a time to {tree.MAX_LEAVES} leaves; the house "
                      "walks root to leaf", tree.build),
    "random_forest": (f"{forest.TREE_COUNTS[-1]} bootstrapped trees added and averaged; every tree votes",
                      forest.build),
    "gradient_boosting": (f"{boosting.N_TREES} small trees fitted to the residuals, added at "
                          f"{boosting.LEARNING_RATE:.0%} strength", boosting.build),
    "knn": ("store the data, sweep k; the house averages its nearest neighbours", knn.build),
    "mlp": (f"1-{'-'.join(map(str, mlp.HIDDEN))}-1 ReLU network trained by Adam; forward pass at inference",
            mlp.build),
}


def lesson_keys() -> tuple[str, ...]:
    """Every lesson's key, in the order they are listed and rendered.

    Returns:
        The keys of ``BUILDERS``.
    """
    return tuple(BUILDERS)


def describe(key: str) -> str:
    """One line about a lesson, for ``--list``.

    Args:
        key: The lesson's key.

    Returns:
        Its description.
    """
    return BUILDERS[key][0]


def build_lesson(key: str, data: IncomeData) -> Lesson:
    """Builds one lesson: trains its model on ``data`` and records its snapshots and story.

    Args:
        key: The lesson's key.
        data: The income view to learn from.

    Returns:
        The lesson, ready to render.

    Raises:
        KeyError: If there is no lesson with that key.
    """
    if key not in BUILDERS:
        raise KeyError(f"unknown lesson '{key}'; choose from {', '.join(BUILDERS)}")
    return BUILDERS[key][1](data)
