"""Pieces every step-by-step training loop shares: the numerical guard and the learning-curve sizes.

Each project writes its own training loops, one per strategy in ``animkit.models``; what they have in
common is here. ``CUMULATIVE_STRATEGIES`` names the strategies whose model carries over from one step
to the next, as opposed to a fresh refit at every step.
"""

from animkit.config import MIN_GROWTH_ROWS
from animkit.models import STRATEGY_INCREMENTAL, STRATEGY_WARM_START

# Strategies whose current model is the product of every step so far (vs. one fresh refit).
CUMULATIVE_STRATEGIES = frozenset({STRATEGY_INCREMENTAL, STRATEGY_WARM_START})


class NumericalError(ValueError):
    """A model produced NaN/inf predictions (for example a diverging learning rate)."""


def growth_sizes(fractions, n_train: int, minimum: int = MIN_GROWTH_ROWS) -> tuple[int, ...]:
    """The training-set sizes of a learning curve, from shares of the training rows.

    Each share becomes a row count: rounded, at least ``minimum`` rows (or every row, when there are
    fewer) and at most ``n_train``. Sizes that come out equal are kept once.

    Args:
        fractions: Shares of the training rows, one per step (for example 0.1 up to 1.0).
        n_train: How many training rows there are.
        minimum: The smallest subset to fit on, in rows.

    Returns:
        The distinct row counts, smallest first.

    Example:
        >>> growth_sizes([0.001, 0.5, 1.0], n_train=1000)
        (20, 500, 1000)
    """
    floor = min(minimum, n_train)
    sizes = {min(n_train, max(floor, round(f * n_train))) for f in fractions}
    return tuple(sorted(sizes))
