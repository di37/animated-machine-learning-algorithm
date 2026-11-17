"""Tiny animation helpers shared by the classifier panels.

During the prediction chapter the director calls a panel with ``t``, the chapter's progress from 0 to 1.
A panel that shows its items one after another (a forest's votes, boosting's tree-by-tree climb) asks
``revealed`` how many to draw at that moment, eased so the reveal starts and ends gently.
"""

import math

from animkit.lessons.timing import ease


def revealed(total: int, t: float) -> int:
    """How many of ``total`` items are on screen at progress ``t`` (eased, at least one).

    Args:
        total: How many items there are.
        t: Progress through the reveal, from 0 to 1 (clamped outside).

    Returns:
        A count from 1 to ``total``.
    """
    return max(1, min(total, math.ceil(total * ease(t))))
