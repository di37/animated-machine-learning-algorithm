"""Read-only arrays, and the small view of a project's dataset splits that shared drawing code reads.

Arrays handed to painters are copied and frozen, so drawing one frame can never change the data the
next frame draws. ``ScaledSplits`` is a ``Protocol``: any object with these three properties will do,
whatever else the project's splits hold.
"""

from typing import Protocol

import numpy as np


class ScaledSplits(Protocol):
    """What shared code needs from a project's train/validation/test splits.

    Features are standardised with the training rows' mean and scale, so a scaled value (a tree
    threshold, say) can be mapped back to real units for a label.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The feature names, in column order.

        Returns:
            One name per feature column.
        """
        ...

    @property
    def feature_mean(self) -> np.ndarray:
        """The training rows' mean of each feature, in the feature's real units.

        Returns:
            One mean per feature.
        """
        ...

    @property
    def feature_scale(self) -> np.ndarray:
        """The training rows' scale of each feature: a standardised value × scale + mean is the real value.

        Returns:
            One scale per feature.
        """
        ...


def read_only(array) -> np.ndarray:
    """A float copy of ``array`` that cannot be modified.

    Args:
        array: Any array-like of numbers.

    Returns:
        A new float array with writing switched off; the caller's array stays as it was.
    """
    copy = np.array(array, dtype=float, copy=True)
    copy.setflags(write=False)
    return copy
