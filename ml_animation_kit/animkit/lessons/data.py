"""Read-only index arrays for the lesson data views (which rows are drawn, highlighted, revealed).

A lesson's data view keeps lists of row numbers that every chapter reads. Freezing them means no
painter can reorder or overwrite them by accident while a video renders.
"""

import numpy as np


def read_only_indices(values) -> np.ndarray:
    """An integer index array that cannot be modified.

    Args:
        values: The indices, as any sequence or array numpy accepts.

    Returns:
        A new integer array (a copy, so the caller's ``values`` stay writable) with writing switched off.
    """
    array = np.array(values, dtype=int, copy=True)
    array.setflags(write=False)
    return array
