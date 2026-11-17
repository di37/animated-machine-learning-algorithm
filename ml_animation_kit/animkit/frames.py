"""The per-feature view of what a model has learned, as an immutable snapshot for the weight bars.

A training dashboard shows one bar per input feature, and what a bar means depends on the model: a
linear model's coefficient on a standardised feature, a tree ensemble's feature importance, or a
network's input sensitivity (the mean absolute first-layer weight leaving that input). ``kind`` says
which, so the bars can choose their title, colours and axis.
"""

from dataclasses import dataclass

import numpy as np

KIND_COEFFICIENT = "coefficient"
KIND_IMPORTANCE = "importance"
KIND_SENSITIVITY = "sensitivity"


@dataclass(frozen=True)
class FeatureWeights:
    """One snapshot of a model's per-feature weights.

    Attributes:
        names: The feature names, in the model's column order.
        values: One weight per feature, in the same order.
        kind: What the weights are: ``KIND_COEFFICIENT`` (signed), ``KIND_IMPORTANCE`` or
            ``KIND_SENSITIVITY`` (both unsigned).
    """

    names: tuple[str, ...]
    values: np.ndarray
    kind: str
