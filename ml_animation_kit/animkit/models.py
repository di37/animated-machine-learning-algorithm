"""How an animated estimator is trained step by step: the four training strategies and the model spec.

A dashboard animates training by showing the model after every step, so each estimator needs a recipe
for what one step is:

- ``incremental``: ``partial_fit`` on shuffled chunks of the training set, epoch after epoch.
- ``warm_start``: keep the fitted ensemble and add more members each step (``warm_start=True``).
- ``param_sweep``: refit from scratch while one hyper-parameter moves along a schedule.
- ``data_growth``: refit from scratch on a growing share of the training rows (a learning curve).

A ``ModelSpec`` names the strategy and its schedule; each project's training loop carries it out.
"""

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.base import BaseEstimator

# partial_fit on shuffled chunks of the training set, epoch after epoch.
STRATEGY_INCREMENTAL = "incremental"
# Keep the fitted ensemble and add more members each step (warm_start=True).
STRATEGY_WARM_START = "warm_start"
# Refit from scratch while one hyper-parameter moves along a schedule.
STRATEGY_PARAM_SWEEP = "param_sweep"
# Refit from scratch on a growing share of the training rows (a learning curve).
STRATEGY_DATA_GROWTH = "data_growth"
STRATEGIES = (STRATEGY_INCREMENTAL, STRATEGY_WARM_START, STRATEGY_PARAM_SWEEP, STRATEGY_DATA_GROWTH)


@dataclass(frozen=True)
class ModelSpec:
    """Everything a dashboard needs to train and label one animated estimator.

    Attributes:
        key: A short identifier for the model.
        title: The model's full name, for headings.
        short_title: A shorter name, for race labels and panel titles.
        step_format: How a step is worded, ``{}`` filled with the formatted step value (``"{} trees"``).
        strategy: How the model is trained step by step: one of ``STRATEGIES``.
        step_label: The axis label for the step value.
        note: One line about the model, shown under the dashboard's title.
        build: Makes a fresh, unfitted estimator.
        steps: The schedule. What an entry means follows the strategy: one per epoch (incremental), the
            member counts (warm start), the parameter values (sweep) or shares of the training rows
            (data growth).
        param_name: The estimator parameter a warm start or a sweep sets at each step.
        chunks_per_epoch: How many ``partial_fit`` calls make one epoch (incremental only); each is a step.
        hold_frames: Video frames per training step, to slow very short schedules.
        log_x: Whether the step axis is logarithmic.
        step_transform: Turns a step value into the value shown on the axis (for example −log10 of a
            regularisation strength); ``None`` shows the value itself.
        no_weights_note: What the weight panel says for a model without per-feature weights.
    """

    key: str
    title: str
    short_title: str
    step_format: str  # e.g. "{} trees", filled with the formatted step value
    strategy: str
    step_label: str
    note: str
    build: Callable[[], BaseEstimator]
    steps: tuple[float, ...]
    param_name: str | None = None
    chunks_per_epoch: int = 1
    hold_frames: int = 1  # video frames per training step, to slow very short schedules
    log_x: bool = False
    step_transform: Callable[[float], float] | None = None
    no_weights_note: str = "This model exposes no per-feature weights."
