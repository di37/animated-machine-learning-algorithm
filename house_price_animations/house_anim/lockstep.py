"""Advance several regressors' training loops together: animkit's lockstep, defaulting to their loop.

Models' schedules have different lengths (18 tree depths, 60 MLP epochs), so a fair side-by-side
picture compares progress, not step counts: at tick k of n, every model has done at least k/n of
its own schedule. This module only fills in the house-price training loop as the default.
"""

from collections.abc import Iterator, Sequence

from animkit.lockstep import Lane, LockstepTick, StepFn
from animkit.lockstep import lockstep as lockstep_any
from house_anim.data import HousingSplits
from house_anim.models import ModelSpec
from house_anim.trainers import training_steps

__all__ = ["Lane", "LockstepTick", "lockstep"]


def lockstep(specs: Sequence[ModelSpec], splits: HousingSplits, ticks: int,
             step_fn: StepFn = training_steps) -> Iterator[LockstepTick]:
    """Yield ``ticks`` snapshots; at tick k every model has done k/ticks of its own schedule.

    ``step_fn(spec, splits)`` is the training loop to advance (the regression loop by default).

    Args:
        specs: The models to train side by side.
        splits: The data every model trains on.
        ticks: How many snapshots to take; the last one has every model at the end of its schedule.
        step_fn: The training loop to advance for each model.

    Returns:
        An iterator of ``LockstepTick`` snapshots, each holding one ``Lane`` (spec, frame and live
        model) per model.
    """
    return lockstep_any(specs, splits, ticks, step_fn)
