"""Advance several training loops together so each frame shows every model equally far along.

Models train on schedules of different lengths: one may run 300 epochs, another grow 40 trees. A race
compares them at the same share of their own schedules, not at the same step number, so at tick k of
n every model has done at least k/n of its schedule. A model with a short schedule simply shows the
same step for several ticks.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from animkit.models import ModelSpec


class TrainingStepLike(Protocol):
    """One step of a training loop: its frame (``step_index``, ``total_steps``, ``is_final``) and model."""

    @property
    def frame(self) -> Any:
        """The step's frame: what the dashboards draw, with its place in the schedule.

        Returns:
            An object with ``step_index`` (counted from 0), ``total_steps`` and ``is_final``.
        """
        ...

    @property
    def model(self) -> Any:
        """The model as it stands after this step.

        Returns:
            The fitted estimator.
        """
        ...


StepFn = Callable[[ModelSpec, Any], Iterator[TrainingStepLike]]


@dataclass(frozen=True)
class Lane:
    """One model's place in the race at a tick: its spec and the latest training step it has reached.

    Attributes:
        spec: The model's spec.
        step: The latest training step it has reached.
    """

    spec: ModelSpec
    step: TrainingStepLike

    @property
    def frame(self) -> Any:
        """The latest step's frame.

        Returns:
            The frame the project's training loop made for that step.
        """
        return self.step.frame

    @property
    def model(self) -> Any:
        """The model after its latest step.

        Returns:
            The fitted estimator.
        """
        return self.step.model

    @property
    def progress(self) -> float:
        """How far through its own schedule this model is.

        Returns:
            The share of its steps done, above 0 and at most 1.
        """
        return (self.frame.step_index + 1) / self.frame.total_steps

    @property
    def finished(self) -> bool:
        """Whether this model has reached the last step of its schedule.

        Returns:
            True once the latest step is the final one.
        """
        return self.frame.is_final


@dataclass(frozen=True)
class LockstepTick:
    """One snapshot of the race: every model at the same share of its schedule.

    Attributes:
        tick: Which tick this is, counted from 1.
        ticks: How many ticks the race has.
        lanes: One lane per model, in the order the models were given.
    """

    tick: int
    ticks: int
    lanes: tuple[Lane, ...]

    @property
    def is_final(self) -> bool:
        """Whether this is the race's last tick.

        Returns:
            True on tick ``ticks``, when every model has finished.
        """
        return self.tick == self.ticks


def _advance(step: TrainingStepLike | None, stream: Iterator[TrainingStepLike],
             target: float) -> TrainingStepLike:
    """Run training steps until this model is at least ``target`` of the way through its schedule."""
    while step is None or (
        not step.frame.is_final and (step.frame.step_index + 1) / step.frame.total_steps < target
    ):
        step = next(stream)
    return step


def lockstep(specs: Sequence[ModelSpec], splits: Any, ticks: int,
             step_fn: StepFn) -> Iterator[LockstepTick]:
    """Yield ``ticks`` snapshots; at tick k every model has done k/ticks of its own schedule.

    ``step_fn(spec, splits)`` is the training loop to advance; any loop that yields objects with
    ``.frame`` (``step_index``, ``total_steps``, ``is_final``) and ``.model`` works.

    Args:
        specs: The racing models.
        splits: The project's data splits, passed to ``step_fn`` as they are.
        ticks: How many snapshots to take.
        step_fn: The project's training loop: ``step_fn(spec, splits)`` yields one step at a time.

    Yields:
        One ``LockstepTick`` per tick, from 1 to ``ticks``.

    Raises:
        ValueError: If ``ticks`` is below 1 or there are no models.
    """
    if ticks < 1:
        raise ValueError("ticks must be at least 1")
    if not specs:
        raise ValueError("lockstep needs at least one model")
    streams = tuple(step_fn(spec, splits) for spec in specs)
    steps: tuple[TrainingStepLike | None, ...] = (None,) * len(specs)
    for tick in range(1, ticks + 1):
        target = tick / ticks
        steps = tuple(_advance(step, stream, target) for step, stream in zip(steps, streams))
        yield LockstepTick(tick, ticks, tuple(Lane(spec, step) for spec, step in zip(specs, steps)))
