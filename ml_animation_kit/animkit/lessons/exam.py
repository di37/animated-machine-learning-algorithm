"""Shared pieces of the final-exam chapter: a fixed scoring order, the reveal pace, and measured speed.

Speed is measured once per video (the best of several timed passes over the whole test set), so the
number on screen is a real measurement from the machine that rendered it.

Test examples are scored on screen in a fixed shuffled order, the same every time the video renders.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from animkit.lessons.data import read_only_indices
from animkit.lessons.timing import phase

SPEED_REPEATS = 5
REVEAL_END = 0.8  # every test example is scored by 80% of the chapter's moving part


@dataclass(frozen=True)
class Speed:
    """Measured prediction speed, in microseconds per example.

    Attributes:
        single_us: One example on its own: the latency of a single predict call.
        batch_us: Per example when the whole test set is predicted in one call.
        batch_size: How many examples that batch held.
    """

    single_us: float  # one example on its own: the latency of a single predict call
    batch_us: float  # per example when the whole test set is predicted in one call
    batch_size: int


def exam_order(count: int, seed: int) -> np.ndarray:
    """The fixed, shuffled order in which test examples are scored on screen.

    Args:
        count: How many test examples there are.
        seed: Seeds the shuffle, so every render uses the same order.

    Returns:
        A read-only permutation of ``0 .. count - 1``.
    """
    return read_only_indices(np.random.default_rng(seed).permutation(count))


def revealed(count: int, t: float) -> int:
    """How many test examples have been scored at chapter progress ``t``.

    Scoring eases in and is complete at ``REVEAL_END``; at least one example is always shown.

    Args:
        count: How many test examples there are.
        t: The chapter's progress, 0 to 1.

    Returns:
        The number scored so far, 1 to ``count``.
    """
    return min(count, max(1, math.ceil(count * phase(t, 0.0, REVEAL_END))))


def _fastest(call: Callable[[], object], repeats: int) -> float:
    """Best of ``repeats`` timed calls after one warm-up; contention can only make a call slower."""
    call()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return max(min(timings), 1e-9)


def measure_speed(predict: Callable[[np.ndarray], np.ndarray], rows: np.ndarray,
                  repeats: int = SPEED_REPEATS) -> Speed:
    """Times the model on one example and on the whole test set, the best of several passes each.

    Args:
        predict: The fitted model's predict function.
        rows: The test set's feature rows.
        repeats: How many timed passes over the whole set; one example, being quicker to time, gets four
            times as many.

    Returns:
        The latency of one example and the per-example cost in one batch, in microseconds.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if len(rows) == 0:
        raise ValueError("cannot time predictions on an empty test set")
    single = _fastest(lambda: predict(rows[:1]), repeats * 4)
    batch = _fastest(lambda: predict(rows), repeats)
    return Speed(single_us=single * 1e6, batch_us=batch / len(rows) * 1e6, batch_size=len(rows))


def duration(us: float) -> str:
    """Two significant figures in a readable unit: '45 ns', '2.6 µs', '1.3 ms'.

    Args:
        us: A time in microseconds.

    Returns:
        The time in nanoseconds, microseconds or milliseconds, whichever reads best.

    Example:
        >>> duration(0.045), duration(2.63), duration(1300)
        ('45 ns', '2.6 µs', '1.3 ms')
    """
    rounded = float(f"{us:.2g}")  # round first, so 999.6 ns becomes 1 µs rather than "1000 ns"
    scale, unit = next((s, u) for s, u in ((1e-3, "ns"), (1.0, "µs"), (1e3, "ms")) if rounded < s * 1000 or u == "ms")
    return f"{float(f'{rounded / scale:.2g}'):g} {unit}"


def speed_line(speed: Speed, noun: str) -> str:
    """The exam's speed line, in plain words.

    Args:
        speed: The measured speed.
        noun: What one example is, such as "row".

    Returns:
        A line such as "speed: 2.6 µs for one row · 45 ns each in a batch of 1,000 (measured here)".
    """
    return (f"speed: {duration(speed.single_us)} for one {noun} · {duration(speed.batch_us)} each "
            f"in a batch of {speed.batch_size:,} (measured here)")
