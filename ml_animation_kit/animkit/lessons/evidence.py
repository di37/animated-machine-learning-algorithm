"""Is the all-features model really better? Paired evidence on the same test rows.

A headline may say "better" or "worse" only when a paired test says the difference is unlikely to be
noise: an exact McNemar test for accuracy (which rows each model gets right), and a paired bootstrap
interval for RMSE. Otherwise the honest verdict is "about the same".

Both tests are paired: the two models are scored on the same rows and compared row by row, so the
luck of which rows landed in the test set affects both alike and cancels out.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

ALPHA = 0.05
RESAMPLES = 2_000
CHUNK = 200  # bootstrap resamples drawn at a time, to bound memory


@dataclass(frozen=True)
class Verdict:
    """What paired evidence says about the richer model, against the lesson's model.

    Attributes:
        better: The richer model is better beyond chance.
        worse: The richer model is worse beyond chance.
        evidence: The numbers behind the verdict, for the note.
    """

    better: bool  # the richer model is better beyond chance
    worse: bool  # the richer model is worse beyond chance
    evidence: str  # the numbers behind the verdict, for the note


def paired_accuracy(lesson_right: np.ndarray, richer_right: np.ndarray, noun: str, alpha: float = ALPHA) -> Verdict:
    """Exact McNemar test on the rows where the two models disagree.

    Rows both models get right, or both get wrong, say nothing about which is better. If the models were
    equally good, each disagreement would go either way with probability 1/2, so the count the richer
    model gains is tested against a fair coin (an exact binomial test).

    Args:
        lesson_right: For each test row, whether the lesson's model gets it right.
        richer_right: The same for the richer model, on the same rows in the same order.
        noun: What one row is, for the evidence text; it is made plural with an "s".
        alpha: The significance level.

    Returns:
        The verdict, with the rows gained and lost and the p-value as its evidence.
    """
    lesson_right, richer_right = np.asarray(lesson_right, bool), np.asarray(richer_right, bool)
    gained = int(np.sum(richer_right & ~lesson_right))
    lost = int(np.sum(lesson_right & ~richer_right))
    p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
    return Verdict(better=gained > lost and p < alpha, worse=lost > gained and p < alpha,
                   evidence=f"{gained} {noun}s gained and {lost} lost against the lesson's model "
                            f"(exact McNemar p = {p:.2f})")


def paired_rmse(lesson_errors: np.ndarray, richer_errors: np.ndarray, show: Callable[[float], str],
                seed: int, alpha: float = ALPHA, resamples: int = RESAMPLES) -> Verdict:
    """Paired bootstrap interval for RMSE(lesson) − RMSE(richer): positive means the richer model is better.

    Each resample draws test rows with replacement, the same rows for both models, and takes the
    difference of their RMSEs on them. The verdict is "better" when the whole interval is above zero and
    "worse" when it is all below.

    Args:
        lesson_errors: The lesson's model's error on each test row (either sign: they are squared).
        richer_errors: The richer model's errors, on the same rows in the same order.
        show: Formats a gain in the target's units, for the evidence text.
        seed: Seeds the resampling, so a video always shows the same interval.
        alpha: The significance level: the interval runs between the alpha/2 and 1 − alpha/2 percentiles.
        resamples: How many bootstrap resamples to draw.

    Returns:
        The verdict, with the interval's ends as its evidence.
    """
    lesson_sq, richer_sq = np.asarray(lesson_errors) ** 2, np.asarray(richer_errors) ** 2
    rng = np.random.default_rng(seed)
    gains = []
    for start in range(0, resamples, CHUNK):
        rows = rng.integers(0, len(lesson_sq), size=(min(CHUNK, resamples - start), len(lesson_sq)))
        gains.append(np.sqrt(lesson_sq[rows].mean(axis=1)) - np.sqrt(richer_sq[rows].mean(axis=1)))
    low, high = np.percentile(np.concatenate(gains), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Verdict(better=low > 0, worse=high < 0,
                   evidence=f"95% paired bootstrap interval for the gain: {show(low)} to {show(high)}")
