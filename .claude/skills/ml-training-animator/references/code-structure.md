# Code that teaches: structure, DRY, OOP and Google-style docstrings

The source is a second deliverable. A reader who opens it should learn the algorithm, how its states were
recorded truthfully, and how a video is built from them. The examples below are one small, runnable k-means
project shown piece by piece; the same patterns apply to every family.

## Folder template

```
kmeans_animations/                 one project per dataset or model family (a sibling of the kits)
  README.md                        what it makes, how to run it, and "How to read this code"
  pyproject.toml                   pytest, coverage and ruff settings (pythonpath to the kits)
  animate_kmeans.py                entry script: a few lines that call cli.main()
  kmeans_anim/
    __init__.py                    one-paragraph overview of the package
    config.py                      every constant, with units: seeds, k, frame rates, paths
    cli.py                         argument parsing and error reporting, nothing else
    data/
      loading.py                   reading the dataset; provenance
      splits.py                    train/validation/test split and scaling (fitted on train only)
    algorithms/
      base.py                      the role every algorithm plays (an ABC or Protocol)
      kmeans.py                    one algorithm, overriding only what differs
    training/
      states.py                    frozen dataclasses for the recorded states
      recorder.py                  runs the real training and records every state
    scenes/
      chapters.py                  what each chapter draws, from recorded states only
    rendering/
      video.py                     canvas and encoding (or reuse animkit / manimkit)
  tests/                           mirrors kmeans_anim/, plus test_boundaries.py
  outputs/<dataset>/               videos and CSVs
```

Map onto what exists: `ml_animation_kit/animkit` and `manim_lesson_kit/manimkit` are the shared kits, and
`house_price_animations/house_anim` is a dataset project (`data.py`, `models.py`, `trainers.py`,
`lessons/stories/`). Reuse the kits' stage, director, encoder and player before writing new ones.

## Google-style docstrings

Every module, class and public function or method has one. The summary line says what it is; the paragraph
after it teaches the idea; the sections say how to use it.

```python
"""k-means clustering, recorded one Lloyd iteration at a time.

k-means repeats two moves until nothing changes: assign every point to its nearest
centroid, then move each centroid to the mean of its points. Each repeat can only
lower the inertia (the sum of squared distances from points to their centroids),
which is why the video's inertia strip never rises.
"""

from dataclasses import dataclass

import numpy as np

MAX_ITERATIONS = 50  # a cap; real runs on lesson data settle in well under 20
TOLERANCE = 1e-6  # centroid movement (in scaled units) below which the run has settled


@dataclass(frozen=True)
class KMeansStep:
    """One recorded state of k-means: centroids, assignments and inertia.

    Attributes:
        step: Iteration number; 0 is the initial centroids, before any move.
        centroids: Centroid positions, shape (k, n_features), in scaled units.
        labels: Cluster index of every training point, shape (n_samples,).
        inertia: Sum of squared distances from each point to its centroid.
    """

    step: int
    centroids: np.ndarray
    labels: np.ndarray
    inertia: float


def assign(points: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, float]:
    """Assign each point to its nearest centroid (the first half of a Lloyd step).

    Args:
        points: Training points, shape (n_samples, n_features), already scaled.
        centroids: Current centroids, shape (k, n_features).

    Returns:
        The nearest centroid's index for every point, and the resulting inertia.

    Example:
        >>> labels, inertia = assign(np.array([[0.0], [9.0]]), np.array([[1.0], [10.0]]))
        >>> labels.tolist(), inertia
        ([0, 1], 2.0)
    """
    squared = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = squared.argmin(axis=1)
    return labels, float(squared[np.arange(len(points)), labels].sum())


def update(points: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Move each centroid to the mean of its points (the second half of a Lloyd step).

    Args:
        points: Training points, shape (n_samples, n_features).
        labels: Each point's cluster index, shape (n_samples,).
        k: Number of clusters.

    Returns:
        The new centroids, shape (k, n_features).

    Raises:
        ValueError: If a cluster has no points, so its mean is undefined.
    """
    empty = [cluster for cluster in range(k) if not np.any(labels == cluster)]
    if empty:
        raise ValueError(f"clusters {empty} lost all their points; restart with another seed")
    return np.stack([points[labels == cluster].mean(axis=0) for cluster in range(k)])
```

## Object-oriented patterns

- **Frozen dataclasses for records.** A recorded state never changes after it is made (as `KMeansStep`
  above). Arrays inside are copied and made read-only when stored.
- **One role, many algorithms (template method).** The base class runs the loop and records states; each
  algorithm supplies only its own step. A new algorithm is a new small subclass, not a copied loop.

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

State = TypeVar("State")


class IterativeRecorder(ABC, Generic[State]):
    """Runs an iterative algorithm and records every real state, for the video to replay.

    The loop and the recording live here once, for every algorithm. A subclass says
    only what its first state is, how one real step changes it, and when it has
    settled.

    Attributes:
        max_iterations: The most steps to run before stopping.
    """

    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        """Creates the recorder.

        Args:
            max_iterations: The most steps to run before stopping.
        """
        self.max_iterations = max_iterations

    def record(self, points: np.ndarray) -> tuple[State, ...]:
        """Trains on ``points`` and returns every state, the initial one first.

        Args:
            points: Training points, shape (n_samples, n_features), already scaled.

        Returns:
            The recorded states, in training order.
        """
        states = [self.initial_state(points)]
        while len(states) <= self.max_iterations and not self.settled(states[-2:]):
            states.append(self.next_state(points, states[-1]))
        return tuple(states)

    @abstractmethod
    def initial_state(self, points: np.ndarray) -> State:
        """The state before any training step."""

    @abstractmethod
    def next_state(self, points: np.ndarray, state: State) -> State:
        """The state after one more real training step."""

    @abstractmethod
    def settled(self, latest: list[State]) -> bool:
        """Whether training has settled, judging by the last one or two states."""


class KMeansRecorder(IterativeRecorder[KMeansStep]):
    """k-means as an iterative recorder: random initial centroids, then Lloyd steps.

    Attributes:
        k: Number of clusters.
        seed: Seed for choosing the initial centroids, shown in the video.
    """

    def __init__(self, k: int, seed: int = 42, max_iterations: int = MAX_ITERATIONS):
        """Creates the recorder.

        Args:
            k: Number of clusters.
            seed: Seed for choosing the initial centroids.
            max_iterations: The most Lloyd steps to run.
        """
        super().__init__(max_iterations)
        self.k = k
        self.seed = seed

    def initial_state(self, points: np.ndarray) -> KMeansStep:
        """Picks ``k`` distinct training points as the first centroids."""
        rows = np.random.default_rng(self.seed).choice(len(points), size=self.k, replace=False)
        centroids = points[rows].copy()
        labels, inertia = assign(points, centroids)
        return KMeansStep(0, centroids, labels, inertia)

    def next_state(self, points: np.ndarray, state: KMeansStep) -> KMeansStep:
        """One Lloyd step: move the centroids to their points' means, then reassign."""
        centroids = update(points, state.labels, self.k)
        labels, inertia = assign(points, centroids)
        return KMeansStep(state.step + 1, centroids, labels, inertia)

    def settled(self, latest: list[KMeansStep]) -> bool:
        """Whether the last step moved every centroid by less than ``TOLERANCE``."""
        if len(latest) < 2:
            return False
        before, after = latest
        return float(np.abs(after.centroids - before.centroids).max()) < TOLERANCE
```

- **Composition for a lesson.** A lesson holds its story (the words), its recorded states and its scene (how
  to draw); none of them inherits from another. animkit's `Lesson`, `Story` and `Scene` are the model.
- **A registry, not if/else chains.** Map each algorithm's key to its builder in one dict (as
  `house_anim.lessons.stories.BUILDERS`); the CLI and the tests read the same registry.
- **No class for its own sake.** A pure calculation (`assign`, `update`) stays a function.

## DRY, concretely

- One `config.py` per project for constants and one `theme` for colours and sizes; nothing hard-coded twice.
- When two chapters draw the same thing, extract a helper or a base method; when two projects need it, move it
  to a kit.
- Tests share fixtures through `conftest.py` instead of rebuilding data in every file.
- Docstrings say a thing once: the module docstring states the idea, a function docstring its own part.

## Before calling the code done

- [ ] Feature folders as in the template; the entry script and `cli.py` are thin.
- [ ] No block appears twice; constants live in `config.py` with units.
- [ ] Every module, class and public function has a Google-style docstring that teaches its idea, with
      `Args`, `Returns`, `Raises`, `Attributes` and an `Example` where it helps.
- [ ] Names use the field's terms; functions under 50 lines; files under 400 lines where possible.
- [ ] Learning code imports no drawing code; recorded states are frozen.
- [ ] Tests named as sentences, 80%+ coverage, `test_boundaries.py` for the folder rules.
- [ ] The README has *How to read this code*.
- [ ] `ruff check --select D` with the Google convention passes.
