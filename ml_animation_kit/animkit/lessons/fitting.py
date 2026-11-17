"""Fixed-schedule training for online learners (SGD, perceptron, MLP), shared by the comparison chapters.

Each ``fit`` trains a fresh clone, so fitting twice gives the same model; the constructor parameters
are never changed.

The wrappers are scikit-learn estimators themselves (``get_params`` reads their constructor
arguments), so they can be cloned, fitted and scored like any other model.
"""

import copy

import numpy as np
from sklearn.base import BaseEstimator, clone


class EpochTrainer(BaseEstimator):
    """Exactly ``epochs`` shuffled passes of partial_fit.

    Attributes:
        model: The online learner to train; it is cloned, never changed.
        epochs: How many passes over the training rows.
        classes: Every class label, which a classifier's first ``partial_fit`` needs; ``None`` for a
            regressor.
        seed: Seeds the shuffling, so the same data always gives the same model.
        model_: The trained clone, set by ``fit``.
    """

    def __init__(self, model, epochs: int, classes=None, seed: int = 0):
        """Stores the settings; nothing is trained until ``fit``.

        Args:
            model: The online learner (anything with ``partial_fit``).
            epochs: How many shuffled passes to make.
            classes: Every class label, for a classifier; ``None`` for a regressor.
            seed: Seeds the shuffle order.
        """
        self.model = model
        self.epochs = epochs
        self.classes = classes
        self.seed = seed

    def _step(self, model, X: np.ndarray, y: np.ndarray, order: np.ndarray) -> None:
        """One pass: ``partial_fit`` on every row, in the given order."""
        if self.classes is None:
            model.partial_fit(X[order], y[order])
        else:
            model.partial_fit(X[order], y[order], classes=self.classes)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EpochTrainer":
        """Trains a fresh clone of ``model`` for ``epochs`` passes, each in a new shuffled order.

        Args:
            X: The training features, one row per example.
            y: The training targets.

        Returns:
            This trainer, with the trained clone in ``model_``.
        """
        model, rng = clone(self.model), np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            self._step(model, X, y, rng.permutation(len(y)))
        self.model_ = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts with the trained clone.

        Args:
            X: The rows to predict, one per example.

        Returns:
            The model's predictions, one per row.
        """
        return self.model_.predict(X)


class BestEpoch(EpochTrainer):
    """Like EpochTrainer, but keeps the epoch with the best validation accuracy.

    A simple form of early stopping: every epoch is still trained, and the model kept is a copy taken
    after the epoch that scored best on the validation rows (the earliest of them, on a tie).

    Attributes:
        X_val: The validation features.
        y_val: The validation labels.
    """

    def __init__(self, model, epochs: int, X_val: np.ndarray, y_val: np.ndarray, classes=None, seed: int = 0):
        """Stores the settings and the validation rows.

        Args:
            model: The online classifier.
            epochs: How many shuffled passes to make.
            X_val: The validation features, scored after every epoch.
            y_val: The validation labels.
            classes: Every class label, for the first ``partial_fit``.
            seed: Seeds the shuffle order.
        """
        super().__init__(model, epochs, classes, seed)
        self.X_val = X_val
        self.y_val = y_val

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BestEpoch":
        """Trains a fresh clone for ``epochs`` passes and keeps the copy that scored best on validation.

        Args:
            X: The training features.
            y: The training labels.

        Returns:
            This trainer, with the best epoch's model in ``model_``.
        """
        model, rng = clone(self.model), np.random.default_rng(self.seed)
        best, best_score = None, -1.0
        for _ in range(self.epochs):
            self._step(model, X, y, rng.permutation(len(y)))
            score = float(np.mean(model.predict(self.X_val) == self.y_val))
            if score > best_score:
                best, best_score = copy.deepcopy(model), score
        self.model_ = best
        return self
