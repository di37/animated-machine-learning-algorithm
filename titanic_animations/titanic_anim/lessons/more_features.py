"""The more-features chapter: the same algorithm, given every input, against the lesson's three.

The all-inputs model is refit on the lesson's training passengers, then judged on the same test
passengers. The lessons see age, fare and sex; the full set adds class, siblings/spouses,
parents/children and port of embarkation (two columns), with fare log-transformed as in the lessons.

``RICHER`` holds, per lesson key, a factory for that algorithm's all-inputs model. ``compare`` fits it and
returns an animkit ``Comparison`` of three test accuracies (always guessing 'did not survive', the
lesson's model, the all-inputs model) for the director's more-features card.

Settings carry over where they mean the same thing (tree size, number of trees, epochs, network shape);
k is re-chosen on validation, never on test. With only a couple of hundred test passengers, the
headline claims a difference only when an exact McNemar test says it is unlikely to be chance.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from animkit.lessons.evidence import paired_accuracy
from animkit.lessons.fitting import BestEpoch, EpochTrainer
from animkit.lessons.lesson import Comparison, ComparisonRow
from titanic_anim.config import RANDOM_STATE
from titanic_anim.lessons.lesson import ClassLesson, threshold
from titanic_anim.lessons.stories import boosting, forest, knn, linear_models, mlp, tree

INPUT_LIST = "class, sex, age, siblings/spouses, parents/children, fare, port of embarkation (8 columns)"
CLASSES = np.array([0, 1])
ONE_RUN = "one training run; other random seeds land a few points either way"

Factory = Callable[[ClassLesson], tuple[Any, str]]


def _rows(lesson: ClassLesson) -> int:
    """How many training passengers the all-inputs model fits on."""
    return len(lesson.data.full.y_train)


def _knn(lesson: ClassLesson) -> tuple[Any, str]:
    """An unfitted kNN for every input, its k re-chosen on the validation passengers, and its settings line."""
    full = lesson.data.full
    options = tuple(k for k in knn.K_VALUES if k <= len(full.y_train))
    scores = [np.mean(KNeighborsClassifier(n_neighbors=k).fit(full.X_train, full.y_train).predict(full.X_val)
                      == full.y_val) for k in options]
    k = options[int(np.argmax(scores))]
    return KNeighborsClassifier(n_neighbors=k), f"k = {k}, re-chosen on validation for every input"


RICHER: dict[str, Factory] = {
    "logistic_regression": lambda lesson: (LogisticRegression(penalty=None, max_iter=10_000),
                                           "no penalty, solved exactly (where the lesson's gradient descent heads)"),
    "perceptron": lambda lesson: (
        BestEpoch(Perceptron(random_state=RANDOM_STATE), linear_models.PERCEPTRON_EPOCHS, lesson.data.full.X_val,
                  lesson.data.full.y_val, classes=CLASSES, seed=RANDOM_STATE),
        f"best of {linear_models.PERCEPTRON_EPOCHS} epochs on validation, {ONE_RUN}"),
    "linear_svm": lambda lesson: (
        EpochTrainer(SGDClassifier(loss="hinge", alpha=linear_models.SVM_ALPHA, random_state=RANDOM_STATE),
                     linear_models.SVM_EPOCHS, classes=CLASSES, seed=RANDOM_STATE),
        f"hinge loss, α = {linear_models.SVM_ALPHA:g}, {linear_models.SVM_EPOCHS} epochs, {ONE_RUN}"),
    "naive_bayes": lambda lesson: (GaussianNB(), "Gaussian naive Bayes"),
    "decision_tree": lambda lesson: (
        DecisionTreeClassifier(max_leaf_nodes=len(lesson.final.detail.leaves),
                               min_samples_leaf=tree.min_leaf(_rows(lesson)), random_state=RANDOM_STATE),
        f"{len(lesson.final.detail.leaves)} leaves, grown best-first"),
    "random_forest": lambda lesson: (
        RandomForestClassifier(n_estimators=forest.TREE_COUNTS[-1], min_samples_leaf=forest.min_leaf(_rows(lesson)),
                               n_jobs=1, random_state=RANDOM_STATE),
        f"{forest.TREE_COUNTS[-1]} trees"),
    "gradient_boosting": lambda lesson: (
        GradientBoostingClassifier(n_estimators=boosting.N_TREES, learning_rate=boosting.LEARNING_RATE,
                                   max_depth=boosting.TREE_DEPTH, min_samples_leaf=boosting.min_leaf(_rows(lesson)),
                                   random_state=RANDOM_STATE),
        f"{boosting.N_TREES} trees at learning rate {boosting.LEARNING_RATE}"),
    "knn": _knn,
    "mlp": lambda lesson: (
        EpochTrainer(MLPClassifier(**mlp.MLP_SETTINGS), mlp.EPOCHS, classes=CLASSES, seed=RANDOM_STATE),
        f"8 → {' → '.join(map(str, mlp.HIDDEN))} → 1, {mlp.EPOCHS} epochs of Adam, {ONE_RUN}"),
}


def richer_model(lesson: ClassLesson) -> tuple[Any, str]:
    """An unfitted model of the lesson's algorithm for every input, and how it was set up.

    Args:
        lesson: The lesson; its story's key picks the algorithm.

    Returns:
        The model, and a line describing its settings for the chapter's note.
    """
    return RICHER[lesson.story.key](lesson)


def _headline(own: float, richer: float, better: bool, worse: bool) -> str:
    """The chapter's headline; it claims a gain or a loss only when the test found one and the percentages differ."""
    shown_own, shown_richer = f"{own:.0%}", f"{richer:.0%}"
    if better and shown_own != shown_richer:
        return f"Four more inputs lift test accuracy from {shown_own} to {shown_richer}."
    if worse and shown_own != shown_richer:
        return f"Here the extra inputs make it worse: {shown_richer} against {shown_own} with age, fare and sex."
    return f"About the same: {shown_richer} with every input against {shown_own} with age, fare and sex."


def compare(lesson: ClassLesson) -> Comparison:
    """Fits the all-inputs model and compares it with the lesson's model on the same test passengers.

    Args:
        lesson: The lesson.

    Returns:
        Three rows (the baseline, the lesson's model, the all-inputs model), a headline, and a note giving
        the McNemar evidence and the settings.
    """
    data, full = lesson.data, lesson.data.full
    model, settings = richer_model(lesson)
    model.fit(full.X_train, full.y_train)
    richer_right = model.predict(full.X_test) == data.y_test
    own_right = (lesson.score(data.x_test) > threshold(lesson.kind)) == (data.y_test == 1)
    richer, own = float(np.mean(richer_right)), float(np.mean(own_right))
    baseline = float(np.mean(data.y_test == 0))
    verdict = paired_accuracy(own_right, richer_right, "passenger")
    rows = (ComparisonRow("Always guess 'did not survive'", baseline, f"{baseline:.0%}", "no inputs at all", "baseline"),
            ComparisonRow("This lesson's model", own, f"{own:.0%}", "age, fare and sex", "lesson"),
            ComparisonRow("Same algorithm, every input", richer, f"{richer:.0%}", INPUT_LIST, "richer"))
    note = (f"Both are judged on the same {len(data.y_test):,} test passengers: {verdict.evidence}. The "
            f"all-inputs model ({settings}) trains on the same {len(full.y_train):,} passengers.")
    return Comparison(metric="Test accuracy (higher is better)", higher_is_better=True, rows=rows,
                      headline=_headline(own, richer, verdict.better, verdict.worse), note=note,
                      takeaway=f"Same algorithm with every input: {richer:.0%} test accuracy "
                               f"(age, fare and sex: {own:.0%}).")
