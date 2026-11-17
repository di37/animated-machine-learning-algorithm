"""Logistic regression trained by full-batch gradient descent on the log loss, written out step by step.

Every frame is a real step: P(survive) = σ(θ₀ + θ₁·age + θ₂·fare + θ₃·woman), inputs scaled.

Gradient descent starts with every θ at 0 and repeats one move: compute the log loss's gradient over all
training passengers, Xᵀ(p − y) / n, and step against it, scaled by the learning rate. The loop is written
out here rather than left to scikit-learn so that every intermediate model can be shown. scikit-learn's
exact fit appears beside the worked prediction, as an independent check of where the descent is heading.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from animkit.data import read_only
from animkit.lessons.lesson import Story
from titanic_anim.lessons.data import PassengerData
from titanic_anim.lessons.lesson import (
    PROBABILITY,
    ClassLesson,
    ClassSnapshot,
    make_snapshot,
    percent,
    verified_score,
)
from titanic_anim.lessons.lesson_diagrams import LinearDiagram, weight_scale
from titanic_anim.lessons.stories.common import WEIGHT_NAMES, Standardiser, WeightsPanel, sigmoid, z_line

LEARNING_RATE = 1.5
STEPS = 60


@dataclass(frozen=True)
class LogisticState:
    """The model after one gradient-descent step.

    Attributes:
        step: How many steps have been taken (0 is the start).
        theta: The bias and the age, fare and woman weights, on the scaled inputs.
        loss: The training log loss at this θ.
        standardise: The input scaling the weights apply to.
    """

    step: int
    theta: np.ndarray  # bias, age, fare, woman (on scaled inputs)
    loss: float  # training log loss at this θ
    standardise: Standardiser


def run_gradient_descent(z: np.ndarray, y: np.ndarray, standardise: Standardiser,
                         learning_rate: float = LEARNING_RATE, steps: int = STEPS) -> tuple[LogisticState, ...]:
    """θ starts at 0 (P = 50% for everyone); each step moves against the mean log-loss gradient.

    Args:
        z: The training passengers' scaled inputs.
        y: Their labels, 1 = survived.
        standardise: The scaling ``z`` came from, kept with each state.
        learning_rate: How far each step moves, as a multiple of the gradient.
        steps: How many steps to take.

    Returns:
        ``steps + 1`` states: the start, then the model after each step.
    """
    design = np.column_stack([np.ones(len(z)), z])
    theta = np.zeros(design.shape[1])
    states = []
    for step in range(steps + 1):
        p = np.clip(sigmoid(design @ theta), 1e-12, 1 - 1e-12)
        loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        states.append(LogisticState(step, read_only(theta), loss, standardise))
        theta = theta - learning_rate * design.T @ (p - y) / len(y)
    return tuple(states)


def _scorer(state: LogisticState):
    """The model at one state, as a scorer: raw feature rows in, P(survive) out."""
    return lambda x: sigmoid(state.theta[0] + state.standardise(x) @ state.theta[1:])


def _note(states: tuple[LogisticState, ...], step: int) -> str:
    """The training callout for one step: the log loss before and after it."""
    if step == 0:
        return "Step 0: every weight is 0\nso everyone gets P(survive) = 50%"
    before, after = states[step - 1], states[step]
    return (f"Step {step}: one step downhill on the log loss\n"
            f"log loss {before.loss:.3f} → {after.loss:.3f}")


def _ranking(theta: np.ndarray) -> str:
    """Which scaled input the trained weights lean on most, read from the weights themselves."""
    order = np.argsort(-np.abs(theta[1:]))
    names = [WEIGHT_NAMES[1 + i] for i in order]
    return f"Largest weight: {names[0]}, then {names[1]}; smallest: {names[2]}."


def _story(data: PassengerData, last: ClassSnapshot) -> Story:
    """The lesson's narration; ``last`` is the final snapshot, for its accuracies and weights."""
    return Story(
        key="logistic_regression", title="Logistic Regression",
        method="trained by batch gradient descent on the log loss",
        objective=("Logistic regression adds up weighted inputs and squashes the total into a "
                   "probability. Watch gradient descent tune four numbers until the 50% boundary "
                   "separates survivors from the rest as well as a straight cut can."),
        initial=("First guess: all weights are 0, so every passenger gets P(survive) = 50% and the "
                 "maps are grey. Circled passengers are the ones it gets wrong."),
        training=(f"Each step computes the log loss's gradient over all {data.n_train} passengers and "
                  f"moves the weights a little downhill (learning rate {LEARNING_RATE}). The boundary "
                  "is where P(survive) = 50%."),
        final=(f"After {STEPS} steps: train accuracy {last.train_accuracy:.0%}, validation "
               f"{last.val_accuracy:.0%}. {_ranking(last.detail.theta)}"),
        inference=("An unseen passenger: scale the inputs, add up bias + weight × input, and squash "
                   "the total with the sigmoid σ to get a probability, compared with 50%."),
        takeaway=("Logistic regression turns a weighted sum into a probability; gradient descent "
                  "finds the weights that make the observed outcomes most likely."),
        limitation=("its boundary is a straight cut in the scaled inputs, so it cannot capture "
                    "patterns such as 'young boys survived' without extra features."),
        progress_label="gradient descent step")


def build(data: PassengerData) -> ClassLesson:
    """Builds the logistic regression lesson: one snapshot per gradient-descent step.

    Earlier steps get more of the training chapter's time (``frame_weights``). The side panel shows the
    weights; the diagram shows the weighted sum and σ.

    Args:
        data: The passengers.

    Returns:
        The lesson.
    """
    standardise = Standardiser.fit(data.x_train)
    states = run_gradient_descent(standardise(data.x_train), data.y_train, standardise)
    snapshots = tuple(make_snapshot(_scorer(s), data, PROBABILITY, label=f"Step {s.step} of {STEPS}",
                                    progress=s.step, note=_note(states, s.step), detail=s)
                      for s in states)
    final = states[-1]
    score = _scorer(final)

    exact = LogisticRegression(penalty=None, tol=1e-10, max_iter=10_000).fit(standardise(data.x_train),
                                                                          data.y_train)

    def explain(features: np.ndarray):
        """Works one passenger's prediction through: scale, weighted sum, σ, compare with 50%.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            The verified explanation, checked against the final step's model.
        """
        z = standardise(features)
        logit = float(final.theta[0] + z @ final.theta[1:])
        worked = float(sigmoid(logit))
        solved = float(exact.predict_proba(z[None, :])[0, 1])
        lines = [z_line(features, z),
                 f"θ₀ + Σ θ·input = {final.theta[0]:+.2f} {' '.join(f'{w * v:+.2f}' for w, v in zip(final.theta[1:], z))}"
                 f" = {logit:+.2f}",
                 f"P(survive) = σ({logit:+.2f}) = {percent(worked)}",
                 f"{percent(worked)} {'>' if worked > 0.5 else '≤'} 50% → predicted: "
                 f"{'survived' if worked > 0.5 else 'did not survive'}",
                 f"independent check: scikit-learn's exact fit gives {percent(solved)}"]
        return verified_score(lines, worked, float(score(features[None, :])[0]), PROBABILITY,
                              call=f"gradient-descent model after {STEPS} steps")

    limit = float(np.max(np.abs(final.theta))) * 1.3
    panel = WeightsPanel(limit, lambda s: s.theta, "Weights on the scaled inputs", final.theta, standardise,
                         note=lambda s: f"training log loss {s.loss:.3f}")
    return ClassLesson(
        story=_story(data, snapshots[-1]), data=data, snapshots=snapshots, final_index=len(snapshots) - 1,
        score=score, kind=PROBABILITY, explain=explain, panel=panel,
        diagram=LinearDiagram("logistic", standardise, lambda s: s.theta, final.theta,
                              weight_scale(s.theta for s in states), probability=sigmoid),
        config={"model": "σ(θ₀ + θ·z), z = standardised age, log fare, woman",
                "optimiser": "full-batch gradient descent", "learning_rate": str(LEARNING_RATE),
                "steps": str(STEPS), "init": "θ = 0"},
        frame_weights=tuple(1.0 / (1.0 + s.step / 6.0) for s in states))
