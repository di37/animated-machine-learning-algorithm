"""A small neural network (3 → 8 → 8 → 1: ReLU, then a sigmoid output) trained by Adam.

Training runs one real ``partial_fit`` chunk at a time. Each snapshot's map is a forward pass with that
moment's weights; the tests check the final one against ``MLPClassifier.predict_proba``.

A forward pass multiplies a layer's activations by its weights and adds its biases, then applies ReLU,
max(0, ·), in the hidden layers and the sigmoid at the output. A unit whose ReLU outputs 0 is silent for
that passenger, which is how the network bends its boundary. Adam is gradient descent with a step size
adapted for each weight; scikit-learn takes one Adam step per mini-batch of ``BATCH`` passengers.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from sklearn.neural_network import MLPClassifier

from animkit.data import read_only
from animkit.lessons import theme
from animkit.lessons.lesson import Story
from animkit.lessons.stage import reset_axes
from animkit.lessons.timing import phase
from titanic_anim.config import RANDOM_STATE
from titanic_anim.lessons.data import Passenger, PassengerData
from titanic_anim.lessons.lesson import (
    PROBABILITY,
    ClassLesson,
    ClassSnapshot,
    make_snapshot,
    percent,
    verified_score,
)
from titanic_anim.lessons.stories.common import Standardiser, sigmoid, z_line

HIDDEN = (8, 8)
BATCH = 32
EPOCHS = 40
FIRST_EPOCH_PARTS = 4
MLP_SETTINGS = dict(hidden_layer_sizes=HIDDEN, activation="relu", solver="adam", learning_rate_init=0.01,
                    batch_size=BATCH, random_state=RANDOM_STATE)
LAYER_X = (0.1, 0.4, 0.7, 0.92)
DIAGRAM_Y = (0.1, 0.78)
LAYER_NAMES = ("inputs", "hidden 1", "hidden 2", "P(survive)")


@dataclass(frozen=True)
class MlpState:
    """The network's parameters after one ``partial_fit`` call.

    Attributes:
        epochs: Passes over the training data so far (a fraction during the first epoch).
        weights: One weight matrix per layer, shape (inputs, outputs), as scikit-learn's ``coefs_``.
        biases: One bias vector per layer, as scikit-learn's ``intercepts_``.
    """

    epochs: float
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]


def forward(state: MlpState, z: np.ndarray) -> list[np.ndarray]:
    """Activations of every layer for scaled inputs ``z``; the last is P(survive).

    Args:
        state: The network's parameters.
        z: Scaled (age, log fare, female) rows, or a single row.

    Returns:
        One array per layer, shape (rows, units): the inputs, both hidden layers, then P(survive).
    """
    layers = [np.atleast_2d(np.asarray(z, dtype=float))]
    for index, (w, b) in enumerate(zip(state.weights, state.biases)):
        pre = layers[-1] @ w + b
        layers.append(sigmoid(pre) if index == len(state.weights) - 1 else np.maximum(pre, 0.0))
    return layers


def _capture(model: MLPClassifier, epochs: float) -> MlpState:
    """Copies the network's weights and biases after ``epochs`` of training."""
    return MlpState(epochs, tuple(read_only(w) for w in model.coefs_), tuple(read_only(b) for b in model.intercepts_))


def _schedule(n_rows: int) -> list[tuple[slice, float, bool]]:
    """(rows of the epoch's shuffled order, epochs done, new epoch?) per partial_fit call."""
    first = min(BATCH, n_rows)
    parts = {round(n_rows * i / FIRST_EPOCH_PARTS) for i in range(1, FIRST_EPOCH_PARTS + 1)}
    bounds = [0, first, *sorted(b for b in parts if b > first)]
    epoch_one = [(slice(a, b), b / n_rows, a == 0) for a, b in zip(bounds, bounds[1:])]
    return epoch_one + [(slice(0, n_rows), float(e), True) for e in range(2, EPOCHS + 1)]


def _epoch(epochs: float) -> str:
    """'Epoch 3', or 'Epoch 0.25' part-way through the first."""
    return f"Epoch {epochs:.0f}" if float(epochs).is_integer() else f"Epoch {epochs:.2f}"


class NetworkPanel:
    """The network's side panel (a ``ClassPanel``): every neuron, joined by every weight.

    An edge is blue for a positive weight and red for a negative one, thicker when larger (each layer
    scaled by its largest final weight). At inference the layers light up in turn, showing which neurons
    fire for the unseen passenger.
    """

    def __init__(self, final: MlpState, standardise: Standardiser):
        """Lays out the neurons.

        Args:
            final: The final network, used at inference and for the edge widths' scale.
            standardise: The input scaling the network was trained on.
        """
        self._final = final
        self._standardise = standardise
        sizes = [3, *HIDDEN, 1]
        self._nodes = [[(x, y) for y in np.linspace(*DIAGRAM_Y, size + 2)[1:-1]] for x, size in zip(LAYER_X, sizes)]
        self._scale = [float(np.abs(w).max()) for w in final.weights]

    def _frame(self, ax: Axes, title: str, state: MlpState) -> None:
        """Draws the title, the layer names and every weight as an edge."""
        reset_axes(ax)
        ax.text(0.04, 0.9, title, fontsize=theme.PANEL_TITLE_SIZE, weight="bold", color=theme.NAVY, va="center")
        for x, name in zip(LAYER_X, LAYER_NAMES):
            ax.text(x, 0.03, name, ha="center", fontsize=11, color=theme.INK_SECONDARY)
        for layer, w in enumerate(state.weights):
            segments, colours, widths = [], [], []
            for i, start in enumerate(self._nodes[layer]):
                for j, end in enumerate(self._nodes[layer + 1]):
                    segments.append((start, end))
                    colours.append(theme.MODEL if w[i, j] >= 0 else theme.ERROR)
                    widths.append(0.4 + 3.2 * min(abs(w[i, j]) / self._scale[layer], 1.0))
            ax.add_collection(LineCollection(segments, colors=colours, linewidths=widths, alpha=0.75, zorder=1))

    def _neurons(self, ax: Axes, fills: list[list]) -> None:
        """Draws every neuron, filled with its colour from ``fills`` (one list per layer)."""
        for layer, nodes in enumerate(self._nodes):
            xs, ys = zip(*nodes)
            ax.scatter(xs, ys, s=200, c=fills[layer], zorder=3, edgecolors=theme.NAVY, linewidths=1.2)

    def draw_training(self, ax: Axes, snapshot: ClassSnapshot) -> None:
        """Draws every weight at one moment of training.

        Args:
            ax: The side panel's axes.
            snapshot: The moment to draw.
        """
        state: MlpState = snapshot.detail
        count = sum(w.size + b.size for w, b in zip(state.weights, state.biases))
        self._frame(ax, f"All {count} weights · {_epoch(state.epochs).lower()}", state)
        self._neurons(ax, [[theme.PANEL] * len(nodes) for nodes in self._nodes])
        ax.text(0.96, 0.84, "blue +, red −, thicker = larger", ha="right", fontsize=11, color=theme.INK_SECONDARY)

    def draw_inference(self, ax: Axes, query: Passenger, t: float) -> None:
        """Draws the forward pass for the unseen passenger, one layer lit at a time, then P(survive).

        Args:
            ax: The side panel's axes.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """
        layers = forward(self._final, self._standardise(query.features))
        self._frame(ax, "Forward pass: which neurons fire", self._final)
        cmap = theme.value_colormap()
        fills = []
        for index, values in enumerate(layers):
            lit = phase(t, index * 0.22, index * 0.22 + 0.2) >= 1.0
            top = max(float(np.abs(values).max()), 1e-9)
            fills.append([cmap(0.25 + 0.75 * abs(v) / top) if lit and v != 0 else
                          (theme.INACTIVE if lit else theme.PANEL) for v in values.ravel()])
        self._neurons(ax, fills)
        if phase(t, 0.66, 0.86) >= 1.0:
            x, y = self._nodes[-1][0]
            ax.text(x, y + 0.08, f"{layers[-1][0, 0]:.0%}", ha="center", fontsize=13, weight="bold", color=theme.NAVY)
        ax.text(0.96, 0.84, "grey = ReLU output 0 (silent)", ha="right", fontsize=11, color=theme.INK_SECONDARY)

    def overlay_training(self, axes, snapshot) -> None:
        """No extras on the maps.

        Args:
            axes: The two maps, women's first.
            snapshot: The moment being drawn.
        """

    def overlay_inference(self, axes, query, t: float) -> None:
        """The panel shows the forward pass.

        Args:
            axes: The two maps, women's first.
            query: The unseen passenger.
            t: Progress through the prediction, from 0 to 1.
        """


def _inference_text(final: MlpState, standardise: Standardiser, query: Passenger) -> str:
    """Narrate the forward pass this passenger really gets."""
    first_layer = forward(final, standardise(query.features))[1]
    if not np.any(first_layer > 0):
        return ("A forward pass for the unseen passenger: for these inputs no first-layer unit fires "
                "(every ReLU outputs 0), so only the later biases reach the sigmoid output.")
    return ("A forward pass for the unseen passenger: scaled inputs flow through both hidden layers; "
            "only units with positive input fire, and the sigmoid gives P(survive).")


def _story(data: PassengerData, snapshots: tuple[ClassSnapshot, ...], inference: str) -> Story:
    """The lesson's narration; ``inference`` is the prediction chapter's text, written for this passenger."""
    first, last = snapshots[0], snapshots[-1]
    return Story(
        key="mlp", title="Neural Network (MLP)",
        method="3 → 8 → 8 → 1 with ReLU and a sigmoid output, trained by Adam",
        objective=("A small neural network bends its boundary by tuning 113 weights together. Watch real "
                   "training: after every mini-batch of 32 passengers, Adam nudges every weight to lower "
                   "the log loss, and the maps take shape."),
        initial=(f"Random starting weights after one mini-batch: the maps barely separate anyone. "
                 f"Validation accuracy {first.val_accuracy:.0%}."),
        training=("Each frame is real training: the first epoch in quarters, then one frame per epoch. "
                  "Edge colour is the weight's sign; thickness is its size."),
        final=(f"{EPOCHS} epochs: train accuracy {last.train_accuracy:.0%}, validation "
               f"{last.val_accuracy:.0%}. ReLU units switch on at different ages and fares, so the boundary bends."),
        inference=inference,
        takeaway="A neural network is many simple units tuned together by gradient descent; together they can bend into almost any boundary.",
        limitation=(f"with {data.n_train} training passengers it can easily over-fit, it needs scaled inputs "
                    "and careful tuning, and its weights are hard to interpret."),
        progress_label="epochs (passes over the training data)")


def build(data: PassengerData) -> ClassLesson:
    """Builds the MLP lesson: snapshots after the first mini-batch, each quarter of epoch 1, then each epoch.

    Each epoch reshuffles the passengers. The snapshots score with ``forward`` on the captured weights;
    the final model and the explanation's check use scikit-learn's own ``predict_proba``.

    Args:
        data: The passengers.

    Returns:
        The lesson, with the last epoch as the final model.
    """
    standardise = Standardiser.fit(data.x_train)
    z = standardise(data.x_train)
    model = MLPClassifier(**MLP_SETTINGS)
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(data.n_train)
    states: list[MlpState] = []
    for rows, epochs, new_epoch in _schedule(data.n_train):
        if new_epoch and states:
            order = rng.permutation(data.n_train)
        model.set_params(batch_size=min(BATCH, rows.stop - rows.start))  # tiny datasets only
        model.partial_fit(z[order[rows]], data.y_train[order[rows]], classes=np.array([0, 1]))
        states.append(_capture(model, epochs))

    def scorer(state: MlpState):
        """One state's network as a scorer.

        Args:
            state: The network's parameters at that moment.

        Returns:
            A function from unscaled feature rows to P(survive), by a forward pass.
        """
        return lambda x: forward(state, standardise(x))[-1].ravel()

    def note(index: int, state: MlpState) -> str:
        """The training callout for one snapshot.

        Args:
            index: The snapshot's index; 0 is after the first mini-batch.
            state: The network at that moment.

        Returns:
            How far training has got, and how many passengers it has seen.
        """
        if index == 0:
            return f"After the first mini-batch ({BATCH} passengers)\nrandom starting weights, one Adam step"
        return (f"{_epoch(state.epochs)}: {round(state.epochs * data.n_train):,} passengers seen\n"
                f"Adam updates every weight after each {BATCH}")

    snapshots = tuple(make_snapshot(scorer(s), data, PROBABILITY, label=_epoch(s.epochs), progress=s.epochs,
                                    note=note(i, s), detail=s) for i, s in enumerate(states))
    final = states[-1]

    def score(x: np.ndarray) -> np.ndarray:
        """The final network's P(survive), from scikit-learn.

        Args:
            x: Unscaled (age, log fare, female) rows.

        Returns:
            One P(survive) per row.
        """
        return model.predict_proba(standardise(x))[:, 1]

    def explain(features: np.ndarray):
        """Works one passenger's prediction through: scaled inputs, which units fire, the output.

        Args:
            features: One passenger's (age, log fare, female) row.

        Returns:
            The verified explanation, checked against scikit-learn's ``predict_proba``.
        """
        zq = standardise(features)
        layers = forward(final, zq)
        worked = float(layers[-1][0, 0])
        lines = [z_line(features, zq),
                 f"hidden layer 1: {int(np.sum(layers[1] > 0))} of {HIDDEN[0]} units fire (ReLU > 0)",
                 f"hidden layer 2: {int(np.sum(layers[2] > 0))} of {HIDDEN[1]} units fire",
                 f"output: σ(Σ weight × activation + bias) = {percent(worked)}"]
        return verified_score(lines, worked, float(score(features[None, :])[0]), PROBABILITY)

    return ClassLesson(
        story=_story(data, snapshots, _inference_text(final, standardise, data.query)), data=data, snapshots=snapshots, final_index=len(snapshots) - 1,
        score=score, kind=PROBABILITY, explain=explain, panel=NetworkPanel(final, standardise),
        config={"model": "MLPClassifier", "layers": "3-8-8-1, ReLU, sigmoid output", "optimiser": "Adam, lr 0.01",
                "batch_size": str(BATCH), "epochs": str(EPOCHS), "inputs": "standardised age, log fare, woman",
                "random_state": str(RANDOM_STATE)})
