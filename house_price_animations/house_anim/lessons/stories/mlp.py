"""A small neural network (1 → 8 → 8 → 1, ReLU) trained with Adam, one real chunk at a time.

Weights are copied after every ``partial_fit`` call; each snapshot's curve is a forward pass with
those exact weights, and the tests check the final one against ``MLPRegressor.predict``.

A multi-layer perceptron (MLP) passes the standardised income through two hidden layers of 8
neurons. Each neuron takes a weighted sum of the layer before plus a bias and applies ReLU,
max(0, ·): it is silent (0) below a threshold and grows in a straight line above it. The output is a
weighted sum of the last hidden layer, so the network's curve is made of straight pieces joined
where neurons switch on. Training is mini-batch gradient descent with Adam: after every 200
districts every weight moves a little to reduce the squared error. One epoch is one pass over all
the training districts. The first epoch is shown in eighths, where the curve changes fastest, then
one snapshot per epoch. The side panel draws the network with every weight as an edge.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from sklearn.neural_network import MLPRegressor

from animkit.lessons import theme
from animkit.lessons.stage import reset_axes
from animkit.lessons.timing import phase
from house_anim.config import RANDOM_STATE
from house_anim.data import read_only
from house_anim.lessons.data import IncomeData, Query
from house_anim.lessons.lesson import (
    Explanation,
    Lesson,
    Panel,
    Snapshot,
    Story,
    make_snapshot,
    money_k,
    predict_one,
    verified,
)
from house_anim.lessons.stories.linear_gd import Standardiser

HIDDEN = (8, 8)
BATCH = 200
EPOCHS = 25
FIRST_EPOCH_CHUNKS = 8  # the first epoch is shown in eighths, where the curve changes fastest
MLP_SETTINGS = dict(hidden_layer_sizes=HIDDEN, activation="relu", solver="adam",
                    learning_rate_init=0.01, batch_size=BATCH, random_state=RANDOM_STATE)
LAYER_X = (0.1, 0.4, 0.7, 0.92)
DIAGRAM_Y = (0.1, 0.78)


@dataclass(frozen=True)
class MlpState:
    """A copy of the network's parameters at one moment of training.

    Attributes:
        epochs: Passes over the training data so far (fractional within the first epoch).
        weights: One weight matrix per layer, shape (inputs, outputs), as scikit-learn's ``coefs_``.
        biases: One bias vector per layer, as scikit-learn's ``intercepts_``.
    """

    epochs: float  # passes over the training data so far
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]


def _epoch(epochs: float) -> str:
    """The snapshot's label: "Epoch 3", or "Epoch 0.25" within the first epoch."""
    return f"Epoch {epochs:.0f}" if float(epochs).is_integer() else f"Epoch {epochs:.2f}"


def forward(state: MlpState, z: np.ndarray) -> list[np.ndarray]:
    """Activations of every layer for standardised inputs ``z``; the last entry is the output.

    Each layer is ``previous @ weights + bias``; hidden layers then apply ReLU, the output layer does not.

    Args:
        state: The network's parameters.
        z: Standardised incomes.

    Returns:
        One array per layer, shape (len(z), layer size): the input, each hidden layer, then the
        predicted prices ($100k).
    """
    layers = [np.asarray(z, dtype=float).reshape(-1, 1)]
    for index, (w, b) in enumerate(zip(state.weights, state.biases)):
        pre = layers[-1] @ w + b
        layers.append(pre if index == len(state.weights) - 1 else np.maximum(pre, 0.0))
    return layers


def _capture(model: MLPRegressor, epochs: float) -> MlpState:
    """A read-only copy of the model's current weights and biases."""
    return MlpState(epochs, tuple(read_only(w) for w in model.coefs_),
                    tuple(read_only(b) for b in model.intercepts_))


def _schedule(n_rows: int) -> list[tuple[slice, float, bool]]:
    """One entry per partial_fit call: (rows of the epoch's shuffled order, epochs done, new epoch?).

    Epoch 1 is cut at the first mini-batch and then at every eighth; later epochs are whole.
    """
    first_batch = min(BATCH, n_rows)
    eighths = {round(n_rows * i / FIRST_EPOCH_CHUNKS) for i in range(1, FIRST_EPOCH_CHUNKS + 1)}
    bounds = [0, first_batch, *sorted(b for b in eighths if b > first_batch)]
    epoch_one = [(slice(a, b), b / n_rows, a == 0) for a, b in zip(bounds, bounds[1:])]
    return epoch_one + [(slice(0, n_rows), float(epoch), True) for epoch in range(2, EPOCHS + 1)]


class NetworkPanel(Panel):
    """The network drawn as neurons and weighted edges; at inference, which neurons fire for the new house.

    Edge colour is the weight's sign (blue +, red −) and its width the weight's size, on the final
    network's largest weight per layer.
    """

    def __init__(self, final: MlpState, standardise: Standardiser):
        """Places the neurons and fixes each layer's edge scale to the final network.

        Args:
            final: The trained network's parameters.
            standardise: Turns raw incomes into the network's input.
        """
        self._final = final
        self._standardise = standardise
        sizes = [1, *HIDDEN, 1]
        self._nodes = [[(x, y) for y in np.linspace(*DIAGRAM_Y, size + 2)[1:-1]]
                       for x, size in zip(LAYER_X, sizes)]
        self._scale = [float(np.abs(w).max()) for w in final.weights]

    def _edges(self, ax: Axes, state: MlpState) -> None:
        """Draws one edge per weight, coloured by sign and as wide as its size."""
        for layer, w in enumerate(state.weights):
            segments, colours, widths = [], [], []
            for i, start in enumerate(self._nodes[layer]):
                for j, end in enumerate(self._nodes[layer + 1]):
                    strength = min(abs(w[i, j]) / self._scale[layer], 1.0)
                    segments.append((start, end))
                    colours.append(theme.MODEL if w[i, j] >= 0 else theme.ERROR)
                    widths.append(0.4 + 3.2 * strength)
            ax.add_collection(LineCollection(segments, colors=colours, linewidths=widths, alpha=0.75,
                                             zorder=1))

    def _neurons(self, ax: Axes, fills: list[list]) -> None:
        """Draws every neuron, filled with its colour from ``fills`` (one list per layer)."""
        for layer, nodes in enumerate(self._nodes):
            xs, ys = zip(*nodes)
            ax.scatter(xs, ys, s=200, c=fills[layer], zorder=3, edgecolors=theme.NAVY, linewidths=1.2)

    def _frame(self, ax: Axes, title: str) -> None:
        """Clears the panel and writes its title and the layer names."""
        reset_axes(ax)
        ax.text(0.04, 0.9, title, fontsize=theme.PANEL_TITLE_SIZE, weight="bold", color=theme.NAVY,
                va="center")
        for x, name in zip(LAYER_X, ("income", "hidden 1", "hidden 2", "price")):
            ax.text(x, 0.03, name, ha="center", fontsize=11, color=theme.INK_SECONDARY)

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws the network with this snapshot's weights, and how many parameters it has.

        Args:
            ax: The side panel.
            snapshot: The moment of training being shown; its ``detail`` is an ``MlpState``.
        """
        state: MlpState = snapshot.detail
        self._frame(ax, f"All {sum(w.size + b.size for w, b in zip(state.weights, state.biases))} "
                        f"weights · {_epoch(state.epochs).lower()}")
        self._edges(ax, state)
        self._neurons(ax, [[theme.PANEL] * len(nodes) for nodes in self._nodes])
        ax.text(0.96, 0.84, "blue +, red −, thicker = larger", ha="right", fontsize=11,
                color=theme.INK_SECONDARY)

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Lights the forward pass layer by layer: firing neurons coloured by strength, silent ones grey.

        Args:
            ax: The side panel.
            query: The unseen district.
            t: How far through the pass, from 0 to 1; the price is written once the output layer lights.
        """
        layers = forward(self._final, self._standardise(np.array([query.income])))
        self._frame(ax, "Forward pass: which neurons fire")
        self._edges(ax, self._final)
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
            ax.text(x, y + 0.08, money_k(float(layers[-1][0, 0])), ha="center", fontsize=13,
                    weight="bold", color=theme.NAVY)
        ax.text(0.96, 0.84, "grey = ReLU output 0 (silent)", ha="right", fontsize=11,
                color=theme.INK_SECONDARY)


def _story(data: IncomeData, snapshots: tuple[Snapshot, ...]) -> Story:
    """Every word of the neural-network video, with the first and last snapshots' errors."""
    last = snapshots[-1]
    return Story(
        key="mlp", title="Neural Network (MLP)",
        method="1 → 8 → 8 → 1 with ReLU, trained by Adam",
        objective=("A small neural network learns a curve by adjusting 97 weights. Watch real "
                   "training: after every mini-batch of 200 districts, Adam nudges every weight "
                   "to reduce the squared error, and the curve bends into shape."),
        initial=(f"Random starting weights after one mini-batch: the curve sits far below the "
                 f"districts. Validation RMSE ${snapshots[0].val_rmse_k:,.0f}k."),
        training=("Each frame is real training: the first epoch is shown in eighths, then one frame "
                  "per epoch. Edge colour is the weight's sign; thickness is its size."),
        final=(f"{EPOCHS} epochs: train RMSE ${last.train_rmse_k:,.0f}k, validation "
               f"${last.val_rmse_k:,.0f}k. ReLU neurons switch on at different incomes, so the curve "
               "is built from straight pieces."),
        inference=("A forward pass for the unseen district: its income flows through both hidden "
                   "layers; only neurons with positive input fire, and the output is the price."),
        takeaway=("A neural network is many simple units whose weights are tuned together by "
                  "gradient descent; together they can bend into almost any shape."),
        limitation=("it needs scaled inputs, careful tuning and more data than the other models, "
                    "and its 97 weights are hard to interpret."),
        progress_label="epochs (passes over the training data)")


def build(data: IncomeData) -> Lesson:
    """Builds the neural-network lesson: trains with ``partial_fit`` on ``_schedule``'s chunks, copying weights.

    Each epoch takes the training districts in a new seeded shuffle. The final model is the last
    snapshot. The lesson has no separate diagram: its panel already is the network.

    Args:
        data: The income view to learn from.

    Returns:
        The lesson.
    """
    standardise = Standardiser(float(data.x_train.mean()), float(data.x_train.std()))
    z = standardise(data.x_train).reshape(-1, 1)
    model = MLPRegressor(**MLP_SETTINGS)
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(data.n_train)
    states: list[MlpState] = []
    for rows, epochs, new_epoch in _schedule(data.n_train):
        if new_epoch and states:
            order = rng.permutation(data.n_train)
        model.set_params(batch_size=min(BATCH, rows.stop - rows.start))  # tiny datasets only
        model.partial_fit(z[order[rows]], data.y_train[order[rows]])
        states.append(_capture(model, epochs))

    def predictor(state: MlpState):
        """The network with these parameters as a predictor, by an explicit forward pass.

        Args:
            state: The copied parameters.

        Returns:
            The predictor: raw incomes ($10k) in, prices ($100k) out.
        """
        return lambda x: forward(state, standardise(np.asarray(x, dtype=float)))[-1].ravel()

    def note(index: int, state: MlpState) -> str:
        """The callout for one snapshot: how much data the network has seen.

        Args:
            index: The snapshot's position; 0 is after the first mini-batch.
            state: The copied parameters.

        Returns:
            Two lines of callout text.
        """
        if index == 0:
            return (f"After the first mini-batch ({BATCH} districts)\n"
                    "random starting weights, one Adam step")
        return (f"{_epoch(state.epochs)}: {round(state.epochs * data.n_train):,} districts seen\n"
                f"Adam updates every weight after each {BATCH} districts")

    snapshots = tuple(make_snapshot(predictor(s), data, label=_epoch(s.epochs),
                                    progress=s.epochs, note=note(i, s), detail=s)
                      for i, s in enumerate(states))
    final = states[-1]

    def predict(x: np.ndarray) -> np.ndarray:
        """The trained network's prices, by scikit-learn's own ``predict``.

        Args:
            x: Raw incomes ($10k).

        Returns:
            The predicted prices ($100k).
        """
        return model.predict(standardise(np.asarray(x, dtype=float)).reshape(-1, 1))

    def explain(income: float) -> Explanation:
        """Works the network's price for one income: its input, how many neurons fire in each layer, the output.

        The worked value comes from the copied weights, so the check also confirms the copy matches the model.

        Args:
            income: The income ($10k).

        Returns:
            The worked lines, checked against the model's own prediction.
        """
        layers = forward(final, standardise(np.array([income])))
        worked = float(layers[-1][0, 0])
        lines = [f"input z = standardised income = {layers[0][0, 0]:.2f}",
                 f"hidden layer 1: {int(np.sum(layers[1] > 0))} of {HIDDEN[0]} neurons fire (ReLU > 0)",
                 f"hidden layer 2: {int(np.sum(layers[2] > 0))} of {HIDDEN[1]} neurons fire",
                 f"output = Σ weight × activation + bias = {money_k(worked)}"]
        return verified(lines, worked, predict_one(predict, income))

    story = _story(data, snapshots)
    return Lesson(
        story=story, data=data, snapshots=snapshots, final_index=len(snapshots) - 1, predict=predict,
        explain=explain, panel=NetworkPanel(final, standardise),
        config={"model": "MLPRegressor", "layers": "1-8-8-1, ReLU", "optimiser": "Adam, lr 0.01",
                "batch_size": str(BATCH), "epochs": str(EPOCHS), "input": "standardised MedInc",
                "random_state": str(RANDOM_STATE)})
