"""Linear regression trained by batch gradient descent, with its real path across the MSE bowl.

The model is a straight line, ŷ = θ₀ + θ₁·z, where z is the income standardised on the training rows
(z = (x − mean) / SD). Standardising makes θ₀ the price at average income and keeps the two numbers on
similar scales, so one learning rate suits both. The loss is the mean squared error (MSE) over every
training district; plotted over (θ₀, θ₁) it is a bowl. Gradient descent starts at θ = (0, 0) and, at
each step, computes the MSE's gradient over all the districts (full batch) and moves θ a little the
other way: θ ← θ − learning rate × gradient. Every snapshot is one of those real steps, and the side
panel draws the bowl with the path the steps took. At the end the line is checked against scikit-learn's
exact least-squares fit.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression

from animkit.lessons import theme
from house_anim.lessons.config import PRICE_TO_K
from house_anim.lessons.data import IncomeData, Query
from house_anim.lessons.lesson import (
    Explanation,
    Lesson,
    Panel,
    Snapshot,
    Story,
    income_k,
    make_snapshot,
    money_k,
    predict_one,
    verified,
)
from house_anim.lessons.lesson_diagrams import LinearDiagram
from house_anim.lessons.plots import K_FORMAT

LEARNING_RATE = 0.1
STEPS = 40
BOWL_RESOLUTION = 120
ARROW_LENGTH = 0.35  # the first step's arrow, as a fraction of the panel's θ₀ span
MIN_ARROW = 0.03  # below 3% of the starting gradient the arrow is hidden (it has converged)


@dataclass(frozen=True)
class GdState:
    """The model before one gradient-descent step.

    Attributes:
        step: The step's number; 0 is the starting guess.
        theta: (θ₀, θ₁) for ŷ = θ₀ + θ₁·z, z = standardised income, in $100k.
        gradient: ∂MSE/∂θ at this θ; the next step moves against it.
        mse: The training MSE at this θ, in ($100k)².
    """

    step: int
    theta: tuple[float, float]  # (θ₀, θ₁) for ŷ = θ₀ + θ₁·z, z = standardised income
    gradient: tuple[float, float]  # ∂MSE/∂θ at this θ (drives the next step)
    mse: float


def run_gradient_descent(z: np.ndarray, y: np.ndarray, learning_rate: float = LEARNING_RATE,
                         steps: int = STEPS) -> tuple[GdState, ...]:
    """Full-batch gradient descent on MSE from θ = (0, 0); returns the state before every step.

    With residuals r = y − (θ₀ + θ₁·z), the gradient is ∂MSE/∂θ₀ = −2·mean(r) and
    ∂MSE/∂θ₁ = −2·mean(r·z).

    Args:
        z: The training incomes, standardised.
        y: The training prices ($100k).
        learning_rate: How far each step moves, as a multiple of the gradient.
        steps: How many steps to take.

    Returns:
        ``steps + 1`` states: the start, then the state after each step.
    """
    theta = np.zeros(2)
    states = []
    for step in range(steps + 1):
        residual = y - (theta[0] + theta[1] * z)
        gradient = np.array([-2.0 * residual.mean(), -2.0 * (residual * z).mean()])
        states.append(GdState(step, (float(theta[0]), float(theta[1])),
                              (float(gradient[0]), float(gradient[1])), float(np.mean(residual ** 2))))
        theta = theta - learning_rate * gradient
    return tuple(states)


@dataclass(frozen=True)
class Standardiser:
    """Turns raw incomes into z-scores, with the training rows' mean and standard deviation.

    Attributes:
        mean: The training incomes' mean ($10k).
        scale: Their standard deviation ($10k).

    Example:
        >>> Standardiser(mean=4.0, scale=2.0)(np.array([6.0]))
        array([1.])
    """

    mean: float
    scale: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Standardises some incomes.

        Args:
            x: Raw incomes ($10k).

        Returns:
            Their z-scores: how many standard deviations each is from the mean.
        """
        return (np.asarray(x, dtype=float) - self.mean) / self.scale


def _line(standardise: Standardiser, theta: tuple[float, float]):
    """The straight line for θ as a predictor: raw income in, price out."""
    return lambda x: theta[0] + theta[1] * standardise(x)


class BowlPanel(Panel):
    """The true MSE surface over (θ₀, θ₁), with the path gradient descent actually took.

    The surface is computed once, over a window around the path, from the training data itself.
    """

    def __init__(self, z: np.ndarray, y: np.ndarray, states: tuple[GdState, ...]):
        """Computes the bowl over a window that frames the whole path.

        Args:
            z: The training incomes, standardised.
            y: The training prices ($100k).
            states: Every gradient-descent state, in order.
        """
        self._states = states
        path = np.array([s.theta for s in states])
        lo, hi = path.min(axis=0), path.max(axis=0)
        span = np.maximum(hi - lo, 0.5)
        self._b = np.linspace(lo[0] - 0.25 * span[0], hi[0] + 0.25 * span[0], BOWL_RESOLUTION)
        self._w = np.linspace(lo[1] - 0.35 * span[1], hi[1] + 0.35 * span[1], BOWL_RESOLUTION)
        b, w = np.meshgrid(self._b, self._w)
        # MSE(b, w) = mean((y - b - w z)^2), expanded so the whole surface costs one pass over the data.
        self._mse = (np.mean(y ** 2) - 2 * b * y.mean() - 2 * w * np.mean(y * z) + b ** 2
                     + 2 * b * w * z.mean() + w ** 2 * np.mean(z ** 2))
        self._levels = np.geomspace(self._mse.min() * 1.02, self._mse.max(), 11)

    def _bowl(self, ax: Axes, title: str) -> None:
        """Draws the MSE bowl as filled contours, with θ₀ and θ₁ in $k on the axes."""
        theme.style_axes(ax, grid_axis=None)
        ax.contourf(self._b * PRICE_TO_K, self._w * PRICE_TO_K, self._mse, levels=self._levels,
                    cmap=theme.value_colormap().reversed(), alpha=0.35)
        ax.contour(self._b * PRICE_TO_K, self._w * PRICE_TO_K, self._mse, levels=self._levels,
                   colors=theme.PANEL, linewidths=1.0)
        ax.set_xlabel("θ₀: price at average income", fontsize=theme.NOTE_SIZE)
        ax.set_ylabel("θ₁: $ per SD of income", fontsize=theme.NOTE_SIZE)
        ax.xaxis.set_major_formatter(K_FORMAT)
        ax.yaxis.set_major_formatter(K_FORMAT)
        ax.set_title(title)

    def _path(self, ax: Axes, upto: int) -> None:
        """Draws the path of θ up to step ``upto``, its latest point highlighted."""
        path = np.array([s.theta for s in self._states[:upto + 1]]) * PRICE_TO_K
        ax.plot(path[:, 0], path[:, 1], color=theme.NAVY, linewidth=1.6, zorder=3)
        ax.scatter(path[:, 0], path[:, 1], s=14, color=theme.NAVY, zorder=4)
        ax.scatter(path[-1:, 0], path[-1:, 1], s=130, color=theme.MODEL, zorder=5,
                   edgecolors=theme.PANEL, linewidths=2.0)

    def draw_training(self, ax: Axes, snapshot: Snapshot) -> None:
        """Draws the bowl, the path so far and an arrow pointing downhill from the current θ.

        The arrow's length follows the gradient's size, relative to the first step's; once it falls
        below ``MIN_ARROW`` of that, descent has converged and the arrow is hidden.

        Args:
            ax: The side panel.
            snapshot: The step being shown; its ``detail`` is the ``GdState``.
        """
        state: GdState = snapshot.detail
        self._bowl(ax, "The loss bowl: MSE for every (θ₀, θ₁)")
        self._path(ax, state.step)
        downhill = -np.array(state.gradient)
        norm = np.linalg.norm(downhill)
        relative = norm / np.linalg.norm(self._states[0].gradient)  # arrow length follows |gradient|
        if relative > MIN_ARROW:
            length = ARROW_LENGTH * relative * (self._b[-1] - self._b[0]) * PRICE_TO_K
            start = np.array(state.theta) * PRICE_TO_K
            ax.annotate("", xy=start + downhill / norm * length, xytext=start, zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=theme.HIGHLIGHT, linewidth=2.6,
                                        mutation_scale=22))
            ax.text(0.98, 0.05, "teal arrow: downhill (−gradient)", transform=ax.transAxes,
                    ha="right", fontsize=12, color=theme.INK_SECONDARY)

    def draw_inference(self, ax: Axes, query: Query, t: float) -> None:
        """Draws the whole path and the two final numbers: the model is just θ₀ and θ₁.

        Args:
            ax: The side panel.
            query: The unseen district (unused: the panel is the same for any district).
            t: How far through the animation (unused: the panel does not move).
        """
        self._bowl(ax, "Training is over: two numbers are the model")
        self._path(ax, len(self._states) - 1)
        theta = np.array(self._states[-1].theta) * PRICE_TO_K
        ax.text(0.98, 0.05, f"θ₀ = ${theta[0]:,.1f}k · θ₁ = ${theta[1]:,.1f}k",
                transform=ax.transAxes, ha="right", fontsize=13, color=theme.NAVY, weight="bold")


def _note(states: tuple[GdState, ...], step: int) -> str:
    """The callout for one step: how far θ₀ and θ₁ moved, and the MSE before and after."""
    if step == 0:
        return "Step 0: θ₀ = θ₁ = 0, a flat line at $0\nevery district is badly under-priced"
    before, after = states[step - 1], states[step]
    moves = [after.theta[i] - before.theta[i] for i in (0, 1)]
    return (f"Step {step}: one step downhill on the error\n"
            f"θ₀ {money_k(moves[0], signed=True)} · θ₁ {money_k(moves[1], signed=True)} · "
            f"MSE {before.mse:.3f} → {after.mse:.3f}")


def _story(data: IncomeData, last: Snapshot, agreement: float) -> Story:
    """Every word of the linear-regression video, with the final step's errors and the gap to least squares."""
    return Story(
        key="linear_regression",
        title="Linear Regression",
        method="trained by batch gradient descent",
        objective=("Watch a straight line learn to price homes. Gradient descent nudges its two "
                   "numbers, the intercept θ₀ and the slope θ₁, downhill on the squared error "
                   "until the line fits the data."),
        initial=("First guess: θ₀ = θ₁ = 0, a flat line at $0. The red lines are its errors "
                 "(residuals). Training shrinks their average square, the MSE."),
        training=(f"Each step measures the slope of the MSE over all {data.n_train:,} districts and "
                  f"moves both numbers a little downhill (learning rate {LEARNING_RATE}). "
                  "The bowl on the right is the real loss surface."),
        final=(f"After {STEPS} steps the line has settled: train RMSE ${last.train_rmse_k:,.0f}k, "
               f"validation ${last.val_rmse_k:,.0f}k. The exact least-squares formula gives the "
               f"same line to within ${agreement * PRICE_TO_K:,.2f}k."),
        inference=("A district the model has never seen: standardise its income, then apply the "
                   "line. Two learned numbers are all it takes to produce a price."),
        takeaway=("Gradient descent finds the best straight line by repeatedly stepping downhill "
                  "on the average squared error."),
        limitation=("a straight line cannot bend to follow curved patterns or the $500k cap, and "
                    "income alone ignores location, house age and size."),
        progress_label="gradient descent step",
    )


def build(data: IncomeData) -> Lesson:
    """Builds the linear-regression lesson: every gradient-descent step is a snapshot.

    The final model is the last step. Early steps get more screen time than late ones, where little
    changes.

    Args:
        data: The income view to learn from.

    Returns:
        The lesson, with the loss-bowl panel and the weighted-sum diagram.
    """
    standardise = Standardiser(float(data.x_train.mean()), float(data.x_train.std()))
    z = standardise(data.x_train)
    states = run_gradient_descent(z, data.y_train)
    snapshots = tuple(
        make_snapshot(_line(standardise, s.theta), data, label=f"Step {s.step} of {STEPS}",
                      progress=s.step, note=_note(states, s.step), detail=s)
        for s in states)
    final = states[-1].theta
    predict = _line(standardise, final)
    ols = LinearRegression().fit(data.x_train.reshape(-1, 1), data.y_train)
    agreement = float(np.max(np.abs(predict(data.grid) - ols.predict(data.grid.reshape(-1, 1)))))

    def explain(income: float) -> Explanation:
        """Works the line's price for one income by hand: standardise it, then apply θ₀ + θ₁·z.

        Args:
            income: The income ($10k).

        Returns:
            The worked lines, checked against the model's own prediction.
        """
        z_query = float(standardise(np.array([income]))[0])
        worked = final[0] + final[1] * z_query
        lines = [f"standardise: z = ({income_k(income)} − {income_k(standardise.mean)}) / "
                 f"{income_k(standardise.scale)} = {z_query:.2f}",
                 f"ŷ = θ₀ + θ₁·z = {money_k(final[0])} + {money_k(final[1])} × {z_query:.2f}",
                 f"ŷ = {money_k(worked)}"]
        return verified(lines, worked, predict_one(predict, income))

    story = _story(data, snapshots[-1], agreement)
    return Lesson(
        story=story, data=data, snapshots=snapshots, final_index=len(snapshots) - 1,
        predict=predict, explain=explain, panel=BowlPanel(z, data.y_train, states),
        diagram=LinearDiagram(standardise, final, scale=max(abs(s.theta[1]) for s in states)),
        config={"model": "ŷ = θ₀ + θ₁·z, z = standardised MedInc", "optimiser": "full-batch gradient descent",
                "learning_rate": str(LEARNING_RATE), "steps": str(STEPS), "init": "θ = (0, 0)"},
        frame_weights=tuple(1.0 / (1.0 + s.step / 6.0) for s in states),
    )
