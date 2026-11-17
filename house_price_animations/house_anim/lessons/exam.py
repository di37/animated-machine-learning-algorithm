"""The regression final exam: the trained model prices every test district, a batch at a time.

Scores on screen are computed from the districts scored so far, using the model's real predictions;
the map shows where it errs, district by district.

The test split is never used to train or to choose a setting, so the exam is the honest measure of
how the model does on new districts. ``prepare_exam`` predicts every test district once, before any
frame, and times the model with animkit's ``measure_speed``; each frame then reveals more of those
predictions in a fixed shuffled order (animkit's ``exam_order``). Three drawings share the work: the
main plot (scored districts against the model's curve), a map of the misses, and a board of running
scores (RMSE, MAE, R² and the share priced within ``GOOD_ENOUGH_K``).
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from animkit.lessons import theme
from animkit.lessons.exam import Speed, exam_order, measure_speed, speed_line
from animkit.lessons.stage import reset_axes
from house_anim.config import GOOD_ENOUGH_K, RANDOM_STATE
from house_anim.data import read_only
from house_anim.lessons import plots
from house_anim.lessons.config import INCOME_TO_K, PRICE_TO_K, X_RANGE
from house_anim.lessons.lesson import Lesson

BATCHES = 60  # the latest batch is drawn with its error lines
ERROR_LIMIT_K = 150.0  # map colour saturates at ±$150k
ERROR_CMAP = LinearSegmentedColormap.from_list("miss", (theme.ERROR, "#eef1f4", theme.MODEL))


@dataclass(frozen=True)
class RegressionExam:
    """The exam, computed once before the chapter: the scoring order, the predictions and their speed.

    Attributes:
        order: The order the test districts are scored on screen (row indices into the test split).
        predicted: The lesson's model on every test district ($100k), in test-split order.
        speed: How long the model takes to price one district alone, and per district in one batch.
    """

    order: np.ndarray  # the order districts are scored on screen
    predicted: np.ndarray  # the lesson's model on every test district (test-split order)
    speed: Speed


@dataclass(frozen=True)
class Scores:
    """The exam's scores over the districts scored so far.

    Attributes:
        rmse_k: The root-mean-square error, in $k: the typical miss, big misses counting extra.
        mae_k: The mean absolute error, in $k: the average miss.
        r2: R², the share of the price spread the model explains; NaN until two districts with
            different prices are scored.
        within_share: The share of districts priced within ±``GOOD_ENOUGH_K``, from 0 to 1.
    """

    rmse_k: float
    mae_k: float
    r2: float  # NaN until two districts with different prices are scored
    within_share: float  # share priced within ±GOOD_ENOUGH_K


def prepare_exam(lesson: Lesson) -> RegressionExam:
    """Prices every test district with the final model, and times it.

    Args:
        lesson: The lesson being played.

    Returns:
        The exam: a seeded shuffled scoring order, the predictions and the measured speed.
    """
    data = lesson.data
    return RegressionExam(order=exam_order(len(data.y_test), RANDOM_STATE),
                          predicted=read_only(lesson.predict(data.x_test)),
                          speed=measure_speed(lesson.predict, data.x_test))


def running_scores(predicted: np.ndarray, actual: np.ndarray) -> Scores:
    """Scores some predictions against the true prices.

    Args:
        predicted: The predicted prices ($100k).
        actual: The true prices ($100k), in the same order.

    Returns:
        Their RMSE, MAE, R² (NaN while it is undefined) and share within ±``GOOD_ENOUGH_K``.
    """
    errors = np.asarray(predicted) - np.asarray(actual)
    spread = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum(errors ** 2)) / spread if len(actual) > 1 and spread > 0 else float("nan")
    return Scores(rmse_k=float(np.sqrt(np.mean(errors ** 2))) * PRICE_TO_K,
                  mae_k=float(np.mean(np.abs(errors))) * PRICE_TO_K, r2=r2,
                  within_share=float(np.mean(np.abs(errors) * PRICE_TO_K <= GOOD_ENOUGH_K)))


def draw_exam_main(ax: Axes, lesson: Lesson, exam: RegressionExam, shown: int) -> None:
    """Draws the scored test districts against the final curve, with the newest batch's error lines.

    The newest batch is the last 1/``BATCHES`` of the districts scored so far. A callout says how many
    districts lie beyond the income axis: they are scored but not drawn.

    Args:
        ax: The main plot.
        lesson: The lesson being played.
        exam: The prepared exam.
        shown: How many districts have been scored so far.
    """
    data = lesson.data
    plots.setup_main(ax)
    plots.draw_cap(ax)
    plots.draw_dots(ax, data, alpha=0.15)
    plots.draw_curve(ax, data, lesson.final.curve)
    rows = exam.order[:shown]
    ax.scatter(data.x_test[rows] * INCOME_TO_K, data.y_test[rows] * PRICE_TO_K, s=10, color=theme.UNSEEN,
               alpha=0.45, linewidths=0, zorder=5)
    latest = exam.order[max(0, shown - max(1, len(exam.order) // BATCHES)):shown]
    x = data.x_test[latest] * INCOME_TO_K
    segments = [((xi, a), (xi, p)) for xi, a, p in zip(x, data.y_test[latest] * PRICE_TO_K,
                                                        exam.predicted[latest] * PRICE_TO_K)]
    ax.add_collection(LineCollection(segments, colors=theme.ERROR, linewidths=1.4, alpha=0.85, zorder=6))
    ax.scatter(x, data.y_test[latest] * PRICE_TO_K, s=40, color=theme.UNSEEN, zorder=7,
               edgecolors=theme.PANEL, linewidths=1.5)
    off_chart = int(np.sum(data.x_test > X_RANGE[1]))
    beyond = f"\n{off_chart} with income above ${X_RANGE[1] * INCOME_TO_K:,.0f}k are scored but off this chart" \
        if off_chart else ""
    plots.callout(ax, "orange: unseen test districts at their real value\n"
                      f"red lines: the newest batch's errors against the blue model{beyond}")


def draw_error_map(ax: Axes, lesson: Lesson, exam: RegressionExam, shown: int) -> None:
    """Maps the scored districts at their longitude and latitude, coloured by actual minus predicted.

    Blue is worth more than predicted, red less; the colour saturates at ±``ERROR_LIMIT_K``. The map's
    extent covers every test district from the start, so it does not move as districts are scored.

    Args:
        ax: The side panel.
        lesson: The lesson being played.
        exam: The prepared exam.
        shown: How many districts have been scored so far.
    """
    data = lesson.data
    theme.style_axes(ax, grid_axis=None)
    lon, lat = data.test_coords[:, 0], data.test_coords[:, 1]
    ax.update_datalim(data.test_coords)  # fixed extent: every test district, whether scored yet or not
    ax.autoscale_view()
    ax.set_aspect(1.0 / np.cos(np.radians(lat.mean())), adjustable="datalim")  # true shape, full panel
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Where it misses: actual − predicted")
    rows = exam.order[:shown]
    misses = (data.y_test[rows] - exam.predicted[rows]) * PRICE_TO_K
    ax.scatter(lon[rows], lat[rows], c=misses, cmap=ERROR_CMAP, s=7, linewidths=0,
               norm=TwoSlopeNorm(0.0, -ERROR_LIMIT_K, ERROR_LIMIT_K))
    ax.text(0.02, 0.03, "blue: worth more than predicted\nred: worth less", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11, color=theme.INK_SECONDARY)  # the empty ocean corner


def draw_board(ax: Axes, lesson: Lesson, exam: RegressionExam, shown: int) -> None:
    """Writes the running scores over the districts scored so far, and the model's measured speed.

    Args:
        ax: The metric strip.
        lesson: The lesson being played.
        exam: The prepared exam.
        shown: How many districts have been scored so far.
    """
    data = lesson.data
    rows = exam.order[:shown]
    scores = running_scores(exam.predicted[rows], data.y_test[rows])
    reset_axes(ax)
    ax.text(0.04, 0.88, "Final exam so far", fontsize=theme.PANEL_TITLE_SIZE, weight="bold", color=theme.NAVY,
            va="center")
    r2 = "—" if np.isnan(scores.r2) else f"{scores.r2:.2f}"
    lines = (f"scored {shown:,} of {len(data.y_test):,} test districts",
             f"RMSE ${scores.rmse_k:,.0f}k · MAE ${scores.mae_k:,.0f}k · R² {r2}",
             f"priced within ±${GOOD_ENOUGH_K:,.0f}k: {scores.within_share:.0%}",
             speed_line(exam.speed, "district"))
    for row, line in enumerate(lines):
        ax.text(0.04, 0.68 - row * 0.17, line, fontsize=13.5 if row < 3 else 10.5, va="center",
                color=theme.NAVY if row == 1 else theme.INK_SECONDARY, weight="bold" if row == 1 else "normal")
