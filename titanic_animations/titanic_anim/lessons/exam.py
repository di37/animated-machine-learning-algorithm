"""The classifier final exam: the trained model judges every test passenger, a batch at a time.

Scores on screen come from the passengers judged so far, using the model's real outputs. The grid
groups them by class of travel, an input the lesson's model never saw.

The exam is prepared once, before the first frame (``prepare_exam``): the model scores every test
passenger, and animkit's ``measure_speed`` times its predictions. Each frame then shows the first
``shown`` passengers of a fixed shuffled order three ways: on the maps, in a grid by class and sex, and
on a running scoreboard. The scene's ``draw_exam`` calls the three drawing functions here.
"""

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import roc_auc_score

from animkit.data import read_only
from animkit.lessons import theme
from animkit.lessons.exam import Speed, exam_order, measure_speed, speed_line
from animkit.lessons.stage import reset_axes
from titanic_anim.config import RANDOM_STATE
from titanic_anim.lessons import maps
from titanic_anim.lessons.config import AGE, LOG_FARE, SEX_TITLES
from titanic_anim.lessons.lesson import ClassLesson, threshold

BATCHES = 30  # the latest batch is drawn larger
GRID_COLUMNS = 10
CLASS_NAMES = ("1st class", "2nd class", "3rd class")
BLOCK = dict(left=0.12, top=0.86, width=0.27, height=0.38, gap=0.02)  # grid blocks, axes fractions
SQUARE_FILL, CROSS_FILL = 0.75, 0.55  # a square's side, and a ✕'s, as a share of the cell pitch
POINTS_PER_INCH = 72.0


@dataclass(frozen=True)
class ClassificationExam:
    """The final exam, computed once before it is drawn.

    Attributes:
        order: The test rows in the order they are judged on screen.
        scores: The lesson's model on every test passenger, in test-split order.
        speed: How long the model takes to predict, for one passenger and per passenger in a batch.
    """

    order: np.ndarray  # the order passengers are judged on screen
    scores: np.ndarray  # the lesson's model on every test passenger (test-split order)
    speed: Speed


@dataclass(frozen=True)
class ClassScores:
    """The exam's scores on the passengers judged so far.

    Attributes:
        accuracy: The share of calls that are right.
        recall: The share of survivors predicted to survive; NaN before any survivor is judged.
        auc: The area under the ROC curve: the chance that a random survivor scores higher than a random
            non-survivor, so it judges the ranking, not the threshold. NaN until both outcomes are judged.
        caught: Survivors predicted to survive.
        missed: Survivors predicted not to.
        false_alarms: Non-survivors predicted to survive.
        right_no: Non-survivors predicted not to.
    """

    accuracy: float
    recall: float  # share of survivors predicted to survive; NaN before any survivor is judged
    auc: float  # ranking quality of the scores; NaN until both outcomes have been judged
    caught: int  # survivors predicted to survive
    missed: int  # survivors predicted not to
    false_alarms: int  # non-survivors predicted to survive
    right_no: int  # non-survivors predicted not to


def prepare_exam(lesson: ClassLesson) -> ClassificationExam:
    """Scores every test passenger with the final model, and times its predictions.

    Args:
        lesson: The lesson.

    Returns:
        The exam: a fixed shuffled order, the scores and the measured speed.
    """
    data = lesson.data
    return ClassificationExam(order=exam_order(len(data.y_test), RANDOM_STATE),
                              scores=read_only(lesson.score(data.x_test)),
                              speed=measure_speed(lesson.score, data.x_test))


def running_scores(scores: np.ndarray, labels: np.ndarray, kind: str) -> ClassScores:
    """The exam's scores on the passengers judged so far.

    Args:
        scores: The model's scores for the judged passengers.
        labels: Their true labels, 1 = survived.
        kind: ``PROBABILITY`` or ``MARGIN``; it sets the threshold.

    Returns:
        Accuracy, recall, AUC and the four counts of right and wrong calls.
    """
    predicted = np.asarray(scores) > threshold(kind)
    survived = np.asarray(labels) == 1
    caught, missed = int(np.sum(predicted & survived)), int(np.sum(~predicted & survived))
    false_alarms, right_no = int(np.sum(predicted & ~survived)), int(np.sum(~predicted & ~survived))
    both = 0 < survived.sum() < len(survived)
    return ClassScores(accuracy=(caught + right_no) / len(survived),
                       recall=caught / (caught + missed) if caught + missed else float("nan"),
                       auc=float(roc_auc_score(labels, scores)) if both else float("nan"),
                       caught=caught, missed=missed, false_alarms=false_alarms, right_no=right_no)


def draw_exam_maps(axes: tuple[Axes, ...], lesson: ClassLesson, exam: ClassificationExam, shown: int) -> None:
    """Draws the judged test passengers on both maps, over the final model's faded shading and boundary.

    The latest batch (one of ``BATCHES``) is drawn larger, so the eye follows the new arrivals; wrong
    calls are circled.

    Args:
        axes: The two maps, women's first.
        lesson: The lesson.
        exam: The prepared exam.
        shown: How many passengers of the exam's order have been judged.
    """
    data = lesson.data
    rows = exam.order[:shown]
    latest = set(exam.order[max(0, shown - max(1, len(exam.order) // BATCHES)):shown].tolist())
    wrong = (exam.scores > threshold(lesson.kind)) != (data.y_test == 1)
    for sex, ax in enumerate(axes):
        maps.setup_map(ax, data, sex)
        maps.draw_surface(ax, data, lesson.final.surface[sex], lesson.kind, alpha=0.35)
        maps.draw_boundary(ax, data, lesson.final.surface[sex], lesson.kind)
        mine = rows[maps.on_map(data.x_test[rows], sex)]
        for outcome, marker, colour in ((1, "o", theme.MODEL), (0, "X", theme.ERROR)):
            group = mine[data.y_test[mine] == outcome]
            sizes = [90 if r in latest else 36 for r in group]
            ax.scatter(data.x_test[group, AGE], data.x_test[group, LOG_FARE], s=sizes, marker=marker, color=colour,
                       zorder=5, edgecolors=theme.PANEL, linewidths=0.9)
        missed = mine[wrong[mine]]
        ax.scatter(data.x_test[missed, AGE], data.x_test[missed, LOG_FARE], s=130, facecolors="none",
                   edgecolors=theme.NAVY, linewidths=1.5, zorder=6)
    axes[0].set_title(f"{SEX_TITLES[0]} · test passengers", fontsize=theme.PANEL_TITLE_SIZE)
    axes[1].set_title(f"{SEX_TITLES[1]} · test passengers", fontsize=theme.PANEL_TITLE_SIZE)


def _cell_pitch_pt(ax: Axes, rows_per_block: int) -> float:
    """Centre-to-centre distance of neighbouring grid cells, in points, on this panel."""
    width_in, height_in = ax.figure.get_size_inches()
    position = ax.get_position()
    return POINTS_PER_INCH * min(BLOCK["width"] / GRID_COLUMNS * position.width * width_in,
                                 BLOCK["height"] / max(rows_per_block, 1) * position.height * height_in)


def _cell(index: int, block: tuple[int, int], rows_per_block: int) -> tuple[float, float]:
    """The centre of a block's ``index``-th cell, in axes fractions; cells fill rows from the top left."""
    column, row = block
    left = BLOCK["left"] + column * (BLOCK["width"] + BLOCK["gap"])
    top = BLOCK["top"] - row * (BLOCK["height"] + BLOCK["gap"] * 3)
    step_x = BLOCK["width"] / GRID_COLUMNS
    step_y = BLOCK["height"] / max(rows_per_block, 1)
    return left + (index % GRID_COLUMNS + 0.5) * step_x, top - (index // GRID_COLUMNS + 0.5) * step_y


def class_blocks(data) -> list[tuple[int, int]]:
    """(class column 0-2, sex row: 0 women / 1 men) for every test passenger, in test-split order.

    Args:
        data: The passengers (a ``PassengerData``).

    Returns:
        One (column, row) block per test passenger.
    """
    return [(int(data.test_class[r]) - 1, maps.map_of(data.x_test[r])) for r in range(len(data.y_test))]


def draw_class_grid(ax: Axes, lesson: ClassLesson, exam: ClassificationExam, shown: int) -> None:
    """Every judged passenger as a square in its class × sex block, shaded by score; ✕ = wrong call.

    Class of travel was never an input, so the grid shows whether the model's mistakes cluster in a group
    it could not see. Squares are sized from the panel's real size, so they fill their cells at any resolution.

    Args:
        ax: The side panel's axes.
        lesson: The lesson.
        exam: The prepared exam.
        shown: How many passengers of the exam's order have been judged.
    """
    data = lesson.data
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("By class and sex (class: never an input)")
    blocks = class_blocks(data)
    sizes = {b: blocks.count(b) for b in set(blocks)}
    rows_per_block = math.ceil(max(sizes.values()) / GRID_COLUMNS)
    filled: dict[tuple[int, int], int] = {}
    xs, ys, values, crosses = [], [], [], []
    for r in exam.order[:shown]:
        block = blocks[r]
        x, y = _cell(filled.get(block, 0), block, rows_per_block)
        filled[block] = filled.get(block, 0) + 1
        xs.append(x)
        ys.append(y)
        values.append(exam.scores[r])
        if (exam.scores[r] > threshold(lesson.kind)) != (data.y_test[r] == 1):
            crosses.append((x, y))
    pitch = _cell_pitch_pt(ax, rows_per_block)
    ax.scatter(xs, ys, c=values, cmap=maps.SCORE_CMAP, norm=maps.score_norm(lesson.kind), marker="s",
               s=(SQUARE_FILL * pitch) ** 2, linewidths=0)
    if crosses:
        cx, cy = zip(*crosses)
        ax.scatter(cx, cy, marker="x", s=(CROSS_FILL * pitch) ** 2, color=theme.NAVY, linewidths=1.4)
    for column, name in enumerate(CLASS_NAMES):
        ax.text(BLOCK["left"] + column * (BLOCK["width"] + BLOCK["gap"]) + BLOCK["width"] / 2, 0.92, name,
                ha="center", fontsize=11, color=theme.INK_SECONDARY)
    for row, name in enumerate(SEX_TITLES):
        ax.text(0.0, BLOCK["top"] - row * (BLOCK["height"] + BLOCK["gap"] * 3) - BLOCK["height"] / 2, name,
                va="center", fontsize=11, color=theme.INK_SECONDARY)


def draw_board(ax: Axes, lesson: ClassLesson, exam: ClassificationExam, shown: int) -> None:
    """Draws the running scoreboard: how many are judged, accuracy, recall, AUC, the counts and the speed.

    Args:
        ax: The strip below the maps.
        lesson: The lesson.
        exam: The prepared exam.
        shown: How many passengers of the exam's order have been judged.
    """
    data = lesson.data
    rows = exam.order[:shown]
    s = running_scores(exam.scores[rows], data.y_test[rows], lesson.kind)
    reset_axes(ax)
    ax.text(0.04, 0.88, "Final exam so far", fontsize=theme.PANEL_TITLE_SIZE, weight="bold", color=theme.NAVY,
            va="center")
    recall = "—" if np.isnan(s.recall) else f"{s.recall:.0%}"
    auc = "—" if np.isnan(s.auc) else f"{s.auc:.2f}"
    lines = (f"judged {shown:,} of {len(data.y_test):,} test passengers",
             f"accuracy {s.accuracy:.0%} · recall {recall} · AUC {auc}",
             f"survivors: {s.caught} caught, {s.missed} missed · others: {s.right_no} right, "
             f"{s.false_alarms} false alarms",
             speed_line(exam.speed, "passenger"))
    for row, line in enumerate(lines):
        ax.text(0.04, 0.68 - row * 0.17, line, fontsize=13.5 if row < 2 else (11.5 if row == 2 else 10.5), va="center",
                color=theme.NAVY if row == 1 else theme.INK_SECONDARY, weight="bold" if row == 1 else "normal")
