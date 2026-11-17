"""Regression drawing on the main plot and the metric strip: dots, model curve, residuals, the new house.

The callout and the worked-prediction panel every lesson shares come from animkit.lessons.plots.

The main plot has income on x and price on y, both written in $k, while the data and the models stay in
the dataset's units (``MedInc`` in $10k, prices in $100k): every function here converts at the moment
it draws. The training picture is built in layers: the census's $500k cap, the training dots, the
algorithm's own overlay, fading "ghost" curves of the last few snapshots, the residual lines from some
districts to the curve, and the current curve on top. The metric strip shows the train and validation
RMSE so far against always guessing the average.
"""

import math
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, NullFormatter

from animkit.lessons import theme
from animkit.lessons.plots import PLAIN_FORMAT, callout
from animkit.lessons.timing import phase
from house_anim.lessons.config import INCOME_TO_K, PRICE_CAP, PRICE_TO_K, X_RANGE, Y_RANGE_K
from house_anim.lessons.data import IncomeData, Query
from house_anim.lessons.lesson import Lesson, baseline_rmse_k, money_k

K_FORMAT = FuncFormatter(lambda value, _: f"{'−' if value < 0 else ''}${abs(value):,.0f}k")
METRIC_CEILING = 2.1  # metric y-axis tops out at 2.1x the always-the-average error
RING = dict(edgecolors=theme.PANEL, linewidths=2.0)  # surface ring around dots


def setup_main(ax: Axes) -> None:
    """Clears the main plot and sets its fixed income and price axes, labelled in $k.

    Args:
        ax: The main plot.
    """
    ax.clear()
    theme.style_axes(ax)
    ax.set_xlim(X_RANGE[0] * INCOME_TO_K, X_RANGE[1] * INCOME_TO_K)
    ax.set_ylim(*Y_RANGE_K)
    ax.set_xlabel("Median household income in the district")
    ax.set_ylabel("Median house value")
    ax.xaxis.set_major_formatter(K_FORMAT)
    ax.yaxis.set_major_formatter(K_FORMAT)


def draw_dots(ax: Axes, data: IncomeData, count: int | None = None, alpha: float = 0.55) -> None:
    """Draws the training districts as dots; a small sample (200 or fewer) gets big, ringed dots.

    Args:
        ax: The main plot.
        data: The lesson's data; its ``display`` rows are drawn, in reveal order.
        count: How many of them to draw; None draws them all.
        alpha: The dots' opacity (a small sample is always drawn at 0.9).
    """
    shown = len(data.display) if count is None else count
    if shown <= 0:
        return
    few = len(data.display) <= 200  # a small training sample gets big, ringed dots
    ax.scatter(data.display_x[:shown] * INCOME_TO_K, data.display_y[:shown] * PRICE_TO_K,
               s=90 if few else 16, color=theme.INK_SECONDARY if few else theme.DOTS,
               alpha=0.9 if few else alpha, zorder=3 if few else 2,
               **(RING if few else dict(linewidths=0)))


def draw_cap(ax: Axes) -> None:
    """Draws the $500k line where the census caps house values, with a label saying so.

    Args:
        ax: The main plot.
    """
    cap_k = PRICE_CAP * PRICE_TO_K
    ax.axhline(cap_k, color=theme.INK_MUTED, linewidth=1.0, linestyle=(0, (3, 4)), zorder=1)
    ax.text(X_RANGE[1] * INCOME_TO_K - 1.0, cap_k + 8, "the census caps values at $500k",
            ha="right", va="bottom", fontsize=theme.NOTE_SIZE, color=theme.INK_MUTED, zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=theme.PANEL, edgecolor="none", alpha=0.9))


def draw_curve(ax: Axes, data: IncomeData, curve: np.ndarray, *, color: str = theme.MODEL,
               width: float = 3.4, alpha: float = 1.0, zorder: int = 6) -> None:
    """Draws a model's curve over the income grid.

    Args:
        ax: The main plot.
        data: The lesson's data, whose ``grid`` the curve was evaluated on.
        curve: The model's prices at each grid income ($100k).
        color: The line colour.
        width: The line width, in points.
        alpha: The line's opacity.
        zorder: Its drawing order; the default puts it above the dots and residuals.
    """
    ax.plot(data.grid * INCOME_TO_K, curve * PRICE_TO_K, color=color, linewidth=width, alpha=alpha,
            zorder=zorder, solid_capstyle="round", solid_joinstyle="round")


def draw_ghosts(ax: Axes, data: IncomeData, curves: Sequence[np.ndarray]) -> None:
    """Earlier genuine snapshots, fading with age.

    Args:
        ax: The main plot.
        data: The lesson's data, whose ``grid`` the curves were evaluated on.
        curves: The earlier snapshots' curves ($100k), oldest first; the newest is drawn strongest.
    """
    for age, curve in enumerate(reversed(curves), start=1):
        draw_curve(ax, data, curve, width=2.0, alpha=0.32 / age, zorder=4)


def draw_residuals(ax: Axes, data: IncomeData, fitted: np.ndarray, grow: float = 1.0) -> None:
    """Red lines from each highlighted district to the model; ``grow`` animates them in.

    Each line starts at the district's true price and grows towards the model's prediction for it.

    Args:
        ax: The main plot.
        data: The lesson's data; its ``residual`` rows get the lines.
        fitted: The model's prices for those rows ($100k).
        grow: How far the lines have grown, from 0 (not drawn) to 1 (reaching the model).
    """
    if grow <= 0:
        return
    x = data.residual_x * INCOME_TO_K
    actual = data.residual_y * PRICE_TO_K
    model = np.asarray(fitted) * PRICE_TO_K
    ends = actual + (model - actual) * grow
    ax.add_collection(LineCollection([((xi, a), (xi, e)) for xi, a, e in zip(x, actual, ends)],
                                     colors=theme.ERROR, linewidths=1.6, alpha=0.8, zorder=3))
    ax.scatter(x, actual, s=46, color=theme.NAVY, zorder=4, **RING)


def draw_training_state(ax: Axes, lesson: Lesson, index: int, ghosts: int = 3,
                        residual_grow: float = 1.0, note: str | None = None) -> None:
    """Draws one training snapshot on the main plot, layer by layer, with its note in a callout.

    Args:
        ax: The main plot.
        lesson: The lesson being played.
        index: Which snapshot to draw.
        ghosts: How many earlier snapshots to draw as fading curves.
        residual_grow: How far the residual lines have grown, from 0 to 1.
        note: The callout text; None shows the snapshot's own note.
    """
    snap = lesson.snapshots[index]
    setup_main(ax)
    draw_cap(ax)
    draw_dots(ax, lesson.data)
    lesson.panel.overlay_training(ax, snap)
    draw_ghosts(ax, lesson.data, [s.curve for s in lesson.snapshots[max(0, index - ghosts):index]])
    draw_residuals(ax, lesson.data, snap.fitted, grow=residual_grow)
    draw_curve(ax, lesson.data, snap.curve)
    callout(ax, snap.note if note is None else note)


def draw_metric(ax: Axes, lesson: Lesson, upto: int | None, *, mark_final: bool = False) -> None:
    """Train vs validation RMSE so far, against the always-predict-the-average baseline.

    The axes are fixed for the whole lesson (every snapshot's progress, and up to ``METRIC_CEILING``
    times the baseline), so the lines grow into a chart that does not move. When the story sets
    ``log_progress`` (the α sweeps do), the x-axis is logarithmic.

    Args:
        ax: The metric strip.
        lesson: The lesson being played.
        upto: The last snapshot to plot; None draws the empty chart with just the baseline.
        mark_final: Whether to draw a line at the final model's progress.
    """
    ax.clear()
    theme.style_axes(ax)
    story, snaps = lesson.story, lesson.snapshots
    baseline = baseline_rmse_k(lesson.data)
    progress = [s.progress for s in snaps]
    if story.log_progress:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(PLAIN_FORMAT)  # log ticks default to LaTeX, which is switched off
        ax.xaxis.set_minor_formatter(NullFormatter())
    pad = (max(progress) - min(progress)) * 0.03 or 0.5
    ax.set_xlim(*(min(progress) / 1.25, max(progress) * 1.25) if story.log_progress else
                (min(progress) - pad, max(progress) + pad))
    ax.set_ylim(0.0, baseline * METRIC_CEILING)
    ax.yaxis.set_major_formatter(K_FORMAT)
    ax.set_xlabel(story.progress_label, fontsize=theme.NOTE_SIZE)
    ax.set_title("Error as it learns (RMSE, lower is better)", fontsize=theme.PANEL_TITLE_SIZE)
    ax.axhline(baseline, color=theme.INK_MUTED, linewidth=1.2, linestyle=(0, (3, 3)))
    ax.text(0.01, 1.0 / METRIC_CEILING - 0.03,
            f"always guess the average, {money_k(lesson.data.baseline_mean)}: ${baseline:,.0f}k",
            transform=ax.transAxes, ha="left", va="top", fontsize=12, color=theme.INK_MUTED)
    if upto is None:
        return
    shown = snaps[:upto + 1]
    xs = [s.progress for s in shown]
    for values, color, name in (([s.train_rmse_k for s in shown], theme.MODEL, "train"),
                                ([s.val_rmse_k for s in shown], theme.UNSEEN, "validation")):
        ax.plot(xs, values, color=color, linewidth=2.2, solid_capstyle="round",
                label=f"{name}  {_k(values[-1])}", clip_on=True)
        ax.scatter([xs[-1]], [min(values[-1], baseline * METRIC_CEILING * 0.985)], s=60, color=color,
                   zorder=5, clip_on=False, **RING)
    if mark_final:
        ax.axvline(lesson.final.progress, color=theme.NAVY, linewidth=1.4)
    ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0.3, ncols=2)


def _k(value: float) -> str:
    """An error in $k for the legend, switching to $M when it runs into the millions."""
    return f"${value:,.0f}k" if value < 10_000 else f"${value / 1000:,.0f}M"


def draw_query(ax: Axes, query: Query, prediction: float, t: float) -> None:
    """The unseen district slides in, rises to the model, reads off a price, then meets the truth.

    The steps are timed on ``t``: it slides in along the floor (0 to 0.15), a dashed line rises to the
    curve (to 0.32), then runs across to the price axis (to 0.45); the true price and the miss appear
    last (0.86 to 0.97).

    Args:
        ax: The main plot.
        query: The unseen district.
        prediction: The final model's price for it ($100k).
        t: How far through the chapter's motion, from 0 to 1.
    """
    x_left, x_right = X_RANGE[0] * INCOME_TO_K, X_RANGE[1] * INCOME_TO_K
    x = query.income * INCOME_TO_K
    predicted, actual = prediction * PRICE_TO_K, query.price * PRICE_TO_K
    floor = Y_RANGE_K[0] + 6.0
    slide = phase(t, 0.0, 0.15)
    x_now = x_right + (x - x_right) * slide
    ax.scatter([x_now], [floor], s=210, marker="D", color=theme.UNSEEN, zorder=9, clip_on=False, **RING)
    ax.text(x_now, floor + 16, f"new district · income ${x:,.1f}k", ha="center", va="bottom",
            fontsize=14, color=theme.NAVY, weight="bold", zorder=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=theme.PANEL, edgecolor="none", alpha=0.9))
    rise = phase(t, 0.15, 0.32)
    if rise > 0:
        ax.plot([x, x], [floor, floor + (predicted - floor) * rise], color=theme.NAVY, linewidth=1.8,
                linestyle=(0, (5, 4)), zorder=8)
    across = phase(t, 0.32, 0.45)
    if across > 0:
        ax.plot([x, x - (x - x_left) * across], [predicted, predicted], color=theme.NAVY,
                linewidth=1.8, linestyle=(0, (5, 4)), zorder=8)
        ax.scatter([x], [predicted], s=170, color=theme.MODEL, zorder=10, **RING)
        ax.text(x_left + 1.5, predicted + 9, f"predicted {money_k(prediction)}", ha="left",
                va="bottom", fontsize=17, weight="bold", color=theme.NAVY, zorder=11,
                alpha=math.sqrt(across))
    truth = phase(t, 0.86, 0.97)
    if truth > 0:
        ax.plot([x, x], [predicted, actual], color=theme.ERROR, linewidth=2.4, alpha=truth, zorder=8)
        ax.scatter([x], [actual], s=210, marker="D", color=theme.UNSEEN, alpha=truth, zorder=10, **RING)
        ax.text(x + 2.5, actual, f"actual {money_k(query.price)} · off by "
                f"{money_k(abs(prediction - query.price))}", ha="left", va="center", fontsize=15,
                weight="bold", color=theme.NAVY, alpha=truth, zorder=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=theme.PANEL, edgecolor="none", alpha=0.9))
