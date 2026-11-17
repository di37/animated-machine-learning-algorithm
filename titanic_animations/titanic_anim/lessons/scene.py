"""The Titanic ``Scene``: the classification picture animkit's lesson director delegates to.

animkit's director (``animkit.lessons.director``) plays every project's lessons through the same
chapters, and asks a ``Scene`` for everything specific to the task. ``ClassificationScene`` implements
that protocol for the Titanic: two maps (women | men), an accuracy strip, and the unseen passenger placed
on their map at inference. It hands the final exam, the 3-D view and the more-features comparison on to
``exam``, ``surface_view`` and ``more_features``. ``CLASSIFICATION`` is the instance the command line
gives the director.
"""

import math

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from animkit.lessons import theme
from animkit.lessons.plots import PLAIN_FORMAT, callout
from animkit.lessons.timing import phase
from animkit.lessons.exam import revealed
from animkit.lessons.lesson import Comparison
from animkit.lessons.surface import Surface
from titanic_anim.lessons import exam, maps, more_features, surface_view
from titanic_anim.lessons.config import AGE, AGE_RANGE, LOG_FARE, SOURCE_NOTE
from titanic_anim.lessons.lesson import (
    PROBABILITY,
    ClassLesson,
    accuracy,
    outcome,
    threshold,
)

METRIC_FLOOR = 0.4  # the accuracy axis starts here unless a frame scores lower


def _legend(ax: Axes, kind: str) -> None:
    """Adds the maps' legend: the two outcomes, the misclassified ring and the boundary line."""
    handles = [Line2D([], [], marker="o", linestyle="", color=theme.MODEL, label="survived"),
               Line2D([], [], marker="X", linestyle="", color=theme.ERROR, label="did not survive"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color=theme.NAVY,
                      label="misclassified"),
               Line2D([], [], color=theme.NAVY, linewidth=2.6,
                      label="boundary: P = 50%" if kind == PROBABILITY else "boundary: score = 0")]
    ax.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.9, frameon=True,
              edgecolor=theme.PANEL_EDGE)


def _no_boundary_note(ax: Axes, surface: np.ndarray, kind: str) -> None:
    """Say so when the boundary falls outside a map, instead of leaving the legend line unexplained."""
    verdict = outcome(int(surface.mean() > threshold(kind)))
    ax.text(0.5, 0.5, f"no boundary on this map:\neveryone here is predicted '{verdict}'",
            transform=ax.transAxes, ha="center", va="center", fontsize=14, weight="bold", color=theme.NAVY,
            zorder=12, bbox=dict(boxstyle="round,pad=0.4", facecolor=theme.PANEL, edgecolor="none", alpha=0.85))


def metric_floor(lesson: ClassLesson) -> float:
    """Where the accuracy axis starts, fixed for the whole lesson so the strip never rescales.

    Args:
        lesson: The lesson.

    Returns:
        0.4, or lower (in 10% steps) if any snapshot's train or validation accuracy is below that.
    """
    lowest = min(min(s.train_accuracy, s.val_accuracy) for s in lesson.snapshots)
    return min(METRIC_FLOOR, math.floor(lowest * 10) / 10)


def _shading_note(ax: Axes, kind: str) -> None:
    """Says in the map's corner what the shading is: P(survive), or a score that is not a probability."""
    text = ("shading: the model's P(survive)" if kind == PROBABILITY
            else "shading: the model's score (not a probability)")
    ax.text(0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            color=theme.INK_SECONDARY, zorder=12,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=theme.PANEL, edgecolor="none", alpha=0.85))


class ClassificationScene:
    """The Titanic's implementation of animkit's ``Scene`` protocol: what each chapter draws.

    Every method takes the ``ClassLesson`` being played; the drawing methods also take the two maps'
    axes (women's first) and, for animated chapters, ``t``, the chapter's progress from 0 to 1.

    Attributes:
        subtitle: The header's subtitle, before the lesson's method.
        source_note: The dataset credit in the footer.
        facets: How many main plots sit side by side: the two maps.
    """

    subtitle = "Who survived the Titanic? Learning from age, fare and sex"
    source_note = SOURCE_NOTE
    facets = 2

    @staticmethod
    def describe_score(value: float, kind: str) -> str:
        """A score as it is written beside the passenger.

        Args:
            value: The model's score.
            kind: ``PROBABILITY`` or ``MARGIN``.

        Returns:
            'P(survive) = 81%' for a probability, 'score +1.23' for a signed score.
        """
        return f"P(survive) = {value:.0%}" if kind == PROBABILITY else f"score {value:+.2f}"

    def data_label(self, lesson: ClassLesson) -> str:
        """The data chapter's header label.

        Args:
            lesson: The lesson.

        Returns:
            How many training passengers there are.
        """
        return f"{lesson.data.n_train:,} training passengers"

    def data_caption(self, lesson: ClassLesson) -> str:
        """The data chapter's caption: how to read a mark on the maps.

        Args:
            lesson: The lesson.

        Returns:
            The caption.
        """
        return ("Each mark is one passenger: age (x) and fare paid (y, log scale); women on the left, "
                "men on the right. Blue dots survived, red crosses did not.")

    def facts(self, lesson: ClassLesson) -> tuple[str, ...]:
        """The dataset facts the data chapter lists in its panel.

        Args:
            lesson: The lesson.

        Returns:
            One line each: the inputs, the target, the three splits and the passengers left out.
        """
        data = lesson.data
        return ("Inputs: age, fare paid, sex",
                f"Target: survived or not ({data.baseline_rate:.0%} of training passengers did)",
                f"Training: {data.n_train:,} passengers",
                f"Validation: {len(data.y_val):,}, to compare settings",
                f"Test: {len(data.y_test):,}, never touched until the end",
                f"Left out: {data.dropped:,} passengers with no recorded age")

    def draw_data(self, axes: tuple[Axes, ...], lesson: ClassLesson, t: float) -> None:
        """Draws the data chapter: the training passengers appear on both maps, all of them by t = 0.7.

        Args:
            axes: The two maps, women's first.
            lesson: The lesson.
            t: The chapter's progress, from 0 to 1.
        """
        count = math.ceil(len(lesson.data.display) * phase(t, 0.0, 0.7))
        for sex, ax in enumerate(axes):
            maps.setup_map(ax, lesson.data, sex)
            maps.draw_passengers(ax, lesson.data, sex, count=count)

    def draw_state(self, axes: tuple[Axes, ...], lesson: ClassLesson, index: int, *, ghosts: int,
                   grow: float, note: str | None) -> None:
        """Draws one snapshot on both maps: shading, boundary, passengers, mistakes ringed, and the note.

        Earlier boundaries stay on as fading ghosts, so the viewer sees which way the boundary is moving.
        A map the boundary never crosses says so instead.

        Args:
            axes: The two maps, women's first.
            lesson: The lesson.
            index: The snapshot to draw.
            ghosts: How many earlier boundaries to draw faintly behind it.
            grow: The opacity of the rings round misclassified passengers (they fade in on the first guess).
            note: The callout text to show instead of the snapshot's own note; None keeps the snapshot's.
        """
        data, snap = lesson.data, lesson.snapshots[index]
        earlier = lesson.snapshots[max(0, index - ghosts):index]
        for sex, ax in enumerate(axes):
            maps.setup_map(ax, data, sex)
            maps.draw_surface(ax, data, snap.surface[sex], lesson.kind)
            for age, old in enumerate(reversed(earlier), start=1):
                maps.draw_boundary(ax, data, old.surface[sex], lesson.kind, width=1.6, alpha=0.45 / age)
            if not maps.draw_boundary(ax, data, snap.surface[sex], lesson.kind):
                _no_boundary_note(ax, snap.surface[sex], lesson.kind)
            maps.draw_passengers(ax, data, sex, wrong=snap.wrong, ring=grow)
        lesson.panel.overlay_training(axes, snap)
        callout(axes[0], snap.note if note is None else note)
        _legend(axes[1], lesson.kind)
        _shading_note(axes[1], lesson.kind)

    def draw_metric(self, ax: Axes, lesson: ClassLesson, upto: int | None, mark_final: bool = False) -> None:
        """Draws the accuracy strip: train and validation accuracy as the model learns.

        The dashed line is the baseline of always guessing 'did not survive', which a model must beat.

        Args:
            ax: The strip's axes.
            lesson: The lesson.
            upto: Plot the snapshots up to this index; None draws only the axes and the baseline.
            mark_final: Whether to mark the final snapshot with a vertical line.
        """
        ax.clear()
        theme.style_axes(ax)
        snaps, story = lesson.snapshots, lesson.story
        baseline = 1.0 - lesson.data.baseline_rate  # always answering "did not survive"
        progress = [s.progress for s in snaps]
        if story.log_progress:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(PLAIN_FORMAT)
            ax.xaxis.set_minor_formatter(NullFormatter())
        pad = (max(progress) - min(progress)) * 0.03 or 0.5
        ax.set_xlim(*(min(progress) / 1.25, max(progress) * 1.25) if story.log_progress else
                    (min(progress) - pad, max(progress) + pad))
        floor = metric_floor(lesson)
        ax.set_ylim(floor, 1.0)
        ax.yaxis.set_major_formatter(maps.PERCENT)
        ax.set_xlabel(story.progress_label, fontsize=theme.NOTE_SIZE)
        ax.set_title("Accuracy as it learns (higher is better)")
        ax.axhline(baseline, color=theme.INK_MUTED, linewidth=1.2, linestyle=(0, (3, 3)))
        ax.text(0.01, (baseline - floor) / (1 - floor) - 0.03,
                f"always guess 'did not survive': {baseline:.0%}", transform=ax.transAxes, ha="left",
                va="top", fontsize=12, color=theme.INK_MUTED)
        if upto is None:
            return
        shown = snaps[:upto + 1]
        xs = [s.progress for s in shown]
        for values, color, name in (([s.train_accuracy for s in shown], theme.MODEL, "train"),
                                    ([s.val_accuracy for s in shown], theme.UNSEEN, "validation")):
            ax.plot(xs, values, color=color, linewidth=2.2, solid_capstyle="round",
                    label=f"{name}  {values[-1]:.0%}")
            ax.scatter([xs[-1]], [values[-1]], s=60, color=color, zorder=5, clip_on=False,
                       edgecolors=theme.PANEL, linewidths=2.0)
        if mark_final:
            ax.axvline(lesson.final.progress, color=theme.NAVY, linewidth=1.4)
        ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0.3, ncols=2)

    def final_note(self, lesson: ClassLesson) -> str:
        """The callout for the final-model chapter.

        Args:
            lesson: The lesson.

        Returns:
            The kept snapshot's label and its train and validation accuracy, on two lines.
        """
        final = lesson.final
        return (f"Final model · {final.label}\n"
                f"train accuracy {final.train_accuracy:.0%} · validation {final.val_accuracy:.0%}")

    def inference_label(self, lesson: ClassLesson) -> str:
        """The prediction chapter's header label.

        Args:
            lesson: The lesson.

        Returns:
            Which test passenger is being predicted.
        """
        return f"Unseen test passenger #{lesson.data.query.row:,}"

    def draw_inference(self, axes: tuple[Axes, ...], lesson: ClassLesson, t: float) -> None:
        """Draws the prediction chapter: the final model's maps, and the unseen passenger arriving on theirs.

        Args:
            axes: The two maps, women's first.
            lesson: The lesson.
            t: The chapter's progress, from 0 to 1.
        """
        data, query = lesson.data, lesson.data.query
        for sex, ax in enumerate(axes):
            maps.setup_map(ax, data, sex)
            maps.draw_surface(ax, data, lesson.final.surface[sex], lesson.kind, alpha=0.45)
            if not maps.draw_boundary(ax, data, lesson.final.surface[sex], lesson.kind) and sex != maps.map_of(query.features):
                _no_boundary_note(ax, lesson.final.surface[sex], lesson.kind)
            maps.draw_passengers(ax, data, sex, alpha=0.25)
        lesson.panel.overlay_inference(axes, query, phase(t, 0.45, 0.85))
        self._draw_query(axes[maps.map_of(query.features)], lesson, t)

    def _draw_query(self, ax: Axes, lesson: ClassLesson, t: float) -> None:
        """Slides the unseen passenger in from the right, then shows the model's call and, last, the truth."""
        query = lesson.data.query
        age, fare = query.features[AGE], query.features[LOG_FARE]
        x_now = AGE_RANGE[1] + (age - AGE_RANGE[1]) * phase(t, 0.0, 0.15)
        ax.scatter([x_now], [fare], s=260, marker="D", color=theme.UNSEEN, zorder=9,
                   edgecolors=theme.PANEL, linewidths=2.0)
        box = dict(boxstyle="round,pad=0.35", facecolor=theme.PANEL, edgecolor="none", alpha=0.92)
        ax.text(x_now, fare + 0.32, f"new passenger · age {age:.0f} · ${np.expm1(fare):,.0f} fare",
                ha="center", va="bottom", fontsize=13, weight="bold", color=theme.NAVY, zorder=11, bbox=box)
        score = lesson.score_one(query.features)
        if phase(t, 0.15, 0.4) > 0:
            guess = outcome(int(score > threshold(lesson.kind)))
            ax.text(age, fare - 0.32, f"{self.describe_score(score, lesson.kind)} → predicted: {guess}",
                    ha="center", va="top", fontsize=14, weight="bold", color=theme.NAVY, zorder=11, bbox=box,
                    alpha=phase(t, 0.15, 0.4))
        truth = phase(t, 0.86, 0.97)
        if truth > 0:
            right = int(score > threshold(lesson.kind)) == query.survived
            ax.text(age, fare - 0.85, f"actually {outcome(query.survived)} {'✓' if right else '✗'}",
                    ha="center", va="top", fontsize=14, weight="bold", color=theme.NAVY, zorder=11, bbox=box,
                    alpha=truth)

    def explain(self, lesson: ClassLesson):
        """The worked prediction for the unseen passenger, shown line by line in the prediction chapter.

        Args:
            lesson: The lesson.

        Returns:
            The lesson's verified ``Explanation``.
        """
        return lesson.explain(lesson.data.query.features)

    def takeaway_facts(self, lesson: ClassLesson) -> tuple[str, ...]:
        """The takeaway card's facts.

        Args:
            lesson: The lesson.

        Returns:
            The final exam's accuracy against always guessing 'did not survive', and the unseen
            passenger's prediction against what really happened.
        """
        data, query = lesson.data, lesson.data.query
        test = accuracy(lesson.score(data.x_test), data.y_test, lesson.kind)
        baseline = float(np.mean(data.y_test == 0))
        score = lesson.score_one(query.features)
        guess = outcome(int(score > threshold(lesson.kind)))
        return (f"Final exam on {len(data.y_test):,} unseen test passengers: {test:.0%} correct "
                f"(always guessing 'did not survive': {baseline:.0%})",
                f"New passenger: {self.describe_score(score, lesson.kind)}, predicted {guess}; "
                f"actually {outcome(query.survived)}")


    def prepare_exam(self, lesson: ClassLesson) -> exam.ClassificationExam:
        """Scores every test passenger once, before the video's first frame.

        Args:
            lesson: The lesson.

        Returns:
            The exam: the order passengers are judged in, the model's scores and its measured speed.
        """
        return exam.prepare_exam(lesson)

    def exam_label(self, lesson: ClassLesson, prepared: exam.ClassificationExam) -> str:
        """The exam chapter's header label.

        Args:
            lesson: The lesson.
            prepared: The exam (not needed for the label).

        Returns:
            How many unseen passengers are judged.
        """
        return f"Final exam · {len(lesson.data.y_test):,} unseen passengers"

    def exam_caption(self, lesson: ClassLesson, prepared: exam.ClassificationExam) -> str:
        """The exam chapter's caption: what is judged, and how a wrong call is marked.

        Args:
            lesson: The lesson.
            prepared: The exam (not needed for the caption).

        Returns:
            The caption.
        """
        return (f"Final exam: the trained model judges all {len(lesson.data.y_test):,} test passengers it has never "
                "seen. Wrong calls are circled on the maps and crossed in the grid, grouped by class of travel.")

    def draw_exam(self, axes: tuple[Axes, ...], panel: Axes, board: Axes, lesson: ClassLesson,
                  prepared: exam.ClassificationExam, t: float) -> None:
        """Draws the exam chapter: more test passengers judged on the maps, the grid and the scoreboard.

        Args:
            axes: The two maps, women's first.
            panel: The side panel, for the grid by class and sex.
            board: The strip below the maps, for the running scores.
            lesson: The lesson.
            prepared: The exam, from ``prepare_exam``.
            t: The chapter's progress, from 0 to 1.
        """
        shown = revealed(len(lesson.data.y_test), t)
        exam.draw_exam_maps(axes, lesson, prepared, shown)
        _legend(axes[1], lesson.kind)
        exam.draw_class_grid(panel, lesson, prepared, shown)
        exam.draw_board(board, lesson, prepared, shown)

    def compare(self, lesson: ClassLesson) -> Comparison:
        """The more-features chapter: the same algorithm refitted on every input, judged on the same test.

        Args:
            lesson: The lesson.

        Returns:
            The comparison of the baseline, the lesson's model and the all-inputs model.
        """
        return more_features.compare(lesson)

    def prepare_surface(self, lesson: ClassLesson) -> Surface:
        """The 3-D chapter: the final model's score as height over age and fare, one sheet per sex.

        Args:
            lesson: The lesson.

        Returns:
            The surface animkit draws.
        """
        return surface_view.prepare_surface(lesson)


CLASSIFICATION = ClassificationScene()
