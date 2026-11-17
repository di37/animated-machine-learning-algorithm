"""The regression picture: one income axis, the model's curve, residual lines and the $500k cap.

The director (animkit.lessons.director) is task-agnostic; everything it draws inside the workspace
comes from a scene like this one.

``RegressionScene`` implements animkit's ``Scene`` protocol for this project: for each chapter the
director asks it for a label, a caption or a drawing, and it answers from the ``Lesson`` alone. The
plotting itself lives in ``plots`` (training and inference), ``exam`` (the final exam), ``surface_view``
(the 3-D view) and ``more_features`` (the all-features comparison); this class is the switchboard.
``REGRESSION`` is the one instance every lesson uses.
"""

import math

from matplotlib.axes import Axes

from animkit.lessons.timing import phase
from animkit.lessons.exam import revealed
from animkit.lessons.lesson import Comparison
from animkit.lessons.surface import Surface
from house_anim.lessons import exam, more_features, plots, surface_view
from house_anim.lessons.config import SOURCE_NOTE
from house_anim.lessons.lesson import Explanation, Lesson, baseline_rmse_k, money_k, predict_one, rmse_k


class RegressionScene:
    """The regression drawing animkit's director delegates to: implements its ``Scene`` protocol.

    Attributes:
        subtitle: The line under every lesson's title.
        source_note: The dataset credit in the footer.
        facets: How many main plots sit side by side: one, the income axis.
    """

    subtitle = "Pricing California homes from one number: the district's median income"
    source_note = SOURCE_NOTE
    facets = 1  # one main plot

    def data_label(self, lesson: Lesson) -> str:
        """The data chapter's header label.

        Args:
            lesson: The lesson being played.

        Returns:
            How many districts the model trains on.
        """
        return f"{lesson.data.n_train:,} training districts"

    def data_caption(self, lesson: Lesson) -> str:
        """The data chapter's caption: what a dot is, and how many of the training districts are drawn.

        Args:
            lesson: The lesson being played.

        Returns:
            The caption, which says so when the lesson trains on a subsample.
        """
        data = lesson.data
        lead = "Each dot is one California district: median income (x) and median house value (y). "
        if data.sample_note:
            return f"{lead}This lesson is {data.sample_note}; {len(data.y_val):,} other districts judge it."
        return f"{lead}Showing {len(data.display):,} of the {data.n_train:,} districts the model trains on."

    def facts(self, lesson: Lesson) -> tuple[str, ...]:
        """The data chapter's side panel: the input, the target and the three splits.

        Args:
            lesson: The lesson being played.

        Returns:
            One line per fact, in the order shown.
        """
        data = lesson.data
        trained = f"{data.n_train:,} districts" + (", one per income band" if data.sample_note else "")
        return ("Input x: median household income",
                "Target y: median house value",
                f"Training: {trained}",
                f"Validation: {len(data.y_val):,} districts, to compare settings",
                f"Test: {len(data.y_test):,} districts, never touched until the end",
                "Split 60/20/20 with seed 42")

    def draw_data(self, axes: tuple[Axes, ...], lesson: Lesson, t: float) -> None:
        """Draws the training dots appearing, then the $500k cap line.

        Args:
            axes: The main plots; this scene uses the first.
            lesson: The lesson being played.
            t: How far through the chapter's motion, from 0 to 1: the dots are all in by 0.7, the cap at 0.75.
        """
        main, data = axes[0], lesson.data
        plots.setup_main(main)
        plots.draw_dots(main, data, count=math.ceil(len(data.display) * phase(t, 0.0, 0.7)))
        if t >= 0.75:
            plots.draw_cap(main)

    def draw_state(self, axes: tuple[Axes, ...], lesson: Lesson, index: int, *, ghosts: int,
                   grow: float, note: str | None) -> None:
        """Draws one training snapshot on the main plot: its curve, recent ghosts and residual lines.

        Args:
            axes: The main plots; this scene uses the first.
            lesson: The lesson being played.
            index: Which snapshot to draw.
            ghosts: How many earlier snapshots to draw as fading curves.
            grow: How far the residual lines have grown, from 0 to 1.
            note: The callout text; None shows the snapshot's own note.
        """
        plots.draw_training_state(axes[0], lesson, index, ghosts=ghosts, residual_grow=grow, note=note)

    def draw_metric(self, ax: Axes, lesson: Lesson, upto: int | None, mark_final: bool = False) -> None:
        """Draws the error chart: train and validation RMSE up to a snapshot, against the baseline.

        Args:
            ax: The metric strip.
            lesson: The lesson being played.
            upto: The last snapshot to plot; None draws the empty chart.
            mark_final: Whether to mark the final model's position with a line.
        """
        plots.draw_metric(ax, lesson, upto, mark_final=mark_final)

    def final_note(self, lesson: Lesson) -> str:
        """The final-model chapter's callout: which state was kept, and its errors.

        Args:
            lesson: The lesson being played.

        Returns:
            Two lines: the final snapshot's label, then its train and validation RMSE.
        """
        final = lesson.final
        return (f"Final model · {final.label}\n"
                f"train RMSE ${final.train_rmse_k:,.0f}k · validation ${final.val_rmse_k:,.0f}k")

    def inference_label(self, lesson: Lesson) -> str:
        """The prediction chapter's header label.

        Args:
            lesson: The lesson being played.

        Returns:
            The unseen district's row in the test split.
        """
        return f"Unseen test district #{lesson.data.query.row:,}"

    def draw_inference(self, axes: tuple[Axes, ...], lesson: Lesson, t: float) -> None:
        """Draws the final model's curve with the unseen district travelling to it and meeting its true price.

        Args:
            axes: The main plots; this scene uses the first.
            lesson: The lesson being played.
            t: How far through the chapter's motion, from 0 to 1.
        """
        main, data, query = axes[0], lesson.data, lesson.data.query
        plots.setup_main(main)
        plots.draw_cap(main)
        plots.draw_dots(main, data, alpha=0.3)
        lesson.panel.overlay_inference(main, query, phase(t, 0.45, 0.85))
        plots.draw_curve(main, data, lesson.final.curve)
        plots.draw_query(main, query, predict_one(lesson.predict, query.income), t)

    def explain(self, lesson: Lesson) -> Explanation:
        """The worked prediction for the unseen district.

        Args:
            lesson: The lesson being played.

        Returns:
            The lesson's own explanation at the query's income, with its check line.
        """
        return lesson.explain(lesson.data.query.income)

    def takeaway_facts(self, lesson: Lesson) -> tuple[str, ...]:
        """The takeaway card's facts: the test error against the baseline, and the unseen district's price.

        Args:
            lesson: The lesson being played.

        Returns:
            Two lines, computed from the final model on the test split.
        """
        data = lesson.data
        test = rmse_k(lesson.predict(data.x_test), data.y_test)
        baseline = baseline_rmse_k(data, data.y_test)
        prediction = predict_one(lesson.predict, data.query.income)
        return (f"Final exam on {len(data.y_test):,} unseen test districts: typical error ${test:,.0f}k "
                f"(always guessing the average, {money_k(data.baseline_mean)}: ${baseline:,.0f}k)",
                f"New district: predicted {money_k(prediction)}, actual {money_k(data.query.price)}")


    def prepare_exam(self, lesson: Lesson) -> exam.RegressionExam:
        """Prices every test district once, before the exam chapter is drawn.

        Args:
            lesson: The lesson being played.

        Returns:
            The scoring order, the predictions and the measured prediction speed.
        """
        return exam.prepare_exam(lesson)

    def exam_label(self, lesson: Lesson, prepared: exam.RegressionExam) -> str:
        """The exam chapter's header label.

        Args:
            lesson: The lesson being played.
            prepared: The prepared exam (unused: the label needs only the test split's size).

        Returns:
            How many unseen districts the exam scores.
        """
        return f"Final exam · {len(lesson.data.y_test):,} unseen districts"

    def exam_caption(self, lesson: Lesson, prepared: exam.RegressionExam) -> str:
        """The exam chapter's caption: what is being scored, and how to read the error map.

        Args:
            lesson: The lesson being played.
            prepared: The prepared exam (unused).

        Returns:
            The caption.
        """
        return (f"Final exam: the trained model prices all {len(lesson.data.y_test):,} test districts it has never "
                "seen, a batch at a time. The map shows where it misses: blue = worth more than predicted.")

    def draw_exam(self, axes: tuple[Axes, ...], panel: Axes, board: Axes, lesson: Lesson,
                  prepared: exam.RegressionExam, t: float) -> None:
        """Draws the exam so far: the scored districts, where the model misses, and the running scores.

        Args:
            axes: The main plots; this scene uses the first.
            panel: The side panel, for the error map.
            board: The metric strip, for the score board.
            lesson: The lesson being played.
            prepared: The prepared exam.
            t: How far through the chapter's motion, from 0 to 1; it sets how many districts are scored.
        """
        shown = revealed(len(lesson.data.y_test), t)
        exam.draw_exam_main(axes[0], lesson, prepared, shown)
        exam.draw_error_map(panel, lesson, prepared, shown)
        exam.draw_board(board, lesson, prepared, shown)

    def compare(self, lesson: Lesson) -> Comparison:
        """The more-features chapter: the lesson's model against the same algorithm given every feature.

        Args:
            lesson: The lesson being played.

        Returns:
            The comparison, scored on the test split.
        """
        return more_features.compare(lesson)

    def prepare_surface(self, lesson: Lesson) -> Surface:
        """The 3-D chapter: the same algorithm trained on income and house age, drawn as a surface.

        Args:
            lesson: The lesson being played.

        Returns:
            The surface, its notes and its scores.
        """
        return surface_view.prepare_surface(lesson)


REGRESSION = RegressionScene()
