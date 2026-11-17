"""Live inference dashboard: the passenger list fills in as test passengers are classified.

One figure per classifier, redrawn after every batch. On the left, the algorithm's diagram works
through the newest passenger. In the middle are the passenger list and, beside it, the running
accuracy and recall, the confusion matrix so far and a histogram of the scores. The equation card
on the right works the same passenger through the model's formula. A ticker under the header names
that passenger and says what the model predicted and what really happened.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from animkit import theme as base
from animkit.config import EQUATION_PANEL_RECT, INFERENCE_ALGORITHM_RECT, INFERENCE_FIGSIZE
from animkit.equation_panel import EquationPanel
from animkit.equations import EquationCard
from animkit.panels import ProgressBar, add_header
from titanic_anim.algorithms import algorithm_view
from titanic_anim.data import TitanicSplits
from titanic_anim.frames import InferenceFrame
from titanic_anim.manifest import PassengerManifest
from titanic_anim.metrics import score
from titanic_anim.models import SCORE_MARGIN, ClassifierSpec
from titanic_anim.panels import ConfusionMatrixPanel, RateLines, ScoreHistogram, percent
from titanic_anim.passengers import describe, outcome

SETTLED_PASSENGERS = 30  # rates over fewer passengers swing too much to set the axis


def format_rate(per_second: float) -> str:
    """A throughput in words.

    Args:
        per_second: Passengers classified per second.

    Returns:
        For example ``"12,345 passengers/s"``, or ``"—"`` when the rate is not finite (no time measured).
    """
    return "—" if not np.isfinite(per_second) else f"{per_second:,.0f} passengers/s"


class InferenceDashboard:
    """The inference figure of one classifier.

    Attributes:
        figure: The matplotlib figure, which animkit's sinks show and record.
        progress: The bar showing how many test passengers have been classified.
        status_text: The line under the header: running scores, latency and throughput.
        ticker_text: The line about the newest passenger.
        algorithm: The classifier's animated diagram.
        manifest: The passenger list.
        running: The running accuracy and recall.
        confusion: The confusion matrix of the test passengers so far.
        histogram: The scores, split by what really happened.
        equations: The equation card.
    """

    def __init__(self, spec: ClassifierSpec, splits: TitanicSplits, figsize=INFERENCE_FIGSIZE):
        """Builds the figure and every panel, empty until the first ``render``.

        Args:
            spec: The trained classifier.
            splits: The splits; the test passengers are the ones classified.
            figsize: The figure's size in inches.
        """
        plt.rcParams.update(base.RC_PARAMS)
        self._spec = spec
        self._splits = splits
        n_total = len(splits.y_test)
        self.figure = plt.figure(figsize=figsize)
        add_header(self.figure, f"{spec.title} · inference on unseen passengers",
                   f"Streaming {n_total:,} held-out test passengers through model.predict(); "
                   "each square fills in the moment the model decides.")
        self.progress = ProgressBar(self.figure)
        self.status_text = self.figure.text(0.04, 0.855, "", fontsize=base.FONT_SUBTITLE,
                                            color=base.INK_PRIMARY)
        self.ticker_text = self.figure.text(0.04, 0.825, "", fontsize=base.FONT_SUBTITLE,
                                            color=base.INK_SECONDARY)
        self.algorithm = algorithm_view(spec, self.figure.add_axes(INFERENCE_ALGORITHM_RECT), splits)
        grid = self.figure.add_gridspec(3, 2, left=0.315, right=0.705, top=0.76, bottom=0.07,
                                        width_ratios=(1.0, 1.05), hspace=0.75, wspace=0.26)
        self.manifest = PassengerManifest(self.figure.add_subplot(grid[:, 0]), splits.raw_test,
                                          spec.score_kind)
        self.running = self._build_running(self.figure.add_subplot(grid[0, 1]), n_total)
        self.confusion = ConfusionMatrixPanel(self.figure.add_subplot(grid[1, 1]))
        self.histogram = ScoreHistogram(self.figure.add_subplot(grid[2, 1]), spec.score_kind)
        self.equations = EquationPanel(self.figure.add_axes(EQUATION_PANEL_RECT))

    @staticmethod
    def _build_running(ax: Axes, n_total: int) -> RateLines:
        """The running-scores panel: accuracy and recall (survivors found) against passengers classified."""
        chart = RateLines(ax, ("accuracy", "survivors found"))
        ax.set_xlim(0, n_total)
        ax.set_xlabel("test passengers predicted so far")
        ax.set_title("Running test scores")
        return chart

    def render(self, frame: InferenceFrame, card: EquationCard | None = None, model=None,
               passenger: np.ndarray | None = None) -> None:
        """Draws one inference frame.

        Args:
            frame: The inference frame to draw.
            card: The equation card for the newest passenger; None leaves the card as it is.
            model: The model, to redraw the algorithm's diagram (with ``passenger``).
            passenger: The newest passenger's standardised features; the diagram is redrawn only when both
                ``model`` and ``passenger`` are given.
        """
        if card is not None:
            self.equations.update(card)
        if model is not None and passenger is not None:
            self.algorithm.show_prediction(model, passenger, frame.n_done)
        counts = np.array([p.n_done for p in frame.history])
        accuracy = np.array([p.accuracy for p in frame.history])
        recall = np.array([p.recall for p in frame.history])

        self.progress.update(frame.n_done / frame.n_total)
        self.manifest.update(frame)
        self.running.update(counts, (accuracy, recall), floor_from=SETTLED_PASSENGERS)
        self.confusion.update(score(frame.y_true, frame.y_pred), "Test so far")
        self.histogram.update(frame.y_true, frame.y_score, "How sure is the model?")
        latest = frame.scores
        self.status_text.set_text(
            f"{frame.n_done:,}/{frame.n_total:,} passengers predicted   ·   "
            f"accuracy {percent(latest.accuracy)}   ·   precision {percent(latest.precision)}   ·   "
            f"recall {percent(latest.recall)}   ·   last batch {frame.batch_latency_ms:.2f} ms   ·   "
            f"throughput {format_rate(frame.throughput_per_s)}"
        )
        self.ticker_text.set_text(self._ticker(frame))

    def _ticker(self, frame: InferenceFrame) -> str:
        """The newest passenger in words, with the model's score, its prediction and what really happened."""
        index = frame.n_done - 1
        who = describe(self._splits.raw_test[index], bool(self._splits.age_known_test[index]))
        value = frame.y_score[index]
        belief = f"f(x) = {value:+.2f}" if self._spec.score_kind == SCORE_MARGIN else f"P(survive) {value:.0%}"
        said, truth = int(frame.y_pred[index]), int(frame.y_true[index])
        mark = "✓" if said == truth else "✗"
        return (f"Latest: passenger #{frame.n_done:,} · {who}   →   {belief}   ·   "
                f"predicted: {outcome(said)}   ·   actually: {outcome(truth)} {mark}")
