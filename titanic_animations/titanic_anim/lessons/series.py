"""The Titanic playlist's introduction and conclusion, built from the real passengers and the nine lessons.

The introduction says who the series is for (not a first course in machine learning), shows the real
passengers, the nine algorithms, how every lesson runs and how to read the screen. The conclusion reports
what the lessons measured: every final model's test accuracy from age, fare and sex, then each algorithm
given every input, and three lessons those numbers support.

``playlist`` does the work once (loads the passengers, builds every lesson and its every-input
comparison); ``introduction`` and ``conclusion`` only arrange those numbers into cards and words.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from animkit.lessons import theme
from animkit.series.cards import Algorithm, Line, Swatch, paint_lineup, paint_swatches, paint_text
from animkit.series.chapters import Series, SeriesChapter, card
from animkit.series.charts import Bar, Cloud, paint_bars, paint_scatter, paint_shifts
from animkit.series.playlist import (audience_chapter, first_lesson_chapter, lesson_plan_chapter, links_chapter,
                                     real_chapter, thank_you_chapter, welcome_chapter)
from animkit.series.results import Result, result_of, scoreboard, shifts, spread, widest_gap
from titanic_anim.data import fetch_titanic
from titanic_anim.lessons.config import AGE, AGE_RANGE, LOG_FARE, SOURCE_NOTE
from titanic_anim.lessons.data import PassengerData, load_passengers
from titanic_anim.lessons.lesson import ClassLesson, accuracy
from titanic_anim.lessons.scene import CLASSIFICATION
from titanic_anim.lessons.stories import build_lesson, lesson_keys

SERIES_TITLE = "Titanic Survival"
KICKER = "MACHINE LEARNING, ANIMATED"
SAME = "About the same"  # how the every-input chapter starts a headline when McNemar's test finds no difference
LOG_FARE_MAX = math.log1p(520.0)  # the scatter's top: just above the highest fare, $512
FARE_TICKS = (0, 10, 50, 100, 500)
AGE_TICKS = (0, 20, 40, 60, 80)
TOP_FEW = 3  # how many leading models the scoreboard's verdict compares
STEP_NOTES = {"data": "the real passengers", "initial": "before any training", "training": "real snapshots only",
              "final": "train and validation accuracy", "inference": "one passenger it never saw",
              "exam": "every test passenger", "surface": "the chance of survival as height",
              "features": "the same algorithm, every input", "takeaway": "what to remember"}
SWATCHES = (
    Swatch(theme.MODEL, "Blue", "survived, and the model's side for survived"),
    Swatch(theme.ERROR, "Red", "did not survive, and the model's side for did not"),
    Swatch(theme.UNSEEN, "Orange", "passengers it never trained on: validation, test, the new one"),
    Swatch(theme.HIGHLIGHT, "Teal", "what the algorithm is looking at right now"),
)


@dataclass(frozen=True)
class Playlist:
    """Everything the two videos show, computed from the real passengers and the nine lessons.

    Attributes:
        algorithms: Each lesson's title and method, in playlist order.
        cloud: The training passengers, placed for the scatter.
        listed: Every passenger on the list, those without a known age included.
        train_rows: Training passengers.
        val_rows: Validation passengers.
        test_rows: Test passengers.
        shown_rows: How many training passengers the scatter shows.
        results: Each lesson's final accuracies and every-input comparison, in playlist order.
        baseline: The test accuracy of always guessing "did not survive".
    """

    algorithms: tuple[Algorithm, ...]
    cloud: Cloud
    listed: int
    train_rows: int
    val_rows: int
    test_rows: int
    shown_rows: int
    results: tuple[Result, ...]
    baseline: float

    @property
    def known(self) -> int:
        """Passengers with a known age: the ones the lessons use."""
        return self.train_rows + self.val_rows + self.test_rows


def _cloud(data: PassengerData) -> Cloud:
    """The training passengers: age across, fare up on a log scale; blue dots survived, red crosses did not."""
    rows = data.display
    return Cloud(x=tuple(float(np.clip(age / AGE_RANGE[1], 0, 1)) for age in data.x_train[rows, AGE]),
                 y=tuple(float(np.clip(fare / LOG_FARE_MAX, 0, 1)) for fare in data.x_train[rows, LOG_FARE]),
                 groups=tuple(int(label) for label in data.y_train[rows]), colours=(theme.ERROR, theme.MODEL),
                 legend=("did not survive", "survived"), markers=("X", "o"), x_label="age",
                 y_label="fare paid (log scale)",
                 x_ticks=tuple((age / AGE_RANGE[1], str(age)) for age in AGE_TICKS),
                 y_ticks=tuple((math.log1p(fare) / LOG_FARE_MAX, f"${fare}") for fare in FARE_TICKS))


def _result(lesson: ClassLesson) -> Result:
    """A lesson's final accuracies and its every-input comparison."""
    data, final = lesson.data, lesson.final
    return result_of(lesson.story.key, lesson.story.title, train=final.train_accuracy, validation=final.val_accuracy,
                     test=accuracy(lesson.score(data.x_test), data.y_test, lesson.kind), training_rows=data.n_train,
                     comparison=CLASSIFICATION.compare(lesson))


def playlist(fetcher: Callable = fetch_titanic) -> Playlist:
    """Loads the passengers, builds the nine lessons and their comparisons, and keeps what the videos show.

    Args:
        fetcher: Loads the Titanic passenger list.

    Returns:
        The playlist's numbers.
    """
    data = load_passengers(fetcher, None)
    lessons = tuple(build_lesson(key, data) for key in lesson_keys())
    known = data.n_train + len(data.y_val) + len(data.y_test)
    return Playlist(algorithms=tuple(Algorithm(lesson.story.title, lesson.story.method) for lesson in lessons),
                    cloud=_cloud(data), listed=known + data.dropped, train_rows=data.n_train,
                    val_rows=len(data.y_val), test_rows=len(data.y_test), shown_rows=len(data.display),
                    results=tuple(_result(lesson) for lesson in lessons),
                    baseline=float(np.mean(data.y_test == 0)))


def _pct(value: float) -> str:
    """A share as the lessons print it, for example "81%"."""
    return f"{value:.0%}"


# ---- the introduction ------------------------------------------------------------------------------

def _intro_chapters(playlist: Playlist) -> tuple[SeriesChapter, ...]:
    """The introduction's nine chapters, from the welcome slide to the hand-over."""
    titles = ", ".join(algorithm.title for algorithm in playlist.algorithms[:-1])
    pitch = ("Nine algorithms learn who survived the Titanic from each passenger's age, fare and sex, and you "
             "watch every one of them train, predict and get tested.")
    return (
        welcome_chapter(SERIES_TITLE),
        card("title", "Introduction", 8.0, f"In this playlist, {pitch[0].lower()}{pitch[1:]}", paint_text,
             kicker=f"{KICKER} · INTRODUCTION", headline=SERIES_TITLE, subhead=pitch, lines=(
                 Line("Nine lessons, one algorithm each, about two and a half minutes with the voice-over",
                      marker="▶"),
                 Line(f"Real data: the passenger list, {playlist.listed:,} passengers", marker="▶"),
                 Line("Every curve, number and prediction comes from the real trained model", marker="▶"))),
        audience_chapter(),
        real_chapter(),
        card("data", "The data", 13.0, f"The data: the Titanic's passenger list. The lessons use the "
             f"{playlist.known:,} passengers whose age is known. Each mark is one passenger, age across and fare "
             "up; blue dots survived and red crosses did not. Every lesson is tuned on the same validation "
             "passengers and tested on the same passengers it never saw.", paint_scatter, kicker="THE DATA",
             headline=f"{playlist.known:,} passengers with a known age", cloud=playlist.cloud, facts=(
                 f"Each mark is one of the {playlist.train_rows:,} training passengers: age across, fare paid up.",
                 f"Every lesson is tuned on the same {playlist.val_rows:,} validation passengers and tested on the "
                 f"same {playlist.test_rows:,} it never saw.",
                 f"{playlist.listed - playlist.known:,} passengers with no recorded age are left out.")),
        card("algorithms", "The nine algorithms", 15.0, f"Nine algorithms, one lesson each, in this order: "
             f"{titles}, and {playlist.algorithms[-1].title}.", paint_lineup, moving=0.75,
             kicker="NINE ALGORITHMS, ONE QUESTION", headline="Did this passenger survive?",
             algorithms=playlist.algorithms),
        lesson_plan_chapter(STEP_NOTES, "Every lesson runs the same ten chapters: the data, a first guess, real "
                            "training, the final model, a prediction for a passenger it never saw, a final exam on "
                            "every test passenger, a 3-D view of the model's chance of survival, the same algorithm "
                            "given every input, and the takeaway."),
        card("colours", "How to read it", 12.0, "How to read the screen: blue means survived, red means did not "
             "survive, orange is a passenger the model never trained on, and teal is what the algorithm is "
             "looking at right now. Every curve, number and prediction comes from the real trained model.",
             paint_swatches, moving=0.75, kicker="HOW TO READ THE SCREEN", headline="Every colour keeps one meaning",
             swatches=SWATCHES, rule="Every curve, number and prediction comes from the real trained model."),
        first_lesson_chapter(playlist.algorithms[0].title, playlist.algorithms[0].how, len(playlist.algorithms)),
    )


def introduction(playlist: Playlist) -> Series:
    """The playlist's introduction video."""
    return Series(name="introduction", title=f"{SERIES_TITLE}: Introduction", source_note=SOURCE_NOTE,
                  chapters=_intro_chapters(playlist))


# ---- the conclusion --------------------------------------------------------------------------------

def _top_gap(playlist: Playlist) -> int:
    """How many test passengers separate the best of the leading models from the last of them."""
    right = sorted((round(result.test * playlist.test_rows) for result in playlist.results), reverse=True)
    return right[0] - right[TOP_FEW - 1]


def _three_inputs(playlist: Playlist) -> SeriesChapter:
    """The scoreboard from age, fare and sex: every final model's test accuracy, and the baseline."""
    low, high = spread(playlist.results, "test")
    gap, base = _top_gap(playlist), _pct(playlist.baseline)
    verdict = (f"All nine score between {_pct(low)} and {_pct(high)}, against {base} for always guessing 'did not "
               f"survive'. One test passenger is {100 / playlist.test_rows:.1f} points, so the top {TOP_FEW} are "
               f"within {gap} passengers of each other.")
    bars = scoreboard(playlist.results, text=_pct, higher_is_better=True,
                      baseline=Bar("Always guess 'did not survive'", playlist.baseline, base, role="baseline"))
    return card("three_inputs", "Age, fare and sex", 17.0, f"From age, fare and sex, all nine algorithms call between "
                f"{_pct(low)} and {_pct(high)} of the {playlist.test_rows} test passengers right, against {base} for "
                "always guessing that a passenger did not survive. With so few test passengers, the top "
                f"{TOP_FEW} are within {gap} passengers of each other.", paint_bars, moving=0.8,
                kicker="SCOREBOARD · AGE, FARE AND SEX", headline="From three inputs, nine algorithms land close",
                metric=f"Share of the {playlist.test_rows} test passengers called right (test accuracy, higher is "
                "better)", bars=bars, verdict=verdict)


def _every_input(playlist: Playlist) -> SeriesChapter:
    """Each algorithm from age, fare and sex and from every input, on the same test passengers."""
    same = sum(result.verdict.startswith(SAME) for result in playlist.results)
    low, high = spread(playlist.results, "every_input")
    if same == len(playlist.results):
        verdict = ("For all nine, an exact McNemar test finds no difference it can confirm: on these test "
                   "passengers, age, fare and sex already do the work.")
    else:
        verdict = (f"For {same} of the nine, an exact McNemar test finds no difference it can confirm; for the "
                   f"other {len(playlist.results) - same}, it does.")
    return card("every_input", "Every input", 17.0, f"Given every input, the nine score between {_pct(low)} and "
                f"{_pct(high)}. {verdict}", paint_shifts, moving=0.8, kicker="SCOREBOARD · EVERY INPUT",
                headline="With every input, about the same", metric="Test accuracy from age, fare and sex (grey) "
                "and from every input (teal), same test passengers",
                shifts=shifts(playlist.results, higher_is_better=True), verdict=verdict)


def _learned(playlist: Playlist) -> SeriesChapter:
    """Three lessons the numbers support, each with its evidence."""
    overfit = widest_gap(playlist.results, higher_is_better=True)
    low, high = spread(playlist.results, "test")
    every_low, every_high = spread(playlist.results, "every_input")
    point = 100 / playlist.test_rows
    lines = (
        Line("Fitting the training passengers is not the goal.", marker="1", detail=(
            f"{overfit.title}: {_pct(overfit.train)} right on its training passengers, {_pct(overfit.validation)} "
            "on validation.")),
        Line("A few strong inputs carry most of the signal.", marker="2", detail=(
            f"Age, fare and sex: {_pct(low)} to {_pct(high)} right on test. Every input: {_pct(every_low)} to "
            f"{_pct(every_high)}.")),
        Line("A small test set blurs the ranking.", marker="3", detail=(
            f"{playlist.test_rows} test passengers: one passenger is {point:.1f} points, so close scores are not a "
            "real ranking.")),
    )
    return card("learned", "What we learned", 18.0, "Three lessons from nine algorithms. One: fitting the training "
                f"passengers is not the goal; {overfit.title} called {_pct(overfit.train)} of them right but only "
                f"{_pct(overfit.validation)} on validation. Two: a few strong inputs carry most of the signal. "
                f"Three: with only {playlist.test_rows} test passengers, close scores are not a real ranking.",
                paint_text, moving=0.8, kicker="WHAT WE LEARNED", headline="Three lessons from nine algorithms",
                lines=lines)


def conclusion(playlist: Playlist) -> Series:
    """The playlist's conclusion video."""
    return Series(name="conclusion", title=f"{SERIES_TITLE}: Conclusion", source_note=SOURCE_NOTE, chapters=(
        card("title", "Conclusion", 6.0, f"Nine algorithms, one question. Here is what they showed us about who "
             "survived the Titanic.", paint_text, kicker=f"{KICKER} · CONCLUSION",
             headline=f"{SERIES_TITLE}: what nine algorithms showed", lines=(
                 Line("How all nine scored from age, fare and sex", marker="▶"),
                 Line("What every input changed, algorithm by algorithm", marker="▶"),
                 Line("Three lessons the numbers support", marker="▶"))),
        _three_inputs(playlist), _every_input(playlist), _learned(playlist),
        links_chapter("California house prices (regression)", "All 76 videos, with live training dashboards and "
                      "algorithm races, are on the project site, and the code is open source. The same nine "
                      "algorithms also price California districts in the other playlist."),
        thank_you_chapter(SERIES_TITLE)))
