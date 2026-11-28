"""The house-price playlist's introduction and conclusion, built from the real data and the nine lessons.

The introduction says who the series is for (not a first course in machine learning), shows the real
districts, the nine algorithms, how every lesson runs and how to read the screen. The conclusion reports
what the lessons measured: every final model's test error with median income alone, then each algorithm
given all 8 inputs, and three lessons that those numbers support.

``playlist`` does the work once (loads the data, builds every lesson and its every-input comparison);
``introduction`` and ``conclusion`` only arrange those numbers into cards and words.
"""

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.datasets import fetch_california_housing

from animkit.lessons import theme
from animkit.series.cards import Algorithm, Line, Swatch, paint_lineup, paint_swatches, paint_text
from animkit.series.chapters import Series, SeriesChapter, card
from animkit.series.charts import Bar, Cloud, paint_bars, paint_scatter, paint_shifts
from animkit.series.playlist import (audience_chapter, first_lesson_chapter, lesson_plan_chapter, links_chapter,
                                     real_chapter, thank_you_chapter, welcome_chapter)
from animkit.series.results import Result, result_of, scoreboard, shifts, spread, widest_gap
from house_anim.lessons.cli import load_income_data
from house_anim.lessons.config import PRICE_TO_K
from house_anim.lessons.data import IncomeData
from house_anim.lessons.lesson import Lesson, baseline_rmse_k, rmse_k
from house_anim.lessons.regression_scene import REGRESSION
from house_anim.lessons.stories import build_lesson, lesson_keys

SERIES_TITLE = "California House Prices"
KICKER = "MACHINE LEARNING, ANIMATED"
SOURCE_NOTE = REGRESSION.source_note
INCOME_MAX, PRICE_MAX = 15.0, 5.0  # the scatter's ranges: median income ($10k) and house value ($100k)
INCOME_TICKS = ((0.0, "$0"), (5.0, "$50k"), (10.0, "$100k"), (15.0, "$150k"))
PRICE_TICKS = tuple((value, f"${value * PRICE_TO_K:.0f}k") for value in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
LINEAR_KEYS = ("linear_regression", "polynomial_regression", "ridge", "lasso")
GAIN = "Seven more inputs cut"  # how the every-input chapter starts a headline the bootstrap backs
STEP_NOTES = {"data": "the real districts", "initial": "before any training", "training": "real snapshots only",
              "final": "train and validation error", "inference": "one district it never saw",
              "exam": "every test district", "surface": "a second input as height",
              "features": "the same algorithm, all 8 inputs", "takeaway": "what to remember"}


@dataclass(frozen=True)
class Playlist:
    """Everything the two videos show, computed from the real data and the nine lessons.

    Attributes:
        algorithms: Each lesson's title and method, in playlist order.
        cloud: The districts the lessons show as dots, placed for the scatter.
        train_rows: Training districts.
        val_rows: Validation districts.
        test_rows: Test districts.
        shown_rows: How many training districts the scatter shows.
        results: Each lesson's final scores and every-input comparison, in playlist order.
        baseline: The test error of always guessing the average price, in $k.
    """

    algorithms: tuple[Algorithm, ...]
    cloud: Cloud
    train_rows: int
    val_rows: int
    test_rows: int
    shown_rows: int
    results: tuple[Result, ...]
    baseline: float

    @property
    def rows(self) -> int:
        """Every district."""
        return self.train_rows + self.val_rows + self.test_rows


def _cloud(data: IncomeData) -> Cloud:
    """The districts the lessons draw as dots: income across, house value up."""
    rows = data.display
    return Cloud(x=tuple(min(value / INCOME_MAX, 1.0) for value in data.x_train[rows]),
                 y=tuple(min(value / PRICE_MAX, 1.0) for value in data.y_train[rows]),
                 groups=tuple(0 for _ in rows), colours=(theme.DOTS,), legend=("one district",),
                 x_label="median income", y_label="median house value",
                 x_ticks=tuple((value / INCOME_MAX, label) for value, label in INCOME_TICKS),
                 y_ticks=tuple((value / PRICE_MAX, label) for value, label in PRICE_TICKS))


def _result(lesson: Lesson) -> Result:
    """A lesson's final errors ($k) and its every-input comparison."""
    data, final = lesson.data, lesson.final
    return result_of(lesson.story.key, lesson.story.title, train=final.train_rmse_k, validation=final.val_rmse_k,
                     test=rmse_k(lesson.predict(data.x_test), data.y_test), training_rows=data.n_train,
                     comparison=REGRESSION.compare(lesson))


def playlist(fetcher: Callable = fetch_california_housing) -> Playlist:
    """Loads the data, builds the nine lessons and their comparisons, and keeps what the videos show.

    Args:
        fetcher: Loads the California housing data.

    Returns:
        The playlist's numbers.
    """
    data = load_income_data(fetcher, None)
    lessons = tuple(build_lesson(key, data) for key in lesson_keys())
    return Playlist(algorithms=tuple(Algorithm(lesson.story.title, lesson.story.method) for lesson in lessons),
                    cloud=_cloud(data), train_rows=data.n_train, val_rows=len(data.y_val),
                    test_rows=len(data.y_test), shown_rows=len(data.display),
                    results=tuple(_result(lesson) for lesson in lessons),
                    baseline=baseline_rmse_k(data, data.y_test))


# ---- the introduction ------------------------------------------------------------------------------

SWATCHES = (
    Swatch(theme.MODEL, "Blue", "the model: its line, curve or prediction"),
    Swatch(theme.UNSEEN, "Orange", "districts it never trained on: validation, test, the new one"),
    Swatch(theme.HIGHLIGHT, "Teal", "what the algorithm is looking at right now"),
    Swatch(theme.ERROR, "Red", "errors: how far a prediction is from the truth"),
    Swatch(theme.DOTS, "Grey-blue", "the training districts"),
)


def _intro_chapters(playlist: Playlist) -> tuple[SeriesChapter, ...]:
    """The introduction's nine chapters, from the welcome slide to the hand-over."""
    first = playlist.algorithms[0].title
    titles = ", ".join(algorithm.title for algorithm in playlist.algorithms[:-1])
    return (
        welcome_chapter(SERIES_TITLE),
        card("title", "Introduction", 8.0, "In this playlist, nine algorithms learn to price a California "
             "district from its census numbers, and you watch every one of them train, predict and get tested.",
             paint_text, kicker=f"{KICKER} · INTRODUCTION",
             headline=SERIES_TITLE, subhead="Nine algorithms learn to price a California district from its census "
             "numbers, and you watch every one of them train, predict and get tested.", lines=(
                 Line("Nine lessons, one algorithm each, about two and a half minutes with the voice-over",
                      marker="▶"),
                 Line(f"Real data: {playlist.rows:,} California districts from the 1990 census", marker="▶"),
                 Line("Every curve, number and prediction comes from the real trained model", marker="▶"))),
        audience_chapter(),
        real_chapter(),
        card("data", "The data", 13.0, f"The data: {playlist.rows:,} California districts from the 1990 census. "
             "Each dot is one district, its median income across and its median house value up. Every lesson "
             "is tuned on the same validation districts and tested on the same districts it never saw.",
             paint_scatter, kicker="THE DATA", headline=f"{playlist.rows:,} California districts",
             cloud=playlist.cloud, facts=(
                 f"Each dot is one district of the 1990 census ({playlist.shown_rows:,} of the "
                 f"{playlist.train_rows:,} training districts).",
                 f"Every lesson is tuned on the same {playlist.val_rows:,} validation districts and tested on "
                 f"the same {playlist.test_rows:,} it never saw.",
                 "Values above $500k were recorded as $500k, so the top edge is flat.")),
        card("algorithms", "The nine algorithms", 15.0, f"Nine algorithms, one lesson each, in this order: "
             f"{titles}, and {playlist.algorithms[-1].title}.", paint_lineup, moving=0.75,
             kicker="NINE ALGORITHMS, ONE QUESTION", headline="What is this district's median house value?",
             algorithms=playlist.algorithms),
        lesson_plan_chapter(STEP_NOTES, "Every lesson runs the same ten chapters: the data, a first guess, real "
                            "training, the final model, a prediction for a district it never saw, a final exam on "
                            "every test district, a 3-D view with a second input, the same algorithm given every "
                            "input, and the takeaway."),
        card("colours", "How to read it", 13.0, "How to read the screen: blue is the model, orange is a district "
             "it never trained on, teal is what the algorithm is looking at right now, red is error, and the "
             "grey-blue dots are the training districts. Every curve, number and prediction comes from the "
             "real trained model.", paint_swatches, moving=0.75, kicker="HOW TO READ THE SCREEN",
             headline="Every colour keeps one meaning", swatches=SWATCHES,
             rule="Every curve, number and prediction comes from the real trained model."),
        first_lesson_chapter(first, playlist.algorithms[0].how, len(playlist.algorithms)),
    )


def introduction(playlist: Playlist) -> Series:
    """The playlist's introduction video."""
    return Series(name="introduction", title=f"{SERIES_TITLE}: Introduction", source_note=SOURCE_NOTE,
                  chapters=_intro_chapters(playlist))


# ---- the conclusion --------------------------------------------------------------------------------

def _k(value: float) -> str:
    """A $k amount as the lessons print it, for example "$84k"."""
    return f"${value:.0f}k"


def _one_input(playlist: Playlist) -> SeriesChapter:
    """The scoreboard with median income alone: every final model's test error, and the baseline."""
    low, high = spread(playlist.results, "test")
    verdict = (f"All nine land between {_k(low)} and {_k(high)}, against {_k(playlist.baseline)} for always "
               "guessing the average price: with income alone, the input limits them more than the algorithm does.")
    bars = scoreboard(playlist.results, text=_k, higher_is_better=False,
                      baseline=Bar("Always guess the average price", playlist.baseline, _k(playlist.baseline),
                                   role="baseline"))
    return card("one_input", "One input", 17.0, f"With median income alone, all nine algorithms land between "
                f"{_k(low)} and {_k(high)} of typical error on the {playlist.test_rows:,} test districts, against "
                f"{_k(playlist.baseline)} for always guessing the average price. With one input, the input limits "
                "them more than the algorithm does.", paint_bars, moving=0.8, kicker="SCOREBOARD · MEDIAN INCOME ONLY",
                headline="With one input, nine algorithms tie", metric=f"Typical error on the {playlist.test_rows:,} "
                "test districts (test RMSE, lower is better)", bars=bars, verdict=verdict)


def _every_input(playlist: Playlist) -> SeriesChapter:
    """Each algorithm with income alone and with all 8 inputs, on the same test districts."""
    best = sorted(playlist.results, key=lambda result: result.every_input)[:3]
    linear = sorted({result.every_input_text for result in playlist.results if result.key in LINEAR_KEYS})
    gains = sum(result.verdict.startswith(GAIN) for result in playlist.results)
    leaders = (f"{best[0].title} reaches {best[0].every_input_text}, {best[1].title} {best[1].every_input_text} and "
               f"{best[2].title} {best[2].every_input_text}")
    linear_text = f"the linear models stop at {' to '.join(linear)}"
    evidence = ("a paired bootstrap backs the gain for all nine" if gains == len(playlist.results)
                else f"a paired bootstrap backs the gain for {gains} of the nine")
    return card("every_input", "Every input", 17.0, f"Given all 8 inputs, the flexible models pull ahead: "
                f"{leaders}, while {linear_text}.", paint_shifts, moving=0.8, kicker="SCOREBOARD · ALL 8 INPUTS",
                headline="With every input, flexible models pull ahead",
                metric="Test RMSE with median income only (grey) and with all 8 inputs (teal), same test districts",
                shifts=shifts(playlist.results, higher_is_better=False),
                verdict=f"{leaders}; {linear_text}. On these test districts, {evidence}.")


def _learned(playlist: Playlist) -> SeriesChapter:
    """Three lessons the numbers support, each with its evidence."""
    overfit = widest_gap(playlist.results, higher_is_better=False)
    low, high = spread(playlist.results, "every_input")
    one_low, one_high = spread(playlist.results, "test")
    lines = (
        Line("Fitting the training data is not the goal.", marker="1", detail=(
            f"{overfit.title} on {overfit.training_rows} districts: {_k(overfit.train)} error on its training "
            f"districts, {_k(overfit.validation)} on validation.")),
        Line("Better inputs beat a cleverer algorithm.", marker="2", detail=(
            f"Median income alone: {_k(one_low)} to {_k(one_high)} for every model. All 8 inputs: {_k(low)} to "
            f"{_k(high)}.")),
        Line("Choose on validation, score on test.", marker="3", detail=(
            "Degrees, penalties and k were chosen on validation; the test districts only ever scored finished "
            "models.")),
    )
    return card("learned", "What we learned", 18.0, "Three lessons from nine algorithms. One: fitting the "
                f"training data is not the goal; {overfit.title} scored {_k(overfit.train)} on its training "
                f"districts but {_k(overfit.validation)} on validation. Two: better inputs beat a cleverer "
                "algorithm. Three: choose settings on validation, and keep the test districts for scoring "
                "finished models.", paint_text, moving=0.8, kicker="WHAT WE LEARNED",
                headline="Three lessons from nine algorithms", lines=lines)


def _keep_going() -> SeriesChapter:
    """Where to go next."""
    return links_chapter("who survived the Titanic (classification)", "All 76 videos, with live training "
                         "dashboards and algorithm races, are on the project site, and the code is open source. "
                         "Next: the same nine algorithms decide who survived the Titanic.")


def conclusion(playlist: Playlist) -> Series:
    """The playlist's conclusion video."""
    return Series(name="conclusion", title=f"{SERIES_TITLE}: Conclusion", source_note=SOURCE_NOTE, chapters=(
        card("title", "Conclusion", 6.0, f"Nine algorithms, one question. Here is what they showed us about "
             f"{SERIES_TITLE.lower()}.", paint_text, kicker=f"{KICKER} · CONCLUSION",
             headline=f"{SERIES_TITLE}: what nine algorithms showed", lines=(
                 Line("How all nine scored with median income alone", marker="▶"),
                 Line("What all 8 inputs changed, algorithm by algorithm", marker="▶"),
                 Line("Three lessons the numbers support", marker="▶"))),
        _one_input(playlist), _every_input(playlist), _learned(playlist), _keep_going(),
        thank_you_chapter(SERIES_TITLE)))
