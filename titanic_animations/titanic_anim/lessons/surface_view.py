"""The 3-D chapter for a Titanic lesson: the final model's score drawn as height over age and fare.

The model is the lesson's own final one, drawn as one sheet per sex. The sheets draw a 40 × 30 subset of
the maps' grid points; statistics in the notes are measured on the full grid.

``prepare_surface`` describes the chapter as an animkit ``Surface`` (the axes, the two sheets, the
training passengers as dots, the new passenger as a marker, and the notes); animkit's ``draw_surface``
draws it while the camera swings round.

Nothing is refitted. The lesson's model already reads age, fare and sex, and its final snapshot holds its
score on every cell of both maps, so the sheets are those maps lifted into the third dimension. A
probability is drawn on a 0–100% axis; a perceptron or SVM score is drawn on its own signed axis and is
never labelled a probability.
"""

import math

import numpy as np

from animkit.lessons import theme
from animkit.lessons.surface import Axis, Dots, Marker, Sheet, Surface
from titanic_anim.lessons import maps
from titanic_anim.lessons.config import AGE, AGE_RANGE, LOG_FARE, SEX_TITLES
from titanic_anim.lessons.lesson import PROBABILITY, ClassLesson, accuracy, threshold

AGE_POINTS, FARE_POINTS = 40, 30  # grid points drawn per sheet (the maps hold 120 × 90)
AGE_TICKS = (0.0, 20.0, 40.0, 60.0, 80.0)
FARE_TICKS = (0, 10, 25, 100, 500)
PROBABILITY_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)

SHAPES = {
    "logistic_regression": ("Logistic regression passes a straight-line score through the S-shaped sigmoid, so "
                            "each sheet is a smooth slope that never leaves the 0–100% range."),
    "perceptron": ("The perceptron's score is a flat, tilted plane for each sex: two parallel planes, offset by "
                   "the effect of sex."),
    "linear_svm": ("The linear SVM's score is a flat, tilted plane for each sex: two parallel planes, offset by "
                   "the effect of sex."),
    "naive_bayes": "Naive Bayes multiplies one bell curve per input, so each sheet is smooth and curved.",
    "decision_tree": ("A tree cuts the floor into rectangles with one flat height each, so the sheets are flat "
                      "terraces."),
    "random_forest": "A forest averages many trees' terraces, so each sheet is a staircase of many small steps.",
    "gradient_boosting": ("Boosting adds up many small trees' steps and turns the total into a probability, so "
                          "each sheet is made of flat blocks."),
    "knn": "kNN takes the vote of the nearest passengers, so each sheet is a patchwork of flat patches.",
    "mlp": ("The network's ReLU units crease its score and the output squashes it into a probability: smooth "
            "slopes with folds."),
}


def share_text(share: float) -> str:
    """'63% of', and never a rounded 'every' or 'no': 'over 99% of' unless it really is every cell.

    Args:
        share: A share of the grid's cells, from 0 to 1.

    Returns:
        The share in words, ready to go before 'the maps' full grid'.
    """
    if share >= 1.0:
        return "every cell of"
    if share <= 0.0:
        return "no cell of"
    if round(share, 2) >= 1.0:
        return "over 99% of"
    if round(share, 2) <= 0.0:
        return "under 1% of"
    return f"{share:.0%} of"


def grid_picks(available: int, wanted: int) -> np.ndarray:
    """``wanted`` evenly spread indices into a grid of ``available`` points, both ends included.

    Args:
        available: How many points the grid has along one axis.
        wanted: How many of them to draw.

    Returns:
        The picked indices, in order (all of them when ``wanted`` is at least ``available``).
    """
    return np.unique(np.linspace(0, available - 1, min(wanted, available)).round().astype(int))


def _score_text(value: float, kind: str) -> str:
    """A score as the notes and the marker write it: 'P(survive) 81%' or 'score +1.23'."""
    return f"P(survive) {value:.0%}" if kind == PROBABILITY else f"score {value:+.2f}"


def _height_axis(lesson: ClassLesson, sheets: np.ndarray) -> Axis:
    """The height axis: 0–100% for a probability; for a score, ± the whole number that holds sheets and query."""
    if lesson.kind == PROBABILITY:
        return Axis("P(survive)", (0.0, 1.0), PROBABILITY_TICKS, tuple(f"{t:.0%}" for t in PROBABILITY_TICKS))
    reach = max(float(np.max(np.abs(sheets))), abs(lesson.score_one(lesson.data.query.features)))
    limit = float(max(1, math.ceil(reach)))
    ticks = tuple(float(t) for t in np.linspace(-limit, limit, 5))
    return Axis("Score (not a probability)", (-limit, limit), ticks, tuple(f"{t:+g}" if t else "0" for t in ticks))


def _sheets(lesson: ClassLesson, ages: np.ndarray, fares: np.ndarray) -> np.ndarray:
    """The final snapshot's two maps at the picked grid points, shape (2, fares, ages)."""
    return np.stack([lesson.final.surface[sex][np.ix_(fares, ages)] for sex in (0, 1)])


def _dots(lesson: ClassLesson, z_axis: Axis) -> Dots:
    """The training passengers as dots: on the top face if they survived, on the bottom face if not."""
    data = lesson.data
    rows = data.display
    survived = data.y_train[rows] == 1
    low, high = z_axis.limits
    return Dots(data.x_train[rows, AGE], data.x_train[rows, LOG_FARE], np.where(survived, high, low),
                tuple(theme.MODEL if alive else theme.ERROR for alive in survived))


def _reading(lesson: ClassLesson, drawn: tuple[int, int]) -> tuple[str, ...]:
    """The 'Reading the 3-D view' notes; ``drawn`` is how many (ages, fares) the sheets show."""
    data, query = lesson.data, lesson.data.query
    women_higher = float(np.mean(lesson.final.surface[0] > lesson.final.surface[1]))
    height = ("P(survive)" if lesson.kind == PROBABILITY else
              "the model's signed score (a score, not a probability)")
    level = "50%" if lesson.kind == PROBABILITY else "score 0"
    sex = SEX_TITLES[maps.map_of(query.features)].lower()
    return ("This is the lesson's own final model, the one the maps showed: nothing is refitted.",
            f"Height is {height} over age and fare; one sheet for women, one for men.",
            f"The women's sheet is higher on {share_text(women_higher)} the maps' full {len(data.ages)} × "
            f"{len(data.log_fares)} grid; the sheets show {drawn[0]} × {drawn[1]} of its points.",
            f"The grey plane is the threshold ({level}): where a sheet rises above it, the model predicts "
            "'survived'.",
            f"Dots: the {data.n_train:,} training passengers, on the top face if they survived and the bottom "
            "face if they did not.",
            f"Orange: the new passenger from the Prediction chapter, on the {sex}'s sheet at "
            f"{_score_text(lesson.score_one(query.features), lesson.kind)}.")


def _scores(lesson: ClassLesson) -> tuple[str, ...]:
    """The 'How this model scores' notes: the same accuracies the other chapters showed."""
    data, final = lesson.data, lesson.final
    test = accuracy(lesson.score(data.x_test), data.y_test, lesson.kind)
    return ("Same model, so the same scores as the other chapters:",
            f"Training accuracy: {final.train_accuracy:.0%}",
            f"Validation accuracy: {final.val_accuracy:.0%}",
            f"Final exam on {len(data.y_test):,} test passengers: {test:.0%}",
            "Next: the same algorithm with every input.")


def _axes(lesson: ClassLesson) -> tuple[Axis, Axis]:
    """The age and fare axes; fare ticks sit at their log positions, labelled in dollars."""
    fares = lesson.data.log_fares
    age = Axis("Age (years)", AGE_RANGE, AGE_TICKS, tuple(f"{t:.0f}" for t in AGE_TICKS))
    fare = Axis("Fare paid (log scale)", (float(fares[0]), float(fares[-1])),
                tuple(float(np.log1p(t)) for t in FARE_TICKS), tuple(f"${t:,}" for t in FARE_TICKS))
    return age, fare


def prepare_surface(lesson: ClassLesson) -> Surface:
    """Describes the 3-D chapter from the lesson's final snapshot: nothing is refitted.

    Args:
        lesson: The lesson.

    Returns:
        The surface for animkit to draw: one sheet per sex, shaded as the maps are, the training
        passengers, the new passenger at their score, the threshold plane, and the notes.
    """
    data, query = lesson.data, lesson.data.query
    ages = grid_picks(len(data.ages), AGE_POINTS)
    fares = grid_picks(len(data.log_fares), FARE_POINTS)
    heights = _sheets(lesson, ages, fares)
    z_axis = _height_axis(lesson, heights)
    norm = maps.score_norm(lesson.kind)
    sheets = tuple(Sheet(heights[sex], maps.SCORE_CMAP(norm(heights[sex])), SEX_TITLES[sex]) for sex in (0, 1))
    score = lesson.score_one(query.features)
    x_axis, y_axis = _axes(lesson)
    height = "P(survive)" if lesson.kind == PROBABILITY else "score"
    return Surface(
        x=data.ages[ages], y=data.log_fares[fares], x_axis=x_axis, y_axis=y_axis, z_axis=z_axis, sheets=sheets,
        dots=_dots(lesson, z_axis), query=Marker(query.features[AGE], query.features[LOG_FARE], score,
                                                 _score_text(score, lesson.kind)),
        title="Age × fare, in 3-D",
        caption=f"{SHAPES[lesson.story.key]} Height is the model's {height}, one sheet per sex, seen by a swinging camera.",
        reading=_reading(lesson, (len(ages), len(fares))), scores=_scores(lesson),
        plane=threshold(lesson.kind), plane_label="50% threshold" if lesson.kind == PROBABILITY else "score 0 threshold")
