"""The two evidence chapters: the final exam on every test example, and the all-features comparison."""

import pytest
from matplotlib.text import Text

from animkit.lessons.config import FIGSIZE, RESOLUTIONS
from animkit.lessons.director import paint_features, paint_title
from animkit.lessons.fit import overflowing_texts
from animkit.lessons.lesson import Comparison, ComparisonRow, Story
from animkit.lessons.stage import TRACKER, Stage, lesson_style, tracker_positions
from animkit.lessons.timing import CHAPTERS

TINY_DPI = RESOLUTIONS["preview"]


def rows() -> tuple[ComparisonRow, ...]:
    return (ComparisonRow("Always guess the average", 1.14, "$114k", "no inputs at all", "baseline"),
            ComparisonRow("This lesson's model", 0.84, "$84k", "median income only", "lesson"),
            ComparisonRow("Same algorithm, all 8 features", 0.51, "$51k", "income, age, rooms, location …", "richer"))


def comparison(**changes) -> Comparison:
    values = dict(metric="Test RMSE (lower is better)", higher_is_better=False, rows=rows(),
                  headline="More inputs cut the typical error from $84k to $51k.",
                  note="Same settings, same training and test districts.", takeaway="All 8 features: $51k.")
    return Comparison(**{**values, **changes})


class FakeLesson:
    story = Story(key="demo", title="Demo", method="m", objective="o", initial="i", training="t", final="f",
                  inference="n", takeaway="k", limitation="l", progress_label="p")


def test_exam_3d_view_and_features_come_after_the_prediction_and_before_the_takeaway():
    """One input, then two (the 3-D view), then every input (more features)."""
    assert CHAPTERS[-5:] == ("inference", "exam", "surface", "features", "takeaway")


def test_tracker_names_every_workspace_chapter_in_order_without_overlap():
    keys = [key for key, _ in TRACKER]
    assert keys == [c for c in CHAPTERS if c != "title"]
    assert dict(TRACKER)["exam"] == "Final exam" and dict(TRACKER)["features"] == "More features"
    xs = tracker_positions()
    assert list(xs) == sorted(xs) and xs[-1] < 0.96


def test_comparison_needs_a_lesson_row_and_a_richer_row():
    with pytest.raises(ValueError, match="lesson"):
        comparison(rows=rows()[:1] + rows()[2:])
    with pytest.raises(ValueError, match="role"):
        comparison(rows=(*rows()[:2], ComparisonRow("x", 1.0, "1", "d", "mystery")))


def test_comparison_reports_the_better_of_lesson_and_richer():
    assert comparison().improvement() == pytest.approx(0.84 - 0.51)
    accuracy = comparison(higher_is_better=True, rows=(
        ComparisonRow("guess", 0.59, "59%", "", "baseline"), ComparisonRow("lesson", 0.81, "81%", "", "lesson"),
        ComparisonRow("all", 0.84, "84%", "", "richer")))
    assert accuracy.improvement() == pytest.approx(0.03)


def test_features_card_shows_every_row_the_headline_and_the_note():
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_features(stage, FakeLesson(), comparison(), 1.0)
        stage.render()
    texts = "\n".join(t.get_text() for t in stage.fig.findobj(match=Text))
    for row in rows():
        assert row.label in texts and row.text in texts
    assert "More inputs cut the typical error" in texts and "Same settings" in texts
    assert "MORE FEATURES · DEMO" in texts


def test_features_card_reveals_rows_one_at_a_time():
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_features(stage, FakeLesson(), comparison(), 0.0)
    texts = "\n".join(t.get_text() for t in stage.fig.findobj(match=Text))
    assert rows()[0].label in texts and rows()[2].label not in texts


def test_speed_is_printed_in_a_readable_unit():
    from animkit.lessons.exam import Speed, duration, speed_line

    assert [duration(v) for v in (0.0451, 2.63, 1349.0, 0.9996)] == ["45 ns", "2.6 µs", "1.3 ms", "1 µs"]
    line = speed_line(Speed(single_us=41.0, batch_us=0.0013, batch_size=4128), "district")
    assert "41 µs for one district ·" in line and "1.3 ns each in a batch of 4,128" in line


def test_title_card_roadmap_lists_every_chapter():
    from animkit.lessons.director import ROADMAP

    assert "Final exam" in ROADMAP and "3-D view" in ROADMAP and "More features" in ROADMAP
    assert ROADMAP.endswith("Takeaway")


def test_paired_accuracy_needs_more_than_a_handful_of_passengers():
    import numpy as np

    from animkit.lessons.evidence import paired_accuracy

    lesson = np.array([True] * 160 + [False] * 50)
    richer = lesson.copy()
    richer[160:173] = True  # 13 gained
    richer[:5] = False  # 5 lost
    verdict = paired_accuracy(lesson, richer, "passenger")
    assert not verdict.better and not verdict.worse and "13 passengers gained and 5 lost" in verdict.evidence
    richer[173:200] = True  # 40 gained, 5 lost
    assert paired_accuracy(lesson, richer, "passenger").better


def test_paired_rmse_detects_a_real_gain_and_ignores_none():
    import numpy as np

    from animkit.lessons.evidence import paired_rmse

    errors = np.random.default_rng(0).normal(0, 1, 2000)
    show = lambda value: f"{value:.2f}"  # noqa: E731
    assert paired_rmse(errors, errors * 0.7, show, seed=1).better
    same = paired_rmse(errors, errors[::-1], show, seed=1)
    assert not same.better and not same.worse and "bootstrap" in same.evidence


def test_epoch_trainers_fit_fresh_every_time():
    import numpy as np
    from sklearn.linear_model import SGDClassifier

    from animkit.lessons.fitting import BestEpoch, EpochTrainer

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 3)), (rng.random(200) > 0.5).astype(int)
    for trainer in (EpochTrainer(SGDClassifier(random_state=0), 5, classes=np.array([0, 1])),
                    BestEpoch(SGDClassifier(random_state=0), 5, X, y, classes=np.array([0, 1]))):
        first = trainer.fit(X, y).predict(X)
        assert np.array_equal(first, trainer.fit(X, y).predict(X))
        assert not hasattr(trainer.model, "coef_")  # the constructor's model is never trained


def test_card_text_grows_with_the_canvas_so_a_card_keeps_its_look():
    with lesson_style():
        stage = Stage(dpi=TINY_DPI)
        paint_title(stage, FakeLesson(), 1.0)
    title = max(text.get_fontsize() for text in stage.card.texts)
    assert title / FIGSIZE[1] == pytest.approx(64 / 10.8)  # the size it was designed at, on a 10.8 in canvas
    assert overflowing_texts(stage.card) == ()
