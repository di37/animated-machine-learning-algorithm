"""The series' data cards: the real rows as a scatter, a scoreboard, and each model before and after."""

import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from animkit.lessons import theme
from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.stage import Stage
from animkit.series.charts import Bar, Cloud, Shift, paint_bars, paint_scatter, paint_shifts

CLOUD = Cloud(x=(0.1, 0.5, 0.9, 0.3), y=(0.2, 0.4, 0.8, 0.6), groups=(0, 1, 0, 1),
              colours=(theme.DOTS, theme.MODEL), legend=("did not", "did"), x_label="age", y_label="fare",
              x_ticks=((0.0, "0"), (1.0, "80")), y_ticks=((0.0, "$5"), (1.0, "$500")))
BARS = (Bar("Model A", 80.0, "$80k"), Bar("Model B", 40.0, "$40k"), Bar("Always the average", 100.0, "$100k",
                                                                     role="baseline"))
SHIFTS = (Shift("Model A", 84.0, 75.0, "$84k", "$75k"), Shift("Model B", 83.0, 55.0, "$83k", "$55k"))


@pytest.fixture
def stage():
    return Stage(dpi=RESOLUTIONS["preview"], source_note="Data: somewhere")


def opacity(text):
    return 1.0 if text.get_alpha() is None else text.get_alpha()


def shown(stage):
    return [text.get_text() for text in stage.card.texts if opacity(text) > 0.99]


def points(stage):
    return sum(len(collection.get_offsets()) for collection in stage.card.collections
               if isinstance(collection, PathCollection) and collection.get_gid() == "rows")


def test_the_scatter_reveals_every_real_row_by_the_end(stage):
    paint_scatter(stage, 0.1, kicker="THE DATA", headline="H", cloud=CLOUD, facts=("1,309 rows",))
    early = points(stage)
    paint_scatter(stage, 1.0, kicker="THE DATA", headline="H", cloud=CLOUD, facts=("1,309 rows",))
    assert early < points(stage) == len(CLOUD.x)
    texts = shown(stage)
    assert {"age", "fare", "$500", "did not", "did"} <= set(texts) and "1,309 rows" in texts


@pytest.mark.parametrize("bad", [{"x": (0.1,)}, {"y": (0.2, 0.4, 1.2, 0.6)}, {"groups": (0, 1, 2, 1)},
                                 {"markers": ("o",)}])
def test_a_cloud_that_does_not_fit_its_box_is_refused(bad):
    fields = {**CLOUD.__dict__, **bad}
    with pytest.raises(ValueError):
        Cloud(**fields)


def test_each_group_keeps_its_own_marker(stage):
    cloud = Cloud(**{**CLOUD.__dict__, "markers": ("X", "o")})
    assert (cloud.marker(0), cloud.marker(1), CLOUD.marker(0)) == ("X", "o", "o")
    paint_scatter(stage, 1.0, kicker="K", headline="H", cloud=cloud, facts=())
    assert points(stage) == len(cloud.x)


def bars_drawn(stage):
    return [patch for patch in stage.card.patches if isinstance(patch, Rectangle) and patch.get_gid() == "bar"]


def test_bars_appear_in_turn_at_their_real_length(stage):
    paint_bars(stage, 0.4, kicker="SCOREBOARD", headline="H", metric="Test error", bars=BARS, verdict="A tie.")
    assert 0 < len(bars_drawn(stage)) < len(BARS)
    paint_bars(stage, 1.0, kicker="SCOREBOARD", headline="H", metric="Test error", bars=BARS, verdict="A tie.")
    widths = [patch.get_width() for patch in bars_drawn(stage)]
    assert widths[0] / widths[1] == pytest.approx(80.0 / 40.0)
    assert "A tie." in shown(stage) and "$100k" in shown(stage)


def test_the_baseline_bar_is_grey_and_the_models_blue(stage):
    paint_bars(stage, 1.0, kicker="K", headline="H", metric="m", bars=BARS, verdict="v")
    colours = [to_rgb(patch.get_facecolor()) for patch in bars_drawn(stage)]
    assert colours[0] == pytest.approx(to_rgb(theme.MODEL)) and colours[2] == pytest.approx(to_rgb(theme.DOTS))


def dots(stage, gid):
    return sum(len(c.get_offsets()) for c in stage.card.collections if c.get_gid() == gid)


def test_each_model_shows_its_one_input_score_before_its_every_input_score(stage):
    paint_shifts(stage, 0.5, kicker="EVERY INPUT", headline="H", metric="Test error", shifts=SHIFTS, verdict="v")
    assert dots(stage, "before") == len(SHIFTS) and dots(stage, "after") == 0
    paint_shifts(stage, 1.0, kicker="EVERY INPUT", headline="H", metric="Test error", shifts=SHIFTS, verdict="v")
    assert dots(stage, "after") == len(SHIFTS)
    assert {"$84k", "$55k", "v"} <= set(shown(stage))


def test_a_lower_score_sits_further_left(stage):
    paint_shifts(stage, 1.0, kicker="K", headline="H", metric="m", shifts=SHIFTS, verdict="v")
    before = [c for c in stage.card.collections if c.get_gid() == "before"]
    after = [c for c in stage.card.collections if c.get_gid() == "after"]
    assert after[1].get_offsets()[0][0] < after[0].get_offsets()[0][0] < before[0].get_offsets()[0][0]


def test_every_chart_renders_a_frame(stage):
    for paint in (lambda: paint_scatter(stage, 0.5, kicker="K", headline="H", cloud=CLOUD, facts=()),
                  lambda: paint_bars(stage, 0.5, kicker="K", headline="H", metric="m", bars=BARS, verdict="v"),
                  lambda: paint_shifts(stage, 0.5, kicker="K", headline="H", metric="m", shifts=SHIFTS,
                                       verdict="v")):
        paint()
        width, height = stage.size_px
        assert len(stage.render()) == width * height * 4
