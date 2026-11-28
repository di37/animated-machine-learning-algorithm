"""The series' text cards: a headline with lines, the algorithm line-up, the road map and the colour key."""

import pytest

from animkit.lessons import theme
from animkit.lessons.config import RESOLUTIONS
from animkit.lessons.stage import Stage
from animkit.series.cards import (Algorithm, Line, Step, Swatch, paint_banner, paint_lineup, paint_roadmap,
                                  paint_swatches, paint_text)

LINES = (Line("For you if you already know training, validation and test data.", marker="✓"),
         Line("New to machine learning? Start with an introductory course.", marker="✗", colour=theme.ERROR,
              detail="Then come back."))
ALGORITHMS = tuple(Algorithm(f"Model {n}", f"learns in way {n}") for n in range(1, 10))
STEPS = (Step("Data", "the real rows"), Step("First guess", "before training"), Step("Training", "real steps"))
SWATCHES = (Swatch(theme.MODEL, "Blue", "the model"), Swatch(theme.UNSEEN, "Orange", "data it has never seen"))


@pytest.fixture
def stage():
    return Stage(dpi=RESOLUTIONS["preview"], source_note="Data: somewhere")


def opacity(text):
    return 1.0 if text.get_alpha() is None else text.get_alpha()


def shown(stage):
    """Every text on the card at full opacity."""
    return [text.get_text() for text in stage.card.texts if text.get_visible() and opacity(text) > 0.99]


def hidden(stage):
    """Every text on the card not yet faded in."""
    return [text.get_text() for text in stage.card.texts if opacity(text) < 0.01]


def test_a_text_card_shows_its_kicker_headline_lines_and_details(stage):
    paint_text(stage, 1.0, kicker="WHO IT IS FOR", headline="Not a first course.", lines=LINES)
    texts = " ".join(shown(stage))
    assert "WHO IT IS FOR" in texts and "Not a first course." in texts
    assert all(line.text.split()[0] in texts for line in LINES) and "Then come back." in texts
    assert "✓" in texts and "✗" in texts


def test_a_text_cards_lines_appear_one_after_another(stage):
    paint_text(stage, 0.0, kicker="K", headline="H", lines=LINES)
    assert any("For you" in text for text in hidden(stage))
    paint_text(stage, 1.0, kicker="K", headline="H", lines=LINES)
    assert not hidden(stage)


def test_a_long_line_is_wrapped_to_fit_the_card(stage):
    long = Line(" ".join(["word"] * 60))
    paint_text(stage, 1.0, kicker="K", headline="H", lines=(long,))
    assert any(text.count("\n") >= 1 for text in shown(stage))


def test_the_line_up_numbers_every_algorithm_in_playlist_order(stage):
    paint_lineup(stage, 1.0, kicker="NINE ALGORITHMS", headline="One question", algorithms=ALGORITHMS)
    texts = shown(stage)
    assert [text for text in texts if text.isdigit()] == [str(n) for n in range(1, 10)]
    assert "Model 9" in texts and any("learns in way 9" in text for text in texts)


def test_the_line_up_reveals_algorithms_in_turn(stage):
    paint_lineup(stage, 0.45, kicker="K", headline="H", algorithms=ALGORITHMS)
    assert "Model 1" in shown(stage) and "Model 9" in hidden(stage)


def test_the_road_map_lights_each_step_in_turn(stage):
    paint_roadmap(stage, 0.0, kicker="K", headline="H", steps=STEPS)
    lit_at_start = [patch for patch in stage.card.patches if patch.get_gid() == "lit"]
    paint_roadmap(stage, 1.0, kicker="K", headline="H", steps=STEPS, note="About 108 seconds.")
    lit_at_end = [patch for patch in stage.card.patches if patch.get_gid() == "lit"]
    assert len(lit_at_start) < len(lit_at_end) == len(STEPS)
    assert "About 108 seconds." in shown(stage) and "First guess" in shown(stage)


def test_the_colour_key_pairs_each_colour_with_its_meaning(stage):
    paint_swatches(stage, 1.0, kicker="HOW TO READ IT", headline="Colours", swatches=SWATCHES,
                   rule="Every number comes from the real model.")
    texts = shown(stage)
    assert "Blue" in texts and "the model" in texts and "Every number comes from the real model." in texts
    colours = {tuple(round(c, 3) for c in patch.get_facecolor()[:3]) for patch in stage.card.patches}
    from matplotlib.colors import to_rgb
    assert tuple(round(c, 3) for c in to_rgb(theme.UNSEEN)) in colours


def test_every_card_renders_a_frame(stage):
    for paint in (lambda: paint_text(stage, 0.5, kicker="K", headline="H", lines=LINES),
                  lambda: paint_lineup(stage, 0.5, kicker="K", headline="H", algorithms=ALGORITHMS),
                  lambda: paint_roadmap(stage, 0.5, kicker="K", headline="H", steps=STEPS),
                  lambda: paint_swatches(stage, 0.5, kicker="K", headline="H", swatches=SWATCHES)):
        paint()
        width, height = stage.size_px
        assert len(stage.render()) == width * height * 4


def test_more_algorithms_than_the_grid_holds_are_refused(stage):
    with pytest.raises(ValueError, match="9"):
        paint_lineup(stage, 1.0, kicker="K", headline="H", algorithms=ALGORITHMS + ALGORITHMS[:1])


def test_a_banner_centres_one_large_headline_over_its_lines(stage):
    paint_banner(stage, 1.0, kicker="A SERIES", headline="Welcome", subhead="Wine Quality", note="By someone")
    texts = {text.get_text(): text for text in stage.card.texts}
    assert {"A SERIES", "Welcome", "Wine Quality", "By someone"} <= set(shown(stage))
    assert all(text.get_ha() == "center" for text in texts.values())
    assert texts["Welcome"].get_fontsize() > 2 * texts["Wine Quality"].get_fontsize()


def test_a_banners_headline_comes_before_its_lines(stage):
    paint_banner(stage, 0.3, kicker="K", headline="Thank you", subhead="for watching")
    assert "for watching" in hidden(stage) and "Thank you" not in hidden(stage)
