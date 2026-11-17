import matplotlib.pyplot as plt
import pytest

from animkit.equation_panel import EquationPanel
from animkit.equations import HEADING, MATH, MONO, NOTE, EquationCard, EquationLine


def make_panel(max_lines=6):
    figure = plt.figure(figsize=(5, 7))
    return EquationPanel(figure.add_axes((0, 0, 1, 1)), max_lines=max_lines)


def card(n_lines):
    kinds = (HEADING, MATH, MONO, NOTE)
    return EquationCard("Title", tuple(EquationLine(f"line {i}", kinds[i % 4]) for i in range(n_lines)))


def test_lines_fill_from_the_top_and_extras_are_hidden():
    panel = make_panel()
    panel.update(card(3))
    shown = [t for t in panel.texts if t.get_visible()]
    assert [t.get_text() for t in shown] == ["line 0", "line 1", "line 2"]
    ys = [t.get_position()[1] for t in shown]
    assert ys == sorted(ys, reverse=True)
    assert panel.title.get_text() == "Title"


def test_long_cards_are_truncated_with_an_ellipsis():
    panel = make_panel(max_lines=4)
    panel.update(card(10))
    visible = [t.get_text() for t in panel.texts if t.get_visible()]
    assert len(visible) == 4 and visible[-1] == "…"


def test_shorter_card_after_longer_one_hides_leftovers():
    panel = make_panel()
    panel.update(card(6))
    panel.update(card(2))
    assert sum(t.get_visible() for t in panel.texts) == 2


def test_kinds_get_distinct_styles():
    panel = make_panel()
    panel.update(card(4))
    heading, math, mono, _ = panel.texts[:4]
    assert heading.get_fontweight() == "bold"
    assert math.get_fontsize() > mono.get_fontsize()
    assert "monospace" in mono.get_fontfamily()


def test_tall_math_gets_more_room_than_inline_math():
    panel = make_panel()
    tall = EquationLine(r"$F = \sum_{m=1}^{M} h_m$", MATH)
    flat = EquationLine(r"$r_i = y_i - F$", MATH)
    panel.update(EquationCard("t", (tall, flat, flat)))
    y0, y1, y2 = (t.get_position()[1] for t in panel.texts[:3])
    assert (y0 - y1) > (y1 - y2)


def gap_after_first(panel, first: EquationLine) -> float:
    panel.update(EquationCard("t", (first, EquationLine("next", NOTE))))
    return panel.texts[0].get_position()[1] - panel.texts[1].get_position()[1]


def test_long_notes_wrap_onto_extra_lines():
    panel = make_panel()
    single = gap_after_first(panel, EquationLine("short", NOTE))
    wrapped = gap_after_first(panel, EquationLine("word " * 40, NOTE))
    assert "\n" in panel.texts[0].get_text()
    assert wrapped > 1.8 * single


def test_spacing_is_fixed_in_points_whatever_the_panel_height():
    short = plt.figure(figsize=(5, 4))
    tall = plt.figure(figsize=(5, 12))
    gaps = []
    for figure in (short, tall):
        panel = EquationPanel(figure.add_axes((0, 0, 1, 1)))
        gap = gap_after_first(panel, EquationLine("one", NOTE))
        gaps.append(gap * figure.get_size_inches()[1])  # back to inches
    assert gaps[0] == pytest.approx(gaps[1])
