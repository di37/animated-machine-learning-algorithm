"""Algorithm diagrams draw what they are given, and the side panel makes room for them."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.text import Text
from sklearn.tree import DecisionTreeRegressor

from animkit.lessons import diagrams, theme
from animkit.lessons.stage import PANEL_RECT, SHARED_PANEL_RECT, Stage


def texts(ax) -> str:
    return "\n".join(t.get_text() for t in ax.findobj(match=Text))


@pytest.fixture
def ax():
    figure = plt.figure(figsize=(6, 2))
    return figure.add_subplot()


@pytest.fixture(scope="module")
def trees():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=200)
    return [DecisionTreeRegressor(max_depth=3, random_state=seed).fit(X[seed:], y[seed:]).tree_ for seed in range(15)]


def colour(value):
    return (0.2, 0.4, 0.8, 1.0)


def test_stage_shares_the_panel_only_when_asked():
    stage = Stage(dpi=20.0)
    stage.share_panel(True)
    assert stage.diagram.get_visible()
    assert stage.panel.get_position().bounds == pytest.approx(SHARED_PANEL_RECT)
    stage.share_panel(False)
    assert not stage.diagram.get_visible()
    assert stage.panel.get_position().bounds == pytest.approx(PANEL_RECT)


def test_the_diagram_is_the_biggest_box_in_the_shared_column():
    stage = Stage(dpi=20.0)
    stage.share_panel(True)
    diagram, panel, metric = (ax.get_position() for ax in (stage.diagram, stage.panel, stage.metric))
    assert diagram.height >= 1.6 * panel.height and diagram.height >= 1.6 * metric.height
    assert diagram.width == panel.width == metric.width
    assert diagram.y0 > panel.y1 and panel.y0 > metric.y1  # stacked top to bottom, never overlapping


def _few_inputs(ax, trees):
    diagrams.weighted_sum(ax, ["age", "fare", "woman"], np.array([-0.2, 0.6, 1.0]), "+ bias −0.44",
                          ["σ", "P(survive)"], title="How it decides", fmt=lambda w: f"{w:+.2f}")


def _voting_grid(ax, trees):
    diagrams.tree_grid(ax, trees[:10], colour, title="Votes", total=15, summary="average\n81%",
                       votes=["81%"] * 10)


def _boosting_chain(ax, trees):
    diagrams.chain(ax, trees[:9], colour, title="Chain", start="F₀\n-0.37", end="σ(F)\n70%", step_label="0.1 ×",
                   total=9, earlier="\n+1.24", notes=["+0.001"] * 4)


@pytest.mark.parametrize("draw", [_few_inputs, _voting_grid, _boosting_chain])
def test_diagram_text_reads_at_least_as_large_as_the_tick_labels(draw, trees):
    """Helpers whose text never has to shrink to fit (many-input rows and box rows can)."""
    stage = Stage(dpi=20.0)
    stage.share_panel(True)
    draw(stage.diagram, trees)
    title, *body = stage.diagram.texts
    assert title.get_fontsize() == theme.PANEL_TITLE_SIZE
    assert min(text.get_fontsize() for text in body) >= theme.TICK_SIZE


@pytest.mark.parametrize("draw", [_voting_grid, _boosting_chain])
def test_tree_thumbnails_nodes_grow_with_the_diagram(draw, trees):
    stage = Stage(dpi=20.0)
    stage.share_panel(True)
    draw(stage.diagram, trees)
    nodes = [size for collection in stage.diagram.collections if isinstance(collection, PathCollection)
             for size in collection.get_sizes()]
    assert max(nodes) > diagrams.TEXT_SCALE ** 2 * 9.0  # the roots, larger than the smallest-node floor


def test_weighted_sum_labels_inputs_weights_and_outputs(ax):
    diagrams.weighted_sum(ax, ["age", "fare", "woman"], np.array([-0.2, 0.6, 1.0]), "+ bias −0.44",
                          ["σ", "P(survive)"], title="How it decides", fmt=lambda w: f"{w:+.2f}")
    shown = texts(ax)
    for word in ("age", "fare", "woman", "+0.60", "−0.44", "σ", "P(survive)", "Σ", "How it decides"):
        assert word in shown


def test_weighted_sum_shows_pushes_values_and_result_at_inference(ax):
    diagrams.weighted_sum(ax, ["age", "fare"], np.array([-0.2, 0.6]), "+ bias", ["σ", "P"], title="t",
                          fmt=lambda w: f"{w:+.2f}", values=["z −0.15", "z −0.43"],
                          pushes=np.array([0.03, -0.26]), result="67%", reveal=1.0)
    shown = texts(ax)
    assert "+0.03" in shown and "z −0.43" in shown and "67%" in shown


def test_switched_off_and_missing_terms_are_drawn_differently(ax):
    diagrams.weighted_sum(ax, ["x", "x²", "x³"], np.array([0.5, 0.0, np.nan]), "+ b", ["ŷ"], title="t",
                          fmt=lambda w: f"{w:+.1f}")
    styles = [line.get_linestyle() for line in ax.get_lines()]
    assert "--" in styles and styles.count("-") == 2


def test_tree_grid_draws_at_most_ten_and_counts_the_rest(ax, trees):
    drawn = diagrams.tree_grid(ax, trees, colour, title="The forest", total=50, summary="average")
    assert drawn == 10 and "+40 more" in texts(ax) and "average" in texts(ax)


def test_chain_shows_the_newest_trees(ax, trees, monkeypatch):
    drawn_trees = []
    monkeypatch.setattr(diagrams, "draw_tree", lambda ax, tree, *args, **kwargs: drawn_trees.append(tree))
    drawn = diagrams.chain(ax, trees[:6], colour, title="Boosting", start="F₀", end="F₆", step_label="0.1 ×",
                           total=6, notes=["+1", "+2", "+3", "+4"])
    assert drawn == 4 and "… 2 earlier" in texts(ax) and "F₆" in texts(ax) and "+4" in texts(ax)
    assert drawn_trees == trees[2:6]


def test_flow_reveals_values_in_order(ax):
    diagrams.flow(ax, ["stored", "nearest", "average"], title="kNN", values=["a", "b", "c"], reveal=0.5)
    shown = texts(ax)
    assert "a" in shown.split("\n") and "b" in shown.split("\n") and "c" not in shown.split("\n")


def test_factor_rows_multiply_then_normalise(ax):
    diagrams.factor_rows(ax, [("survived", ["41%", "p(age)"]), ("did not", ["59%", "p(age)"])],
                         title="Bayes", merge="normalise", result="82%")
    shown = texts(ax)
    assert "×" in shown and "normalise" in shown and "82%" in shown


def test_factor_rows_join_logs_with_plus(ax):
    diagrams.factor_rows(ax, [("survived", ["ln prior\n-0.9", "age\n-3.9"]), ("did not", ["ln prior\n-0.5", "age\n-3.8"])],
                         title="Bayes", merge="normalise", joiner="+")
    shown = texts(ax).split("\n")
    assert "+" in shown and "×" not in shown


def test_chain_summarises_the_trees_it_does_not_draw(ax, trees):
    diagrams.chain(ax, trees, colour, title="Boosting", start="F₀", end="F", step_label="0.1 ×", total=15,
                   earlier="\n+$102k")
    assert "… 11 earlier\n+$102k" in texts(ax)


def test_tree_grid_says_how_many_trees_are_not_drawn(ax, trees):
    diagrams.tree_grid(ax, trees, colour, title="t", total=15, summary="average")
    assert "+5 more trees" in texts(ax)
    diagrams.tree_grid(ax, trees[:11], colour, title="t", total=11, summary="average")
    assert "+1 more tree" in texts(ax).split("\n")


def test_long_titles_shrink_to_fit_and_short_ones_do_not():
    stage = Stage(dpi=20.0)
    stage.share_panel(True)
    diagrams.frame(stage.diagram, "short")
    assert stage.diagram.texts[0].get_fontsize() == diagrams.TITLE_SIZE
    diagrams.frame(stage.diagram, "a very long title " * 6)
    assert stage.diagram.texts[0].get_fontsize() < diagrams.TITLE_SIZE


def test_training_edges_follow_a_fixed_scale_so_growth_shows(ax):
    diagrams.weighted_sum(ax, ["a", "b"], np.array([0.5, -1.0]), "+ b", ["ŷ"], title="t", fmt=str, scale=2.0)
    widths = [line.get_linewidth() for line in ax.get_lines()]
    assert widths == pytest.approx([0.8 + 4.2 * 0.25, 0.8 + 4.2 * 0.5])


def test_inference_edges_follow_the_pushes_on_their_own_scale(ax):
    diagrams.weighted_sum(ax, ["a", "b"], np.array([0.5, -1.0]), "+ b", ["ŷ"], title="t", fmt=str, scale=10.0,
                          values=["1", "1"], pushes=np.array([2.0, -0.5]))
    lines = ax.get_lines()
    assert [line.get_linewidth() for line in lines] == pytest.approx([5.0, 0.8 + 4.2 * 0.25])
    assert [line.get_color() for line in lines] == [theme.MODEL, theme.ERROR]


def test_edges_wait_in_grey_until_their_input_is_revealed(ax):
    diagrams.weighted_sum(ax, ["a", "b", "c"], np.array([0.5, -1.0, 0.3]), "+ b", ["ŷ"], title="t",
                          fmt="{:+.1f}".format, values=["1", "2", "3"], pushes=np.array([2.0, -0.5, 0.9]),
                          reveal=0.3)
    assert [line.get_color() for line in ax.get_lines()] == [theme.MODEL, theme.INACTIVE, theme.INACTIVE]
    shown = texts(ax).split("\n")
    assert "+2.0" in shown and "-1.0" not in shown and "+0.3" not in shown  # no stale weights beside pushes


def test_rounding_error_weights_are_drawn_switched_off(ax):
    diagrams.weighted_sum(ax, ["a", "b"], np.array([1.0, -7.3e-15]), "+ b", ["ŷ"], title="t", fmt=str)
    assert [line.get_linestyle() for line in ax.get_lines()] == ["-", "--"]


def test_missing_terms_are_grey_and_faded(ax):
    diagrams.weighted_sum(ax, ["x", "x²"], np.array([0.5, np.nan]), "+ b", ["ŷ"], title="t", fmt=str)
    present, missing = ax.get_lines()
    assert present.get_color() == theme.MODEL and missing.get_color() == theme.INACTIVE
    assert missing.get_alpha() < present.get_alpha()


def test_log_widths_keep_small_weights_visible(ax):
    names = [f"x{i}" for i in range(5)]
    diagrams.weighted_sum(ax, names, np.array([1.0, 0.1, 0.01, 0.001, 1e-4]), "+ b", ["ŷ"], title="t", fmt=str,
                          scale=1.0, decades=3.0)
    widths = [line.get_linewidth() for line in ax.get_lines()]
    assert widths == pytest.approx([5.0, 0.8 + 4.2 * 2 / 3, 0.8 + 4.2 / 3, 0.8, 0.8])


@pytest.mark.parametrize("draw", [
    lambda ax: diagrams.weighted_sum(ax, ["a", "b"], np.zeros(3), "+ b", ["ŷ"], title="t", fmt=str),
    lambda ax: diagrams.weighted_sum(ax, ["a", "b"], np.zeros(2), "+ b", ["ŷ"], title="t", fmt=str,
                                     values=["1"], pushes=np.zeros(2)),
    lambda ax: diagrams.flow(ax, ["a", "b"], title="t", values=["1"]),
])
def test_mismatched_lengths_are_refused(ax, draw):
    with pytest.raises(ValueError):
        draw(ax)


def test_mismatched_notes_are_refused(ax, trees):
    with pytest.raises(ValueError):
        diagrams.chain(ax, trees, colour, title="t", start="F₀", end="F", step_label="×", total=15, notes=["+1"])
