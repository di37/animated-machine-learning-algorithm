"""The layout checks catch texts that leave their card or collide, and every diagram helper passes them
inside the real shared side panel."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from animkit.lessons import diagrams
from animkit.lessons.fit import colliding_texts, overflowing_texts
from animkit.lessons.lesson import Explanation
from animkit.lessons.plots import EXPLANATION_SIZE, draw_explanation
from animkit.lessons.stage import Stage


@pytest.fixture(scope="module")
def stage():
    shared = Stage(dpi=100.0)  # the video's own resolution, where the slack is tightest
    shared.share_panel(True)
    return shared


@pytest.fixture(scope="module")
def trees():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] - X[:, 1]
    return [DecisionTreeRegressor(max_depth=3, random_state=s).fit(X[s:], y[s:]).tree_ for s in range(15)]


def colour(value):
    return (0.2, 0.4, 0.8, 1.0)


def test_a_text_past_the_edge_overflows_and_two_texts_on_one_spot_collide(stage):
    ax = stage.diagram
    diagrams.frame(ax, "title")
    ax.text(0.99, 0.5, "far too long to fit")
    ax.text(0.3, 0.3, "one")
    ax.text(0.3, 0.3, "two")
    assert overflowing_texts(ax) == ("far too long to fit",)
    assert colliding_texts(ax) == (("one", "two"),)


def _cases(ax, trees):
    weights = np.array([-0.2, 0.6, 1.0])
    yield lambda: diagrams.weighted_sum(ax, ["age", "fare", "woman"], weights, "+ bias -0.44", ["σ", "P(survive)"],
                                        title="The model: P(survive) = σ(bias + Σ weight × input)", fmt="{:+.2f}".format)
    yield lambda: diagrams.weighted_sum(ax, ["age", "fare", "woman"], weights, "+ bias -0.44", ["sign", "survived?"],
                                        title="t", fmt="{:+.2f}".format, values=["22 y → -0.55", "$7 → -0.88",
                                                                                 "man → -0.74"],
                                        pushes=np.array([0.1, -0.5, -0.7]), result="-1.23 → did not survive")
    yield lambda: diagrams.weighted_sum(ax, ["income"], np.array([80.0]), "+ θ₀ $206k", ["ŷ"], title="t",
                                        fmt="+${:.0f}k".format, values=["$52k → z +1.20"], pushes=np.array([96.0]),
                                        result="$302k")
    yield lambda: diagrams.weighted_sum(ax, [f"x{i}" for i in range(12)], np.linspace(-1, 1, 12), "+ b $206k", ["ŷ"],
                                        title="t", fmt=str, values=[""] * 12, pushes=np.linspace(1, -1, 12),
                                        result="$302k")
    yield lambda: diagrams.tree_grid(ax, trees, colour, title="The forest: 50 trees", total=50, summary="average\nprice",
                                     votes=[f"${v}k" for v in range(100, 110)])
    yield lambda: diagrams.chain(ax, trees, colour, title="Boosting", start="F₀\n$206k", end="F₁₅\n$300k",
                                 step_label="0.1 ×", total=15, earlier="\n+$102k", notes=["+$1.2k"] * 4)
    yield lambda: diagrams.flow(ax, ["store all\n16,512 districts", "find the k = 9\nnearest", "average\ntheir prices",
                                     "ŷ"], title="How kNN decides", values=["16,512 stored", "9 found", "$250k", "$250k"])
    yield lambda: diagrams.factor_rows(ax, [("survived", ["prior\n41%", "age\nmean 28\nsd 14", "log fare\nmean 2.9\nsd 1.0",
                                                          "women\n68%"]),
                                            ("did not", ["prior\n59%", "age\nmean 30\nsd 14", "log fare\nmean 2.5\nsd 0.9",
                                                         "women\n15%"])],
                                       title="Bayes' rule", merge="normalise", result="P(survive)\n82%")


@pytest.mark.parametrize("count", range(1, 13))
def test_weighted_sum_fits_any_number_of_inputs(stage, count):
    names = [f"term{i}"[:6] for i in range(count)]
    weights = np.linspace(-1.0, 1.0, count)
    for pushes in (None, weights[::-1]):
        diagrams.weighted_sum(stage.diagram, names, weights, "+ bias -0.44", ["σ", "P(survive)"], title="t",
                              fmt="{:+.2f}".format, values=None if pushes is None else ["v -0.55"] * count,
                              pushes=pushes, result="-1.23 → did not survive")
        assert overflowing_texts(stage.diagram) == () and colliding_texts(stage.diagram) == (), count


def test_every_diagram_helper_fits_the_shared_panel(stage, trees):
    for draw in _cases(stage.diagram, trees):
        draw()
        assert overflowing_texts(stage.diagram) == ()
        assert colliding_texts(stage.diagram) == ()


def test_explanation_lines_fit_the_shared_panel_and_shrink_only_when_there_are_many(stage):
    lines = ("passenger: 22 y, $7, man", "score = -0.44 + ...", "σ(-1.23) = 23%", "23% ≤ 50% → did not survive")
    many = (*lines[:3], "the log-odds, summed", "one more worked step", "and another", lines[-1])
    for ax in (stage.panel, stage.metric):  # the worked prediction is drawn where the metric chart was
        for shown, shrinks in ((lines, False), (many, True)):
            draw_explanation(ax, Explanation(lines=shown, prediction=0.23), len(shown), "How it predicts")
            assert (max(text.get_fontsize() for text in ax.texts[1:]) < EXPLANATION_SIZE) == shrinks
            assert overflowing_texts(ax) == () and colliding_texts(ax) == ()
