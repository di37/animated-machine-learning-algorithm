"""animkit stays dataset-neutral: it imports no project package and names no dataset."""

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from animkit.algorithms.common import draw_tree
from animkit.lessons.stage import Stage
from animkit.models import ModelSpec
from animkit.race import colour_slots

PACKAGE = Path(__file__).resolve().parents[1] / "animkit"
PROJECT_PACKAGES = ("house_anim", "titanic_anim")
DATASET_WORDS = re.compile(
    r"california|titanic|housing|house|medinc|medhouseval|district|passenger|surviv(e|ed|al)\b|census|openml",
    re.IGNORECASE)


def sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    return names | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}


def test_the_kit_imports_no_project_package():
    offenders = [f"{path.name}: {module}" for path in sources() for module in imported_modules(path)
                 if module.split(".")[0] in PROJECT_PACKAGES]
    assert sources() and not offenders


def test_the_kit_names_no_dataset():
    offenders = [f"{path.relative_to(PACKAGE)}:{number}: {line.strip()}" for path in sources()
                 for number, line in enumerate(path.read_text().splitlines(), start=1) if DATASET_WORDS.search(line)]
    assert not offenders


def test_stage_footer_carries_no_dataset_credit_by_default():
    assert Stage(dpi=20.0).fig.texts[-1].get_text() == ""


def test_colour_slots_shares_no_hue_unless_asked():
    specs = [ModelSpec(key, key, key, "{}", "param_sweep", "step", "", object, (1.0,)) for key in ("a", "b")]
    assert colour_slots(specs) == ((0, False), (1, False))
    assert colour_slots(specs, {"b": "a"}) == ((0, False), (0, True))


def test_draw_tree_needs_a_colour_rule():
    from sklearn.tree import DecisionTreeRegressor

    tree = DecisionTreeRegressor(max_depth=1).fit(np.arange(8.0).reshape(-1, 1), np.arange(8.0)).tree_
    ax = plt.figure().add_subplot()
    with pytest.raises(ValueError, match="node_colour"):
        draw_tree(ax, tree, (0.0, 0.0, 1.0, 1.0), max_depth=1)
    assert draw_tree(ax, tree, (0.0, 0.0, 1.0, 1.0), max_depth=1, signed_scale=1.0).nodes == 3
