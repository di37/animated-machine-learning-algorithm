"""manimkit stays dataset-neutral: it imports no project package and names no dataset."""

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "manimkit"
PROJECT_PACKAGES = ("house_anim", "titanic_anim", "house_manim", "titanic_manim")
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
