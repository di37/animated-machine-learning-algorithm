"""The house Manim lessons stand alone: shared code from the kits, lessons from house_anim, nothing Titanic."""

import ast
import sys
from pathlib import Path

import pytest

from house_manim import kit_path

PROJECT = Path(__file__).resolve().parents[1]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    return {name.split(".")[0] for name in names}


def test_no_module_imports_a_titanic_project():
    files = [p for p in PROJECT.rglob("*.py") if ".venv" not in p.parts and
             imported_roots(p) & {"titanic_anim", "titanic_manim"}]
    assert not files


def test_the_siblings_are_the_folders_next_to_this_one():
    for package, folder in kit_path.SIBLINGS.items():
        assert (folder / package / "__init__.py").is_file() and folder.parent == PROJECT.parent


def test_a_missing_sibling_is_reported_by_name(tmp_path):
    with pytest.raises(ImportError, match="manimkit"):
        kit_path.add_siblings_to_path({"manimkit": tmp_path})


def test_each_sibling_is_added_once(monkeypatch):
    folders = {str(folder) for folder in kit_path.SIBLINGS.values()}
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in folders])
    kit_path.add_siblings_to_path()
    kit_path.add_siblings_to_path()
    assert all(sys.path.count(folder) == 1 for folder in folders)
