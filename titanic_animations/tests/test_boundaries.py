"""The Titanic project stands alone: shared code comes from animkit, never from the house project."""

import ast
import sys
from pathlib import Path

import pytest

from titanic_anim import kit_path

PROJECT = Path(__file__).resolve().parents[1]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    return {name.split(".")[0] for name in names}


def test_no_titanic_module_imports_the_house_project():
    files = [p for p in PROJECT.rglob("*.py") if "house_anim" in imported_roots(p)]
    assert not files


def test_kit_path_points_at_the_sibling_kit():
    assert (kit_path.KIT_DIR / "animkit" / "__init__.py").is_file()
    assert kit_path.KIT_DIR.parent == PROJECT.parent


def test_kit_path_reports_a_missing_kit_folder(tmp_path):
    with pytest.raises(ImportError, match="ml_animation_kit"):
        kit_path.add_kit_to_path(tmp_path)


def test_kit_path_adds_the_folder_once(monkeypatch):
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(kit_path.KIT_DIR)])
    kit_path.add_kit_to_path()
    kit_path.add_kit_to_path()
    assert sys.path.count(str(kit_path.KIT_DIR)) == 1
