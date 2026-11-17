"""Make the three sibling folders importable, none of them installed as a package.

- ``ml_animation_kit/`` holds animkit, the matplotlib lesson engine;
- ``manim_lesson_kit/`` holds manimkit, which plays animkit's lessons with Manim;
- ``house_price_animations/`` holds house_anim: the data, the models and the lessons themselves.

Entry points (``animate_manim_lessons.py``, ``python -m house_manim``) call ``add_siblings_to_path``
before importing anything that needs them. The tests get the same folders from pytest's ``pythonpath``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIBLINGS = {"animkit": ROOT / "ml_animation_kit", "manimkit": ROOT / "manim_lesson_kit",
            "house_anim": ROOT / "house_price_animations"}


def add_siblings_to_path(siblings: dict[str, Path] = SIBLINGS) -> None:
    """Puts each sibling folder on ``sys.path`` (once), after checking its package is there.

    Args:
        siblings: Package name: the folder that holds it.

    Raises:
        ImportError: If a folder does not hold its package, with the folder to put back.
    """
    for package, folder in siblings.items():
        if not (folder / package / "__init__.py").is_file():
            raise ImportError(f"{package} was not found in {folder}; keep the {folder.name} folder next to "
                              "house_price_manim_lessons")
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
