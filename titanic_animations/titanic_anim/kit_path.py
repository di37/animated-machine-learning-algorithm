"""Make the shared engine importable: animkit lives in the sibling ml_animation_kit folder, uninstalled.

Entry points (animate_titanic*.py, ``python -m titanic_anim``) call ``add_kit_to_path`` before
importing anything that needs animkit. The tests get the same folder from pytest's ``pythonpath``.
"""

import sys
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parents[2] / "ml_animation_kit"


def add_kit_to_path(kit_dir: Path = KIT_DIR) -> None:
    """Puts the folder that holds animkit at the front of ``sys.path``, once.

    Args:
        kit_dir: The folder with the ``animkit`` package in it; by default the sibling
            ``ml_animation_kit`` folder.

    Raises:
        ImportError: If ``kit_dir`` holds no ``animkit`` package.
    """
    if not (kit_dir / "animkit" / "__init__.py").is_file():
        raise ImportError(f"the shared animation engine was not found at {kit_dir}; keep the "
                          "ml_animation_kit folder next to titanic_animations")
    if str(kit_dir) not in sys.path:
        sys.path.insert(0, str(kit_dir))
