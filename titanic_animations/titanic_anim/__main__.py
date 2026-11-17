"""``python -m titanic_anim``: the same command line as ``animate_titanic.py``.

animkit lives, uninstalled, in the sibling ``ml_animation_kit`` folder, so it goes on ``sys.path``
before ``titanic_anim.cli``, which imports it, is loaded.
"""

import sys

from titanic_anim.kit_path import add_kit_to_path

add_kit_to_path()

from titanic_anim.cli import main  # noqa: E402  (animkit must be on sys.path first)

sys.exit(main())
