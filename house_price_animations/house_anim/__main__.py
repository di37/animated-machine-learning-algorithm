"""``python -m house_anim``: the same as ``python animate.py``.

It puts the shared engine (animkit, in the sibling ml_animation_kit folder) on ``sys.path``,
then runs the command line and exits with its status.
"""

import sys

from house_anim.kit_path import add_kit_to_path

add_kit_to_path()

from house_anim.cli import main  # noqa: E402  (animkit must be on sys.path first)

sys.exit(main())
