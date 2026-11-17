"""Run: python -m titanic_anim.lessons --model knn   (see --help).

The shared engine, animkit, lives uninstalled in the sibling ``ml_animation_kit`` folder, so this entry
point puts it on ``sys.path`` before importing the command line, then exits with ``cli.main``'s status.
"""

import sys

from titanic_anim.kit_path import add_kit_to_path

add_kit_to_path()

from titanic_anim.lessons.cli import main  # noqa: E402  (animkit must be on sys.path first)

sys.exit(main())
