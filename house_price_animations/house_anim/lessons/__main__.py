"""Run: python -m house_anim.lessons --model knn   (see --help).

The shared engine, animkit, lives in the sibling ml_animation_kit folder rather than an installed
package, so this entry point puts it on ``sys.path`` before importing the command line, then exits with
the command line's status.
"""

import sys

from house_anim.kit_path import add_kit_to_path

add_kit_to_path()

from house_anim.lessons.cli import main  # noqa: E402  (animkit must be on sys.path first)

sys.exit(main())
