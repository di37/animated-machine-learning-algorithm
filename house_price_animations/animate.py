"""The entry point for the dashboards and the two scenes: it puts animkit on the path, then runs the CLI.

Run: python animate.py --model random_forest   (see --help for every option).

The shared engine, animkit, is not installed as a package: it lives in the sibling ml_animation_kit
folder. So ``add_kit_to_path`` runs first, and only then is ``house_anim.cli`` imported, since it
imports animkit. ``python -m house_anim`` does the same thing.
"""

import sys

from house_anim.kit_path import add_kit_to_path

add_kit_to_path()

from house_anim.cli import main  # noqa: E402  (animkit must be on sys.path first)

if __name__ == "__main__":
    sys.exit(main())
