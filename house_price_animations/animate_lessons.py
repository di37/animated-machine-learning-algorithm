"""The entry point for the narrated lesson videos: it puts animkit on the path, then runs the lessons' CLI.

Run: python animate_lessons.py --model linear_regression   (see --help for every option).

The lessons (one algorithm per video, learning price from median income) live in
``house_anim.lessons``; their director, encoder and command line come from ``animkit.lessons``.
animkit is not installed, so ``add_kit_to_path`` runs before ``house_anim.lessons.cli`` is imported.
``python -m house_anim.lessons`` does the same thing.
"""

import sys

from house_anim.kit_path import add_kit_to_path

add_kit_to_path()

from house_anim.lessons.cli import main  # noqa: E402  (animkit must be on sys.path first)

if __name__ == "__main__":
    sys.exit(main())
