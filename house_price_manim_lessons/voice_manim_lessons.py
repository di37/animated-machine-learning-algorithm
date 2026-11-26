"""Voice the Manim lessons with the same narration as the matplotlib ones.

The two players share lessons, captions, chapters and timing, so this uses the matplotlib project as it
is; only the silent videos differ (this folder's outputs/california_housing/lessons/). It needs no Manim, so
plain ``python3`` runs it. The voice clips are shared too: whichever project is voiced second takes
every clip from the cache and pays nothing.

Run (from this folder): python3 voice_manim_lessons.py --model linear_regression   (see --help).
The voiced videos go to outputs/california_housing/lessons_voiced/.
"""

import sys

from house_manim.kit_path import add_siblings_to_path

add_siblings_to_path()

from sklearn.datasets import fetch_california_housing  # noqa: E402

from animkit.lessons import voice_cli  # noqa: E402  (the sibling folders must be on sys.path first)
from house_anim.lessons.cli import project  # noqa: E402

if __name__ == "__main__":
    sys.exit(voice_cli.run(project(), fetcher=fetch_california_housing, label=" (Manim)"))
