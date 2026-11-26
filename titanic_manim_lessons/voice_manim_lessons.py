"""Voice the Manim lessons with the same narration as the matplotlib ones.

The two players share lessons, captions, chapters and timing, so this uses the matplotlib project as it
is; only the silent videos differ (this folder's outputs/titanic/lessons/). It needs no Manim, so
plain ``python3`` runs it. The voice clips are shared too: whichever project is voiced second takes
every clip from the cache and pays nothing.

Run (from this folder): python3 voice_manim_lessons.py --model logistic_regression   (see --help).
The voiced videos go to outputs/titanic/lessons_voiced/.
"""

import sys

from titanic_manim.kit_path import add_siblings_to_path

add_siblings_to_path()

from animkit.lessons import voice_cli  # noqa: E402  (the sibling folders must be on sys.path first)
from titanic_anim.data import fetch_titanic  # noqa: E402
from titanic_anim.lessons.cli import project  # noqa: E402

if __name__ == "__main__":
    sys.exit(voice_cli.run(project(), fetcher=fetch_titanic, label=" (Manim)"))
