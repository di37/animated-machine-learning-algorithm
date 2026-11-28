"""Voice the playlist's introduction and conclusion, as the lessons are voiced.

Run (from this folder): python3 voice_manim_series.py   (see --help; needs no Manim)
The voiced videos go to outputs/titanic/series_voiced/.
"""

import sys

from titanic_manim.kit_path import add_siblings_to_path

add_siblings_to_path()

from animkit.series.voice import run_series_voice  # noqa: E402  (the sibling folders must be on sys.path first)
from titanic_anim.lessons.scene import CLASSIFICATION  # noqa: E402

SOURCE_DIR = "outputs/titanic/series"

if __name__ == "__main__":
    sys.exit(run_series_voice(["--source-dir", SOURCE_DIR, *sys.argv[1:]], phrases=CLASSIFICATION.spoken_phrases))
