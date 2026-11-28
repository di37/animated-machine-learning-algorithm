"""Voice the playlist's introduction and conclusion, as the lessons are voiced.

Run (from this folder): python3 voice_manim_series.py   (see --help; needs no Manim)
The voiced videos go to outputs/california_housing/series_voiced/.
"""

import sys

from house_manim.kit_path import add_siblings_to_path

add_siblings_to_path()

from animkit.series.voice import run_series_voice  # noqa: E402  (the sibling folders must be on sys.path first)

SOURCE_DIR = "outputs/california_housing/series"

if __name__ == "__main__":
    sys.exit(run_series_voice(["--source-dir", SOURCE_DIR, *sys.argv[1:]]))
