"""Render the playlist's introduction and conclusion with Manim.

Run (from this folder): ../manim_lesson_kit/.venv/bin/python animate_manim_series.py
The videos go to outputs/california_housing/series/, each with the chapters file its voice-over reads.
"""

import sys
from pathlib import Path

from house_manim.kit_path import add_siblings_to_path

add_siblings_to_path()

from animkit.cli import silence_spurious_matmul_warnings  # noqa: E402  (the sibling folders must be on sys.path first)
from house_anim.lessons.series import conclusion, introduction, playlist  # noqa: E402
from manimkit.series import render_playlist  # noqa: E402

OUTPUT_DIR = Path("outputs/california_housing/series")

if __name__ == "__main__":
    silence_spurious_matmul_warnings()
    facts = playlist()
    sys.exit(render_playlist((introduction(facts), conclusion(facts)), OUTPUT_DIR))
