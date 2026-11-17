"""Run: python -m house_manim --model linear_regression   (see --help)."""

import sys

from house_manim.kit_path import add_siblings_to_path

add_siblings_to_path()

from house_manim.cli import main  # noqa: E402  (the sibling folders must be on sys.path first)

sys.exit(main())
