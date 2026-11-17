"""Run: python -m titanic_manim --model logistic_regression   (see --help)."""

import sys

from titanic_manim.kit_path import add_siblings_to_path

add_siblings_to_path()

from titanic_manim.cli import main  # noqa: E402  (the sibling folders must be on sys.path first)

sys.exit(main())
