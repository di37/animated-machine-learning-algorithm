"""Run: python animate_manim_lessons.py --model logistic_regression   (see --help for every option)."""

import sys

from titanic_manim.kit_path import add_siblings_to_path

add_siblings_to_path()

from titanic_manim.cli import main  # noqa: E402  (the sibling folders must be on sys.path first)

if __name__ == "__main__":
    sys.exit(main())
