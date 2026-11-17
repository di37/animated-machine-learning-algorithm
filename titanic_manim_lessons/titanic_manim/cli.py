"""Command line: the narrated Titanic lessons, played with Manim.

The options are the matplotlib lessons' (``titanic_anim.lessons.cli``), and so are the lessons, the data
and the summary CSV; only the player differs. This module derives the Manim project from the
matplotlib one; ``manimkit.cli`` runs it, as it runs every lesson project.
"""

import argparse
from collections.abc import Callable

from manimkit import cli as manim_cli
from manimkit.director import render_lesson
from titanic_anim.data import fetch_titanic
from titanic_anim.lessons import cli as matplotlib_cli

DESCRIPTION = ("Render narrated MP4 lessons with Manim: a classifier learning who survived the Titanic "
               "from age, fare and sex, then predicting an unseen passenger.")


def project() -> manim_cli.LessonProject:
    """The Titanic lessons, played with Manim: the matplotlib project under this description.

    Returns:
        The project.
    """
    return manim_cli.LessonProject.of(matplotlib_cli.project(), DESCRIPTION)


def build_parser() -> argparse.ArgumentParser:
    """The matplotlib lessons' parser, described as the Manim lessons.

    Returns:
        A parser with the same options as ``titanic_anim.lessons.cli``.
    """
    return manim_cli.build_parser(project())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Reads the command line.

    Args:
        argv: The arguments, without the program name; None reads ``sys.argv``.

    Returns:
        The parsed options.
    """
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None, fetcher: Callable = fetch_titanic) -> int:
    """Renders the chosen lessons, one after another; a lesson that fails does not stop the rest.

    Args:
        argv: The arguments, without the program name; None reads ``sys.argv``.
        fetcher: Loads the Titanic passenger list (``titanic_anim.data.fetch_titanic``); tests pass a
            small stand-in.

    Returns:
        The exit status: 0 if every lesson was rendered, 1 if the data or a lesson failed, 130 if stopped.
    """
    # render_lesson is looked up here on every call, so a test can stand in for it on this module
    return manim_cli.run(project(), argv, fetcher=fetcher, render=render_lesson)
