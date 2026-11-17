"""Command line: render narrated 16:9 lesson videos, one Titanic survival classifier each.

This module only says what the Titanic lessons are, as a ``MatplotlibProject``; animkit's
``animkit.lessons.cli.run`` lists and renders them, as it does every lesson project's.
"""

import argparse
from collections.abc import Callable

from animkit.lessons import cli as lesson_cli
from animkit.lessons.director import render_lesson
from titanic_anim.config import MIN_ROWS
from titanic_anim.data import DataLoadError, fetch_titanic
from titanic_anim.lessons.config import DEFAULT_OUTPUT_DIR, SUMMARY_FILE
from titanic_anim.lessons.data import load_passengers
from titanic_anim.lessons.scene import CLASSIFICATION
from titanic_anim.lessons.stories import build_lesson, describe, lesson_keys
from titanic_anim.lessons.summary import merge_summary, summary_row

MAX_ROWS = (MIN_ROWS, 100_000)  # the smallest and largest --max-rows


def project() -> lesson_cli.MatplotlibProject:
    """The Titanic lessons, as the shared command line needs them.

    It is made on every call from this module's names, so a test can stand in for any of them.

    Returns:
        The project.
    """
    return lesson_cli.MatplotlibProject(
        description="Render narrated MP4 lessons: a classifier learning who survived the Titanic from age, "
                    "fare and sex, then predicting an unseen passenger.",
        load_data=load_passengers, data_errors=(DataLoadError, ValueError), lesson_keys=lesson_keys,
        describe=describe, build_lesson=build_lesson, scene=CLASSIFICATION, summary_file=SUMMARY_FILE,
        summary_row=summary_row, merge_summary=merge_summary, default_output_dir=DEFAULT_OUTPUT_DIR,
        max_rows=MAX_ROWS, max_rows_help="subsample the passengers for a quick run")


def build_parser() -> argparse.ArgumentParser:
    """The command line's parser.

    Returns:
        The standard lesson options, with this project's lessons, row limits and output folder.
    """
    return project().parser()


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
    return lesson_cli.run(project(), argv, fetcher=fetcher, render=render_lesson)
