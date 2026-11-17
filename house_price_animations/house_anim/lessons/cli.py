"""Command line: render narrated 16:9 lesson videos, one regression algorithm each.

This module only says what the house-price lessons are, as a ``MatplotlibProject``; animkit's
``animkit.lessons.cli.run`` lists and renders them, as it does every lesson project's.
"""

import argparse
from collections.abc import Callable

from sklearn.datasets import fetch_california_housing

from animkit.lessons import cli as lesson_cli
from house_anim.data import DataLoadError, load_splits
from house_anim.lessons.config import DEFAULT_OUTPUT_DIR, SUMMARY_FILE
from house_anim.lessons.data import IncomeData, income_view
from house_anim.lessons.director import render_lesson
from house_anim.lessons.regression_scene import REGRESSION
from house_anim.lessons.stories import build_lesson, describe, lesson_keys
from house_anim.lessons.summary import merge_summary, summary_row

MAX_ROWS = (50, 1_000_000)  # the smallest and largest --max-rows


def load_income_data(fetcher: Callable, max_rows: int | None) -> IncomeData:
    """The lessons' data: median income and price, split into train and test.

    Args:
        fetcher: Loads the California housing data (scikit-learn's ``fetch_california_housing``).
        max_rows: Subsamples the data to this many rows; None keeps them all.

    Returns:
        The income view of the scaled splits.

    Raises:
        DataLoadError: If the data cannot be loaded.
    """
    return income_view(load_splits(fetcher=fetcher, max_rows=max_rows))


def project() -> lesson_cli.MatplotlibProject:
    """The house-price lessons, as the shared command line needs them.

    It is made on every call from this module's names, so a test can stand in for any of them.

    Returns:
        The project.
    """
    return lesson_cli.MatplotlibProject(
        description="Render narrated MP4 lessons: a regression algorithm learning California house "
                    "prices from median income, then predicting an unseen district.",
        load_data=load_income_data, data_errors=(DataLoadError, ValueError), lesson_keys=lesson_keys,
        describe=describe, build_lesson=build_lesson, scene=REGRESSION, summary_file=SUMMARY_FILE,
        summary_row=summary_row, merge_summary=merge_summary, default_output_dir=DEFAULT_OUTPUT_DIR,
        max_rows=MAX_ROWS, max_rows_help="subsample the dataset for a quick run")


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


def main(argv: list[str] | None = None, fetcher: Callable = fetch_california_housing) -> int:
    """Renders the chosen lessons, one after another; a lesson that fails does not stop the rest.

    Args:
        argv: The arguments, without the program name; None reads ``sys.argv``.
        fetcher: Loads the California housing data (scikit-learn's ``fetch_california_housing``); tests
            pass a small stand-in.

    Returns:
        The exit status: 0 if every lesson was rendered, 1 if the data or a lesson failed, 130 if stopped.
    """
    return lesson_cli.run(project(), argv, fetcher=fetcher, render=render_lesson)
