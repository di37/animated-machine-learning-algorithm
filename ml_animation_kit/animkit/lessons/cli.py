"""The lesson command line every project shares, and the argument types it uses.

A project describes itself as a ``LessonProject``: its lessons, how to load its data, its ``Scene`` and
its summary CSV. ``run`` does the rest, the same way for every project: it lists the lessons, or loads
the data once and renders each chosen lesson, reporting a lesson that fails and going on with the rest.

``LessonProject`` leaves one thing to its subclasses: where the command line's options come from.
``MatplotlibProject`` builds the standard lesson options itself; manimkit's Manim lessons take a
matplotlib project's options as they are.

Example:
    A project's ``cli.py`` needs only a ``project()`` and a ``main``::

        def project() -> MatplotlibProject:
            return MatplotlibProject(description=..., load_data=..., lesson_keys=..., ...)

        def main(argv=None, fetcher=fetch_my_data):
            return run(project(), argv, fetcher=fetcher, render=render_lesson)
"""

import argparse
import csv
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from animkit.cli import int_in_range, silence_spurious_matmul_warnings
from animkit.lessons.config import DEFAULT_RESOLUTION, DURATION_SCALE_RANGE, RESOLUTIONS
from animkit.lessons.director import RenderReport, Scene
from animkit.lessons.encoder import EncoderError, probe
from animkit.lessons.lesson import LessonError
from animkit.lessons.timing import Timing

FAILED = 1  # the exit status when the data or a lesson failed
INTERRUPTED = 130  # the exit status of a program stopped with Ctrl-C (128 + SIGINT)
LESSON_ERRORS = (LessonError, EncoderError, OSError, csv.Error)  # a lesson failing with these is skipped
KEY_WIDTH = 22  # the column the lesson descriptions start in, for --list

Renderer = Callable[..., RenderReport]  # render(lesson, path, *, scene, timing, dpi), as the directors take it


def float_in_range(low: float, high: float) -> Callable[[str], float]:
    """An argparse type for a number between ``low`` and ``high``, both allowed.

    Args:
        low: The smallest number allowed.
        high: The largest number allowed.

    Returns:
        A function that turns an option's text into its number, or raises ``argparse.ArgumentTypeError``.
    """
    def parse(text: str) -> float:
        """Reads one option's number.

        Args:
            text: The option's text.

        Returns:
            Its number.

        Raises:
            argparse.ArgumentTypeError: If it is not a number, or not within the range.
        """
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{text}' is not a number") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}, got {value}")
        return value

    return parse


@dataclass(frozen=True)
class LessonProject(ABC):
    """What a lesson project tells the command line; everything here is the project's own.

    Subclasses say where the options come from, in ``parser``.

    Attributes:
        description: The ``--help`` description.
        load_data: Loads the lessons' data: ``load_data(fetcher=..., max_rows=...)``.
        data_errors: What ``load_data`` raises when the data cannot be loaded.
        lesson_keys: The lessons' keys, in the order they are rendered.
        describe: One line about a lesson, for ``--list``.
        build_lesson: Builds a lesson from its key and the data.
        scene: The project's ``Scene``: it draws the task-specific parts.
        summary_file: The summary CSV's file name, in the output folder.
        summary_row: The summary's row for a rendered lesson: ``summary_row(lesson, report, info, resolution)``.
        merge_summary: Writes rows into the summary CSV, keeping the other lessons' rows.
    """

    description: str
    load_data: Callable[..., Any]
    data_errors: tuple[type[Exception], ...]
    lesson_keys: Callable[[], Sequence[str]]
    describe: Callable[[str], str]
    build_lesson: Callable[[str, Any], Any]
    scene: Scene
    summary_file: str
    summary_row: Callable[..., dict[str, str]]
    merge_summary: Callable[..., None]

    @abstractmethod
    def parser(self) -> argparse.ArgumentParser:
        """The command line's parser, under the project's description.

        Returns:
            A parser with the options ``run`` reads: ``--model``, ``--list``, ``--resolution``,
            ``--duration-scale``, ``--max-rows`` and ``--output-dir``.
        """


@dataclass(frozen=True)
class MatplotlibProject(LessonProject):
    """A lesson project whose command line has the standard lesson options.

    Attributes:
        default_output_dir: Where the videos and the summary go unless ``--output-dir`` says otherwise.
        max_rows: The smallest and largest ``--max-rows`` allowed.
        max_rows_help: The ``--help`` line for ``--max-rows``.
    """

    default_output_dir: Path
    max_rows: tuple[int, int]
    max_rows_help: str

    def parser(self) -> argparse.ArgumentParser:
        """The standard lesson options, with this project's lessons, row limits and output folder.

        Returns:
            The parser.
        """
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument("--model", default="all", choices=(*self.lesson_keys(), "all"),
                            help="which lesson to render (default: all)")
        parser.add_argument("--list", action="store_true", help="list the lessons and exit")
        parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, choices=tuple(RESOLUTIONS),
                            help="1440p (2560x1440, default), 1080p, 720p, or preview (384x216)")
        parser.add_argument("--duration-scale", type=float_in_range(*DURATION_SCALE_RANGE), default=1.0,
                            help="stretch or shrink every chapter (1.0 = about 108 s per lesson)")
        parser.add_argument("--max-rows", type=int_in_range(*self.max_rows), default=None, help=self.max_rows_help)
        parser.add_argument("--output-dir", type=Path, default=self.default_output_dir)
        return parser


def run(project: LessonProject, argv: list[str] | None = None, *, fetcher: Callable, render: Renderer,
        label: str = "") -> int:
    """Lists the lessons, or renders the chosen ones; a lesson that fails does not stop the rest.

    Args:
        project: The lesson project.
        argv: The arguments, without the program name; None reads ``sys.argv``.
        fetcher: Downloads the project's data; tests pass a small stand-in.
        render: Renders one lesson to an MP4 (a test can pass a stand-in).
        label: Added after each lesson's key on its progress line (manimkit's is ``" (Manim)"``).

    Returns:
        The exit status: 0 if every lesson was rendered, ``FAILED`` if the data or a lesson failed,
        ``INTERRUPTED`` if stopped.
    """
    args = project.parser().parse_args(argv)
    if args.list:
        return _list_lessons(project)
    silence_spurious_matmul_warnings()
    try:
        data = project.load_data(fetcher=fetcher, max_rows=args.max_rows)
    except project.data_errors as exc:
        print(f"error: {exc}", file=sys.stderr)
        return FAILED
    except KeyboardInterrupt:
        print("\nStopped while loading the dataset.", file=sys.stderr)
        return INTERRUPTED
    keys = project.lesson_keys() if args.model == "all" else (args.model,)
    return _render_all(project, keys, data, args, render, label)


def _list_lessons(project: LessonProject) -> int:
    """Prints each lesson's key and description; returns the exit status, 0."""
    for key in project.lesson_keys():
        print(f"{key:<{KEY_WIDTH}} {project.describe(key)}")
    return 0


def _render_all(project: LessonProject, keys: Sequence[str], data: Any, args: argparse.Namespace,
                render: Renderer, label: str) -> int:
    """Renders each lesson in turn, reporting the ones that fail; returns the exit status."""
    timing = Timing(scale=args.duration_scale)
    failed: list[str] = []
    for index, key in enumerate(keys, start=1):
        print(f"[{index}/{len(keys)}] {key}{label} …", flush=True)
        try:
            _render_one(project, key, data, args, timing, render)
        except KeyboardInterrupt:
            print("\nStopped; the unfinished video was discarded.", file=sys.stderr)
            return INTERRUPTED
        except LESSON_ERRORS as exc:
            print(f"error: {key}: {exc}", file=sys.stderr)
            failed = [*failed, key]
    if failed:
        print(f"{len(failed)} of {len(keys)} lessons failed: {', '.join(failed)}", file=sys.stderr)
        return FAILED
    return 0


def _render_one(project: LessonProject, key: str, data: Any, args: argparse.Namespace, timing: Timing,
                render: Renderer) -> None:
    """Builds one lesson, renders it, and adds its row to the summary CSV."""
    lesson = project.build_lesson(key, data)
    report = render(lesson, args.output_dir / f"{key}_lesson.mp4", scene=project.scene, timing=timing,
                    dpi=RESOLUTIONS[args.resolution])
    info = probe(report.path)
    summary_row = project.summary_row(lesson, report, info, args.resolution)
    project.merge_summary(args.output_dir / project.summary_file, [summary_row])
    print(f"    saved {report.path} · {info.width}x{info.height} · {info.fps:g} fps · "
          f"{info.duration_s:.1f} s · {info.codec}/{info.pix_fmt}", flush=True)
