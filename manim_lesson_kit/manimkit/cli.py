"""The Manim lesson command line: animkit's lesson command line, with Manim as the player.

A Manim ``LessonProject`` is animkit's ``LessonProject`` whose options are a matplotlib project's, as
they are: the same lessons, data and summary, under its own description. ``run`` is animkit's
``run`` with manimkit's ``render_lesson``, and each lesson's progress line says ``(Manim)``.

Example:
    A project's ``cli.py`` derives its Manim lessons from its matplotlib ones::

        def project() -> LessonProject:
            return LessonProject.of(matplotlib_cli.project(), "Render the lessons with Manim ...")

        def main(argv=None, fetcher=fetch_my_data):
            return run(project(), argv, fetcher=fetcher, render=render_lesson)
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, fields

from animkit.lessons import cli as lesson_cli
from animkit.lessons.cli import Renderer
from manimkit.director import render_lesson

MANIM_LABEL = " (Manim)"  # added after each lesson's key on its progress line


@dataclass(frozen=True)
class LessonProject(lesson_cli.LessonProject):
    """A lesson project played with Manim, its options taken from a matplotlib project.

    Attributes:
        matplotlib_parser: Builds the matplotlib lessons' parser; the Manim one keeps its options
            (``--model``, ``--list``, ``--resolution``, ``--duration-scale``, ``--max-rows``,
            ``--output-dir``) under this project's description.
    """

    matplotlib_parser: Callable[[], argparse.ArgumentParser]

    @classmethod
    def of(cls, project: lesson_cli.LessonProject, description: str) -> "LessonProject":
        """The Manim lessons of a matplotlib project: its lessons, data and summary, under a new description.

        Args:
            project: The matplotlib lesson project.
            description: The Manim lessons' ``--help`` description.

        Returns:
            A Manim project sharing everything else with ``project``, its options included.
        """
        shared = {field.name: getattr(project, field.name) for field in fields(lesson_cli.LessonProject)}
        return cls(**{**shared, "description": description}, matplotlib_parser=project.parser)

    def parser(self) -> argparse.ArgumentParser:
        """The matplotlib lessons' parser, described as the Manim lessons.

        Returns:
            A parser with exactly the matplotlib lessons' options and defaults.
        """
        parser = self.matplotlib_parser()
        parser.description = self.description
        return parser


def build_parser(project: LessonProject) -> argparse.ArgumentParser:
    """The project's parser.

    Args:
        project: The lesson project.

    Returns:
        A parser with exactly the matplotlib lessons' options and defaults, under the project's description.
    """
    return project.parser()


def run(project: LessonProject, argv: list[str] | None = None, *, fetcher: Callable,
        render: Renderer = render_lesson) -> int:
    """Lists the lessons, or renders the chosen ones with Manim; a lesson that fails does not stop the rest.

    Args:
        project: The lesson project.
        argv: The arguments, without the program name; None reads ``sys.argv``.
        fetcher: Downloads the project's data; tests pass a small stand-in.
        render: Renders one lesson to an MP4 (a test can pass a stand-in).

    Returns:
        The exit status, as animkit's ``run`` gives it: 0 if every lesson was rendered, 1 if the data or
        a lesson failed, 130 if stopped.
    """
    return lesson_cli.run(project, argv, fetcher=fetcher, render=render, label=MANIM_LABEL)
