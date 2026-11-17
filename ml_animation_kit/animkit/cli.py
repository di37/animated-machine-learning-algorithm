"""Command-line helpers every dashboard CLI shares: argument types, run options and failure reports.

A dashboard project parses its own options; these helpers turn them into ``RunOptions``, falling back
to recording only when there is no display and to GIF when ffmpeg is missing. They also report a
failed run in one line with the right exit status, and keep the last live window open until the user
closes it.
"""

import argparse
import subprocess
import sys
import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib import animation

from animkit.recorder import RenderingStopped
from animkit.runner import PHASE_INFERENCE, PHASE_TRAINING, RunOptions
from animkit.trainers import NumericalError

NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}
PHASES = {"training": (PHASE_TRAINING,), "inference": (PHASE_INFERENCE,),
          "both": (PHASE_TRAINING, PHASE_INFERENCE)}
# Failures that end a run but should still report the models that finished before them.
RUN_FAILURES = (RenderingStopped, KeyboardInterrupt, NumericalError, OSError,
                subprocess.CalledProcessError)


def int_in_range(low: int, high: int) -> Callable[[str], int]:
    """An argparse type for a whole number between ``low`` and ``high``, both allowed.

    Args:
        low: The smallest number allowed.
        high: The largest number allowed.

    Returns:
        A function that turns an option's text into its number, or raises ``argparse.ArgumentTypeError``.
    """
    def parse(text: str) -> int:
        """Reads one option's number.

        Args:
            text: The option's text.

        Returns:
            Its number.

        Raises:
            argparse.ArgumentTypeError: If it is not a whole number, or not within the range.
        """
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{text}' is not an integer") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}, got {value}")
        return value

    return parse


def silence_spurious_matmul_warnings() -> None:
    """Hides the RuntimeWarnings numpy raises inside matmul for finite inputs on some machines.

    Only warnings whose message ends "encountered in matmul" are hidden; a real NaN or infinity in a
    prediction is still caught by the projects' own checks.
    """
    # numpy 2.x on Apple Accelerate emits divide/overflow/invalid RuntimeWarnings inside
    # matmul for perfectly finite inputs. Genuine NaN/inf output is still caught by each
    # project's prediction checks, which raise NumericalError.
    warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)


def _resolve_live(requested: bool) -> bool:
    """Whether a live window can be shown; switches matplotlib to the off-screen Agg backend when it cannot."""
    if requested and plt.get_backend().lower() in NON_INTERACTIVE_BACKENDS:
        print("No interactive display available; recording videos only.", file=sys.stderr)
        requested = False
    if not requested:
        plt.switch_backend("Agg")
    return requested


def _resolve_format(requested: str, saving: bool) -> str:
    """The video format to use: GIF when MP4 is asked for but ffmpeg is missing (saying so when saving)."""
    if requested == "mp4" and not animation.writers.is_available("ffmpeg"):
        if saving:
            print("ffmpeg not found on PATH; saving GIFs instead.", file=sys.stderr)
        return "gif"
    return requested


def options_from_args(args: argparse.Namespace) -> RunOptions:
    """The run's options from the parsed command line, after checking what this machine can do.

    Without a live window, matplotlib is switched to its off-screen Agg backend.

    Args:
        args: The parsed options: ``no_live``, ``no_save``, ``format``, ``fps``, ``dpi``, ``output_dir``,
            ``batch_size`` and ``phase``.

    Returns:
        The resolved options: a live window is dropped when there is no display, and MP4 becomes GIF
        when ffmpeg is missing.
    """
    return RunOptions(
        live=_resolve_live(not args.no_live),
        save=not args.no_save,
        video_format=_resolve_format(args.format, saving=not args.no_save),
        fps=args.fps,
        dpi=args.dpi,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        phases=PHASES[args.phase],
    )


def report_failure(failure: BaseException, options: RunOptions) -> int:
    """Prints a one-line report of what ended the run, and picks the exit status.

    Args:
        failure: What ended the run: one of ``RUN_FAILURES``.
        options: The run's options; a stopped run says whether its frames were saved.

    Returns:
        130 for a run the user stopped (a closed window or Ctrl-C), 1 for any other failure.
    """
    if isinstance(failure, (RenderingStopped, KeyboardInterrupt)):
        saved = " Frames recorded so far were saved (the last video is partial)." if options.save else ""
        print(f"\nStopped early.{saved}", file=sys.stderr)
        return 130
    if isinstance(failure, subprocess.CalledProcessError):
        print(f"error: the video encoder failed ({failure}); try --format gif", file=sys.stderr)
    else:
        print(f"error: {failure}", file=sys.stderr)
    return 1


def wait_for_window(options: RunOptions) -> None:
    """Keeps the last live window open until the user closes it; returns at once when there is none.

    Args:
        options: The run's options.
    """
    if not (options.live and plt.get_fignums()):
        return
    print("Close the window to exit.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass  # Ctrl-C while the final window is open is just another way to close it
