"""Command line entry point: animate training and inference for one or all regressors.

``main`` reads the options, loads the data once, and runs one of three scenes:

- ``dashboards`` (the default): for each chosen model, the training dashboard and then the
  inference dashboard, adding the model's row to ``metrics_summary.csv`` as soon as it finishes;
- ``race``: every model trains side by side, then sits a final exam on the test houses;
- ``map``: four models repaint a map of California as they train.

A run stopped part-way (the window closed, Ctrl-C, a model diverging) keeps the frames recorded so
far, and the dashboards scene still prints the table of the models that finished. The option
types, the failure report and the wait for the last window are animkit's (``animkit.cli``).
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from sklearn.datasets import fetch_california_housing

from animkit.cli import (
    PHASES,
    RUN_FAILURES,
    int_in_range,
    options_from_args,
    report_failure,
    silence_spurious_matmul_warnings,
    wait_for_window,
)
from house_anim.config import (
    DEFAULT_DPI,
    DEFAULT_FPS,
    DEFAULT_INFERENCE_BATCH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DPI_RANGE,
    FPS_RANGE,
    VIDEO_FORMATS,
)
from house_anim.data import DataLoadError, HousingSplits, load_splits
from house_anim.models import MODEL_SPECS, ModelSpec, get_spec, model_keys
from house_anim.runner import ModelResult, RunOptions, run_model
from house_anim.scenes import animate_map, animate_race, format_exam
from house_anim.summary import format_table, merge_summary_csv, result_row

SUMMARY_FILE = "metrics_summary.csv"
SCENES = ("dashboards", "race", "map")


def build_parser() -> argparse.ArgumentParser:
    """The command line's options.

    Returns:
        A parser for ``--scene``, ``--model``, ``--list``, ``--phase``, the live and save switches,
        the video settings, ``--batch-size``, ``--max-rows`` and ``--output-dir``.
    """
    parser = argparse.ArgumentParser(
        description="Animate scikit-learn regressors training and predicting California house prices.")
    parser.add_argument("--scene", default="dashboards", choices=SCENES,
                        help="dashboards: per-model training + inference (default); "
                             f"race: all {len(MODEL_SPECS)} models race; map: watch 4 models learn California")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=(*model_keys(), "all"),
                        help=f"which regressor to animate (default: {DEFAULT_MODEL})")
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    parser.add_argument("--phase", default="both", choices=tuple(PHASES),
                        help="animate training, inference, or both (default: both)")
    parser.add_argument("--no-live", action="store_true", help="don't open a live window")
    parser.add_argument("--no-save", action="store_true", help="don't record video files")
    parser.add_argument("--format", default="mp4", choices=VIDEO_FORMATS, help="video format")
    parser.add_argument("--fps", type=int_in_range(*FPS_RANGE), default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int_in_range(*DPI_RANGE), default=DEFAULT_DPI)
    parser.add_argument("--batch-size", type=int_in_range(1, 100_000),
                        default=DEFAULT_INFERENCE_BATCH, help="test houses predicted per frame")
    parser.add_argument("--max-rows", type=int_in_range(50, 1_000_000), default=None,
                        help="subsample the dataset for a quick run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: The arguments, without the program name; None reads ``sys.argv``.

    Returns:
        The parsed options.
    """
    return build_parser().parse_args(argv)


def _list_models() -> int:
    """Prints each model's key, title and what one training frame shows; returns the exit status, 0."""
    for key, spec in MODEL_SPECS.items():
        print(f"{key:<18} {spec.title}\n{'':<18} {spec.note}")
    return 0


def _run_all(specs: list[ModelSpec], splits: HousingSplits, options: RunOptions,
             context: dict[str, int]) -> tuple[list[ModelResult], BaseException | None]:
    """Run models in turn; on failure, return what finished plus the error instead of losing it."""
    results: list[ModelResult] = []
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.title} …", flush=True)
        try:
            result = run_model(spec, splits, options, keep_open=options.live and index == len(specs))
            if options.save:  # persist per model, so a later failure can't lose this row
                merge_summary_csv(options.output_dir / SUMMARY_FILE, [result_row(result, **context)])
        except RUN_FAILURES as exc:
            return results, exc
        results = [*results, result]
    return results, None


def _run_dashboards(args: argparse.Namespace, splits: HousingSplits, options: RunOptions) -> int:
    """Runs the chosen model's dashboards (or every model's), prints the table; returns the exit status."""
    specs = list(MODEL_SPECS.values()) if args.model == "all" else [get_spec(args.model)]
    context = {"rows_used": splits.n_train + len(splits.y_val) + len(splits.y_test),
               "batch_size": options.batch_size}
    results, failure = _run_all(specs, splits, options, context)
    if results:
        print("\n" + format_table([result_row(r, **context) for r in results]))
    if failure is not None:
        return report_failure(failure, options)
    if options.save:
        print(f"\nVideos and {SUMMARY_FILE} saved in {options.output_dir.resolve()}")
    return 0


def _run_scene(scene: str, splits: HousingSplits, options: RunOptions) -> int:
    """Records the race or the map scene and prints where the video went; returns the exit status."""
    print(f"Scene: {scene} …", flush=True)
    try:
        if scene == "race":
            exam, path = animate_race(splits, options, keep_open=options.live)
            print("\n" + format_exam(exam, n_test=len(splits.y_test)))
        else:
            _, path = animate_map(splits, options, keep_open=options.live)
    except RUN_FAILURES as exc:
        return report_failure(exc, options)
    if path is not None:
        print(f"\nVideo saved to {path.resolve()}")
    return 0


def main(argv: list[str] | None = None, fetcher: Callable = fetch_california_housing) -> int:
    """Runs the program: lists the models, or loads the data and records the chosen scene.

    Args:
        argv: The arguments, without the program name; None reads ``sys.argv``.
        fetcher: Downloads the dataset; tests pass a small stand-in.

    Returns:
        The exit status: 0 on success, 1 if the data could not be loaded or a run failed, 2 if
        there is nothing to do (no live window and ``--no-save``), 130 if stopped early.
    """
    args = parse_args(argv)
    if args.list:
        return _list_models()
    silence_spurious_matmul_warnings()
    options = options_from_args(args)
    if not (options.live or options.save):
        print("error: nothing to do: no live window (--no-live or no display) and --no-save",
              file=sys.stderr)
        return 2
    try:
        splits = load_splits(fetcher=fetcher, max_rows=args.max_rows)
    except (DataLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.scene == "dashboards":
        status = _run_dashboards(args, splits, options)
    else:
        status = _run_scene(args.scene, splits, options)
    if status == 0:
        wait_for_window(options)
    return status
