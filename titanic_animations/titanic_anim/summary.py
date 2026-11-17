"""Console table and CSV summary of every classifier run.

Each run adds one row per model to ``metrics_summary.csv``: its fit times and validation scores,
and its test scores when inference ran. Rows are merged by model key, so running the models one
at a time still builds one table, in the registry's order. The file is written to a temporary
name and then moved into place, so a reader never sees a half-written file.
"""

import csv
import os
from pathlib import Path

from animkit.runner import ModelResult
from titanic_anim.models import model_keys

SUMMARY_FIELDS = (
    "model", "title", "rows_used", "batch_size", "fit_seconds", "total_fit_seconds",
    "val_accuracy", "val_auc", "test_accuracy", "test_precision", "test_recall", "test_f1",
    "throughput_per_s", "videos",
)
TEST_FIELDS = ("test_accuracy", "test_precision", "test_recall", "test_f1", "throughput_per_s")


def result_row(result: ModelResult, *, rows_used: int, batch_size: int) -> dict[str, str]:
    """One model's CSV row.

    Args:
        result: The finished run of one model.
        rows_used: How many passengers the run used, in all three splits.
        batch_size: Test passengers per inference frame.

    Returns:
        Every ``SUMMARY_FIELDS`` column as text. ``fit_seconds`` is the cost of the final model,
        ``total_fit_seconds`` adds up every refit of a sweep, and the test columns are empty when
        inference did not run.
    """
    training, inference = result.final_training, result.final_inference
    row = {
        "model": result.key,
        "title": result.title,
        "rows_used": str(rows_used),
        "batch_size": str(batch_size),
        "fit_seconds": f"{training.model_fit_seconds:.3f}",
        "total_fit_seconds": f"{training.fit_seconds:.3f}",
        "val_accuracy": f"{training.val_scores.accuracy:.4f}",
        "val_auc": f"{training.val_scores.auc:.4f}",
        **{field: "" for field in TEST_FIELDS},
        "videos": ";".join(str(p) for p in result.videos),
    }
    if inference is not None:
        scores = inference.scores
        row = {
            **row,
            "test_accuracy": f"{scores.accuracy:.4f}",
            "test_precision": f"{scores.precision:.4f}",
            "test_recall": f"{scores.recall:.4f}",
            "test_f1": f"{scores.f1:.4f}",
            "throughput_per_s": f"{inference.throughput_per_s:.0f}",
        }
    return row


def format_table(rows: list[dict[str, str]]) -> str:
    """The summary rows as a console table.

    Args:
        rows: CSV rows, as ``result_row`` makes them.

    Returns:
        Aligned columns under a header and a rule; an empty cell shows as "—".
    """
    columns = ("model", "fit_seconds", "val_accuracy", "test_accuracy", "test_precision", "test_recall")
    headers = ("model", "fit s", "val accuracy", "test accuracy", "precision", "recall")
    table = [headers, *[tuple(row[c] or "—" for c in columns) for row in rows]]
    widths = [max(len(line[i]) for line in table) for i in range(len(columns))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(line, widths)) for line in table]
    return "\n".join([lines[0], "  ".join("─" * w for w in widths), *lines[1:]])


def _merge_row(old: dict[str, str] | None, new: dict[str, str]) -> dict[str, str]:
    """Keep earlier non-empty test scores when this run skipped inference on the same data."""
    if old is None or old.get("rows_used") != new["rows_used"]:
        return new
    kept = {field: old[field] for field in TEST_FIELDS if not new[field] and old.get(field)}
    return {**new, **kept}


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    """The summary CSV's rows by model key; empty if the file does not exist yet."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["model"]: row for row in csv.DictReader(handle) if row.get("model")}


def merge_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Upserts this run's rows, so running models one at a time still builds one table.

    A model's new row replaces its old one, but keeps the old test scores when this run skipped
    inference on the same number of passengers.

    Args:
        path: The summary CSV; it and its folder are created if needed.
        rows: This run's rows, as ``result_row`` makes them.
    """
    existing = _read_rows(path)
    merged = {**existing, **{row["model"]: _merge_row(existing.get(row["model"]), row) for row in rows}}
    order = {key: index for index, key in enumerate(model_keys())}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        for key in sorted(merged, key=lambda k: order.get(k, len(order))):
            writer.writerow(merged[key])
    os.replace(temporary, path)  # readers never see a half-written file
