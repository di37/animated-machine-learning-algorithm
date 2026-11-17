"""Console table and CSV summary of every model run.

``metrics_summary.csv`` holds one row per model and is rewritten after every model, so running
the models one at a time still builds a single table. A row from a run that skipped inference
keeps the test scores an earlier run measured on the same data (the same ``rows_used``). The file
is written to a temporary file first and then moved into place, so a reader never sees half of it.
"""

import csv
import os
from pathlib import Path

from house_anim.config import TARGET_TO_THOUSANDS
from house_anim.models import model_keys
from house_anim.runner import ModelResult

SUMMARY_FIELDS = (
    "model", "title", "rows_used", "batch_size", "fit_seconds", "total_fit_seconds",
    "val_rmse_k", "val_r2", "test_rmse_k", "test_mae_k", "test_r2", "throughput_per_s", "videos",
)
TEST_FIELDS = ("test_rmse_k", "test_mae_k", "test_r2", "throughput_per_s")


def result_row(result: ModelResult, *, rows_used: int, batch_size: int) -> dict[str, str]:
    """One CSV row. fit_seconds is the cost of the final model; total_fit_seconds sums a sweep.

    Prices are in $k. The test columns are empty when inference did not run.

    Args:
        result: One model's run.
        rows_used: Rows in all three splits, which tells runs on different subsamples apart.
        batch_size: Test houses per predict call in this run.

    Returns:
        The row as text, keyed by ``SUMMARY_FIELDS``.
    """
    training, inference = result.final_training, result.final_inference
    row = {
        "model": result.key,
        "title": result.title,
        "rows_used": str(rows_used),
        "batch_size": str(batch_size),
        "fit_seconds": f"{training.model_fit_seconds:.3f}",
        "total_fit_seconds": f"{training.fit_seconds:.3f}",
        "val_rmse_k": f"{training.val_scores.rmse * TARGET_TO_THOUSANDS:.2f}",
        "val_r2": f"{training.val_scores.r2:.4f}",
        **{field: "" for field in TEST_FIELDS},
        "videos": ";".join(str(p) for p in result.videos),
    }
    if inference is not None:
        row = {
            **row,
            "test_rmse_k": f"{inference.scores.rmse * TARGET_TO_THOUSANDS:.2f}",
            "test_mae_k": f"{inference.scores.mae * TARGET_TO_THOUSANDS:.2f}",
            "test_r2": f"{inference.scores.r2:.4f}",
            "throughput_per_s": f"{inference.throughput_per_s:.0f}",
        }
    return row


def format_table(rows: list[dict[str, str]]) -> str:
    """The rows as an aligned console table of fit time and scores; an empty score shows as a dash.

    Args:
        rows: Rows from ``result_row``.

    Returns:
        The table: a header, a rule, and one line per row.
    """
    columns = ("model", "fit_seconds", "val_rmse_k", "test_rmse_k", "test_mae_k", "test_r2")
    headers = ("model", "fit s", "val RMSE $k", "test RMSE $k", "test MAE $k", "test R²")
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
    """The summary's existing rows by model key; empty when there is no file yet."""
    if not path.exists():
        return {}
    with open(path, newline="") as handle:
        return {row["model"]: row for row in csv.DictReader(handle) if row.get("model")}


def merge_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Upsert this run's rows so running models one at a time still builds one table.

    Rows for other models are kept, and every row is written in ``model_keys()`` order.

    Args:
        path: The CSV file; it and its folder are created if missing.
        rows: This run's rows, from ``result_row``.
    """
    existing = _read_rows(path)
    merged = {**existing, **{row["model"]: _merge_row(existing.get(row["model"]), row) for row in rows}}
    order = {key: index for index, key in enumerate(model_keys())}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        for key in sorted(merged, key=lambda k: order.get(k, len(order))):
            writer.writerow(merged[key])
    os.replace(temporary, path)  # readers never see a half-written file
