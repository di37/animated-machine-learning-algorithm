"""lessons_summary.csv: one row per rendered lesson, with the numbers and settings needed to reproduce it.

Each row holds the final model's train, validation and test errors (in $k) beside the average-price
baseline, the unseen district's predicted and true price, the video's own properties as ffprobe
reports them, and the lesson's model settings as JSON. Lessons can be rendered one at a time:
``merge_summary`` replaces a lesson's old row and keeps the others, in the lessons' own order. The
shared lesson command line (animkit.lessons.cli) calls both functions after each render.
"""

import csv
import json
import os
from pathlib import Path

from animkit.lessons.encoder import VideoInfo
from house_anim.lessons.config import INCOME_TO_K, PRICE_TO_K
from house_anim.lessons.director import RenderReport
from house_anim.lessons.lesson import Lesson, baseline_rmse_k, predict_one, rmse_k
from house_anim.lessons.stories import lesson_keys

SUMMARY_FIELDS = (
    "lesson", "title", "final_model", "training_rows", "train_rmse_k", "val_rmse_k", "test_rmse_k",
    "baseline_test_rmse_k", "query_row", "query_income_k", "query_predicted_k", "query_actual_k",
    "frames", "seconds", "width", "height", "fps", "codec", "resolution", "video", "config",
)


def summary_row(lesson: Lesson, report: RenderReport, info: VideoInfo, resolution: str) -> dict[str, str]:
    """The summary's row for one rendered lesson.

    Args:
        lesson: The lesson that was rendered.
        report: The director's report: where the video was written.
        info: The video's properties, as probed from the file.
        resolution: The resolution name it was rendered at, e.g. "1440p".

    Returns:
        One text value per field of ``SUMMARY_FIELDS``; money is in $k, to two decimal places.
    """
    data, final = lesson.data, lesson.final
    baseline = baseline_rmse_k(data, data.y_test)
    return {
        "lesson": lesson.story.key,
        "title": lesson.story.title,
        "final_model": final.label,
        "training_rows": str(data.n_train),
        "train_rmse_k": f"{final.train_rmse_k:.2f}",
        "val_rmse_k": f"{final.val_rmse_k:.2f}",
        "test_rmse_k": f"{rmse_k(lesson.predict(data.x_test), data.y_test):.2f}",
        "baseline_test_rmse_k": f"{baseline:.2f}",
        "query_row": str(data.query.row),
        "query_income_k": f"{data.query.income * INCOME_TO_K:.2f}",
        "query_predicted_k": f"{predict_one(lesson.predict, data.query.income) * PRICE_TO_K:.2f}",
        "query_actual_k": f"{data.query.price * PRICE_TO_K:.2f}",
        "frames": str(info.frames),
        "seconds": f"{info.duration_s:.2f}",
        "width": str(info.width),
        "height": str(info.height),
        "fps": f"{info.fps:g}",
        "codec": f"{info.codec}/{info.pix_fmt}",
        "resolution": resolution,
        "video": str(report.path),
        "config": json.dumps(dict(lesson.config), ensure_ascii=False),
    }


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    """The summary's existing rows by lesson key; none if the file does not exist yet."""
    if not path.exists():
        return {}
    with open(path, newline="") as handle:
        return {row["lesson"]: row for row in csv.DictReader(handle) if row.get("lesson")}


def merge_summary(path: Path, rows: list[dict[str, str]]) -> None:
    """Upsert rows by lesson key, so rendering lessons one at a time still builds one table.

    The rename is atomic, but the read-modify-write is not locked: run one render per folder.

    The table is written to a temporary file and then renamed over the old one, with its rows in the
    lessons' registry order (unknown keys last).

    Args:
        path: The summary CSV; it and its folder are created if missing.
        rows: The new rows, each with a "lesson" key; they replace any old rows with the same key.
    """
    merged = {**_read_rows(path), **{row["lesson"]: row for row in rows}}
    order = {key: index for index, key in enumerate(lesson_keys())}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        writer.writeheader()
        for key in sorted(merged, key=lambda k: order.get(k, len(order))):
            writer.writerow(merged[key])
    os.replace(temporary, path)  # readers never see a half-written file
