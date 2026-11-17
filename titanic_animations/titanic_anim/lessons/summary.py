"""The summary CSV for the classifier lessons: final setting, accuracies, the unseen passenger, video facts.

Each rendered lesson adds one row to ``lessons_summary.csv`` in the output folder. A row is keyed by its
lesson, so rendering one lesson again replaces its row and keeps the others'. The command line hands
``summary_row`` and ``merge_summary`` to animkit's shared ``run``, which calls them after each video.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np

from animkit.lessons.director import RenderReport
from animkit.lessons.encoder import VideoInfo
from titanic_anim.lessons.lesson import ClassLesson, accuracy
from titanic_anim.lessons.stories import lesson_keys
from titanic_anim.lessons.stories.common import describe_passenger

SUMMARY_FIELDS = (
    "lesson", "title", "final_model", "score_kind", "training_rows", "train_accuracy", "val_accuracy",
    "test_accuracy", "baseline_test_accuracy", "query_row", "query_passenger", "query_score", "query_survived",
    "frames", "seconds", "width", "height", "fps", "codec", "resolution", "video", "config",
)


def summary_row(lesson: ClassLesson, report: RenderReport, info: VideoInfo, resolution: str) -> dict[str, str]:
    """The summary's row for one rendered lesson.

    Args:
        lesson: The lesson that was rendered.
        report: The renderer's report: where the video was written.
        info: What ffprobe read back from the video: frames, length, size, frame rate and codec.
        resolution: The ``--resolution`` it was rendered at, such as '1440p'.

    Returns:
        One text value per ``SUMMARY_FIELDS`` column.
    """
    data, final, query = lesson.data, lesson.final, lesson.data.query
    return {
        "lesson": lesson.story.key,
        "title": lesson.story.title,
        "final_model": final.label,
        "score_kind": lesson.kind,
        "training_rows": str(data.n_train),
        "train_accuracy": f"{final.train_accuracy:.4f}",
        "val_accuracy": f"{final.val_accuracy:.4f}",
        "test_accuracy": f"{accuracy(lesson.score(data.x_test), data.y_test, lesson.kind):.4f}",
        "baseline_test_accuracy": f"{float(np.mean(data.y_test == 0)):.4f}",
        "query_row": str(query.row),
        "query_passenger": describe_passenger(query.features),
        "query_score": f"{lesson.score_one(query.features):.4f}",
        "query_survived": str(query.survived),
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
    """The summary's existing rows, keyed by lesson; none if the file does not exist yet."""
    if not path.exists():
        return {}
    with open(path, newline="") as handle:
        return {row["lesson"]: row for row in csv.DictReader(handle) if row.get("lesson")}


def merge_summary(path: Path, rows: list[dict[str, str]]) -> None:
    """Upsert rows by lesson key (atomic rename; one render per folder at a time).

    The file is written in full to a temporary name, then renamed over the old one, so a render that stops
    halfway never leaves half a CSV. Rows are kept in the lessons' order.

    Args:
        path: The summary CSV.
        rows: The new rows; each replaces any existing row for its lesson.
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
    os.replace(temporary, path)
