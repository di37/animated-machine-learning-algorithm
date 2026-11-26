"""Write a YouTube upload pack for the voiced Manim lessons: everything to paste into YouTube Studio.

    python docs/make_youtube_pack.py      (after the lessons are voiced; needs ffmpeg and Pillow)

For each lesson, in launch/youtube/<dataset>/ (git-ignored, like the rest of the launch kit):

- ``NN_<key>.txt``: the title, the description (with chapters, links and hashtags) and the tags
- ``NN_<key>.srt``: English subtitles of exactly what the voice says, timed to it
- ``NN_<key>.jpg``: a 1280×720 thumbnail made from the lesson's 3-D view

plus ``metadata.csv`` (one row per video) and ``UPLOAD.md`` (the steps and settings). Chapters and subtitles
come from each video's narration file, so they match the voiced video to the frame. Abbreviations are
written with their full names, as the voice says them.
"""

import csv
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS))
sys.path.insert(0, str(ROOT / "ml_animation_kit"))

from animkit.lessons.speech_text import JARGON  # noqa: E402
from animkit.lessons.youtube import chapter_list, subtitles, timestamp, youtube_chapters  # noqa: E402
from catalogue import (  # noqa: E402
    HOUSE_LESSONS_LIST, HOUSE_MANIM_VOICED, PAGES_URL, REPO_URL, TITANIC_LESSONS_LIST, TITANIC_MANIM_VOICED,
)

OUT = ROOT / "launch" / "youtube"
TITLE_MAX = 100  # YouTube's limits
DESCRIPTION_MAX = 5000
TAGS_MAX = 500
THUMB_SIZE = (1280, 720)
THUMB_MAX_BYTES = 2_000_000
THUMB_QUALITY = 90
THUMB_AT = 0.55  # how far into the 3-D view chapter the thumbnail frame is taken
FONT_BOLD = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf")
NAVY, ORANGE, WHITE, PALE = (15, 37, 65), (235, 104, 52), (255, 255, 255), (207, 227, 255)
MARGIN = 56
TITLE_SIZES = (84, 76, 68, 60)
BAND_FROM = 0.42  # the dark band behind the text starts this far down the thumbnail
VIDEO_COUNT = 76

# The lesson names as titles say them: abbreviations in full, as the voice says them.
NAMES = {
    "linear_regression": "Linear Regression", "polynomial_regression": "Polynomial Regression",
    "ridge": "Ridge Regression", "lasso": "Lasso Regression", "decision_tree": "Decision Tree",
    "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting", "knn": "k-Nearest Neighbours",
    "mlp": "Multi-Layer Perceptron (Neural Network)", "logistic_regression": "Logistic Regression",
    "perceptron": "Perceptron", "linear_svm": "Linear Support Vector Machine", "naive_bayes": "Naive Bayes",
}


@dataclass(frozen=True)
class Dataset:
    """One playlist's worth of lessons and the words that describe them."""

    key: str
    folder: str
    lessons: tuple
    learns: str
    headline: str
    playlist: str
    credit: str
    unseen: str
    tags: tuple[str, ...]


DATASETS = (
    Dataset("house", HOUSE_MANIM_VOICED, HOUSE_LESSONS_LIST, "California house prices", "California House Prices",
            "Machine Learning Algorithms, Animated: California House Prices",
            "California Housing (scikit-learn), from the 1990 US census", "a district it has never seen",
            ("california housing", "regression", "house price prediction")),
    Dataset("titanic", TITANIC_MANIM_VOICED, TITANIC_LESSONS_LIST, "who survived the Titanic",
            "Who Survived the Titanic",
            "Machine Learning Algorithms, Animated: Titanic Survival",
            "Titanic passenger list (OpenML 'titanic', version 1)", "a passenger it has never seen",
            ("titanic", "classification", "titanic survival prediction")),
)
COMMON_TAGS = ("machine learning", "machine learning algorithms", "scikit-learn", "python", "data science",
               "manim", "animation", "explained", "tutorial", "how it learns")
WRITTEN = re.compile(r"\b(" + "|".join(sorted(map(re.escape, JARGON), key=len, reverse=True)) + r")\b")


def written(text: str) -> str:
    """``text`` with each abbreviation given its full name: "on the real MSE bowl" as "… mean squared error
    (MSE) bowl"."""
    return WRITTEN.sub(lambda match: f"{JARGON[match.group(1)]} ({match.group(1)})", text)


def safe(text: str) -> str:
    """YouTube refuses titles and descriptions with angle brackets."""
    return text.replace("<", "less than").replace(">", "more than")


def youtube_title(name: str, dataset: Dataset) -> str:
    """For example "Linear Regression, Animated: Watch It Learn California House Prices"."""
    title = safe(f"{name}, Animated: Watch It Learn {dataset.headline}")
    if len(title) > TITLE_MAX:
        raise SystemExit(f"title over {TITLE_MAX} characters: {title}")
    return title


def description(name: str, words: str, dataset: Dataset, chapters: str, minutes: str, index: int,
                site_id: str) -> str:
    """The description: what it is, what you will see, chapters, links and hashtags."""
    text = safe(f"""{name} learns {dataset.learns} from real data, step by step, in a {minutes} narrated animation. \
Every number and curve on screen comes from the real trained scikit-learn model.

What you'll see: {written(words)}. Then the final model, a prediction for {dataset.unseen}, and the final exam on \
data it never saw in training.

Chapters
{chapters}

▶ All {VIDEO_COUNT} videos, with live training dashboards and algorithm races: {PAGES_URL}#v={site_id}
💻 Open-source code: {REPO_URL}
📊 Data: {dataset.credit}

Lesson {index} of {len(dataset.lessons)} in "{dataset.playlist}".

#MachineLearning #DataScience #Python""")
    if len(text) > DESCRIPTION_MAX:
        raise SystemExit(f"description over {DESCRIPTION_MAX} characters for {name}")
    return text


def tags(name: str, dataset: Dataset) -> str:
    """The tags, comma-separated, within YouTube's 500 characters."""
    chosen = ", ".join(dict.fromkeys((name.lower(), *dataset.tags, *COMMON_TAGS)))
    if len(chosen) > TAGS_MAX:
        raise SystemExit(f"tags over {TAGS_MAX} characters for {name}")
    return chosen


def _font(size: int) -> ImageFont.FreeTypeFont:
    """The bold font, Arial where it is installed."""
    for path in FONT_BOLD:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise SystemExit("no bold font found for the thumbnails")


def _lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """``text`` wrapped to ``width`` pixels."""
    lines: list[str] = []
    for word in text.split():
        trial = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(trial, font=font) <= width:
            lines = [*lines[:-1], trial]
        else:
            lines = [*lines, word]
    return lines


def _fit_title(draw: ImageDraw.ImageDraw, name: str, width: int) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """The largest title size at which ``name`` fits on two lines (the smallest size if none does)."""
    for size in TITLE_SIZES:
        font = _font(size)
        lines = _lines(draw, name, font, width)
        if len(lines) <= 2:
            return font, lines
    return font, lines


def _band(size: tuple[int, int]) -> Image.Image:
    """A navy band fading in from transparent, behind the thumbnail's text."""
    width, height = size
    band = Image.new("RGBA", size, (*NAVY, 0))
    top = int(height * BAND_FROM)
    pixels = ImageDraw.Draw(band)
    for y in range(top, height):
        alpha = int(240 * min(1.0, (y - top) / (height * 0.22)))
        pixels.line([(0, y), (width, y)], fill=(*NAVY, alpha))
    return band


def thumbnail(video: Path, at_s: float, name: str, learns: str, out: Path) -> None:
    """A 1280×720 JPEG: the frame at ``at_s`` with the lesson's name over a navy band."""
    with tempfile.TemporaryDirectory() as work:
        frame = Path(work) / "frame.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at_s:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={THUMB_SIZE[0]}:{THUMB_SIZE[1]}", str(frame)], check=True)
        image = Image.alpha_composite(Image.open(frame).convert("RGBA"), _band(THUMB_SIZE))
    draw = ImageDraw.Draw(image)
    font, lines = _fit_title(draw, name, THUMB_SIZE[0] - 2 * MARGIN)
    sub_font, pill_font = _font(38), _font(26)
    y = THUMB_SIZE[1] - MARGIN - 46 - len(lines) * (font.size + 8)
    pill = "ANIMATED LESSON"
    pill_w = draw.textlength(pill, font=pill_font) + 32
    draw.rounded_rectangle([MARGIN, y - 58, MARGIN + pill_w, y - 14], radius=10, fill=ORANGE)
    draw.text((MARGIN + 16, y - 52), pill, font=pill_font, fill=WHITE)
    for line in lines:
        draw.text((MARGIN, y), line, font=font, fill=WHITE)
        y += font.size + 8
    draw.text((MARGIN, y + 6), f"learns {learns}, step by step", font=sub_font, fill=PALE)
    image.convert("RGB").save(out, "JPEG", quality=THUMB_QUALITY, optimize=True)
    if out.stat().st_size > THUMB_MAX_BYTES:
        raise SystemExit(f"thumbnail over 2 MB: {out}")


def pack_one(dataset: Dataset, index: int, lesson: tuple, out_dir: Path) -> dict[str, str]:
    """Writes one lesson's text, subtitles and thumbnail; returns its metadata row."""
    key, _emoji, _title, words = lesson
    video = ROOT / dataset.folder / f"{key}_lesson.mp4"
    plan = json.loads(video.with_suffix(".narration.json").read_text(encoding="utf-8"))
    rows, seconds = plan["chapters"], float(plan["seconds"])
    name, stem = NAMES[key], f"{index:02d}_{key}"
    chapters = chapter_list(youtube_chapters(rows, seconds))
    title = youtube_title(name, dataset)
    body = description(name, words, dataset, chapters, timestamp(seconds), index,
                       f"{dataset.key}-manim-{key}")
    tag_text = tags(name, dataset)
    (out_dir / f"{stem}.txt").write_text(f"TITLE\n{title}\n\nDESCRIPTION\n{body}\n\nTAGS\n{tag_text}\n",
                                         encoding="utf-8")
    (out_dir / f"{stem}.srt").write_text(subtitles(rows), encoding="utf-8")
    surface = next(row for row in rows if row["chapter"] == "surface")
    thumbnail(video, surface["voice_at_s"] + THUMB_AT * surface["voice_s"], name, dataset.learns,
              out_dir / f"{stem}.jpg")
    return {"order": str(index), "file": str(video.relative_to(ROOT)), "title": title, "length": timestamp(seconds),
            "playlist": dataset.playlist, "subtitles": f"{stem}.srt", "thumbnail": f"{stem}.jpg", "tags": tag_text}


def upload_guide(dataset: Dataset, rows: list[dict[str, str]]) -> str:
    """UPLOAD.md: the playlist, then the same steps for each video, then the videos in order."""
    table = "\n".join(f"| {row['order']} | `{row['file']}` | {row['title']} | {row['length']} |" for row in rows)
    return f"""# YouTube upload: {dataset.playlist}

Everything here is ready to paste into YouTube Studio. Nothing has been uploaded.

## Once: the playlist

YouTube Studio → **Content** → **Playlists** → **New playlist**: title "{dataset.playlist}", visibility
Public. Upload the videos in the order below, so the playlist plays them in lesson order.

## For each video (01 to {len(rows):02d})

1. **Create → Upload videos**, and pick the video file from the table below (paths are from the repo root).
   Studio takes up to 15 files at once; each still needs its details.
2. **Details:** paste the title and description from `NN_<lesson>.txt`, upload `NN_<lesson>.jpg` as the
   thumbnail, choose the playlist, and set **Audience** to "No, it's not made for kids".
3. **Show more:** paste the tags from the same `.txt`. Language: English. Caption certification: "This
   content has never aired on television in the US". Category: Education. Altered or synthetic content:
   "No" for a stock synthetic narrator voice; "Yes" if the voice is a clone of a real person.
4. **Video elements → Add subtitles → Upload file → With timing:** `NN_<lesson>.srt` (English). These
   are exactly the words the voice says, so they are better than YouTube's automatic captions.
5. **Visibility:** Public (or Schedule, to release one lesson a day).

The description's chapter list makes YouTube split the progress bar into the lesson's chapters.

## Videos

| # | File | Title | Length |
|---|---|---|---|
{table}
"""


def main() -> None:
    for dataset in DATASETS:
        out_dir = OUT / Path(dataset.folder).parts[-2]  # the dataset's name, e.g. california_housing
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [pack_one(dataset, index, lesson, out_dir) for index, lesson in enumerate(dataset.lessons, start=1)]
        with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out_dir / "UPLOAD.md").write_text(upload_guide(dataset, rows), encoding="utf-8")
        print(f"{out_dir.relative_to(ROOT)}: {len(rows)} videos")


if __name__ == "__main__":
    main()
