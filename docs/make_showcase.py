"""Build the showcase media from the project's own videos.

    python docs/make_showcase.py

- docs/hero.webp                      looping 2x2 hero animation at the top of the README
- launch/highlight_video_1080p.mp4    ~50 s, 1920x1080, 30 fps highlight video with captions, for posting
- docs/social-card.png               link-preview card (Pages og:image; also for GitHub Settings -> Social preview)

Every clip is cut from a real video in this repository; nothing is re-rendered or invented.
"""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS, LAUNCH = ROOT / "docs", ROOT / "launch"
REPO_URL = "github.com/di37/animated-machine-learning-algorithm"
FOOTER = "IR · Isham Rashik · AI Engineer · Making AI simplified for all"

NAVY, BLUE, PAPER, MUTED = "#0f2541", "#1557d6", "#f2f8fb", "#6b7f95"  # animkit/lessons/theme.py
HEADER = "Engineering AI with Clarity"
FONT = ["Arial", "Helvetica", "DejaVu Sans"]
WIDTH, HEIGHT, FPS = 1920, 1080, 30

HOUSE = "house_price_animations/outputs/california_housing"
HOUSE_MANIM = "house_price_manim_lessons/outputs/california_housing/lessons"
TITANIC = "titanic_animations/outputs/titanic"


@dataclass(frozen=True)
class Clip:
    path: str
    start: float  # seconds into the source
    length: float  # seconds of source used
    speed: float  # >1 plays faster
    caption: str

    @property
    def seconds(self) -> float:
        return self.length / self.speed


HIGHLIGHTS = (
    Clip(f"{HOUSE}/decision_tree_training.mp4", 0, 8, 1.0,
         "Training, live: every frame is a real training step. The tree grows one level deeper."),
    Clip(f"{TITANIC}/gradient_boosting_inference.mp4", 0, 13, 2.0,
         "Inference: each passenger's prediction worked out, and checked against the model ✓"),
    Clip(f"{HOUSE_MANIM}/gradient_boosting_lesson.mp4", 18, 12, 2.0,
         "Narrated lessons: boosting fixes the last tree's mistakes, one correction at a time"),
    Clip(f"{HOUSE_MANIM}/gradient_boosting_lesson.mp4", 81, 8, 1.0,
         "3-D: the fitted model as a surface over income and house age"),
    Clip(f"{TITANIC}/survival_landscape.mp4", 0, 13, 2.0,
         "Who survives? Four models learn \"women and children first\" from the data"),
    Clip(f"{TITANIC}/algorithm_race.mp4", 0, 21, 3.0,
         "The Algorithm Race: 9 classifiers, one leaderboard, one final exam"),
)

HERO = (  # 2x2 grid, 8 seconds each
    Clip(f"{HOUSE}/decision_tree_training.mp4", 0, 8, 1.0, "Decision tree · training"),
    Clip(f"{TITANIC}/gradient_boosting_inference.mp4", 0, 13, 13 / 8, "Gradient boosting · inference"),
    Clip(f"{HOUSE_MANIM}/gradient_boosting_lesson.mp4", 81, 8, 1.0, "Narrated lesson · 3-D view"),
    Clip(f"{TITANIC}/survival_landscape.mp4", 0, 13, 13 / 8, "Who survives? · 4 models learning"),
)


def run(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def card(path: Path, lines: list[tuple[str, float, str, str]], size=(WIDTH, HEIGHT), alpha_bg=False) -> Path:
    """Draw text lines (text, y in 0..1, fontsize in px, colour) on a paper or transparent card."""
    w, h = size
    figure = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    figure.patch.set_alpha(0.0 if alpha_bg else 1.0)
    figure.patch.set_facecolor(PAPER)
    for text, y, fontsize, colour in lines:
        weight = "bold" if colour in (BLUE, NAVY) and fontsize >= 40 else "normal"
        figure.text(0.5, y, text, ha="center", va="center", fontsize=fontsize * 0.72, color=colour,
                    family=FONT, weight=weight)
    figure.savefig(path, transparent=alpha_bg)
    plt.close(figure)
    return path


def caption_overlay(path: Path, text: str) -> Path:
    """A lower-third bar: navy band with white text, transparent elsewhere."""
    figure = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    figure.patch.set_alpha(0.0)
    band = figure.add_axes((0.0, 0.0, 1.0, 0.11))
    band.set_axis_off()
    band.add_patch(plt.Rectangle((0, 0), 1, 1, transform=band.transAxes, color=NAVY, alpha=0.88))
    band.text(0.5, 0.5, text, ha="center", va="center", fontsize=30 * 0.72 * 1.4, color="white",
              family=FONT, weight="bold", transform=band.transAxes)
    figure.savefig(path, transparent=True)
    plt.close(figure)
    return path


def render_clip(clip: Clip, overlay: Path, out: Path) -> Path:
    fit = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,"
           f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color={PAPER},setsar=1")
    run("-ss", str(clip.start), "-t", str(clip.length), "-i", str(ROOT / clip.path), "-i", str(overlay),
        "-filter_complex", f"[0:v]setpts=PTS/{clip.speed},fps={FPS},{fit}[v];[v][1:v]overlay=0:0,format=yuv420p",
        "-an", "-c:v", "libx264", "-crf", "19", "-preset", "medium", str(out))
    return out


def still_clip(image: Path, seconds: float, out: Path) -> Path:
    run("-loop", "1", "-t", str(seconds), "-i", str(image), "-vf", f"fps={FPS},format=yuv420p",
        "-c:v", "libx264", "-crf", "19", "-preset", "medium", str(out))
    return out


def highlight_video(work: Path) -> Path:
    title = card(work / "title.png", [
        ("MACHINE LEARNING, ANIMATED", 0.66, 30, BLUE),
        ("Watch the algorithms learn.", 0.54, 84, NAVY),
        ("Then watch them predict.", 0.43, 84, BLUE),
        ("Real training states · real predictions · every answer checked", 0.30, 34, MUTED),
    ])
    end = card(work / "end.png", [
        ("76 videos · 2 datasets · regression and classification", 0.66, 40, NAVY),
        ("Every frame from a real model state.", 0.56, 40, MUTED),
        (REPO_URL, 0.42, 64, BLUE),
        ("Star it, share it, learn from it", 0.32, 36, NAVY),
        (FOOTER, 0.12, 26, MUTED),
    ])
    parts = [still_clip(title, 3.5, work / "00_title.mp4")]
    for index, clip in enumerate(HIGHLIGHTS, start=1):
        overlay = caption_overlay(work / f"caption_{index}.png", clip.caption)
        parts.append(render_clip(clip, overlay, work / f"{index:02d}_clip.mp4"))
    parts.append(still_clip(end, 4.5, work / "99_end.mp4"))
    listing = work / "parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    out = LAUNCH / "highlight_video_1080p.mp4"
    run("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out))
    return out


def hero(work: Path) -> Path:
    cells = []
    for index, clip in enumerate(HERO):
        overlay = work / f"hero_label_{index}.png"
        figure = plt.figure(figsize=(6.4, 3.6), dpi=100)
        figure.patch.set_alpha(0.0)
        figure.text(0.02, 0.05, f" {clip.caption} ", fontsize=13, color="white", family=FONT, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=NAVY, edgecolor="none", alpha=0.9))
        figure.savefig(overlay, transparent=True)
        plt.close(figure)
        cell = work / f"hero_{index}.mp4"
        run("-ss", str(clip.start), "-t", str(clip.length), "-i", str(ROOT / clip.path), "-i", str(overlay),
            "-filter_complex",
            f"[0:v]setpts=PTS/{clip.speed},fps=10,scale=640:360:flags=lanczos,setsar=1[v];[v][1:v]overlay=0:0",
            "-t", "8", "-an", "-c:v", "libx264", "-crf", "16", str(cell))
        cells.append(cell)
    out = DOCS / "hero.webp"
    inputs = [arg for cell in cells for arg in ("-i", str(cell))]
    run(*inputs, "-filter_complex",
        "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=white[v]", "-map", "[v]",
        "-c:v", "libwebp_anim", "-lossless", "0", "-q:v", "60", "-compression_level", "6", "-loop", "0",
        str(out))
    return out


def social_card() -> Path:
    """1280x640: title on the left, four real frames on the right."""
    figure = plt.figure(figsize=(12.8, 6.4), dpi=100)
    figure.patch.set_facecolor(PAPER)
    figure.text(0.05, 0.80, "MACHINE LEARNING, ANIMATED", fontsize=15, color=BLUE, family=FONT, weight="bold")
    figure.text(0.05, 0.62, "Watch the algorithms\nlearn, then predict.", fontsize=34, color=NAVY,
                family=FONT, weight="bold", va="center", linespacing=1.15)
    figure.text(0.05, 0.40, "21 scikit-learn models · 2 datasets\n76 videos · every frame a real model state",
                fontsize=15, color=MUTED, family=FONT, va="center", linespacing=1.5)
    figure.text(0.05, 0.12, REPO_URL, fontsize=14, color=BLUE, family=FONT, weight="bold")
    frames = [(f"{HOUSE}/decision_tree_inference.mp4", 9), (f"{TITANIC}/gradient_boosting_inference.mp4", 10),
              (f"{HOUSE_MANIM}/gradient_boosting_lesson.mp4", 85), (f"{TITANIC}/survival_landscape.mp4", 12)]
    with tempfile.TemporaryDirectory() as tmp:
        for index, (path, second) in enumerate(frames):
            still = Path(tmp) / f"frame_{index}.png"
            run("-ss", str(second), "-i", str(ROOT / path), "-frames:v", "1", "-vf", "scale=640:-2", str(still))
            row, column = divmod(index, 2)
            ax = figure.add_axes((0.47 + column * 0.265, 0.53 - row * 0.40, 0.255, 0.36))
            ax.imshow(plt.imread(still))
            ax.set_axis_off()
    figure.text(0.95, 0.92, HEADER, ha="right", fontsize=13, color=NAVY, family=FONT, weight="bold")
    out = DOCS / "social-card.png"
    figure.savefig(out, facecolor=PAPER)
    plt.close(figure)
    return out


def main() -> None:
    LAUNCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for made in (hero(work), highlight_video(work), social_card()):
            print(f"{made.relative_to(ROOT)}  {made.stat().st_size / 1024:,.0f} KB")


if __name__ == "__main__":
    main()
