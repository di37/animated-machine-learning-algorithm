"""Make a small looping animated WebP preview for every finished video, for the GitHub README.

Usage: python docs/make_previews.py [video paths relative to the repo root ...]
Previews go to docs/previews/<same path>.webp. Dashboards and scenes play at real speed (sampled at
6 fps); 108-second lessons become a ~9-second time-lapse (one frame every 1.2 s, played at 10 fps).
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "docs" / "previews"
WIDTH = 640
QUALITY = 50
LESSON_SAMPLE = "fps=1/1.2"
LESSON_PLAYBACK_FPS = 10
REALTIME_FPS = 6
TIME_LAPSED = {"lessons", "series_voiced"}  # folders whose long videos become a time-lapse


def preview_path(video: Path) -> Path:
    return PREVIEW_DIR / video.relative_to(ROOT).with_suffix(".webp")


def encode(video: Path) -> tuple[str, int]:
    lesson = bool(TIME_LAPSED & set(video.parts))
    sample, fps = (LESSON_SAMPLE, LESSON_PLAYBACK_FPS) if lesson else (f"fps={REALTIME_FPS}", REALTIME_FPS)
    out = preview_path(video)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"{sample},scale={WIDTH}:-2:flags=lanczos,setpts=N/({fps}*TB)",
         "-r", str(fps), "-c:v", "libwebp_anim", "-lossless", "0", "-q:v", str(QUALITY),
         "-compression_level", "6", "-loop", "0", "-an", str(out)],
        check=True,
    )
    return str(out.relative_to(ROOT)), out.stat().st_size


def finished_videos() -> list[Path]:
    return sorted(p for p in ROOT.glob("*/outputs/**/*.mp4") if ".partial." not in p.name)


def main() -> None:
    videos = [ROOT / v for v in sys.argv[1:]] or finished_videos()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(encode, videos))
    total = sum(size for _, size in results)
    for path, size in results if len(results) <= 6 else []:
        print(f"{size / 1024:7.0f} KB  {path}")
    print(f"{len(results)} previews, {total / 1024 / 1024:.1f} MB total, largest {max(s for _, s in results) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
