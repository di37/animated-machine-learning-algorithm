"""Write the GitHub Pages site, docs/index.html, in the lesson videos' theme.

    python docs/build_site.py     (after docs/build_readme.py; needs the `markdown` package and ffprobe)

The gallery comes from catalogue.py; the explanatory sections are rendered from README.md, so the two
never disagree. Full videos stream from jsDelivr (the Pages site serves docs/ only); previews, the
hero and the social card are served from docs/.
"""

import html
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS))

from catalogue import (  # noqa: E402
    CDN, HOUSE_DASH, HOUSE_LESSONS, HOUSE_LESSONS_LIST, HOUSE_MANIM, HOUSE_MODELS, HOUSE_SCENES,
    PAGES_URL, REPO_URL, TITANIC_DASH, TITANIC_LESSONS, TITANIC_LESSONS_LIST, TITANIC_MANIM,
    TITANIC_MODELS, TITANIC_SCENES,
)

HEADER = "Engineering AI with Clarity"
FOOTER = "IR · Isham Rashik · AI Engineer · Making AI simplified for all"
TITLE = "Animated Machine Learning Algorithms"
DESCRIPTION = ("Watch 21 scikit-learn models learn two classic datasets, frame by frame, then explain every prediction. "
               "76 videos: live dashboards, algorithm races and narrated lessons. The black box, opened.")
# README sections shown on the site, by the start of their heading (emoji included).
SECTIONS = {
    "why": "🆚 Why this is different",
    "gain": "🎁 What you'll get",
    "black-box": "🔍 Opening the black box",
    "how": "👀 How well training",
    "scoreboard": "🏆 Scoreboard",
    "truth": "🔒 Truthfulness rules",
    "run": "🚀 Quick start",
}
DATASETS = {"house": "🏠 California Housing", "titanic": "🚢 Titanic"}
KINDS = {"dashboards": "📊 Live dashboards", "scenes": "🏁 Comparison scenes",
         "lessons": "🎓 Narrated lessons", "manim": "🎞️ Lessons in Manim"}


@dataclass(frozen=True)
class Video:
    id: str
    dataset: str
    kind: str
    title: str
    badge: str
    description: str
    path: str

    @property
    def preview(self) -> str:
        return "previews/" + str(Path(self.path).with_suffix(".webp"))


def video_facts(path: str) -> str:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(ROOT / path)],
                         capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    stream = info["streams"][0]
    return f"{float(info['format']['duration']):.0f} s · {stream['width']}×{stream['height']}"


def catalogue() -> list[Video]:
    videos: list[Video] = []
    for dataset, folder, models in (("house", HOUSE_DASH, HOUSE_MODELS), ("titanic", TITANIC_DASH, TITANIC_MODELS)):
        for key, emoji, title, training, inference in models:
            for phase, words in (("training", training), ("inference", inference)):
                videos.append(Video(f"{dataset}-{key}-{phase}", dataset, "dashboards", f"{emoji} {title}",
                                    phase.capitalize(), words, f"{folder}/{key}_{phase}.mp4"))
    for dataset, scenes in (("house", HOUSE_SCENES), ("titanic", TITANIC_SCENES)):
        for title, blurb, path in scenes:
            videos.append(Video(f"{dataset}-{Path(path).stem}", dataset, "scenes", title, "Scene", blurb, path))
    for dataset, kind, folder, lessons in (("house", "lessons", HOUSE_LESSONS, HOUSE_LESSONS_LIST),
                                           ("house", "manim", HOUSE_MANIM, HOUSE_LESSONS_LIST),
                                           ("titanic", "lessons", TITANIC_LESSONS, TITANIC_LESSONS_LIST),
                                           ("titanic", "manim", TITANIC_MANIM, TITANIC_LESSONS_LIST)):
        badge = "Manim lesson" if kind == "manim" else "Lesson"
        for key, emoji, title, words in lessons:
            videos.append(Video(f"{dataset}-{kind}-{key}", dataset, kind, f"{emoji} {title}", badge, words,
                                f"{folder}/{key}_lesson.mp4"))
    return videos


def readme_sections() -> dict[str, str]:
    """Render the chosen README sections; repo-relative links point at GitHub instead."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## ", text)
    found = {}
    for name, prefix in SECTIONS.items():
        body = next((p for p in parts if p.startswith(prefix)), None)
        if body is None:
            raise SystemExit(f"README has no section starting '## {prefix}'")
        rendered = markdown.markdown("## " + body, extensions=["tables", "fenced_code"])
        found[name] = re.sub(r'href="(?!https?:|#|mailto:)([^"]+)"', rf'href="{REPO_URL}/blob/main/\1"', rendered)
    return found


def card(video: Video) -> str:
    e = html.escape
    return (f'<button class="card" type="button" data-id="{e(video.id)}" data-src="{CDN}/{e(video.path)}" '
            f'data-title="{e(video.title)} · {e(video.badge)}" data-meta="{e(video_facts(video.path))}">'
            f'<span class="thumb"><img src="{e(video.preview)}" loading="lazy" alt="Preview: {e(video.title)}, '
            f'{e(video.badge.lower())}"><span class="play" aria-hidden="true">▶</span>'
            f'<span class="badge badge-{e(video.badge.split()[0].lower())}">{e(video.badge)}</span></span>'
            f'<span class="card-title">{e(video.title)}</span>'
            f'<span class="card-desc">{e(video.description)}</span></button>')


def gallery(videos: list[Video]) -> str:
    groups = []
    for dataset, dataset_name in DATASETS.items():
        for kind, kind_name in KINDS.items():
            members = [v for v in videos if v.dataset == dataset and v.kind == kind]
            groups.append(f'<div class="group" data-dataset="{dataset}" data-kind="{kind}">'
                          f'<h3>{dataset_name} · {kind_name} <span class="count">{len(members)}</span></h3>'
                          f'<div class="grid">{"".join(card(v) for v in members)}</div></div>')
    return "\n".join(groups)


def chips(attribute: str, options: dict[str, str]) -> str:
    buttons = [f'<button type="button" class="chip" data-{attribute}="all" aria-pressed="true">All</button>']
    buttons += [f'<button type="button" class="chip" data-{attribute}="{key}" aria-pressed="false">{label}</button>'
                for key, label in options.items()]
    return "".join(buttons)


def page(videos: list[Video], sections: dict[str, str]) -> str:
    e = html.escape
    nav = "".join(f'<a href="#{key}">{label}</a>' for key, label in
                  (("why", "Why"), ("gallery", "Gallery"), ("black-box", "Black box"),
                   ("scoreboard", "Scoreboard"), ("run", "Run it")))
    prose = {name: f'<section id="{name}" class="panel prose">{body}</section>' for name, body in sections.items()}
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE} · Isham Rashik</title>
<meta name="description" content="{e(DESCRIPTION)}">
<link rel="canonical" href="{PAGES_URL}">
<meta property="og:type" content="website">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{e(DESCRIPTION)}">
<meta property="og:url" content="{PAGES_URL}">
<meta property="og:image" content="{PAGES_URL}social-card.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230f2541'/%3E%3Ctext x='32' y='42' font-family='Arial' font-weight='700' font-size='28' fill='%23ffffff' text-anchor='middle'%3EIR%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<header class="topbar">
  <a class="brand" href="#top"><span class="mark">IR</span><span class="brand-name">{TITLE}</span></a>
  <nav aria-label="Sections">{nav}</nav>
  <span class="header-line">{HEADER}</span>
  <a class="btn btn-small" href="{REPO_URL}">★ Star on GitHub</a>
</header>
<main id="top">
<section class="hero">
  <div class="hero-text">
    <p class="eyebrow">MACHINE LEARNING, ANIMATED</p>
    <h1>Watch the algorithm itself learn.<br><span>Then watch it explain every prediction.</span></h1>
    <p class="lead">21 scikit-learn models learn California house prices and Titanic survival, two classic datasets you
    already know, frame by frame. Every frame is a real model state, and every prediction is worked out and checked
    against the model's own answer. <b>The black box, opened.</b></p>
    <div class="cta"><a class="btn btn-primary" href="#gallery">▶ Watch the {len(videos)} videos</a>
    <a class="btn" href="{REPO_URL}">★ Star on GitHub</a></div>
    <ul class="stats">
      <li><b>{len(videos)}</b><span>videos</span></li><li><b>21</b><span>models</span></li>
      <li><b>2</b><span>classic datasets</span></li><li><b>✓</b><span>every prediction checked</span></li>
    </ul>
  </div>
  <figure class="hero-media panel"><img src="hero.webp" alt="Four of the videos looping: a decision tree growing,
  gradient boosting explaining a Titanic passenger, a lesson's 3-D view, and four models learning who survived">
  <figcaption>Real clips from the gallery below. Every frame comes from a real training state.</figcaption></figure>
</section>
{prose["why"]}
{prose["gain"]}
<section id="gallery">
  <h2>🎬 The gallery</h2>
  <p class="section-lead">Every card loops a preview of the real video. Click one to play it in full;
  the link in the player opens straight to that video, ready to share.</p>
  <div class="filters"><div class="chips" role="group" aria-label="Dataset">{chips("dataset", DATASETS)}</div>
  <div class="chips" role="group" aria-label="Kind of video">{chips("kind", KINDS)}</div></div>
  {gallery(videos)}
</section>
{prose["black-box"]}
{prose["how"]}
{prose["scoreboard"]}
{prose["truth"]}
{prose["run"]}
<section class="panel closing">
  <h2>⭐ If an algorithm finally clicked for you</h2>
  <p>Star the repository so more learners find it, and share a video with someone learning ML.</p>
  <a class="btn btn-primary" href="{REPO_URL}">★ Star on GitHub</a>
</section>
</main>
<footer class="footer"><span>{FOOTER}</span><span>Data: California Housing (scikit-learn) · Titanic (OpenML)</span></footer>
<dialog id="player" aria-labelledby="player-title">
  <div class="player-head"><h3 id="player-title"></h3>
  <button type="button" class="icon-btn" data-close aria-label="Close">✕</button></div>
  <video controls playsinline preload="none"></video>
  <div class="player-foot"><span class="player-meta"></span>
  <span><button type="button" class="btn btn-small" data-copy>🔗 Copy link</button>
  <a class="btn btn-small" data-open target="_blank" rel="noopener">Open MP4 ↗</a></span></div>
</dialog>
<script src="assets/site.js"></script>
</body>
</html>
"""


def main() -> None:
    videos = catalogue()
    finished = {str(p.relative_to(ROOT)) for p in ROOT.glob("*/outputs/**/*.mp4") if ".partial." not in p.name}
    listed = [v.path for v in videos]
    if set(listed) != finished or len(listed) != len(set(listed)):
        raise SystemExit(f"catalogue and outputs differ: {sorted(finished ^ set(listed))}")
    missing = [v.preview for v in videos if not (DOCS / v.preview).is_file()]
    if missing:
        raise SystemExit(f"missing previews (run docs/make_previews.py): {missing[:3]}")
    (DOCS / "index.html").write_text(page(videos, readme_sections()), encoding="utf-8")
    (DOCS / ".nojekyll").touch()  # serve docs/ as-is: no Jekyll build
    print(f"docs/index.html written: {len(videos)} videos")


if __name__ == "__main__":
    main()
