"""Write the top-level README, showing every finished video as a looping preview.

    python docs/make_previews.py    # once, or after new videos: docs/previews/**/*.webp
    python docs/make_showcase.py    # docs/hero.webp (and the local launch/ kit)
    python docs/build_readme.py     # README.md

GitHub strips <video> players from READMEs, so each video is shown as its animated WebP preview,
linked to the full MP4 on jsDelivr's CDN, which serves the repository's files as video/mp4 so they
play in the browser. Every path is checked to exist, and every .mp4 committed under an outputs/
folder must appear exactly once; a silent Manim lesson counts as shown when its voiced version is.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
PREVIEWS = "docs/previews"

from catalogue import (  # noqa: E402  (docs/ is on sys.path when run as a script)
    CDN, HOUSE_DASH, HOUSE_LESSONS, HOUSE_LESSONS_LIST, HOUSE_MANIM, HOUSE_MANIM_VOICED, HOUSE_MODELS,
    HOUSE_SCENES, TITANIC_DASH, TITANIC_LESSONS, TITANIC_LESSONS_LIST, TITANIC_MANIM, TITANIC_MANIM_VOICED,
    TITANIC_MODELS, TITANIC_SCENES,
)

EMBEDDED: list[str] = []
SHOWN_VOICED: list[str] = []  # silent lessons the README shows in their voiced version instead
SECONDS_PER_MINUTE = 60


def video(path: str, label: str = "Watch the full video", preview_of: str = "") -> str:
    """A looping preview that opens the full MP4 in the browser.

    ``preview_of`` names the video the preview was made from when it is another one: a voiced lesson loops
    the preview of its silent twin, which shows the same pictures.
    """
    preview = f"{PREVIEWS}/{Path(preview_of or path).with_suffix('.webp')}"
    for needed in (path, preview):
        if not (ROOT / needed).is_file():
            raise SystemExit(f"missing file: {needed}")
    EMBEDDED.append(path)
    full = f"{CDN}/{path}"
    return (f'<a href="{full}"><img src="{preview}" width="100%" alt="{Path(path).stem} preview"></a>'
            f'<br>▶️ <a href="{full}">{label}</a> · <code>{Path(path).name}</code>')


def dashboard_pairs(folder: str, models) -> str:
    blocks = []
    for key, emoji, title, training, inference in models:
        blocks.append(f"""#### {emoji} {title}

<table>
  <tr><th width="50%">🏋️ Training: {training}</th><th width="50%">🔮 Inference: {inference}</th></tr>
  <tr>
    <td>{video(f"{folder}/{key}_training.mp4")}</td>
    <td>{video(f"{folder}/{key}_inference.mp4")}</td>
  </tr>
</table>
""")
    return "\n".join(blocks)


def voiced_length(path: str) -> str:
    """A voiced lesson's length from its narration file, as "2:26"."""
    plan = json.loads((ROOT / path).with_suffix(".narration.json").read_text(encoding="utf-8"))
    minutes, seconds = divmod(int(plan["seconds"]), SECONDS_PER_MINUTE)
    return f"{minutes}:{seconds:02d}"


def lesson_cell(folder: str, key: str, silent: str) -> str:
    """One lesson's preview and link; a voiced lesson loops its silent twin's preview."""
    path = f"{folder}/{key}_lesson.mp4"
    if not silent:
        return video(path, "Watch the full 108-second lesson")
    twin = f"{silent}/{key}_lesson.mp4"
    SHOWN_VOICED.append(twin)
    return video(path, f"Watch the full voiced lesson ({voiced_length(path)}) 🔊", preview_of=twin)


def lesson_grid(folder: str, lessons, silent: str = "") -> str:
    """The lessons two to a row; ``silent`` is the folder of their silent twins when these are voiced."""
    rows = []
    for start in range(0, len(lessons), 2):
        cells = [f'    <td width="50%"><b>{emoji} {title}</b><br><sub>{frame}</sub><br>'
                 f'{lesson_cell(folder, key, silent)}</td>' for key, emoji, title, frame in lessons[start:start + 2]]
        rows.append("  <tr>\n" + "\n".join(cells) + "\n  </tr>")
    return "<table>\n" + "\n".join(rows) + "\n</table>\n"


def scene_pair(first: tuple[str, str, str], second: tuple[str, str, str]) -> str:
    cells = [f'    <td width="50%"><b>{title}</b><br><sub>{blurb}</sub><br>{video(path)}</td>'
             for title, blurb, path in (first, second)]
    return "<table>\n  <tr>\n" + "\n".join(cells) + "\n  </tr>\n</table>\n"


INTRO = """<h1 align="center">🎬 Machine Learning Algorithms, Animated</h1>

<p align="center"><b>Stop staring at loss curves. Watch the algorithm itself learn,<br>
then watch it explain every prediction it makes.</b></p>

<p align="center">
  <a href="https://di37.github.io/animated-machine-learning-algorithm/"><img src="https://img.shields.io/badge/▶%20watch-the%20website-0b8f9c" alt="Watch on the website"></a>
  <img src="https://img.shields.io/badge/videos-76-1557d6" alt="76 videos">
  <img src="https://img.shields.io/badge/models-21-0f2541" alt="21 models">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/matplotlib%20%2B%20Manim-11557c" alt="matplotlib and Manim">
  <a href="https://github.com/di37/animated-machine-learning-algorithm/stargazers"><img src="https://img.shields.io/github/stars/di37/animated-machine-learning-algorithm?style=social" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="#-california-housing-regression"><img src="docs/hero.webp" width="100%" alt="Four of the 76 videos: a decision tree growing, gradient boosting explaining a Titanic passenger, a narrated lesson's 3-D view, and four models learning who survived"></a>
</p>

<p align="center"><sub>21 scikit-learn models · 2 classic datasets · 76 videos · every frame a real model state · every prediction checked ✅</sub></p>

<p align="center">🌐 <b><a href="https://di37.github.io/animated-machine-learning-algorithm/">Watch all 76 videos in your browser on the website →</a></b></p>

> 🔍 **The black box, opened.** Watch scikit-learn models **learn** two classic datasets, frame by frame, then
> **predict** on data they have never seen. You see what each model stores, how it combines its
> inputs, and exactly why it gave *this* answer for *this* example, worked through and checked
> against the model's own output.

> ⭐ **If an algorithm finally clicks for you here, star the repo and share it with someone learning ML.**

Two classic datasets, two renderers, **76 videos**:

| | 🏠 California Housing | 🚢 Titanic |
|---|---|---|
| **Task** | regression: a district's median house value | classification: who survived |
| **Data** | 20,640 districts, 8 features | 1,309 passengers (OpenML `titanic`) |
| **📊 Live dashboards** | 9 regressors × training + inference | 9 classifiers × training + inference |
| **🏁 Comparison scenes** | The Algorithm Race · Watch the model learn California | The Algorithm Race · Who survives? |
| **🎓 Narrated lessons** | 9 lessons, matplotlib + voiced Manim 🔊 | 9 lessons, matplotlib + voiced Manim 🔊 |
| **🎞️ Videos** | 38 | 38 |

> [!TIP]
> Every picture below is a live loop of the real video, and the lessons play as a 9-second
> time-lapse. **Click any preview, or its ▶️ link, to play the full video in your browser.**

## 🆚 Why this is different from other ML explainers

Most explainers show you an *idea* of an algorithm. This shows you **the algorithm itself, running**.

| 📺 Most ML explainers | 🎬 Here |
|---|---|
| 🎨 Hand-animated or idealised curves, drawn to look right | 🔁 Every frame is a snapshot of a real scikit-learn model, taken during real training |
| 🧸 Synthetic blobs, generated to behave for the demo | 📚 Two classic datasets of real records, familiar on purpose: 20,640 California census districts and 1,309 Titanic passengers. You already know the data, so all your attention goes to the algorithm |
| 🏁 The finished model only | 🎞️ A model's whole life: data → first guess → training → final model → inference → test exam |
| 📉 A loss curve and an accuracy number | 🔍 The internals themselves: weights, splits, votes, support vectors, bell curves, activations |
| 🪄 "The model says 0.82", taken on trust | ✅ The prediction worked out step by step, and checked against the model's own `predict` before it may show ✓ |
| 🧮 After-the-fact approximations of what the model did | 🧠 The model's own parameters, read straight from the fitted estimator |
| 🔬 One algorithm at a time, each on its own example | ⚖️ Many models on the same data, side by side: races, landscapes, a scoreboard |
| 🤷 Training accuracy, or no score at all | 🧪 Train vs validation vs held-out test, no leakage, and statistical tests before any "better" |
| 👀 Watch-only | 🛠️ Open source and reproducible: one command regenerates any video, and tests guard every rule |

## 🎁 What you'll get out of it

| If you are… | You'll walk away with… |
|---|---|
| 🌱 **An ML enthusiast or student** | Intuition that formulas alone don't give: why a deep tree over-fits, why boosting takes many small steps, why KNN "learns" nothing, and what a neural network's weights actually do. |
| 🧑‍💻 **A data scientist or ML engineer** | A sharper eye for model behaviour as it happens: bias and variance in motion, regularisation paths, overconfident scores, and honest evaluation habits (validation for choices, the test set once). |
| 🏢 **A lead who explains models to stakeholders** | Pictures that answer "why did the model say *that*?" for a single case, with numbers that match the model exactly. |
| 🧑‍🏫 **An educator or content creator** | Ready-made lessons, one algorithm each: 108-second versions that work with the sound off, voiced Manim versions, and the code to make your own. |
| 💼 **Someone preparing for interviews** | A visual memory of 20+ algorithms: how each one trains, how it predicts, and where it breaks. |
| 🛠️ **A builder of ML tools** | A tested pattern for truthful visualisation: immutable snapshots, checked predictions, no invented frames. |

## 🧭 Contents

- [🆚 Why this is different](#-why-this-is-different-from-other-ml-explainers)
- [🎁 What you'll get out of it](#-what-youll-get-out-of-it)
- [🔍 Opening the black box](#-opening-the-black-box)
- [👀 How well training and inference are visualised](#-how-well-training-and-inference-are-visualised)
- [🏠 California Housing: regression](#-california-housing-regression)
- [🚢 Titanic: classification](#-titanic-classification)
- [🏆 Scoreboard](#-scoreboard)
- [📂 Project layout](#-project-layout)
- [🚀 Quick start](#-quick-start)
- [🔒 Truthfulness rules](#-truthfulness-rules)

## 🔍 Opening the black box

A trained model is usually a black box: numbers go in, a prediction comes out, and you are asked to
trust it. These videos open the box. They draw the model's **own internals**, read straight from the
fitted estimator (`coef_`, `tree_`, `estimators_`, `support_vectors_`, `theta_` and `var_`, `coefs_`),
not an after-the-fact approximation such as a surrogate model or a feature-attribution estimate. That
is what makes every explanation checkable: the worked equation is the model's real computation, so
it has to land on `predict`'s exact answer before it prints ✓.

| Model | 🗃️ What is inside the box | 🔎 Why it gave *this* answer |
|---|---|---|
| 📏 Linear, Ridge, Lasso, SGD | every weight, drawn as an edge whose width and colour follow it; Lasso's zeroed features go dashed | each feature's push w·x, summed with the bias into the price |
| 📈 Logistic regression, perceptron, linear SVM | the same weighted sum, plus the S-curve or the ±1 margin | the score z, its side of the threshold, and σ(z) where it is a real probability |
| 🌳 Decision tree | every split, with its threshold in real units ("age ≤ 12.5?"), nodes coloured by their examples' average price or survival share | the exact path of yes/no questions, and the training examples that share its leaf |
| 🌲 Random forest | the individual trees, not just their average | every tree's vote, and the average the forest returns |
| 🚀 Gradient boosting | each tree's correction, shrinking as the errors shrink | the running total from the base rate F₀, tree by tree, to the final answer |
| 📍 k-nearest neighbours | the stored training examples, which is all KNN keeps | the k nearest examples and their prices or fates |
| 🛡️ SVM (RBF kernel) | which training passengers became support vectors, and how much each weighs (α) | the strongest pulls α·y·K(xᵢ, x) and the margin f(x) they add up to |
| 🔔 Naive Bayes | the base rates and one bell curve per feature and outcome | how strongly each feature's value favours "survived" over "did not" (its log-likelihood ratio) |
| 🧠 Neural network (MLP) | every weight, re-coloured each epoch as training moves it | the forward pass: which hidden units fire and how much signal each edge carries |

Seeing the box open teaches things a score never shows:

- **Why linear models plateau:** they can only tilt one weighted sum.
- **Why one deep tree over-fits:** its leaves shrink to single passengers.
- **Why averaging trees helps:** the votes spread out, and their mean is steadier than any one tree.
- **What KNN actually learns:** nothing but its memory.

Honest limits:

- An MLP's hidden units have no human meaning. The diagram shows the computation, not a concept.
- The KNN and SVM maps squeeze 8 features onto 2 axes (PCA), so "near" on the map is only roughly
  "near" to the model.
- A margin model's score (SVM, perceptron) is never called a probability.

## 👀 How well training and inference are visualised

Most "ML animations" are a loss curve and a confusion matrix. These videos show the **algorithm
itself** working, synchronised with the numbers that prove it, at both stages of a model's life.

### 🏋️ At training time

- **Every frame is real training work.** A frame is drawn the moment the model finishes a real unit
  of work: a `partial_fit` chunk, a few more trees, one level deeper, a weaker penalty, or more
  training rows. Nothing is interpolated between states that never existed.
- **The algorithm animates, not just its scores.** Weighted-sum edges thicken and change sign,
  decision trees grow level by level, forests fill in tree by tree, boosting's corrections shrink,
  support vectors get ringed, Naive Bayes' bell curves sharpen, and every one of an MLP's weights
  changes colour each epoch.
- **Live equations with the current numbers.** A side panel shows the model's formula, the objective
  it is minimising (with its value right now), and what this step changed: SGD's update rule, the
  boosting update F_M = F₀ + ν Σ h_m, a tree's root split and its impurity drop.
- **Under- and over-fitting are visible.** Train and validation curves share one chart, so you watch
  a decision tree's training score keep improving while its validation score turns back.
- **Honest timing.** Only the model's own work is timed; scoring and drawing are excluded.

### 🔮 At inference time

- **Unseen data streams through `predict()`.** Held-out test rows are predicted batch by batch, and
  latency and throughput are measured on the real calls.
- **Every newest prediction is explained.** The diagram traces the example through the model (the
  tree path, each tree's vote, the nearest neighbours, the forward pass), and the equation panel
  works the formula through with that example's numbers.
- **Every explanation is checked.** Each card ends with
  `check: model.predict(x) = …  ✓` (or `predict_proba` / `decision_function`). An independent review
  re-ran it on all 262 Titanic test passengers: the worked equation matched the model to within 1e-13.
- **Results pile up where you can read them.** A California map colours every priced district, and
  the Titanic passenger list fills in square by square, grouped by class and by women and men, with
  ✗ on every wrong call. You can watch "women and children first" appear.
- **Running scores update live:** RMSE, MAE and R² for prices; accuracy, recall, the confusion matrix
  and a confidence histogram for survival.

### 🎓 In the narrated lessons

Each lesson is a **108-second, 2560×1440, 30 fps** story of one algorithm that works with the sound
off, in ten chapters: **title → data → first guess → training → final model → prediction on an
unseen example → final exam → 3-D view → more features → takeaway.**

- Training cuts from one real snapshot to the next, and the final curve equals `model.predict`.
- The 3-D chapter draws the model's output as a surface over two inputs while the camera swings.
- The "more features" chapter refits on every input and says "better" or "worse" only when a paired
  bootstrap (prices) or an exact McNemar test (survival) says the difference is real.
- The Manim versions replay the same lessons with smooth reveals and transitions; every chapter
  still ends on exactly the matplotlib picture.
- 🔊 The Manim versions have a voice-over: a narrator reads each chapter's own captions, with every
  symbol and abbreviation said in words (θ₀ as "theta zero", MLP as "multi-layer perceptron"). Where
  a chapter's voice runs long, its last frame holds, so a voiced lesson runs about 2½ minutes.
"""

HOUSE = """## 🏠 California Housing: regression

`sklearn.datasets.fetch_california_housing`: 20,640 census districts, 8 features, target = median
house value (capped at $500k). Split 60/20/20, scaled on the training rows only.
Code: [`house_price_animations/`](house_price_animations/README.md).

### 📊 Live dashboards: 9 regressors, training and inference

Each dashboard is **2560×1440**, in three columns: the animated algorithm, the live charts (learning
curve, predicted vs actual, learned weights, residuals) and the equations at work.

{dashboards}

### 🏁 Comparison scenes

{scenes}

### 🎓 Narrated lessons (matplotlib): one algorithm each

The model learns house value from **median income alone**, so the model itself is the curve you watch
move; the 3-D chapter then adds house age. Code: `house_price_animations/house_anim/lessons/`.

{lessons}

### 🎞️ Voiced lessons (Manim) 🔊

The same nine lessons, played with Manim, with a voice-over: turn the sound on.
Code: [`house_price_manim_lessons/`](house_price_manim_lessons/README.md).

{manim}
"""

TITANIC = """## 🚢 Titanic: classification

OpenML's `titanic`: 1,309 passengers. The `boat` and `body` columns are dropped, because they are only
known after the sinking and would leak the answer. Stratified 60/20/20 split, so every split keeps
the same 38% survival rate; gaps are filled from the training passengers only.
Code: [`titanic_animations/`](titanic_animations/README.md).

### 📊 Live dashboards: 9 classifiers, training and inference

Training shows the accuracy learning curve, the ROC curve with AUC, the learned weights and the
confusion matrix. Inference shows the passenger list, running accuracy and recall, and how sure the
model is. Blue means survived, red means did not, and the neutral grey midpoint is the 0.5 threshold.

{dashboards}

### 🏁 Comparison scenes

{scenes}

### 🎓 Narrated lessons (matplotlib): one classifier each

Each classifier learns survival from **age, fare and sex**: two maps (women | men), age across, fare on
a log scale, shaded by the model's score with its decision boundary drawn and every misclassified
training passenger circled. Code: `titanic_animations/titanic_anim/lessons/`.

{lessons}

### 🎞️ Voiced lessons (Manim) 🔊

The same nine lessons, played with Manim, with a voice-over: turn the sound on.
Code: [`titanic_manim_lessons/`](titanic_manim_lessons/README.md).

{manim}
"""

OUTRO = """## 🏆 Scoreboard

Test-set results from the live dashboards (all 8 features). The lessons use one or three features on
purpose, so each picture tells one story; their scores are in each lessons folder's `lessons_summary.csv`.

| 🏠 Regressor | Test RMSE | Test R² | | 🚢 Classifier | Test accuracy |
|---|---:|---:|---|---|---:|
| 🥇 Gradient Boosting | $49.7k | 0.812 | | 🥇 Gradient Boosting | 83.6% |
| 🥈 Random Forest | $50.9k | 0.802 | | 🥇 MLP (32-16) | 83.6% |
| 🥉 MLP (64-32) | $53.2k | 0.784 | | 🥉 KNN (k=11) | 82.8% |
| KNN (k=10) | $64.1k | 0.686 | | SVM (RBF) | 82.4% |
| Decision Tree (depth 18) | $69.6k | 0.631 | | Random Forest | 79.8% |
| Lasso | $74.8k | 0.573 | | Logistic Regression | 79.4% |
| Linear Regression | $75.0k | 0.571 | | SGD (logistic) | 77.1% |
| Ridge | $75.0k | 0.571 | | Naive Bayes | 77.1% |
| SGD Regressor | $77.3k | 0.544 | | Decision Tree (depth 14) | 74.0% |

The decision trees are reported at the end of their depth sweeps, which over-fit on purpose: that is
the lesson their videos teach.

## 📂 Project layout

Each dataset lives in its own folder; anything shared lives in a dataset-neutral kit.

```
animated-machine-learning-algorithm/
├── ml_animation_kit/           🧰 animkit: recorder, playback, dashboard chrome, equation panel,
│                                  tree and network drawing, and the narrated-lesson director
├── house_price_animations/     🏠 house_anim: regressor dashboards, scenes and lessons
├── titanic_animations/         🚢 titanic_anim: classifier dashboards, scenes and lessons
├── manim_lesson_kit/           🎞️ manimkit: plays any lesson with Manim instead of matplotlib
├── house_price_manim_lessons/  🏠 the house-price lessons, played with Manim
└── titanic_manim_lessons/      🚢 the Titanic lessons, played with Manim
```

| Folder | README |
|---|---|
| `ml_animation_kit/` | [shared matplotlib engine](ml_animation_kit/README.md) |
| `house_price_animations/` | [regressor dashboards and lessons](house_price_animations/README.md) |
| `titanic_animations/` | [classifier dashboards and lessons](titanic_animations/README.md) |
| `manim_lesson_kit/` | [shared Manim player](manim_lesson_kit/README.md) |
| `house_price_manim_lessons/` | [house-price lessons in Manim](house_price_manim_lessons/README.md) |
| `titanic_manim_lessons/` | [Titanic lessons in Manim](titanic_manim_lessons/README.md) |

## 🚀 Quick start

You need Python 3.10+ and `ffmpeg` on your PATH (`brew install ffmpeg`). Nothing is installed as a
package: each project finds the kit next to it.

```bash
git clone https://github.com/di37/animated-machine-learning-algorithm.git
cd animated-machine-learning-algorithm
```

🏠 **California Housing**

```bash
cd house_price_animations
pip install -r requirements.txt
python animate.py --model gradient_boosting       # live window + training and inference videos
python animate.py --scene race                    # The Algorithm Race
python animate.py --scene map                     # Watch the model learn California
python animate_lessons.py --model decision_tree   # one narrated lesson
```

🚢 **Titanic**

```bash
cd titanic_animations
pip install -r requirements.txt
python animate_titanic.py --model random_forest
python animate_titanic.py --scene race
python animate_titanic.py --scene landscape       # Who survives?
python animate_titanic_lessons.py --model knn
```

🎞️ **Manim lessons** (set up `manim_lesson_kit/.venv` first; its README shows how)

```bash
cd house_price_manim_lessons
../manim_lesson_kit/.venv/bin/python animate_manim_lessons.py --model linear_regression
```

🔊 **Voice-over** (ElevenLabs: put `ELEVENLABS_API_KEY=...` in a `.env` file at the repo root, which git ignores)

```bash
cd house_price_manim_lessons
python3 voice_manim_lessons.py --model linear_regression   # --script-only shows the words first, free
```

Every command takes `--list` and `--help`. Dashboards also accept `--no-live` (record only),
`--phase training|inference|both` and `--format gif`.

## 🔒 Truthfulness rules

These are enforced by the test suites, not just promised:

- 🔁 **Real states only.** Every curve, boundary, weight and metric on screen comes from a real model
  state; frames are held, never interpolated into states that did not exist.
- ✅ **Checked predictions.** A worked prediction must reproduce `predict`, `predict_proba` or
  `decision_function` before it may print ✓.
- 🧪 **No leakage.** Scalers and imputation are fitted on training rows only; test rows appear only at
  inference and in the final exam.
- 🎯 **Honest labels.** Margin models (SVM, perceptron) show a signed score, never a "probability".
- ⚖️ **Honest verdicts.** "Better" or "worse" is claimed only when a paired bootstrap or exact McNemar
  test supports it.
- ⏱️ **Honest timing.** Fit and predict times count only the model's own work.

## ⭐ Spread the word

If these videos helped an algorithm click for you:

- ⭐ **Star the repo**, so more learners find it.
- 🔁 **Share a video** with a friend, a class or a study group. Every ▶️ link plays in the browser.
- 🧑‍🏫 **Teach with it:** the matplotlib lessons are 108 seconds and work with the sound off; the Manim
  lessons have a voice-over.

---

<p align="center"><sub>IR · Isham Rashik · AI Engineer · Making AI simplified for all</sub></p>
"""


def main() -> None:
    house = HOUSE.format(
        dashboards=dashboard_pairs(HOUSE_DASH, HOUSE_MODELS),
        scenes=scene_pair(*HOUSE_SCENES),
        lessons=lesson_grid(HOUSE_LESSONS, HOUSE_LESSONS_LIST),
        manim=lesson_grid(HOUSE_MANIM_VOICED, HOUSE_LESSONS_LIST, silent=HOUSE_MANIM),
    )
    titanic = TITANIC.format(
        dashboards=dashboard_pairs(TITANIC_DASH, TITANIC_MODELS),
        scenes=scene_pair(*TITANIC_SCENES),
        lessons=lesson_grid(TITANIC_LESSONS, TITANIC_LESSONS_LIST),
        manim=lesson_grid(TITANIC_MANIM_VOICED, TITANIC_LESSONS_LIST, silent=TITANIC_MANIM),
    )
    # every committed video (jsDelivr serves only those) is embedded, or shown in its voiced version
    finished = subprocess.run(["git", "ls-files", "--", ":(glob)*/outputs/**/*.mp4"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.split()
    missing = sorted(set(finished) - set(EMBEDDED) - set(SHOWN_VOICED))
    duplicates = len(EMBEDDED) - len(set(EMBEDDED))
    if missing or duplicates:
        raise SystemExit(f"not embedded: {missing}; duplicates: {duplicates}; embedded {len(EMBEDDED)}")
    (ROOT / "README.md").write_text(INTRO + "\n" + house + "\n" + titanic + "\n" + OUTRO, encoding="utf-8")
    print(f"README.md written: {len(EMBEDDED)} videos embedded, all finished videos covered")


if __name__ == "__main__":
    main()
