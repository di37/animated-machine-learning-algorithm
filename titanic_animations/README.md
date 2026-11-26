# Titanic survival classifier animations

These animations show nine scikit-learn **classifiers** learning who survived the Titanic live,
while the code runs. Each frame is drawn right after a real piece of training work, then again
while the trained model classifies the held-out passengers batch by batch. The dataset is OpenML's
`titanic` (version 1): 1,309 passengers, downloaded once and cached in `~/scikit_learn_data`.

The features are class, sex, age, siblings/spouses aboard, parents/children aboard, fare, and
port of embarkation (Cherbourg and Queenstown one-hot encoded, Southampton as the baseline).
The `boat` and `body` columns are dropped, because they are only known after the sinking and
would leak the answer. Missing ages, fares and ports are filled from the training passengers
only. The split is a stratified 60/20/20, so every split keeps the same 38% survival rate.

## Run it

Keep `ml_animation_kit/` next to this folder: it holds `animkit`, the shared, dataset-neutral engine
(recorder, playback, dashboard chrome, equation panel, tree and network drawing, lesson director).
The entry scripts and `python -m titanic_anim` put it on `sys.path` themselves, and pytest gets it
from `pythonpath` in `pyproject.toml`. Nothing needs to be installed as a package. Run every command
from this folder.

```bash
pip install -r requirements.txt                    # ffmpeg on PATH for .mp4 (brew install ffmpeg)

python animate_titanic.py                          # random forest: live window + mp4 files
python animate_titanic.py --model logistic_regression
python animate_titanic.py --model all --no-live    # record all 9 classifiers
python animate_titanic.py --scene race             # The Algorithm Race: 9 classifiers, test-set podium
python animate_titanic.py --scene landscape        # Who survives? 4 models paint P(survive) by age, class, sex
python animate_titanic.py --list
```

| Option | Meaning |
|---|---|
| `--scene dashboards\|race\|landscape` | per-model dashboards (default), the race, or the survival landscapes |
| `--model KEY\|all` | which classifier to animate (default `random_forest`) |
| `--phase training\|inference\|both` | which part to animate; `inference` trains silently first |
| `--no-live` / `--no-save` | skip the on-screen window / skip writing video files |
| `--format mp4\|gif` | video format (falls back to GIF if ffmpeg is missing) |
| `--fps`, `--dpi` | video frame rate and resolution |
| `--batch-size N` | test passengers classified per inference frame (default 4) |
| `--max-rows N` | subsample the passenger list for a quick run (at least 100) |
| `--output-dir DIR` | where videos and `metrics_summary.csv` go (default `outputs/titanic/`) |

Closing the live window, or pressing Ctrl-C, stops the run. Any frames recorded so far
are still saved as a playable video.

## What the animations show

| Key | Estimator | What one training frame is | The algorithm, animated |
|---|---|---|---|
| `logistic_regression` | `LogisticRegression` | larger C (weaker L2 penalty) | inputs → weighted edges → Σ → σ → P, above the S-curve with survivors at 1 and the others at 0 |
| `sgd` | `SGDClassifier(loss="log_loss")` | `partial_fit` on 1/8 of an epoch | same diagram; weights walk downhill |
| `svm` | `SVC(kernel="rbf")` | refit on more passengers | support vectors ringed on a PCA map; at prediction, the strongest α·y·K pulls |
| `naive_bayes` | `GaussianNB` | count more passengers | one pair of bell curves (survived / not) per feature; the passenger's value and its log-likelihood ratio |
| `decision_tree` | `DecisionTreeClassifier` | `max_depth` + 1 | the tree grows, nodes coloured by survival share; the passenger's path of yes/no questions |
| `random_forest` | `RandomForestClassifier` | more trees (`warm_start`) | 12 tree tiles; every tree's P(survive) and their average |
| `gradient_boosting` | `GradientBoostingClassifier` | 5 more stages (`warm_start`) | the newest tree's log-odds corrections; P(survive) climbing from the base rate tree by tree |
| `knn` | `KNeighborsClassifier(k=11)` | more passengers memorised | stored passengers on a PCA map; the 11 nearest and how many survived |
| `mlp` | `MLPClassifier(32, 16)` | one Adam epoch | the network's weights; activations firing into a sigmoid output |

- **Training dashboard:** an accuracy learning curve (train vs validation), the validation ROC
  curve with AUC, learned weights (log-odds coefficients, importances, first-layer sensitivity,
  or Naive Bayes' class separation), and the validation confusion matrix.
- **Inference dashboard:** the passenger list. Each test passenger is a square grouped by class
  and by women and men, coloured by predicted P(survive), with ✗ marking wrong predictions. It
  also shows running accuracy and recall, the confusion matrix so far, and a histogram of
  P(survive) split by what really happened.
- **Equations:** every card ends with `check: model.predict_proba(x)[survived] = …  ✓`, or
  `decision_function` for the SVM. The tests assert the worked equation equals the model's own
  answer for all 9 classifiers.
- **Colours:** blue means survived and red means did not survive. The neutral grey midpoint is
  the 0.5 decision threshold. SGD shares Logistic Regression's hue in the race and is told apart
  by its hatched bar and dashed curve.

## Narrated classifier lessons

These are narrated, 108-second, 2560×1440, 30 fps H.264 lessons, one per classifier, built on animkit's
lesson director, stage and encoder. Each model learns survival from **age, fare and sex**.
The picture is two maps, women on the left and men on the right, with age on x and fare on a log y-axis.
Shading shows the model's score and the solid line is its decision boundary. Misclassified training
passengers are circled. The 263 passengers with no recorded age are left out instead of being filled
with the median, which would paint a false stripe at one age.

After the prediction on one unseen passenger come the final exam, a 3-D view and the more-features
comparison:

- **Final exam:** all 210 test passengers are judged batch by batch.
  - The scoreboard shows the running accuracy, recall, AUC and the confusion counts, plus prediction
    speed measured once per video.
  - A grid groups the passengers by class of travel and sex, an input the lesson's model never sees,
    with ✕ on every wrong call.
- **3-D view:** the lesson's own final model, not a refit. Its score is drawn as height over age and fare,
  one sheet for women and one for men, while the camera swings to and fro.
  - A grey plane marks the threshold: where a sheet rises above it, the model predicts "survived".
  - Training passengers sit on the top face if they survived and on the bottom face if not.
  - The perceptron and linear SVM are drawn on a signed score axis, never labelled a probability.
- **More features:** the same algorithm is refit on every input: class, sex, age, siblings/spouses,
  parents/children, log fare and port, 8 columns in all. It is judged on the same test passengers and
  compared with the lesson's model and the always-'did not survive' baseline.
  - k is re-chosen on validation; the other settings carry over.
  - With only 210 test passengers, the headline says "lift" or "worse" only when an exact McNemar
    test on the passengers the two models disagree about gives p < 0.05. Otherwise it says "about the
    same", with the gained and lost counts in the note.

```bash
python animate_titanic_lessons.py                        # all nine (about 14 minutes at 1440p)
python animate_titanic_lessons.py --model decision_tree
python animate_titanic_lessons.py --list
```

| Lesson | What a training frame is | Score shown |
|---|---|---|
| `logistic_regression` | one full-batch gradient-descent step on the log loss | P(survive) |
| `perceptron` | online updates (first epoch every 60 passengers, then whole epochs); validation picks the epoch | signed score, not a probability |
| `linear_svm` | one SGD epoch on hinge loss + L2; dashed lines are the ±1 margin | signed score, not a probability |
| `naive_bayes` | `partial_fit` on more passengers: priors and per-class bell curves update | P(survive) |
| `decision_tree` | one more best-first split (1 → 12 leaves) | leaf survival share |
| `random_forest` | more bootstrapped trees, probabilities averaged | P(survive) |
| `gradient_boosting` | the trees added since the last frame, in log-odds | σ(F) |
| `knn` | a refit with larger k (1 → 256); validation picks k | share of the k nearest who survived |
| `mlp` | real Adam training of a 3→8→8→1 network | P(survive) |

The unseen passenger is the test passenger closest to a 30-year-old woman on a $15 ticket. It is
chosen by features alone and is the same passenger in every lesson. Score-only models carry
`kind = "margin"`, so no frame, caption or check line calls their score a probability. Truthfulness
rules, enforced by `tests/lessons/`:
- the final map equals the model's own scores;
- the worked prediction reproduces `predict_proba` or `decision_function`;
- the tree grows as a prefix;
- forest maps are tree averages;
- boosting is additive in log-odds.

Videos and `lessons_summary.csv` go to `outputs/titanic/lessons/`.

### Voice-over

`voice_titanic_lessons.py` speaks each chapter's caption and lays it over the rendered video. The voice says the
chapter's own on-screen words, with the symbols written out (σ as "sigma", P(survive) as "chance
of survival"), so it says nothing the video does not show. Jargon is always said by its full name (SVM as
"support vector machine", MLP as "multi-layer perceptron"), never letter by letter; a lesson with an
abbreviation that has no full name yet is not voiced. The fine print (confidence intervals, settings)
stays on screen. Where a chapter's voice runs longer than the chapter, its last frame is held until
the voice has finished, so a voiced lesson runs about 2 minutes 20 seconds instead of 108 seconds.

```bash
python voice_titanic_lessons.py --script-only                 # write what the voice will say: free, no key needed
python voice_titanic_lessons.py --model logistic_regression   # one lesson, ElevenLabs (the default engine)
python voice_titanic_lessons.py                               # all nine (about 14,500 characters of speech)
python voice_titanic_lessons.py --voice <voice-id>            # another ElevenLabs voice
python voice_titanic_lessons.py --engine openai               # OpenAI's gpt-4o-mini-tts instead (--voice marin, ...)
python voice_titanic_lessons.py --engine say                  # macOS's offline voice: free, for checking the timing
```

ElevenLabs speaks with `eleven_multilingual_v2` in the Mixture-of-Experts videos' voice and settings,
and bills by the character (about $0.08 per 1,000, or included in a plan's monthly allowance). Its key
is read from `ELEVENLABS_API_KEY`, and OpenAI's from `OPENAI_API_KEY`: from the environment or from a
`.env` file in this folder or one above it (`.env` is git-ignored). Voice clips are cached in
`~/.cache/animkit/tts/`, keyed by voice, settings and text, so running again costs nothing and the
Manim lessons reuse the same clips. Voiced videos go
to `outputs/titanic/lessons_voiced/`, each with a `<lesson>_lesson.narration.json` that lists every
chapter's on-screen text, spoken text and timing.

## Layout

```
animate_titanic.py          entry point (also: python -m titanic_anim)
animate_titanic_lessons.py  lesson videos entry point (also: python -m titanic_anim.lessons)
voice_titanic_lessons.py    voice-over entry point: narration laid over the rendered lessons
titanic_anim/
  kit_path.py               puts ../ml_animation_kit on sys.path for the entry points
  config.py, theme.py       Titanic constants and the survival colours (shared tokens come from animkit)
  data.py                   load OpenML titanic, drop leaky columns, impute, stratified split, scale
  passengers.py             plain-language wording for passengers, features and tree splits
  models.py                 registry: classifier + training schedule per model
  trainers.py               training loops that yield a frame per real step
  inference.py              batched, timed predictions on the test passengers
  frames.py, metrics.py     immutable snapshots and classification scores
  panels.py, manifest.py    ROC, confusion matrix, rates, score histogram, the passenger list
  algorithms/               animated classifier diagrams (logistic, SVM, Naive Bayes, trees, KNN, network)
  equations/                per-family equation cards ending in a predict_proba / decision_function check
  training_dashboard.py     training figure
  inference_dashboard.py    inference figure
  race.py, race_dashboard.py           the Algorithm Race scene
  landscape.py, landscape_dashboard.py   the survival landscape scene
  projection.py             fixed PCA map of passengers for the distance-based diagrams
  scenes.py                 records the two scenes
  runner.py, cli.py         orchestration and command line
  summary.py                console table + metrics_summary.csv
  lessons/                  age / fare / sex lessons: passenger data view, women | men maps, the
                            classification scene and stories/ for the nine classifiers; cli.py
                            only describes the lessons, and animkit.lessons runs them
tests/                      pytest suite (synthetic passengers, no network); tests/lessons/ for the lessons
outputs/titanic/            videos and CSVs (lessons/ for the lesson videos)
```

## How to read this code

Read each module with its test open beside it: the tests (in `tests/`) are short worked examples of
what the module promises. Each step builds on the ones before. For the engine underneath, read the
table in `../ml_animation_kit/README.md` first or alongside.

**The live animations** (`titanic_anim/`):

| # | Read | What it teaches | Beside it |
|---|------|-----------------|-----------|
| 1 | `kit_path.py`, `config.py`, `theme.py` | using a shared package that is not installed; probability against margin scores and their thresholds; a score as a colour | `test_boundaries.py` |
| 2 | `data.py` | avoiding leakage: leaky columns dropped, gaps filled from the training rows only, a stratified split, standardising | `test_data.py` |
| 3 | `passengers.py` | putting raw features and tree splits into plain words | `test_dashboards.py` |
| 4 | `models.py` | the four step-by-step training strategies, as animkit `ModelSpec`s | `test_trainers.py` |
| 5 | `metrics.py`, `frames.py` | the classification scores; immutable snapshots handed from the loops to the drawing | `test_metrics.py` |
| 6 | `trainers.py`, `inference.py` | generators that yield a frame after each real step; timing `predict` on its own | `test_trainers.py` |
| 7 | `panels.py`, `manifest.py` | making chart artists once, then updating their data every frame | `test_dashboards.py` |
| 8 | `equations/` (cards, linear, kernels, trees, others) | working a model's formula by hand, checked against `predict_proba` | `test_equations.py` |
| 9 | `projection.py`, `algorithms/` | drawing an algorithm from the fitted model's own internals | `test_algorithm_views.py` |
| 10 | `training_dashboard.py`, `inference_dashboard.py`, `runner.py` | putting the panels into figures and playing the streams through animkit's sinks | `test_dashboards.py`, `test_cli.py` |
| 11 | `race.py`, `race_dashboard.py`, `landscape.py`, `landscape_dashboard.py`, `scenes.py` | comparing models in lockstep; easing the leaderboard; a model's score over a grid of typical passengers | `test_scenes.py` |
| 12 | `summary.py`, `cli.py`, `__main__.py`, `animate_titanic.py` | adding rows to the summary CSV safely; command-line exit statuses | `test_cli.py` |

**The narrated lessons** (`titanic_anim/lessons/`, tests in `tests/lessons/`):

| # | Read | What it teaches | Beside it |
|---|------|-----------------|-----------|
| 1 | `config.py`, `data.py` | three inputs, why rows with unknown ages are dropped, the 60/20/20 split, the map grid | `test_passenger_data.py` |
| 2 | `lesson.py` | snapshots of a real model; probability against signed score; a worked prediction that is verified | `test_class_stories.py` |
| 3 | `stories/__init__.py`, `stories/common.py` | the lesson registry; standardising the inputs; the weight-bar panel | `test_class_stories.py` |
| 4 | `stories/logistic.py`, `stories/linear_models.py` | gradient descent on the log loss, step by step; the perceptron's mistake updates; the SVM's hinge loss and margin | `test_class_stories.py` |
| 5 | `stories/naive_bayes.py` | Bayes' rule as counting (`partial_fit`) | `test_class_stories.py` |
| 6 | `stories/tree.py`, `forest.py`, `boosting.py` | best-first splits and Gini; bootstrapping and averaging; additive log-odds corrections | `test_class_stories.py` |
| 7 | `stories/knn.py`, `stories/mlp.py` | prediction from memory and choosing k on validation; the forward pass, ReLU and Adam | `test_class_stories.py` |
| 8 | `lesson_diagrams.py`, `timing_helpers.py` | implementing animkit's `Diagram` protocol, and wrapping its model diagrams | `test_lesson_diagrams.py` |
| 9 | `maps.py`, `scene.py` | a three-input model drawn as two 2-D maps; implementing the engine's `Scene` protocol | `test_class_render.py` |
| 10 | `exam.py`, `surface_view.py`, `more_features.py` | test-set scoring (running accuracy, recall, AUC) and grouping by a variable the model never saw; the scores in 3-D; a fair all-inputs comparison with a McNemar test | `test_exam_and_features.py`, `test_surface_view.py` |
| 11 | `summary.py`, `cli.py`, `__main__.py` | an atomic CSV upsert; the shared lesson command line | `test_class_cli.py` |

How the code explains itself:

- Every module opens with a docstring that explains the idea before any code: start there.
- Every class and function has a Google-style docstring: a one-line summary, then `Args`, `Returns`,
  `Raises`, `Yields`, `Attributes` or `Example` as they apply. Private helpers (a leading underscore)
  have a one-line summary.
- A method that overrides a library's says `See base class.`: its contract is the library's.

## Tests

```bash
python -m pytest -q
python -m coverage run -m pytest -q && python -m coverage report
```

On macOS, numpy 2.x with Apple Accelerate can raise spurious
`divide by zero encountered in matmul` warnings on finite data. The CLI and the
tests filter only that message. Scores are separately checked for NaN or infinite
values, so a real numerical failure still raises an error.
