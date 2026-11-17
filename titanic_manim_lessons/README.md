# Titanic lessons, played with Manim

The narrated lesson videos of `../titanic_animations/` (one classifier learning who survived the Titanic
from age, fare and sex), played with Manim instead of rasterized by matplotlib. Everything the lessons
show (the passengers, the trained models, the words, every number) comes from `titanic_anim` unchanged;
`../manim_lesson_kit/` (`manimkit`) plays them, and its README says how.

All nine lessons run through the same player: logistic regression, perceptron, linear SVM, naive Bayes,
decision tree, random forest, gradient boosting, kNN and the MLP.

## Run it

Use the shared Manim environment in `../manim_lesson_kit/.venv` (its README sets it up), from this folder:

```bash
../manim_lesson_kit/.venv/bin/python animate_manim_lessons.py --model logistic_regression
../manim_lesson_kit/.venv/bin/python animate_manim_lessons.py                     # all nine
../manim_lesson_kit/.venv/bin/python animate_manim_lessons.py --list
../manim_lesson_kit/.venv/bin/python animate_manim_lessons.py --model knn --resolution 1080p
```

The options are the matplotlib lessons' (`python animate_titanic_lessons.py --help` there). Videos and
`lessons_summary.csv` go to `outputs/titanic/lessons/` in this folder. The passenger list is OpenML's
`titanic`, cached in `~/scikit_learn_data` after the first run.

Each video has the matplotlib video's frame count, size (2560×1440 by default), 30 fps and encoding
(H.264, yuv420p, CRF 19). Every chapter ends on the matplotlib picture; in between, Manim fades the last
chapter out before drawing the next one in, writes large text, draws lines in, and moves things
smoothly. Training cuts from one real snapshot to the next, as the matplotlib video does.

## Layout

```
animate_manim_lessons.py    entry point (also: python -m titanic_manim)
titanic_manim/
  kit_path.py               puts ml_animation_kit, manim_lesson_kit and titanic_animations on sys.path
  cli.py                    the matplotlib project, played with Manim, and its main
tests/                      boundaries, the command line, and chapter-by-chapter parity with matplotlib
```

## How to read this code

This folder is deliberately thin: it chooses *what* to play, and the sibling folders do the work.
Read it top-down, then follow the imports to where the lessons and the player live.

| # | Read | What it teaches |
|---|------|-----------------|
| 1 | `animate_manim_lessons.py` | an entry point that does one thing: make the sibling folders importable, then hand over to `main` |
| 2 | `titanic_manim/kit_path.py` | using code from folders that are not installed packages, and failing early with a message that says which folder to put back |
| 3 | `titanic_manim/cli.py` | reuse instead of copying: the parser, the lessons, the data and the summary CSV all come from `titanic_anim`, and the command line itself is animkit's, shared with every lesson folder. This file only derives the Manim project from `titanic_anim.lessons.cli.project()`; read `../ml_animation_kit/animkit/lessons/cli.py` next for how `run` lists, renders, and goes on past a lesson that fails |
| 4 | `tests/test_cli.py` | testing a command line without the network: `main` takes the dataset loader as an argument, and the tests pass a small stand-in |
| 5 | `tests/test_boundaries.py` | keeping a project's dependencies honest: it fails if this folder imports the other dataset's code |
| 6 | `tests/test_parity.py` | the proof that the Manim videos show what the matplotlib ones do, chapter by chapter |

Where to go next:

- The lessons themselves (data, models, the words on screen): `../titanic_animations/titanic_anim/lessons/`: `stories/` holds one file per algorithm, and `scene.py` draws the classification-specific parts.
- How a matplotlib frame becomes Manim animations: the *How to read this code* table in
  `../manim_lesson_kit/README.md`.

Every function has a Google-style docstring (`Args`, `Returns`, `Raises` as they apply); each module's
opening docstring says what it is for.

## Tests

```bash
../manim_lesson_kit/.venv/bin/python -m pytest -q
```

`tests/test_parity.py` renders every lesson at 1280×720 and compares the last frame of every chapter with
the one animkit's own players draw for it (a few minutes).
