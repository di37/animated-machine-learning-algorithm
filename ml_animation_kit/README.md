# ml_animation_kit (`animkit`)

The dataset-neutral engine behind the two animation projects next to this folder:

- `../house_price_animations/`: regressors on California Housing (`house_anim`)
- `../titanic_animations/`: classifiers on the Titanic passenger list (`titanic_anim`)

`animkit` knows nothing about either dataset and imports neither project. Each project
imports what it needs from here, and keeps everything dataset-specific (data loading, model
registries, charts with real units, equations, scenes and stories) in its own package. The kit's
tests check the boundary.

## What is in it

```
animkit/
  config.py, theme.py       frame rates, canvas sizes, panel layout, the validated colour palette
  data.py                   read_only arrays and the ScaledSplits protocol shared drawing reads
  models.py, trainers.py    ModelSpec, the four training strategies, NumericalError, growth_sizes
  frames.py                 FeatureWeights snapshots
  recorder.py, runner.py    live window / mp4 / gif sinks; play a frame stream into a dashboard
  cli.py                    argument types, run options, failure reports, wait-for-window
  inference.py              prepare a fitted model for timed, single-threaded predict
  lockstep.py, race.py      advance several training loops together; colour slots for a race
  panels.py, race_dashboard.py, training_dashboard.py   header, progress bar, weight bars, easing
  equations/, equation_panel.py   equation cards and the column that draws them
  algorithms/               base AlgorithmView, signed colours, tree drawing, the network diagram
  lessons/                  narrated lesson videos: director (with the Scene protocol), stage,
                            ffmpeg encoder, timing, lesson theme, the Story/Explanation/verified
                            contract, callout, tree layout, the shared lesson command line, and
                            the 3-D view (surface.py: a model's output as height over two inputs,
                            swinging camera), and the voice-over: narration.py (each chapter's
                            caption as a line for the voice), speech_text.py (symbols as words),
                            tts.py (ElevenLabs, OpenAI or macOS voices, cached clips), voiceover.py (holds a
                            chapter's last frame while its voice finishes; ffmpeg dub) and voice_cli.py
  series/                   a playlist's introduction and conclusion, as full-screen cards: chapters.py,
                            cards.py and charts.py (the cards), playlist.py (the chapters every playlist
                            shares), results.py (the lessons' scores), render.py and voice.py
```

`lessons/cli.py` is the command line every lesson project runs. A project describes itself as a
`MatplotlibProject` (its lessons, data loader, `Scene`, summary CSV, row limits and output folder) and
`run` does the rest: it lists the lessons, or loads the data once and renders each chosen lesson,
reporting a lesson that fails and going on with the rest. `LessonProject`, the abstract base, leaves
only the options to its subclasses, which is how manimkit's Manim lessons reuse the same loop.

A lesson's dataset credit, axes and pictures come from the project's `Scene`; the kit's defaults
(for example `Stage(source_note="")`) are deliberately blank.

## Using it

There is nothing to install. Each project puts this folder on `sys.path`: the entry scripts and
`python -m <package>` do it at start-up, and pytest does it through
`pythonpath = [".", "../ml_animation_kit"]` in the project's `pyproject.toml`.

## How to read this code

Read the modules in this order, each with its test open beside it where it has one (in `tests/`):
the tests are short worked examples of what a module promises. Each step builds on the ones before.

| # | Read | What it teaches | Beside it |
|---|------|-----------------|-----------|
| 1 | `__init__.py`, `config.py`, `theme.py` | an engine that knows no dataset; canvas sizes; a colour-blind-safe palette with quiet chrome | `test_boundaries.py` |
| 2 | `models.py`, `trainers.py`, `frames.py` | four ways to turn training into visible steps: `partial_fit`, warm starts, a parameter sweep, a learning curve | |
| 3 | `lockstep.py`, `race.py`, `race_dashboard.py` | comparing models at the same share of their schedules; colour follows the model, not its rank | `test_boundaries.py` |
| 4 | `recorder.py`, `runner.py`, `inference.py`, `cli.py` | generator-driven rendering: draw once, send to many sinks; clean failures and exit codes | `test_recorder.py` |
| 5 | `panels.py`, `training_dashboard.py`, `equations/`, `equation_panel.py` | updating artists in place instead of redrawing; layout in points | `test_equation_panel.py` |
| 6 | `algorithms/` | drawing a model from its live parameters (a tree's paths, an MLP's activations times weights) | `test_boundaries.py` |
| 7 | `lessons/config.py`, `lessons/timing.py` | seconds to frames, easing, and sharing a frame budget out by largest remainder | `lessons/test_timing.py` |
| 8 | `lessons/lesson.py`, `lessons/data.py` | the lesson contract: a worked prediction must reproduce the model's own | `lessons/test_chapters.py` |
| 9 | `lessons/stage.py`, `lessons/theme.py`, `lessons/encoder.py` | layout in inches; raw RGBA frames piped into ffmpeg | `lessons/test_stage.py`, `lessons/test_encoder.py` |
| 10 | `lessons/diagrams.py`, `lessons/model_diagrams.py`, `lessons/tree_layout.py`, `lessons/fit.py` | algorithm diagrams, and checking a layout with the real renderer | `lessons/test_diagrams.py`, `lessons/test_model_diagrams.py`, `lessons/test_fit.py` |
| 11 | `lessons/plots.py`, `lessons/exam.py`, `lessons/evidence.py`, `lessons/fitting.py`, `lessons/surface.py` | the chapters' parts: measured speed, paired McNemar and bootstrap tests, a 3-D view's painting order | `lessons/test_surface.py` |
| 12 | `lessons/director.py`, `lessons/cli.py` | plug-ins through a `Protocol` (a project's `Scene`); chapter painters and players; one command line for every project | `lessons/test_chapters.py`, `lessons/test_cli.py` |

How the code explains itself:

- Every module opens with a docstring that explains the idea before any code: start there.
- Every class and function has a Google-style docstring: a one-line summary, then `Args`, `Returns`,
  `Raises`, `Yields`, `Attributes` or `Example` as they apply. Private helpers (a leading underscore)
  have a one-line summary.
- A method that overrides a library's says `See base class.`: its contract is the library's.
- The `>>>` examples in `trainers.py`, `lessons/tree_layout.py`, `lessons/timing.py`, `lessons/exam.py`
  and `lessons/lesson.py` run as doctests (`python -m pytest --doctest-modules animkit`).

## Tests

```bash
python -m pytest -q
python -m coverage run -m pytest -q && python -m coverage report
```
