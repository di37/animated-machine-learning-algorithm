# manim_lesson_kit (`manimkit`)

The dataset-neutral engine that plays the narrated lesson videos with Manim. The lessons themselves
(data, models, words, every number on screen) stay in the projects next to this folder, and so does the
matplotlib engine that draws them:

- `../ml_animation_kit/` (`animkit`): the matplotlib lesson engine; its director paints every frame
- `../house_price_manim_lessons/` (`house_manim`): the house-price lessons, played with this kit
- `../titanic_manim_lessons/` (`titanic_manim`): the Titanic lessons, played with this kit

`manimkit` imports animkit and Manim, and nothing dataset-specific; `tests/test_boundaries.py` checks that.

## How it plays a lesson

animkit's director paints each chapter on a matplotlib `Stage` as a function of time. manimkit runs the
same painters on the same stage, but never rasterizes it:

```
animkit painter ──▶ recorder ──▶ timeline ──▶ motion ──▶ player ──▶ ffmpeg
(paints the stage    (every path,    (follows each     (Manim          (Manim draws
 for this frame)      marker, text    shape from frame  animations:     the frame with
                      line and image  to frame: appears, reveal, morph,  Cairo; still
                      as a Manim      moves, changes,   cut, fade, on    frames are
                      shape)          goes)             chapter frames)  written again)
```

- **recorder.py, batches.py**: a matplotlib renderer that records instead of rasterizing. matplotlib
  lays the frame out exactly as for Agg (text is measured by the figure's own Agg renderer), and every
  draw call becomes a *leaf* in drawing order: paths transformed and snapped as Agg does them, a
  scatter's thousands of markers in one numpy pass, text lines placed on the pixel Agg rounds them to.
  Each leaf is keyed by the artist groups it was drawn in (the stage's axes and texts are named), so
  the same thing keeps its key on the next frame.
- **leaves.py, glyphs.py, paths.py**: a leaf builds its Manim shape. Text is drawn from the fonts' own
  glyph outlines (with matplotlib's fallback fonts), spaced to the width Agg's hinted text takes.
- **timeline.py**: on each sampled frame a leaf either draws the same, is the same thing moved or
  recoloured, or is something else in its place. Things carry over from one chapter to the next.
- **motion.py**: lives become Manim animations. A line is drawn in with `Create`, a large text with
  `Write`, the rest fade in; a thing matplotlib grows or fades in is revealed over those same frames;
  a thing that moves is transformed along the frames it moves on; a thing that goes fades out unless
  something takes its place. At a chapter's first frame the old picture fades out before the new one
  is drawn in.
- **player.py, camera.py**: plays the tracks frame by frame on a Manim `Scene`. The camera draws
  hard-edged shapes without antialiasing (the 3-D sheet), clips shapes to their axes as Agg does, and
  keeps each still shape's Cairo path until its points change.
- **director.py**: the ten chapters as animkit plays them, with the same frame counts. A moving chapter
  is painted on exactly the frames animkit renders it on (every second one, then t = 1 for the hold), and
  Manim moves things on every frame in between. Training holds each real snapshot and cuts to the next,
  as the matplotlib video does: nothing between two snapshots is invented. The 3-D view is *live*: each
  sample shows the box, the sheet and the dots where matplotlib's camera has them.

Every chapter ends on exactly the picture the matplotlib video ends on; in between, Manim moves things.

**cli.py** is animkit's lesson command line (`animkit.lessons.cli`) with Manim as the player: a Manim
`LessonProject` is the matplotlib project's (`LessonProject.of`), under its own description, and `run`
is animkit's run loop with manimkit's `render_lesson`. A project's own `cli.py` is a few lines.

## How to read this code

Read the modules in the order the data flows through them, with each one's test open beside it: the
tests are short worked examples of what the module promises. Each step builds on the one before.

| # | Read | What it teaches | Beside it |
|---|------|-----------------|-----------|
| 1 | `manimkit/__init__.py` | the map: every module in one line, in this order | |
| 2 | `frame.py` | two coordinate systems (matplotlib's pixels, Manim's units) and the one mapping between them; a frozen dataclass that checks itself | `tests/conftest.py` (`small_frame`, used by every test) |
| 3 | `paths.py` | Bézier curves: lines and quadratics as cubics (degree elevation); one numpy pass for the common case, a plain loop for the rest | `test_paths.py` |
| 4 | `glyphs.py` | how text is drawn: font outlines, hinting, and why Agg's text is a little wider | `test_glyphs.py` |
| 5 | `leaves.py` | the central idea: a drawn thing with a *key* (where it sits) and a *signature* (how it looks); one base class, three kinds that each say how to compare and build themselves | `test_leaves.py` |
| 6 | `recorder.py`, then `batches.py` | plugging into a library through its interface: a matplotlib renderer that records instead of drawing; then the same work for thousands of items at once with `einsum` | `test_recorder.py` |
| 7 | `timeline.py` | following objects across frames: when a thing is the same thing moved, and when it is something new | `test_timeline.py` |
| 8 | `motion.py` | planning as plain data: tracks are frozen records, and a bad plan is refused before a frame is drawn | `test_motion.py` |
| 9 | `camera.py`, then `player.py` | driving Manim by hand: an animation's `begin`, `interpolate` and `finish`; a cache that forgets with its shapes; a context manager that puts global state back | `test_camera.py`, `test_player.py` |
| 10 | `director.py` | composition: the pieces above wired into a whole lesson, one chapter plan at a time | `test_director.py` |
| 11 | `cli.py`, after `../ml_animation_kit/animkit/lessons/cli.py` | reuse through inheritance: animkit's abstract `LessonProject` holds what every project shares and its `run` is the one run loop; manimkit's `LessonProject` subclass only says where its options come from. The renderer and the data fetcher are passed in, so tests can stand in for both | `test_cli.py` |
| 12 | `parity.py` | testing against a reference: a subclass that watches animkit work, and pictures compared tile by tile | `test_parity_tools.py` |

How the code explains itself:

- Every module opens with a docstring that explains the idea before any code: start there.
- Every public class and function has a Google-style docstring: a one-line summary, then `Args`,
  `Returns`, `Raises`, `Yields`, `Attributes` or `Example` as they apply. Private helpers (a leading
  underscore) have a one-line summary.
- A method that overrides matplotlib's or Manim's says `See base class.`: its contract is the library's.
- Constants sit at the top of their module, each with a comment saying where its number comes from.
- The `Example` in `paths.cubic_points` runs as a doctest
  (`.venv/bin/python -m pytest --doctest-modules manimkit`); the others need a real figure or lesson,
  so they are marked `+SKIP` and show the call.

## Setup

The Manim folders share one environment, made from the base Python so the models are the same ones
the matplotlib lessons train:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

Manim is pinned to 0.22.1: the player and camera use Manim's `Scene` and Cairo `Camera` internals, and
the recorder uses two private matplotlib font helpers (`TextToPath._get_font`, `RendererAgg._prepare_font`)
so that text uses the fonts Agg uses. matplotlib is the base environment's (3.10).

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m coverage run -m pytest -q && .venv/bin/python -m coverage report
```

**parity.py** checks a project's lessons against animkit itself: it runs animkit's own chapter players
(with its own `Recorder`) for the reference frames, renders the Manim lesson from the same prepared exam,
comparison and 3-D view, and compares each chapter's last frame tile by tile. Both lesson folders' parity
tests use it for every lesson.

The player checks itself: a plan that would leave an animation running past its chapter is refused, and
`Player.mismatches(chapter)` lists anything on screen that is not the chapter's last recorded picture (the
house-price tests call it after every chapter).

## Known differences from the matplotlib frames

- Text: Agg draws hinted glyph bitmaps, Manim the unhinted outlines, placed within a pixel of them; the
  difference shows most on the cards' largest headlines.
- The 3-D sheet: Agg's hard-edged tile mesh shows thin seams along slanted tile edges; Cairo fills
  the mesh without them.
- Dashed lines: Manim spaces the dashes evenly along each line, so their phase differs slightly.
- One-pixel outlines (a random forest's region maps) are rasterized a little differently.
