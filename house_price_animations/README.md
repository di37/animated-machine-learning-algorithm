# House price regression animations

These animations show scikit-learn regressors learning California house prices live,
while the code runs. Each frame is drawn right after a real piece of training work
(a `partial_fit` chunk, a few more trees, one level deeper, more training rows), then
again while the trained model predicts the held-out houses batch by batch.

Dataset: `sklearn.datasets.fetch_california_housing`. It has 20,640 census block groups
and 8 features. The target is the median house value in $100k, capped at $500k. It is
downloaded once and cached in `~/scikit_learn_data`. The split is 60/20/20
train/validation/test. Features are standardised using the training rows only.

The Titanic classifier animations are a separate project in the sibling `../titanic_animations/`
folder. Both projects draw with the shared, dataset-neutral engine in `../ml_animation_kit/`
(package `animkit`), and neither imports the other.

## Run it

Keep `ml_animation_kit/` next to this folder: the entry scripts and `python -m house_anim` put it
on `sys.path` themselves, and pytest gets it from `pythonpath` in `pyproject.toml`. Nothing needs to
be installed as a package. Run every command from this folder.

```bash
pip install -r requirements.txt          # ffmpeg on PATH for .mp4 (brew install ffmpeg)

python animate.py                        # random forest: live window + mp4 files
python animate.py --model gradient_boosting
python animate.py --model all --no-live  # record every model without opening windows
python animate.py --list                 # what each model's animation shows
```

Two scenes compare several models in one video (16:9, 1600×900):

```bash
python animate.py --scene race   # The Algorithm Race: all 9 models, sliding leaderboard, test-set podium
python animate.py --scene map    # Watch the model learn California: 4 models repaint the state
```

- **Race:** all 9 models train in lockstep, so each frame shows every model equally far
  through its own schedule. On the left, each model has its own validation-error curve.
  On the right, a leaderboard re-sorts as models overtake each other. Ridge shares Lasso's
  colour, since the palette has 8 validated hues; it is told apart by a hatched bar, a dashed
  curve and its name. The video ends on a final-exam podium scored on the unseen test houses.
- **Map:** Linear Regression, Decision Tree, Gradient Boosting and MLP each price the
  *same typical house* (median income, age, rooms and occupancy) at every point in
  California. As they train, you can watch the plane tilt, the tree cut the state into
  boxes, boosting sharpen, and the MLP blend.

| Option | Meaning |
|---|---|
| `--scene dashboards\|race\|map` | per-model dashboards (default), or one of the two scenes above |
| `--model KEY\|all` | which regressor to animate (default `random_forest`) |
| `--phase training\|inference\|both` | which part to animate; `inference` trains silently first |
| `--no-live` / `--no-save` | skip the on-screen window / skip writing video files |
| `--format mp4\|gif` | video format (falls back to GIF if ffmpeg is missing) |
| `--fps`, `--dpi` | video frame rate and resolution |
| `--batch-size N` | test houses predicted per inference frame (default 40) |
| `--max-rows N` | subsample the dataset for a quick run |
| `--output-dir DIR` | where videos and `metrics_summary.csv` go (default `outputs/california_housing/`) |

Closing the live window, or pressing Ctrl-C, stops the run. Any frames recorded so far
are still saved as a playable video.

## What the animations show

Model dashboards record at **2560×1440**. The live window opens smaller so it fits a laptop
screen. Each dashboard has three columns: an animated diagram of the algorithm, the charts,
and the live equations.

**The algorithm, animated (left column):**

| Model | While training | While predicting the latest house |
|---|---|---|
| Linear, Ridge, Lasso, SGD | 8 inputs → weighted edges → Σ → ŷ; edge width and colour follow the weights (Lasso's zeroed edges turn dashed) | inputs light up with the house's values; edges carry each w·x push into Σ |
| Decision Tree | the tree diagram grows a level per frame, nodes coloured by mean price | the house's path through the splits, highlighted to its leaf |
| Random Forest | a grid of trees fills in as trees are added | every tree highlights its path and casts a vote; a strip shows all votes and their average |
| Gradient Boosting | the newest tree's corrections, and a strip of shrinking per-tree corrections | F_m(x) climbing from F₀ to ŷ tree by tree, plus each tree's correction |
| KNN | stored houses pile up on the map | the house linked to its 10 nearest neighbours |
| MLP | all 2,592 weights of the 8→64→32→1 network changing each epoch | activations firing through the layers (ReLU > 0), edges showing activation × weight |

**Training dashboard**, updated after every training step:
- **Learning curve:** train vs validation RMSE in $k. You can see underfitting, convergence and overfitting.
- **Validation predicted vs actual:** the cloud tightens onto the diagonal as the model learns.
- **Learned weights:** coefficients (blue +, red −) for linear models, impurity importances
  for trees, and mean first-layer |weight| for the MLP.
- **Residual histogram:** the share of houses priced within ±$50k.

- **Equations at work (right-hand column):** the model's formula with its current numbers,
  the objective training is minimising and its current value, and what this step changed.
  Examples: SGD's update rule with η and the update count, the boosting update
  F_M = F_0 + ν Σ h_m, the tree's root split and its variance reduction.

**Inference dashboard**, updated after every `model.predict()` batch on unseen test houses:
- **Equation worked through for the latest house:** linear terms w·x summed, the tree's
  decision path in real units, the forest's tree votes, boosting's corrections, the 10 nearest
  neighbours, or the MLP's forward pass. Each card ends with `check: model.predict(x) = …  ✓`.
  The tests assert that the worked equation equals the model's own prediction for all 9 models.
- **California map:** dots appear coloured by predicted price, and the latest batch is ringed.
- **Predicted vs actual, running RMSE/MAE/R², residuals:** these fill in house by house.
- **Live ticker:** the latest house's prediction vs its true value, batch latency and throughput.

| Key | Estimator | What one training frame is |
|---|---|---|
| `linear_regression` | `LinearRegression` | refit on more rows (learning curve) |
| `ridge` | `Ridge` | weaker L2 penalty: coefficients grow back smoothly, never exactly zero |
| `lasso` | `Lasso` | weaker L1 penalty: features switch on |
| `sgd` | `SGDRegressor` | `partial_fit` on 1/16 of an epoch |
| `decision_tree` | `DecisionTreeRegressor` | `max_depth` + 1 (watch it overfit) |
| `random_forest` | `RandomForestRegressor` | more trees (`warm_start`) |
| `gradient_boosting` | `GradientBoostingRegressor` | 5 more boosting stages (`warm_start`) |
| `knn` | `KNeighborsRegressor` | more houses memorised |
| `mlp` | `MLPRegressor` | one Adam epoch (`partial_fit`) |

Timings count only the model's own work. Scoring and drawing are excluded.
- **Live "total fit time":** all fitting so far. A sweep, such as tree depth, Lasso alpha
  or more data, sums every refit.
- **`fit s` in the summary:** what it cost to produce the *final* model. Warm-start and
  epoch-based models are cumulative, so their total counts. Sweep models count only their
  last refit. The CSV keeps the sweep total in `total_fit_seconds`.
- **Inference:** runs single-threaded (`n_jobs=1`). On 40-house batches, joblib's thread
  dispatch would otherwise cost several times more than the prediction itself.
- **Which model is reported:** test scores come from the model at the end of its schedule.
  For `decision_tree` that is depth 18, which overfits on purpose, and for `lasso` it is
  the weakest penalty.

`outputs/california_housing/metrics_summary.csv` is updated after every model, so runs of single models add
up to one table. The `rows_used` and `batch_size` columns show which run each row came from.

## Narrated lesson videos (one algorithm each)

The dashboards above show everything at once. The lessons tell one story per algorithm, built
to the `ml-training-animator` skill. Each is a 108-second, 2560×1440, 30 fps H.264 MP4 that works
with the sound off. It learns house value from a single feature, median income, so the model
itself is the curve you watch move; a 3-D chapter then adds house age as a second input.

```bash
python animate_lessons.py                         # all nine lessons (about 11 minutes at 1440p)
python animate_lessons.py --model decision_tree   # one lesson
python animate_lessons.py --resolution 1080p      # 1920x1080 (also 720p; 'preview' is 384x216)
python animate_lessons.py --duration-scale 0.3    # shorter chapters for a quick look
python animate_lessons.py --list
```

Every lesson plays ten chapters: **title → data → first guess → training → final model →
prediction on an unseen district → final exam → 3-D view → more features → takeaway**. The unseen district is
the test district whose income is nearest $60k. It is chosen by income alone and is the same house
in every lesson.

The 3-D view and the two evidence chapters around it go from one input, to two, to all eight:

- **Final exam:** the trained model prices all 4,128 test districts batch by batch.
  - The scoreboard shows the running RMSE, MAE, R², the share priced within ±$50k, and the prediction
    speed, measured once on the rendering machine.
  - A California map colours every district by its error: blue means worth more than predicted, red
    means worth less.
- **3-D view:** the same algorithm is refit on income and house age, and its predicted price is drawn as
  a surface over both while the camera swings to and fro. House age is the second input because it lowers
  a straight line's validation error the most.
  - Training dots, red error sticks and the unseen district (at this model's price) sit on the surface.
  - It uses the More-features settings below, except that Ridge and Lasso keep the lessons' degree-12
    polynomial (now of both inputs), with α re-chosen on validation.
  - The side panel judges house age against the same recipe refitted on income alone, on the same rows,
    with a paired bootstrap verdict (the polynomial lessons themselves train on only 30 districts).
- **More features:** the same algorithm with the same settings is refit on all 8 features and scored
  on the same test districts. A bar card compares it with the lesson's income-only model and the
  always-guess-the-average baseline.
  - Choices tuned for one input (polynomial degree, Ridge/Lasso α, k) are re-chosen on validation,
    never on test.
  - Ridge and Lasso use linear terms only, without the lessons' polynomial expansion.
  - The headline says "cut" or "worse" only when a paired bootstrap interval for the RMSE gain
    excludes zero; otherwise it says "about the same".

| Lesson | What a training frame is | What the side panel shows |
|---|---|---|
| `linear_regression` | one full-batch gradient-descent step (lr 0.1, 40 steps) | the real MSE bowl over (θ₀, θ₁) and the path taken |
| `polynomial_regression` | a fresh least-squares fit, degree 1 → 12 | one weight per term (x … x¹²) |
| `ridge` | a refit with a stronger L2 penalty α | weights shrinking smoothly |
| `lasso` | a refit with a stronger L1 penalty α | weights switching off (exactly 0, drawn ○) |
| `decision_tree` | one more best-first split (1 → 14 leaves) | the tree diagram growing; the house's root-to-leaf path |
| `random_forest` | more bootstrapped trees, averaged | newest tree vs the average; every tree's vote |
| `gradient_boosting` | the trees added since the last frame, each fitted to the residuals | residuals and the added trees' correction |
| `knn` | a refit with larger k (1 → 1,024) | the k nearest districts to a $35k income |
| `mlp` | real Adam training (`partial_fit`), 1→8→8→1 ReLU | all 97 weights; a forward pass at inference |

Polynomial, Ridge and Lasso fit only 30 districts, one from each income band, so over-fitting is
visible and validation never asks them to extrapolate. Sweeps (degree, α, k) keep the setting with
the lowest validation RMSE. The others keep the end of training.

**Truthfulness rules, enforced by `tests/lessons/`:**
- Every curve, metric and weight on screen comes from a real model state. A frame is held, never
  interpolated into a state that did not exist.
- The final frame's curve equals `model.predict`, and its RMSEs are recomputed from it.
- A forest snapshot equals a cold fit of that many trees.
- Every boosting frame names all the trees it added.
- The worked prediction at inference must reproduce `model.predict` before it may print ✓.
- Lasso refuses to show a solver that did not converge.
- Test districts appear only at the end: the prediction, the takeaway card and the summary.
- "Always guess the average" uses the full training split's mean in every lesson.

Videos and `lessons_summary.csv` go to `outputs/california_housing/lessons/`. The CSV holds the final
setting, train, validation and test RMSE, the prediction for the unseen district, the video
facts from ffprobe, and the model configuration.

## Layout

```
animate.py                  entry point (also: python -m house_anim)
animate_lessons.py          lesson videos entry point (also: python -m house_anim.lessons)
house_anim/
  kit_path.py               puts ../ml_animation_kit on sys.path for the entry points
  config.py, theme.py       California constants and the price colour ramp (shared tokens come from animkit)
  data.py                   load, validate, split, scale (read-only arrays)
  models.py                 registry: estimator + training schedule per model
  trainers.py               training loops that yield a frame per real step
  inference.py              batched, timed predictions on the test set
  frames.py, metrics.py     immutable snapshots passed to the dashboards
  panels.py                 price charts (predicted vs actual, residuals, learning curve)
  algorithms/               animated algorithm diagrams (weighted sum, trees, forest, boosting, KNN, network)
  equations/                per-family equation cards (linear, trees, KNN, MLP)
  training_dashboard.py     training figure
  inference_dashboard.py    inference figure
  lockstep.py               advance several training loops together
  race.py, race_dashboard.py         the Algorithm Race scene
  geo_grid.py, map_scene.py, map_dashboard.py   the California map scene
  scenes.py                 records the two scenes
  runner.py, cli.py         orchestration and command line
  summary.py                console table + metrics_summary.csv
  lessons/                  one-feature lessons: data view, regression scene, plots and stories/
                            (one module per algorithm family); cli.py only describes the lessons,
                            and the director, stage, encoder, timing and command line come from
                            animkit.lessons
tests/                      pytest suite (synthetic data, no network); tests/lessons/ for the lessons
outputs/california_housing/ videos and CSVs (lessons/ for the lesson videos)
```

The recorder, equation panel, weight bars, tree drawing, network diagram, lockstep, playback and
CLI helpers live in `../ml_animation_kit/animkit/`; see its README.

## How to read this code

Read each module with its test open beside it: the tests (in `tests/`) are short worked examples of
what the module promises. Each step builds on the ones before. For the engine underneath, read the
table in `../ml_animation_kit/README.md` first or alongside.

**The live animations** (`house_anim/`):

| # | Read | What it teaches | Beside it |
|---|------|-----------------|-----------|
| 1 | `kit_path.py`, `config.py`, `theme.py` | using a shared package that is not installed; units ($100k in the model, $k on screen); a single-hue ramp for amounts | `test_boundaries.py` |
| 2 | `data.py` | a train / validation / test split, scaled on the training rows only so nothing leaks; read-only arrays | `test_data.py` |
| 3 | `metrics.py`, `frames.py` | RMSE, MAE and R²; frozen snapshots handed from the training loops to the drawing | `test_metrics.py` |
| 4 | `models.py` | the four strategies that make learning visible: more data, a parameter sweep, incremental fitting, warm starts | `test_models.py` |
| 5 | `trainers.py` | generator loops that score the model after each real step of training | `test_trainers.py` |
| 6 | `inference.py` | timed, batched prediction on houses the model never saw | `test_inference.py` |
| 7 | `panels.py`, `training_dashboard.py`, `inference_dashboard.py` | building artists once and updating them every frame; what each chart diagnoses | `test_dashboards.py` |
| 8 | `algorithms/` | drawing each algorithm from the fitted estimator's real internals | `test_algorithm_views.py` |
| 9 | `equations/` | each family's formula, worked through for one house and checked against `model.predict` | `test_equations.py` |
| 10 | `runner.py`, `summary.py` | interleaving training with drawing; updating a CSV in place and replacing it in one step | `test_summary.py` |
| 11 | `lockstep.py`, `race.py`, `race_dashboard.py` | comparing models with schedules of different lengths fairly; ranking on validation, testing once | `test_lockstep.py`, `test_race.py`, `test_scene_dashboards.py` |
| 12 | `geo_grid.py`, `map_scene.py`, `map_dashboard.py` | what a model learned about location: one typical house, priced everywhere | `test_geo_grid.py`, `test_map_scene.py` |
| 13 | `scenes.py`, `cli.py`, `__main__.py`, `animate.py` | wiring the pieces into a command line | `test_cli.py` |

**The narrated lessons** (`house_anim/lessons/`, tests in `tests/lessons/`):

| # | Read | What it teaches | Beside it |
|---|------|-----------------|-----------|
| 1 | `config.py`, `data.py` | narrowing the splits to one input, and choosing what to show (the dots, the residuals, the query district) | `test_lesson_data.py` |
| 2 | `lesson.py` | the contract: immutable snapshots of real model states, the final pick by validation, a worked prediction checked against `predict` | `test_exam_and_features.py` |
| 3 | `stories/__init__.py`, `stories/linear_gd.py` | the builder pattern; full-batch gradient descent on the squared-error bowl | `test_stories.py` |
| 4 | `stories/polynomial.py` | a degree sweep and over-fitting; Ridge (L2 shrinks) against Lasso (L1 zeroes) | `test_stories.py` |
| 5 | `stories/tree.py`, `forest.py`, `boosting.py` | greedy splits; bagging and averaging; fitting the residuals with shrinkage | `test_stories.py` |
| 6 | `stories/knn.py`, `stories/mlp.py` | prediction from memory and the bias-variance choice of k; a ReLU network trained by Adam | `test_stories.py` |
| 7 | `plots.py`, `lesson_diagrams.py` | drawing in screen units from model units; weighted-sum diagrams | `test_lesson_diagrams.py` |
| 8 | `regression_scene.py`, `director.py` | implementing the engine's `Scene` protocol, so its task-agnostic director can draw a regression | `test_director.py` |
| 9 | `exam.py`, `more_features.py`, `surface_view.py` | honest test-set scoring; re-tuning on validation; a paired-bootstrap verdict against a matched control | `test_exam_and_features.py`, `test_surface_view.py` |
| 10 | `summary.py`, `cli.py`, `__main__.py` | a reproducible results CSV with an atomic merge; the shared lesson command line | `test_lesson_cli.py` |

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
tests filter only that message. Predictions are separately checked for NaN or
infinite values, so a real numerical failure, such as SGD diverging, still raises an error.
