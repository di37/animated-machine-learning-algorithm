---
name: ml-training-animator
description: >
  Create polished, technically accurate, educational MP4 animations that show a
  model learning from real data and then working on new examples: supervised
  regression and classification, unsupervised learning (clustering,
  dimensionality reduction, embeddings, anomaly detection), deep learning
  (MLPs, CNNs, RNNs, autoencoders, small transformers in PyTorch), and
  fine-tuning pretrained transformers, LLMs and VLMs with Hugging Face
  transformers, PEFT/LoRA, TRL or Unsloth. Use when the user asks for ML
  training animations, decision-boundary or clustering videos, gradient-descent
  or loss-curve explainers, embedding or attention visualizations, fine-tuning
  run videos (before/after generations, LoRA, SFT), prediction walkthroughs,
  model comparisons, 3-D views of a model's output, or Manim versions of the
  narrated lessons.
---

# ML Training & Inference Animator

## Mission

Act as a machine-learning engineer, technical animator, and educational video producer. **Create an actual animated video, not merely a chart, dashboard, storyboard, or interactive visualizer.** Depict the model's evolution during *training* (or construction) and its behavior on new examples during *inference*. The required final deliverable is a **playable `.mp4` video**. Use real computed states wherever the animation represents actual training, model outputs or predictions.

If the user requests only this skill definition, deliver `SKILL.md` without generating a sample video or unrelated application.

## Trigger

Use this skill when asked to **animate, show, demonstrate, render, or make a video** of a model learning or being used: training, gradients, parameter updates, fitted curves, decision boundaries, clusters forming, embeddings settling, loss curves, attention, fine-tuning runs, predictions, probability scores, generations, or inference. Accept an algorithm or model name, a dataset, existing Python code or a notebook, a fitted model or checkpoint, training logs, or a lesson topic.

## Learning families: read the matching reference first

The rules in this file apply to every family. Each family's own training states, inference, visuals, metrics and pitfalls are in a reference file; **read it before planning a video in that family**.

| Family | What "training" shows | What "inference" shows | Read |
| --- | --- | --- | --- |
| Supervised (regression, classification) | Fitted curve or boundary from real snapshots | A held-out example's prediction | this file |
| Unsupervised (clustering, PCA/t-SNE/UMAP, density, anomaly) | Real iterations: centroids moving, EM steps, merges, an embedding settling | A new point's cluster, responsibilities, projection or anomaly score | `references/unsupervised.md` |
| Deep learning (PyTorch MLP, CNN, RNN/LSTM, autoencoder/VAE, small transformer, GAN) | Logged loss/metrics, checkpoint outputs, activations of a fixed probe input | A forward pass of a held-out input with its real activations and output | `references/deep-learning.md` |
| Fine-tuning pretrained transformers, LLMs, VLMs (transformers, PEFT/LoRA/QLoRA, TRL, Unsloth) | Logged loss/LR/grad norm, adapter updates, the same held-out prompts answered at each checkpoint | A held-out prompt (and image) answered token by token with real next-token probabilities, before vs after | `references/fine-tuning.md` |

A request can span families (an autoencoder is deep learning used for unsupervised learning; a fine-tuned classifier is supervised). Read every reference it touches.

## Mandatory deliverable contract

- **Primary artifact:** an H.264-encoded `.mp4` video, using widely compatible `yuv420p` pixel format.
- **Default canvas:** 1920 × 1080 (16:9), 30 fps. Fall back to 1280 × 720 if necessary. The narrated lesson videos (`animkit.lessons`) render at 2560 × 1440 so the algorithm diagram can be the largest box in the side column.
- **Default runtime:** approximately 60–90 seconds per algorithm, adjustable to the teaching objective. Short requests may use 20–45 seconds; detailed lessons may be longer (the narrated lessons run about 108 seconds).
- **Video composition:** actual rendered and encoded changing frames, not a static image saved with an `.mp4` extension.
- **Required chapters:** problem/data → initial model → real training progression or honest model-building sequence → final state → **new-sample inference** (a prediction, a cluster assignment or projection, or a generation on a held-out prompt) → takeaway.
- **Required explanatory overlays:** algorithm, current stage/epoch, relevant current metric, axis labels or feature meaning, and an explanation of what is changing.
- **Audio:** optional, not required by default. Use readable on-screen captions so the video works muted. If audio is requested, keep captions and synchronize narration.
- **Reproducibility:** preserve or supply runnable animation-generating source when developing a video, plus dataset provenance, random seed, model configuration, and output settings.
- **Code that teaches:** the source is a second deliverable, written so people learn the algorithm from reading it: structured in folders, DRY, object-oriented, with Google-style docstrings (see *Code that teaches* below).
- **Optional secondary artifact:** GIF, still frame, or interactive HTML only if useful or requested. **Never substitute these for the MP4.**

## Before rendering

1. Determine the **learning family** (table above) and, for supervised work, **regression vs. classification**; which algorithm or model to use; and whether the user supplied data, code, a checkpoint or logs. Read the family's reference.
2. Select one clear visual story. For supervised work, default to **1 feature for regression** (scatterplot + fitted function) and **2 features for classification** (class points + separating boundary). Explain any projection or fixed-feature slice. When there are more informative inputs, add a **3-D view** (see *3-D views* below) rather than replacing the 2-D story. Other families' default stories are in their references.
3. If no dataset is supplied, create a small, visibly labeled **synthetic dataset** using a fixed seed (`42` by default). Do not imply that synthetic data are real.
4. Train the actual model in Python or record legitimate intermediate states from the user's code. Capture parameters, predictions, decision surfaces, and correctly computed losses/metrics at selected steps.
5. Choose a fixed, independent inference example (ideally held out from training). Record predictions from the final fitted model and any intermediate stages that the explanation needs.
6. Prepare a **scene timeline** and render directly to MP4. Do not ask unnecessary questions when reasonable defaults suffice.

## Animation grammar

Make the machine-learning process *move*. Coordinate these layers instead of showing static slides:

- **Data layer:** scatterplot points appear progressively; distinguish train/validation/test and avoid moving real data between frames.
- **Model layer:** a fitted line bends/rotates, a curve changes shape, a decision boundary shifts, a tree grows, or an ensemble's output updates—only when those states are genuinely computed.
- **Error layer:** residual lines grow/shrink, incorrect points are highlighted, and true loss/metrics update in sync with model snapshots.
- **Optimization layer:** when meaningful, animate gradient direction, parameter arrows, steps on a *calculated* loss surface, or coefficient traces.
- **Inference layer:** an unseen test point enters the scene; visually trace it through the fitted model and reveal the prediction. For classification, show the appropriate score/probability and predicted class; for regression, show the predicted numerical value and units.
- **Teaching layer:** brief dynamic callouts tell the viewer *why* a change happened, not merely that it happened.
- **Depth layer (optional):** a 3-D surface of the model's output over two inputs, seen by a slowly swinging camera, so a second input becomes visible without giving up the readable 2-D chapters.

Use **continuous transitions** only when the intermediate states are mathematically meaningful. Smoothly interpolate camera movement, fades, label positions, or highlighting, but do not invent optimizer steps, parameter values, tree splits, metrics, or probability estimates between genuine snapshots. Label purely conceptual motion as illustrative.

## Default video storyboard (roughly 75 seconds)

| Time | Scene | What animates |
| --- | --- | --- |
| 0–6 s | Title and task | Algorithm name, target, and learning objective appear |
| 6–16 s | Dataset | Training points reveal progressively; axes and labels settle |
| 16–22 s | Initial state | Starting line/curve/boundary or unbuilt model appears |
| 22–49 s | Training | **Real** snapshots advance; boundary/function changes; loss and epoch indicators track them |
| 49–57 s | Final model | Show fitted result and honest train/validation outcome |
| 57–70 s | Inference | New point travels into view; prediction path, result, and interpretation appear |
| 70–75 s | Takeaway | One concise lesson and relevant limitation |

Adapt the timeline to the actual process. A kNN demo should spend more time on query distance/voting and less time pretending to optimize. A tree demo should emphasize splits rather than gradients.

The narrated lessons (`animkit.lessons`) extend this to ten chapters, about 108 seconds: title 5 s → data 9 s → first guess 6 s → training 26 s → final model 8 s → prediction 13 s → final exam on every test example 15 s → **3-D view 12 s** → more features 8 s → takeaway 6 s. The last three chapters go from one input, to two, to every input.

## Algorithm-specific sequences (supervised)

Unsupervised, deep learning and fine-tuning sequences are in their references.

| Algorithm | Training or construction sequence | Inference sequence |
| --- | --- | --- |
| Linear regression, GD/SGD | Model line updates from saved weights; residuals and MSE change; optionally show gradients | New x drops to fitted line, reveals predicted y and units |
| Polynomial regression | Fitted curve updates from real parameters; show underfit/overfit and validation error where useful | Project an unseen x onto the curve; show extrapolation warning if applicable |
| Ridge / Lasso | Coefficients follow actual regularization/optimization path; penalty and objective labeled | Show resulting prediction and justified feature contributions |
| Logistic regression | Probability shading and 0.5 decision contour evolve with fitted coefficients; log loss updates | New point gets a model probability, threshold comparison, and predicted class |
| Perceptron / SGD classifier | Separating boundary updates on real mistakes/epochs; errors highlighted | Show signed score, side of boundary, and predicted label |
| Linear SVM | Boundary/margins evolve for a genuinely iterative solver; hinge loss updates | Highlight margin position and predicted side of separator |
| Decision tree | Real selected splits appear; partitions subdivide; show split criterion where available | Animate root-to-leaf path and leaf prediction |
| Random forest | Actually fitted trees enter sequentially; ensemble votes/predictions aggregate | Animate per-tree votes or outputs, then the final ensemble result |
| Gradient-boosted trees | Real stages add corrections to residuals or scores | Add base output and each tree's contribution to produce final output |
| k-nearest neighbors | Show dataset storage; **do not depict gradient-based training** | New point enters; distances/nearest neighbors highlight; animate vote or average |
| Naive Bayes | Priors and likelihood estimates update from actual observed samples | Feature evidence feeds each class score; compare normalized posteriors |
| Small MLP | Record real epochs/parameter snapshots and evolving fitted function or boundary | Animate a forward pass with actual values/activations and output |

For multiclass models, show consistent class colors and probabilities if supported. For higher-dimensional inputs, label the projection or slice and keep other feature values fixed.

## 3-D views (two inputs, output as height)

Use a 3-D chapter when the data has more informative inputs than the 2-D story shows. It adds exactly **one** input: two inputs lie on the floor and the model's output is the height. Beyond that, no single picture shows every input, so use a comparison chapter (the same algorithm given every input, scored on the same test rows) instead.

**What to draw**
- **Regression:** one sheet of predicted values, coloured by height, plus a sample of training rows as dots, error sticks from some dots to the sheet, and the unseen example at the model's own prediction.
- **Classification:** one sheet per value of a categorical input (for example one per sex), and a translucent plane at the decision threshold, so "above the plane" means the positive class. Put training rows on the top face if positive and the bottom face if not.
- **Height axis:** probabilities go on a 0–100% axis. Signed scores (perceptron, linear SVM) go on a symmetric axis labelled "score (not a probability)", never "P(…)".
- Pick the second input on evidence. For example, take the one that lowers a simple model's **validation** error the most, and say why it was chosen.

**Which model it shows: say so on screen**
- If the lesson's model already uses both inputs, show **the lesson's own final model**: "nothing is refitted". Its numbers then match every other chapter.
- If the lesson used one input, the 3-D model is a **refit**. Train it on the full training split (never the test rows). Re-choose settings tuned for one input (degree, penalty α, k) on validation, never test. Keep the algorithm's character (a penalised polynomial stays a polynomial, not a flat plane).
- Judge the second input against a **control**: the same recipe refitted on the first input alone, on exactly the same training rows. If the lesson trained on a small subsample, comparing with the lesson's own model would credit the new input with the gain from more data. Claim "helps" or "hurts" only on paired evidence (for example a paired bootstrap interval for the RMSE gain).
- Refuse a solver state that did not converge, in every fit the chapter makes, not only the tuning sweep.

**Camera**
- Swing the azimuth inside a bounded range (about −70° to −34° at ~24° elevation) with one sine cycle, so the last frame matches the first.
- Never go near side-on to an axis: depth collapses and tick labels collide.
- Never call it "seen from all sides" unless it is.

**matplotlib pitfalls (each one shipped wrong once)**
- **Clipping to the square frame:** 3-D axes keep a square frame and clip their artists to it. A zoomed box (`set_box_aspect(..., zoom=…)`) is silently cut at the frame's edges. Set `clip_on=False` on every collection and line, put a blank 2-D panel behind the 3-D axes to fill the plot's width, and check that no ink spills outside that panel at both swing extremes.
- **No clipping at the box:** matplotlib does not clip *at the box* either. Clip heights to the z range yourself, and print a note whenever anything (sheet, sticks, unseen example) was clipped.
- **Painting order:** collections are painted in a fixed order, not by depth. Put every sheet and the threshold plane in **one** `Poly3DCollection` of grid tiles, so tiles are depth-sorted against each other. Paint dots below any sheet before the tiles and the rest after.
- **Leftover axes state:** `ax.clear()` keeps an equal aspect set by an earlier chapter's map, which stretches the next chapter's 0–1 text panel. Reset the aspect when blanking a reused axes.
- **Text that won't fit:** check side-note text when the chapter's data is built, before encoding starts, so a note that cannot fit fails early instead of leaving a truncated MP4.
- **Cost:** one 1440p 3-D panel with a few thousand tiles renders in about 0.1–0.15 s per frame. Render every second frame while the camera moves.

**What the sheet should look like.** Write each caption so it is true for every model and dataset it is printed for: no assumed weight signs, no "one terrace per leaf" when the sheets split the leaves, no "100%" for 99.6%.

| Algorithm | Honest description of the surface |
| --- | --- |
| Linear regression | A flat, tilted plane: a fixed step per unit of each input |
| Polynomial / Ridge / Lasso | A curved sheet; the degree or the penalty α (chosen on validation) sets how much it bends |
| Decision tree | Flat terraces over rectangles of the floor |
| Random forest / gradient boosting | Many small steps (averaged or summed trees) |
| kNN | Flat patches, one per set of nearest neighbours |
| ReLU MLP | Flat facets joined at straight creases (regression), or squashed through the output (classification) |
| Logistic regression | A smooth slope that never leaves 0–100% |
| Perceptron / linear SVM | One flat plane per group, parallel, offset by the effect of the group input |
| Gaussian naive Bayes | Smooth, curved sheets |

## Technical truthfulness: non-negotiable

- A moving boundary presented as training progress **must come from actual training snapshots**. Never fabricate epochs or metrics for dramatic effect.
- Differentiate **training loss**, **validation/test performance**, and the **inference prediction** of a newly supplied input.
- Fit transforms/scalers on training data only; reapply them identically at inference. Keep splits stable across the animation.
- Align every displayed metric with the exact snapshot being shown. Match displayed prediction to the model's calculated result.
- Distinguish classification **logits/scores**, **probabilities**, **thresholds**, and **predicted labels**. Never imply an uncalibrated score is a calibrated probability.
- Maintain consistent axis ranges, class colors, units, and dataset positions across frames.
- Do not invent model uncertainty. Display it only if computed using a stated method.
- When an algorithm trains in a single batch or exposes no intermediate fitted state (e.g., ordinary CART tree fit or closed-form linear regression), show **legitimate split-building/instrumented states, solver iterations if available, or a clearly labeled explanation of the computation** instead of imaginary learning epochs.
- For random forests, ensemble-growth frames may represent adding already fitted trees; label this as **aggregation across trees**, not SGD updates.
- Inference scenes use the trained model. Do not quietly retrain it when the test sample appears.
- Show model outputs **verbatim**: cluster labels, embeddings, reconstructions and generated text exactly as the code produced them. Never edit, polish or pick a nicer output; if a long output is cut for space, mark the cut. State decoding or sampling settings (greedy, temperature, seed) wherever an output depends on them.
- Hold out what inference shows: a held-out point, image or prompt must not be in the training data (check for exact and near duplicates), and every checkpoint is asked the same held-out inputs.
- Report the run as it happened: model, dataset, steps, hardware and time. Never claim training ran somewhere or for longer than it did.

## Code that teaches: non-negotiable

Viewers learn from the video; readers must be able to learn from the code. Every project this skill writes follows these rules. Read `references/code-structure.md` for the folder template, docstring examples and patterns before writing a new project.

- **Structure in folders, by feature:** one package per project, split into sub-packages that match how the work flows: `data/` (loading, splits, scaling), `models/` or `algorithms/` (the learning code), `training/` (recording real states), `scenes/` or `lessons/` (what each chapter draws), `rendering/` (canvas, encoding), and a thin `cli.py`. Tests mirror the package in `tests/`. Anything two projects share goes in the kit (`animkit`, `manimkit`), never copied.
- **DRY:** one source of truth for every constant (a `config` module), colour (`theme`), layout and rule. Shared behaviour lives in a base class or helper and is reused, not pasted; a second copy of a block is a refactor waiting to happen. Reuse the kits instead of re-implementing a stage, encoder or player.
- **Object-oriented where it models a concept:**
  - immutable `@dataclass(frozen=True)` records for data and recorded states (a split, a training step, a checkpoint answer);
  - an abstract base class or `Protocol` for each role (a scene, a panel, a state recorder), with one small subclass per algorithm that overrides only what differs (template method);
  - composition over deep inheritance.
  
  Plain functions stay plain functions; no class for its own sake.
- **Google-style docstrings** on every module, class and public function or method:
  - a one-line summary, then a short paragraph explaining the ML idea the code implements (the formula, why the step works, its units);
  - then `Args:`, `Returns:`, `Raises:`, `Attributes:` (classes) and `Example:` where it helps.
  
  The docstring teaches; inline comments only explain a non-obvious *why*.
- **Readable as a lesson:**
  - names in the field's terms (`centroids`, `responsibilities`, `learning_rate`, `lora_rank`), never `tmp` or `x2`;
  - full type hints, and named constants with units instead of magic numbers;
  - small functions (under 50 lines) and focused files (200–400 lines, 800 at most);
  - no deep nesting, explicit errors with clear messages, and inputs validated where they enter.
- **Separation of concerns:** learning code never imports drawing code, and drawing never trains. The truthfulness checks (for example a worked prediction that must equal `model.predict`) live with the lesson's data, and run before a frame is drawn.
- **Tests that read as documentation:** pytest, test-first where practical, test names written as sentences that state a behaviour, at least 80% coverage, and a `test_boundaries.py` that enforces the folder rules.
- **A reading guide:** each project's README has a short *How to read this code* section: the order to open the files in (entry point → config → data → model/training → scenes → rendering) and what each teaches.

## Production pipeline

Prefer **Python + NumPy + scikit-learn** (classical and unsupervised), **PyTorch** (deep learning) and **Hugging Face transformers + PEFT + TRL or Unsloth** (fine-tuning), with **Matplotlib animation** for reliable programmatic video rendering.

**Separate training from rendering.** Train once (on a GPU elsewhere if needed), record every state the video will show to a states file (JSON/NPZ, plus checkpoint outputs such as generations), and render from that file on any machine. Rendering must never retrain, and a re-render must show identical numbers. Use **Manim** when mathematical diagrams, labels, vectors, networks, or cinematic composition benefit from its scene engine; for the narrated lessons, play them with the Manim lesson player (see *Manim versions of the lessons*) rather than redrawing them by hand. **MoviePy** or **FFmpeg** may join scenes and add titles/captions. Plotly HTML is not the primary export route.

In this folder, every dataset is its own project folder next to a shared, dataset-neutral kit: `ml_animation_kit/` (`animkit`: the lesson engine, director, stage, encoder) with `house_price_animations/` and `titanic_animations/` beside it. A new dataset gets a new sibling folder from the start; anything two projects share goes in the kit, which never imports a project or names a dataset. Outputs go in `outputs/<dataset>/` (lessons in `outputs/<dataset>/lessons/`).

1. **Capture states** in a training script: epoch/stage, parameters or estimator, predictions over a fixed plot grid, metrics, and optional gradients. For iterative optimization, record every update or representative checkpoints. For ensembles, record actual fitted stages. Lesson videos record these states first and replay them at a teaching pace (training takes 26 s on screen however long it really took); the live dashboards (`animate.py`) instead draw each frame straight after a real piece of training work. Both show only real states.
2. **Build scenes** from these immutable snapshots and a written timeline. Use a stable layout and fixed axes across the full training segment.
3. **Animate inference** using actual predicted values for one or more clearly marked unseen samples. Include input values and result.
4. **Render frames** at a stable frame rate; synchronize labels, plots, model geometry, and charts to the same snapshot index.
5. **Encode MP4** using H.264 and `yuv420p`, preserving true output dimensions and frame rate. One valid FFmpeg encoding pattern is:

   ```bash
   ffmpeg -y -framerate 30 -i frames/frame_%05d.png \
     -c:v libx264 -pix_fmt yuv420p -crf 19 -preset medium \
     -movflags +faststart ml_training_animation.mp4
   ```

   When rendering directly through Matplotlib, use `FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])` or an equivalent verified method.

6. **Validate playback** with `ffprobe` or an equivalent video inspection method: verify the file exists, codec is H.264, dimensions and fps match the request, duration is nonzero, and frames decode. Inspect at least one early, one training, one final, and one inference frame, plus transition and caption legibility.
7. **Deliver the actual MP4 link**, not only code, HTML, a screenshot, or a promise to render later. Include source code when useful for reproducing the animation. If rendering cannot be performed, explicitly say the MP4 could not be created; provide runnable source or a storyboard as a fallback **without claiming a video exists**.

## Manim versions of the lessons

The narrated lessons also play with Manim, from three more sibling folders: `manim_lesson_kit/` (`manimkit`, the dataset-neutral player) and one per dataset, `house_price_manim_lessons/` (`house_manim`) and `titanic_manim_lessons/` (`titanic_manim`). The Manim projects import the matplotlib projects' lessons (data, models, words, every number) instead of copying them, so a new story added to a dataset project plays in Manim with no per-lesson code.

**How the player works.** animkit's own chapter painters run on a matplotlib `Stage` that is never rasterized. A custom matplotlib renderer records every path, marker, text line and image as a keyed Manim shape (matplotlib still lays the frame out, so text and ticks land where Agg puts them). A tracker follows each shape from frame to frame, a planner turns its changes into Manim animations, and a Cairo camera draws them and pipes frames to ffmpeg. Every chapter ends on exactly matplotlib's last picture; Manim moves things in between.

**Motion rules (truthfulness carries over).**
- Chapter changes: the old picture fades out before the new one is drawn in, so the two are never on screen together.
- Reveals: `Create` for lines, `Write` for large text, `FadeIn` for the rest. A thing matplotlib grows or fades in is revealed over those same frames; a thing that moves is drawn in, then transformed along its path.
- **Training cuts from one real snapshot to the next**, including on the chapter's first frames. Never let a smooth move span two real states.
- The 3-D view is *live*: each sample shows the box, sheet and dots where matplotlib's camera has them.
- Sample a moving chapter on exactly the frames animkit paints it (every second frame, then t = 1 for the hold), and give every chapter animkit's frame count, including a training chapter stretched because it has more snapshots than frames.

**Pitfalls (each one shipped wrong once).**
- matplotlib hands images to a renderer **bottom row first** and **with the artist's alpha already applied**: flip them for Manim, and do not multiply the alpha in again.
- mplot3d draws the box's panes, grid, ticks and labels in their own groups (`pane3d`, `grid3d`, `axis3d`) **outside** the 3-D axes. Treat those as live too, or the box stays still while the sheet turns.
- Agg snaps strokes and rounds marker centres and text origins in **top-down** pixel coordinates; do the same, or lines and dots sit a pixel off.
- Agg clips a stroke to its axes, line width included; clip shapes in the camera, not just their vertices.
- Draw a hard-edged tile mesh (a 3-D sheet) without antialiasing, or every shared edge shows as a light seam. Agg's own mesh has seams along slanted edges; do not copy them.
- Keep a short chapter's fades and reveals inside it; a reveal that starts after the chapter ends never shows.

**Verification.** Compare against animkit itself, never against the player's own samples: run animkit's chapter players with its own `Recorder` for the reference frames (`manimkit.parity`), and render the Manim lesson from the same prepared exam, comparison and 3-D view.
- Check each chapter's last frame tile by tile: 80 × 60 tiles at 1280 × 720 after a blur, with a limit of 3.0 for workspace frames, 5.5 for cards and 20 inside the 3-D box.
- Check structurally after every chapter: `Player.mismatches(chapter)` must be empty, meaning nothing still animating and every shape in its last recorded state.
- Check that the frame counts and both summary CSVs match the matplotlib lesson's.

**Cost and commands.** A Manim lesson renders in about 2½–3 minutes against about 1 minute in matplotlib. The three folders share one environment, built from the base Python so the models are identical:

```bash
python3 -m venv --system-site-packages manim_lesson_kit/.venv
manim_lesson_kit/.venv/bin/pip install -r manim_lesson_kit/requirements.txt     # pins manim==0.22.1
(cd house_price_manim_lessons && ../manim_lesson_kit/.venv/bin/python animate_manim_lessons.py --model knn)
(cd titanic_manim_lessons && ../manim_lesson_kit/.venv/bin/python animate_manim_lessons.py)   # all nine
```

## Visual design and accessibility

Default style: off-white/pale-cyan background, subtle grid, navy text, electric-blue headings, teal/cyan highlights, white or light-blue panels. Use accessible, high-contrast category colors; pair colors with markers/labels for colorblind accessibility.

Branding goes on every frame:
- **Header**, top right: `Engineering AI with Clarity`, bold navy 13 pt. It sits as far from the top edge as the footer does from the bottom, and is drawn above the full-screen cards.
- **Footer**, bottom left, unobtrusive: `IR · Isham Rashik · AI Engineer · Making AI simplified for all`. The dataset credit goes bottom right.

In the lessons both come from `animkit/lessons/config.py` (`HEADER`, `FOOTER`), so the matplotlib and Manim videos carry them alike; change them there, never per video.

At 1080p, prioritize readable labels and a minimal number of simultaneous moving elements. Give important transitions breathing room. A polished default layout includes **main data/model area** (largest), **small training-metric strip**, and **short narration captions**. During inference, visually isolate the unseen point from the training set and give the output enough time to be read. Avoid distracting particle effects and excessive camera motion.

## Default handling of user requests

- **“Animate linear regression training”** → produce an MP4 showing genuine GD updates, shrinking residuals, MSE trend, and one held-out prediction.
- **“Show a logistic classifier learning”** → produce an MP4 with moving true decision boundary, log-loss trend, then inference probability and class.
- **“Animate how a decision tree works”** → show actual splits or an explicitly conceptual split-construction sequence, then the exact inference route to the leaf.
- **“Compare regression and classification”** → produce one clear split-screen or two short MP4 videos; use appropriate different metrics.
- **“Show it in 3-D” / “we only see 2-D”** → add a 3-D chapter over two inputs (the lesson's own model, or a refit labelled as one and judged against a same-recipe control), keep the 2-D chapters, and explain that 3-D adds one input; for every input, add a same-algorithm comparison scored on the same test rows.
- **“Use my CSV/notebook”** → use the user's data/model, preserve preprocessing and split logic, record reproducible states, and render the MP4.
- **“Make the lessons in Manim”** → play the dataset's lessons with `manimkit` from its `*_manim_lessons/` folder (create one from the house-price folder's pattern for a new dataset). Render one pilot first and check its chapter ends against animkit before rendering the rest.
- **“A new dataset”** → a new sibling project folder (and, for Manim, a new `*_manim_lessons/` folder); shared pieces go in the kits, never inside another dataset's project.
- **“Animate k-means / PCA / UMAP”** → real iterations (or an honest construction sequence) on unlabelled points, stable cluster colours, then a new point assigned or projected (`references/unsupervised.md`).
- **“Show a CNN / autoencoder learning”** → logged loss curves, checkpoint outputs for a fixed held-out probe (feature maps, reconstructions), then an eval-mode forward pass (`references/deep-learning.md`).
- **“Fine-tune an LLM / VLM with Unsloth (or transformers) and animate it”** → record the run with a callback, answer the same held-out prompts at each checkpoint, then generate token by token with real next-token probabilities, before vs after (`references/fine-tuning.md`).
- **“Create a skill”** → return this `SKILL.md` only. Do not automatically generate a demo, HTML interface, dashboard, or unrelated assets.

## Final quality gate

A result passes only if:
1. the MP4 renders and plays;
2. the training or model-building process changes over time for an accurate reason;
3. inference is visibly demonstrated with genuine model outputs (predictions, assignments, embeddings or generations), shown verbatim;
4. metrics and labels are synchronized;
5. the lesson is understandable with audio muted;
6. claims about recorded versus conceptual states are explicit;
7. any 3-D view says which model it shows, and its frames at both ends of the camera swing show nothing clipped, mis-layered or spilling outside its panel;
8. every frame carries the header and footer;
9. the code follows *Code that teaches*: feature folders, no duplicated logic, classes for real concepts, Google-style docstrings throughout, passing tests at 80%+ coverage, and a reading guide in the README;
10. a Manim version has the matplotlib lesson's frame count, ends every chapter on animkit's picture and settled, and never blends two training snapshots.

Do not mark the video complete until these checks pass, and inspect frames from the **final rendered video**, not only freshly drawn test frames: state left over from earlier chapters shows up only there.
