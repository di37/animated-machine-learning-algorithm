# 🎬 Machine Learning Algorithms, Animated

> Watch scikit-learn models **learn** from real data, frame by frame, and then **predict** on data
> they have never seen. Every frame is drawn from a real model state, and every worked prediction is
> checked against the model's own answer. ✅
>
> 🔍 **The black box, opened:** you see what each model stores, how it combines its inputs, and exactly
> why it gave *this* answer for *this* example.

Two classic datasets, two renderers, **76 videos**:

| | 🏠 California Housing | 🚢 Titanic |
|---|---|---|
| **Task** | regression: a district's median house value | classification: who survived |
| **Data** | 20,640 districts, 8 features | 1,309 passengers (OpenML `titanic`) |
| **📊 Live dashboards** | 9 regressors × training + inference | 9 classifiers × training + inference |
| **🏁 Comparison scenes** | The Algorithm Race · Watch the model learn California | The Algorithm Race · Who survives? |
| **🎓 Narrated lessons** | 9 lessons, matplotlib and Manim | 9 lessons, matplotlib and Manim |
| **🎞️ Videos** | 38 | 38 |

> [!NOTE]
> The videos are embedded with HTML `<video>` players and play inline in VS Code or Cursor's Markdown
> preview and other local Markdown viewers. GitHub does not play repository videos inside a README, so
> every player also has a ▶️ link that opens the file.

## 🧭 Contents

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

## 🏠 California Housing: regression

`sklearn.datasets.fetch_california_housing`: 20,640 census districts, 8 features, target = median
house value (capped at $500k). Split 60/20/20, scaled on the training rows only.
Code: [`house_price_animations/`](house_price_animations/README.md).

### 📊 Live dashboards: 9 regressors, training and inference

Each dashboard is **2560×1440**, in three columns: the animated algorithm, the live charts (learning
curve, predicted vs actual, learned weights, residuals) and the equations at work.

#### 📏 Linear Regression

<table>
  <tr><th width="50%">🏋️ Training: refit on more houses each frame: a straight line plateaus early</th><th width="50%">🔮 Inference: each feature's w·x push flows into Σ and lands on the price</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/linear_regression_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/linear_regression_training.mp4"><code>linear_regression_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/linear_regression_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/linear_regression_inference.mp4"><code>linear_regression_inference.mp4</code></a></td>
  </tr>
</table>

#### 🪢 Ridge (L2)

<table>
  <tr><th width="50%">🏋️ Training: a weaker L2 penalty each frame: weights grow back smoothly, never exactly 0</th><th width="50%">🔮 Inference: the same weighted sum, with the shrunken weights</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/ridge_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/ridge_training.mp4"><code>ridge_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/ridge_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/ridge_inference.mp4"><code>ridge_inference.mp4</code></a></td>
  </tr>
</table>

#### ✂️ Lasso (L1)

<table>
  <tr><th width="50%">🏋️ Training: a weaker L1 penalty each frame: features switch on one by one (zeroed edges go dashed)</th><th width="50%">🔮 Inference: only the surviving features push the price</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/lasso_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lasso_training.mp4"><code>lasso_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/lasso_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lasso_inference.mp4"><code>lasso_inference.mp4</code></a></td>
  </tr>
</table>

#### 🎯 SGD Regressor

<table>
  <tr><th width="50%">🏋️ Training: partial_fit on 1/16 of an epoch: averaged weights walk downhill on squared error</th><th width="50%">🔮 Inference: the learned weighted sum, house by house</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/sgd_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/sgd_training.mp4"><code>sgd_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/sgd_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/sgd_inference.mp4"><code>sgd_inference.mp4</code></a></td>
  </tr>
</table>

#### 🌳 Decision Tree

<table>
  <tr><th width="50%">🏋️ Training: one level deeper each frame: training error falls, validation error turns back up</th><th width="50%">🔮 Inference: the house's root-to-leaf path through the splits, in real units</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/decision_tree_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/decision_tree_training.mp4"><code>decision_tree_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/decision_tree_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/decision_tree_inference.mp4"><code>decision_tree_inference.mp4</code></a></td>
  </tr>
</table>

#### 🌲 Random Forest

<table>
  <tr><th width="50%">🏋️ Training: more bootstrapped trees each frame (warm_start)</th><th width="50%">🔮 Inference: 12 trees light up their paths and vote; a strip shows all 100 votes and the average</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/random_forest_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/random_forest_training.mp4"><code>random_forest_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/random_forest_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/random_forest_inference.mp4"><code>random_forest_inference.mp4</code></a></td>
  </tr>
</table>

#### 🚀 Gradient Boosting

<table>
  <tr><th width="50%">🏋️ Training: 5 more shallow trees each frame, each fitted to the residuals</th><th width="50%">🔮 Inference: the price climbing from F₀ to the prediction, tree by tree</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/gradient_boosting_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/gradient_boosting_training.mp4"><code>gradient_boosting_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/gradient_boosting_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/gradient_boosting_inference.mp4"><code>gradient_boosting_inference.mp4</code></a></td>
  </tr>
</table>

#### 📍 K-Nearest Neighbours (k=10)

<table>
  <tr><th width="50%">🏋️ Training: more houses memorised on the California map</th><th width="50%">🔮 Inference: the house (a star) linked to its 10 nearest neighbours</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/knn_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/knn_training.mp4"><code>knn_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/knn_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/knn_inference.mp4"><code>knn_inference.mp4</code></a></td>
  </tr>
</table>

#### 🧠 MLP (64-32)

<table>
  <tr><th width="50%">🏋️ Training: one Adam epoch: all 2,592 weights change on screen</th><th width="50%">🔮 Inference: hidden units firing (ReLU > 0) and the signal each edge carries</th></tr>
  <tr>
    <td><video src="house_price_animations/outputs/california_housing/mlp_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/mlp_training.mp4"><code>mlp_training.mp4</code></a></td>
    <td><video src="house_price_animations/outputs/california_housing/mlp_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/mlp_inference.mp4"><code>mlp_inference.mp4</code></a></td>
  </tr>
</table>


### 🏁 Comparison scenes

<table>
  <tr>
    <td width="50%"><b>🏁 The Algorithm Race</b><br><sub>all 9 regressors train in lockstep; the leaderboard re-sorts, then a test-set podium</sub><br><video src="house_price_animations/outputs/california_housing/algorithm_race.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/algorithm_race.mp4"><code>algorithm_race.mp4</code></a></td>
    <td width="50%"><b>🗺️ Watch the model learn California</b><br><sub>4 models price the same typical house at every point in the state</sub><br><video src="house_price_animations/outputs/california_housing/california_map.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/california_map.mp4"><code>california_map.mp4</code></a></td>
  </tr>
</table>


### 🎓 Narrated lessons (matplotlib): one algorithm each

The model learns house value from **median income alone**, so the model itself is the curve you watch
move; the 3-D chapter then adds house age. Code: `house_price_animations/house_anim/lessons/`.

<table>
  <tr>
    <td width="50%"><b>📏 Linear regression</b><br><sub>40 full-batch gradient-descent steps, on the real MSE bowl</sub><br><video src="house_price_animations/outputs/california_housing/lessons/linear_regression_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/linear_regression_lesson.mp4"><code>linear_regression_lesson.mp4</code></a></td>
    <td width="50%"><b>〰️ Polynomial regression</b><br><sub>degree 1 → 12: watch it over-fit 30 districts</sub><br><video src="house_price_animations/outputs/california_housing/lessons/polynomial_regression_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/polynomial_regression_lesson.mp4"><code>polynomial_regression_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🪢 Ridge</b><br><sub>a stronger L2 penalty: weights shrink smoothly</sub><br><video src="house_price_animations/outputs/california_housing/lessons/ridge_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/ridge_lesson.mp4"><code>ridge_lesson.mp4</code></a></td>
    <td width="50%"><b>✂️ Lasso</b><br><sub>a stronger L1 penalty: weights switch off (exactly 0)</sub><br><video src="house_price_animations/outputs/california_housing/lessons/lasso_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/lasso_lesson.mp4"><code>lasso_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🌳 Decision tree</b><br><sub>one more best-first split, 1 → 14 leaves</sub><br><video src="house_price_animations/outputs/california_housing/lessons/decision_tree_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/decision_tree_lesson.mp4"><code>decision_tree_lesson.mp4</code></a></td>
    <td width="50%"><b>🌲 Random forest</b><br><sub>more bootstrapped trees, averaged</sub><br><video src="house_price_animations/outputs/california_housing/lessons/random_forest_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/random_forest_lesson.mp4"><code>random_forest_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🚀 Gradient boosting</b><br><sub>trees fitted to the residuals</sub><br><video src="house_price_animations/outputs/california_housing/lessons/gradient_boosting_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/gradient_boosting_lesson.mp4"><code>gradient_boosting_lesson.mp4</code></a></td>
    <td width="50%"><b>📍 k-nearest neighbours</b><br><sub>k from 1 to 1,024; validation picks k</sub><br><video src="house_price_animations/outputs/california_housing/lessons/knn_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/knn_lesson.mp4"><code>knn_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🧠 Neural network (MLP)</b><br><sub>real Adam training of a 1→8→8→1 ReLU network, all 97 weights</sub><br><video src="house_price_animations/outputs/california_housing/lessons/mlp_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_animations/outputs/california_housing/lessons/mlp_lesson.mp4"><code>mlp_lesson.mp4</code></a></td>
  </tr>
</table>


### 🎞️ Narrated lessons (Manim)

The same nine lessons, played with Manim. Code: [`house_price_manim_lessons/`](house_price_manim_lessons/README.md).

<table>
  <tr>
    <td width="50%"><b>📏 Linear regression</b><br><sub>40 full-batch gradient-descent steps, on the real MSE bowl</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/linear_regression_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/linear_regression_lesson.mp4"><code>linear_regression_lesson.mp4</code></a></td>
    <td width="50%"><b>〰️ Polynomial regression</b><br><sub>degree 1 → 12: watch it over-fit 30 districts</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/polynomial_regression_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/polynomial_regression_lesson.mp4"><code>polynomial_regression_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🪢 Ridge</b><br><sub>a stronger L2 penalty: weights shrink smoothly</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/ridge_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/ridge_lesson.mp4"><code>ridge_lesson.mp4</code></a></td>
    <td width="50%"><b>✂️ Lasso</b><br><sub>a stronger L1 penalty: weights switch off (exactly 0)</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/lasso_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/lasso_lesson.mp4"><code>lasso_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🌳 Decision tree</b><br><sub>one more best-first split, 1 → 14 leaves</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/decision_tree_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/decision_tree_lesson.mp4"><code>decision_tree_lesson.mp4</code></a></td>
    <td width="50%"><b>🌲 Random forest</b><br><sub>more bootstrapped trees, averaged</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/random_forest_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/random_forest_lesson.mp4"><code>random_forest_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🚀 Gradient boosting</b><br><sub>trees fitted to the residuals</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/gradient_boosting_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/gradient_boosting_lesson.mp4"><code>gradient_boosting_lesson.mp4</code></a></td>
    <td width="50%"><b>📍 k-nearest neighbours</b><br><sub>k from 1 to 1,024; validation picks k</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/knn_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/knn_lesson.mp4"><code>knn_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🧠 Neural network (MLP)</b><br><sub>real Adam training of a 1→8→8→1 ReLU network, all 97 weights</sub><br><video src="house_price_manim_lessons/outputs/california_housing/lessons/mlp_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="house_price_manim_lessons/outputs/california_housing/lessons/mlp_lesson.mp4"><code>mlp_lesson.mp4</code></a></td>
  </tr>
</table>


## 🚢 Titanic: classification

OpenML's `titanic`: 1,309 passengers. The `boat` and `body` columns are dropped, because they are only
known after the sinking and would leak the answer. Stratified 60/20/20 split, so every split keeps
the same 38% survival rate; gaps are filled from the training passengers only.
Code: [`titanic_animations/`](titanic_animations/README.md).

### 📊 Live dashboards: 9 classifiers, training and inference

Training shows the accuracy learning curve, the ROC curve with AUC, the learned weights and the
confusion matrix. Inference shows the passenger list, running accuracy and recall, and how sure the
model is. Blue means survived, red means did not, and the neutral grey midpoint is the 0.5 threshold.

#### 📈 Logistic Regression

<table>
  <tr><th width="50%">🏋️ Training: a larger C each frame: weights grow from zero, validation passengers separate on the S-curve</th><th width="50%">🔮 Inference: inputs → weighted edges → Σ → σ → P(survive), and the passenger's point on the S-curve</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/logistic_regression_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/logistic_regression_training.mp4"><code>logistic_regression_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/logistic_regression_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/logistic_regression_inference.mp4"><code>logistic_regression_inference.mp4</code></a></td>
  </tr>
</table>

#### 🎯 SGD Classifier (logistic loss)

<table>
  <tr><th width="50%">🏋️ Training: partial_fit on 1/8 of an epoch: averaged weights walk downhill on log loss</th><th width="50%">🔮 Inference: the same σ diagram with the SGD weights</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/sgd_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/sgd_training.mp4"><code>sgd_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/sgd_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/sgd_inference.mp4"><code>sgd_inference.mp4</code></a></td>
  </tr>
</table>

#### 🛡️ Support Vector Machine (RBF)

<table>
  <tr><th width="50%">🏋️ Training: refit on more passengers: support vectors ringed on a PCA map, sized by α</th><th width="50%">🔮 Inference: the 10 strongest α·y·K pulls, and the signed margin f(x)</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/svm_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/svm_training.mp4"><code>svm_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/svm_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/svm_inference.mp4"><code>svm_inference.mp4</code></a></td>
  </tr>
</table>

#### 🔔 Gaussian Naive Bayes

<table>
  <tr><th width="50%">🏋️ Training: count more passengers: a survived / did-not-survive bell curve per feature sharpens</th><th width="50%">🔮 Inference: the passenger's value on each curve and its log-likelihood ratio</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/naive_bayes_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/naive_bayes_training.mp4"><code>naive_bayes_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/naive_bayes_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/naive_bayes_inference.mp4"><code>naive_bayes_inference.mp4</code></a></td>
  </tr>
</table>

#### 🌳 Decision Tree

<table>
  <tr><th width="50%">🏋️ Training: one level deeper each frame, nodes coloured by survival share</th><th width="50%">🔮 Inference: the yes/no questions: “male? yes · age ≤ 12.5? no …” down to the leaf</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/decision_tree_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/decision_tree_training.mp4"><code>decision_tree_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/decision_tree_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/decision_tree_inference.mp4"><code>decision_tree_inference.mp4</code></a></td>
  </tr>
</table>

#### 🌲 Random Forest

<table>
  <tr><th width="50%">🏋️ Training: more bootstrapped trees each frame (warm_start)</th><th width="50%">🔮 Inference: every tree's P(survive), and the forest's average</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/random_forest_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/random_forest_training.mp4"><code>random_forest_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/random_forest_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/random_forest_inference.mp4"><code>random_forest_inference.mp4</code></a></td>
  </tr>
</table>

#### 🚀 Gradient Boosting

<table>
  <tr><th width="50%">🏋️ Training: 5 more trees each frame, correcting the log-odds</th><th width="50%">🔮 Inference: P(survive) climbing from the 38% base rate, tree by tree</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/gradient_boosting_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/gradient_boosting_training.mp4"><code>gradient_boosting_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/gradient_boosting_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/gradient_boosting_inference.mp4"><code>gradient_boosting_inference.mp4</code></a></td>
  </tr>
</table>

#### 📍 K-Nearest Neighbours (k=11)

<table>
  <tr><th width="50%">🏋️ Training: more passengers memorised on a PCA map</th><th width="50%">🔮 Inference: the 11 most similar passengers and how many survived</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/knn_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/knn_training.mp4"><code>knn_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/knn_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/knn_inference.mp4"><code>knn_inference.mp4</code></a></td>
  </tr>
</table>

#### 🧠 MLP (32-16)

<table>
  <tr><th width="50%">🏋️ Training: one Adam epoch per frame</th><th width="50%">🔮 Inference: activations firing into a sigmoid output</th></tr>
  <tr>
    <td><video src="titanic_animations/outputs/titanic/mlp_training.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/mlp_training.mp4"><code>mlp_training.mp4</code></a></td>
    <td><video src="titanic_animations/outputs/titanic/mlp_inference.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/mlp_inference.mp4"><code>mlp_inference.mp4</code></a></td>
  </tr>
</table>


### 🏁 Comparison scenes

<table>
  <tr>
    <td width="50%"><b>🏁 The Algorithm Race</b><br><sub>9 classifiers race on validation error, then a test-set podium</sub><br><video src="titanic_animations/outputs/titanic/algorithm_race.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/algorithm_race.mp4"><code>algorithm_race.mp4</code></a></td>
    <td width="50%"><b>🧭 Who survives?</b><br><sub>4 models paint P(survive) for every age, class and sex as they train</sub><br><video src="titanic_animations/outputs/titanic/survival_landscape.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/survival_landscape.mp4"><code>survival_landscape.mp4</code></a></td>
  </tr>
</table>


### 🎓 Narrated lessons (matplotlib): one classifier each

Each classifier learns survival from **age, fare and sex**: two maps (women | men), age across, fare on
a log scale, shaded by the model's score with its decision boundary drawn and every misclassified
training passenger circled. Code: `titanic_animations/titanic_anim/lessons/`.

<table>
  <tr>
    <td width="50%"><b>📈 Logistic regression</b><br><sub>full-batch gradient descent on the log loss</sub><br><video src="titanic_animations/outputs/titanic/lessons/logistic_regression_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/logistic_regression_lesson.mp4"><code>logistic_regression_lesson.mp4</code></a></td>
    <td width="50%"><b>⚡ Perceptron</b><br><sub>online updates; a signed score, never called a probability</sub><br><video src="titanic_animations/outputs/titanic/lessons/perceptron_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/perceptron_lesson.mp4"><code>perceptron_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🛡️ Linear SVM</b><br><sub>SGD epochs on hinge loss; dashed ±1 margins</sub><br><video src="titanic_animations/outputs/titanic/lessons/linear_svm_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/linear_svm_lesson.mp4"><code>linear_svm_lesson.mp4</code></a></td>
    <td width="50%"><b>🔔 Naive Bayes</b><br><sub>priors and per-class bell curves update</sub><br><video src="titanic_animations/outputs/titanic/lessons/naive_bayes_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/naive_bayes_lesson.mp4"><code>naive_bayes_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🌳 Decision tree</b><br><sub>one more best-first split, 1 → 12 leaves</sub><br><video src="titanic_animations/outputs/titanic/lessons/decision_tree_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/decision_tree_lesson.mp4"><code>decision_tree_lesson.mp4</code></a></td>
    <td width="50%"><b>🌲 Random forest</b><br><sub>more bootstrapped trees, probabilities averaged</sub><br><video src="titanic_animations/outputs/titanic/lessons/random_forest_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/random_forest_lesson.mp4"><code>random_forest_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🚀 Gradient boosting</b><br><sub>trees added in log-odds</sub><br><video src="titanic_animations/outputs/titanic/lessons/gradient_boosting_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/gradient_boosting_lesson.mp4"><code>gradient_boosting_lesson.mp4</code></a></td>
    <td width="50%"><b>📍 k-nearest neighbours</b><br><sub>k from 1 to 256; validation picks k</sub><br><video src="titanic_animations/outputs/titanic/lessons/knn_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/knn_lesson.mp4"><code>knn_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🧠 Neural network (MLP)</b><br><sub>real Adam training of a 3→8→8→1 network</sub><br><video src="titanic_animations/outputs/titanic/lessons/mlp_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_animations/outputs/titanic/lessons/mlp_lesson.mp4"><code>mlp_lesson.mp4</code></a></td>
  </tr>
</table>


### 🎞️ Narrated lessons (Manim)

The same nine lessons, played with Manim. Code: [`titanic_manim_lessons/`](titanic_manim_lessons/README.md).

<table>
  <tr>
    <td width="50%"><b>📈 Logistic regression</b><br><sub>full-batch gradient descent on the log loss</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/logistic_regression_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/logistic_regression_lesson.mp4"><code>logistic_regression_lesson.mp4</code></a></td>
    <td width="50%"><b>⚡ Perceptron</b><br><sub>online updates; a signed score, never called a probability</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/perceptron_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/perceptron_lesson.mp4"><code>perceptron_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🛡️ Linear SVM</b><br><sub>SGD epochs on hinge loss; dashed ±1 margins</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/linear_svm_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/linear_svm_lesson.mp4"><code>linear_svm_lesson.mp4</code></a></td>
    <td width="50%"><b>🔔 Naive Bayes</b><br><sub>priors and per-class bell curves update</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/naive_bayes_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/naive_bayes_lesson.mp4"><code>naive_bayes_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🌳 Decision tree</b><br><sub>one more best-first split, 1 → 12 leaves</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/decision_tree_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/decision_tree_lesson.mp4"><code>decision_tree_lesson.mp4</code></a></td>
    <td width="50%"><b>🌲 Random forest</b><br><sub>more bootstrapped trees, probabilities averaged</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/random_forest_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/random_forest_lesson.mp4"><code>random_forest_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🚀 Gradient boosting</b><br><sub>trees added in log-odds</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/gradient_boosting_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/gradient_boosting_lesson.mp4"><code>gradient_boosting_lesson.mp4</code></a></td>
    <td width="50%"><b>📍 k-nearest neighbours</b><br><sub>k from 1 to 256; validation picks k</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/knn_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/knn_lesson.mp4"><code>knn_lesson.mp4</code></a></td>
  </tr>
  <tr>
    <td width="50%"><b>🧠 Neural network (MLP)</b><br><sub>real Adam training of a 3→8→8→1 network</sub><br><video src="titanic_manim_lessons/outputs/titanic/lessons/mlp_lesson.mp4" controls muted playsinline preload="metadata" width="100%"></video><br>▶️ <a href="titanic_manim_lessons/outputs/titanic/lessons/mlp_lesson.mp4"><code>mlp_lesson.mp4</code></a></td>
  </tr>
</table>


## 🏆 Scoreboard

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
machine-learning-algorithms/
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

---

<p align="center"><sub>IR · Isham Rashik · AI Engineer · Making AI simplified for all</sub></p>
