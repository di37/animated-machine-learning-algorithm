"""Every finished video, with the words that describe it: shared by the README and the Pages site."""

REPO = "di37/animated-machine-learning-algorithm"
REPO_URL = f"https://github.com/{REPO}"
PAGES_URL = "https://di37.github.io/animated-machine-learning-algorithm/"
# jsDelivr serves the repository's files with Content-Type video/mp4, so a link plays in the browser.
CDN = f"https://cdn.jsdelivr.net/gh/{REPO}@main"

HOUSE_DASH = "house_price_animations/outputs/california_housing"
HOUSE_LESSONS = f"{HOUSE_DASH}/lessons"
HOUSE_MANIM = "house_price_manim_lessons/outputs/california_housing/lessons"
TITANIC_DASH = "titanic_animations/outputs/titanic"
TITANIC_LESSONS = f"{TITANIC_DASH}/lessons"
TITANIC_MANIM = "titanic_manim_lessons/outputs/titanic/lessons"
# The Manim lessons with an ElevenLabs voice-over, which the Pages site plays (the README keeps the silent ones).
HOUSE_MANIM_VOICED = "house_price_manim_lessons/outputs/california_housing/lessons_voiced"
TITANIC_MANIM_VOICED = "titanic_manim_lessons/outputs/titanic/lessons_voiced"
# Each playlist's voiced introduction and conclusion (Manim), shown around its Manim lessons.
HOUSE_SERIES = "house_price_manim_lessons/outputs/california_housing/series_voiced"
TITANIC_SERIES = "titanic_manim_lessons/outputs/titanic/series_voiced"
# key, emoji, title, what it covers
SERIES_VIDEOS = (
    ("introduction", "👋", "Introduction", "who the series is for, the data, the nine algorithms and how a lesson runs"),
    ("conclusion", "🎓", "Conclusion", "both scoreboards, and three lessons the numbers support"),
)

# key, emoji, title, what one training frame is, what the inference video adds
HOUSE_MODELS = (
    ("linear_regression", "📏", "Linear Regression",
     "refit on more houses each frame: a straight line plateaus early",
     "each feature's w·x push flows into Σ and lands on the price"),
    ("ridge", "🪢", "Ridge (L2)",
     "a weaker L2 penalty each frame: weights grow back smoothly, never exactly 0",
     "the same weighted sum, with the shrunken weights"),
    ("lasso", "✂️", "Lasso (L1)",
     "a weaker L1 penalty each frame: features switch on one by one (zeroed edges go dashed)",
     "only the surviving features push the price"),
    ("sgd", "🎯", "SGD Regressor",
     "partial_fit on 1/16 of an epoch: averaged weights walk downhill on squared error",
     "the learned weighted sum, house by house"),
    ("decision_tree", "🌳", "Decision Tree",
     "one level deeper each frame: training error falls, validation error turns back up",
     "the house's root-to-leaf path through the splits, in real units"),
    ("random_forest", "🌲", "Random Forest",
     "more bootstrapped trees each frame (warm_start)",
     "12 trees light up their paths and vote; a strip shows all 100 votes and the average"),
    ("gradient_boosting", "🚀", "Gradient Boosting",
     "5 more shallow trees each frame, each fitted to the residuals",
     "the price climbing from F₀ to the prediction, tree by tree"),
    ("knn", "📍", "K-Nearest Neighbours (k=10)",
     "more houses memorised on the California map",
     "the house (a star) linked to its 10 nearest neighbours"),
    ("mlp", "🧠", "MLP (64-32)",
     "one Adam epoch: all 2,592 weights change on screen",
     "hidden units firing (ReLU > 0) and the signal each edge carries"),
)

TITANIC_MODELS = (
    ("logistic_regression", "📈", "Logistic Regression",
     "a larger C each frame: weights grow from zero, validation passengers separate on the S-curve",
     "inputs → weighted edges → Σ → σ → P(survive), and the passenger's point on the S-curve"),
    ("sgd", "🎯", "SGD Classifier (logistic loss)",
     "partial_fit on 1/8 of an epoch: averaged weights walk downhill on log loss",
     "the same σ diagram with the SGD weights"),
    ("svm", "🛡️", "Support Vector Machine (RBF)",
     "refit on more passengers: support vectors ringed on a PCA map, sized by α",
     "the 10 strongest α·y·K pulls, and the signed margin f(x)"),
    ("naive_bayes", "🔔", "Gaussian Naive Bayes",
     "count more passengers: a survived / did-not-survive bell curve per feature sharpens",
     "the passenger's value on each curve and its log-likelihood ratio"),
    ("decision_tree", "🌳", "Decision Tree",
     "one level deeper each frame, nodes coloured by survival share",
     "the yes/no questions: “male? yes · age ≤ 12.5? no …” down to the leaf"),
    ("random_forest", "🌲", "Random Forest",
     "more bootstrapped trees each frame (warm_start)",
     "every tree's P(survive), and the forest's average"),
    ("gradient_boosting", "🚀", "Gradient Boosting",
     "5 more trees each frame, correcting the log-odds",
     "P(survive) climbing from the 38% base rate, tree by tree"),
    ("knn", "📍", "K-Nearest Neighbours (k=11)",
     "more passengers memorised on a PCA map",
     "the 11 most similar passengers and how many survived"),
    ("mlp", "🧠", "MLP (32-16)",
     "one Adam epoch per frame",
     "activations firing into a sigmoid output"),
)

# key, emoji, title, what a training frame is
HOUSE_LESSONS_LIST = (
    ("linear_regression", "📏", "Linear regression", "40 full-batch gradient-descent steps, on the real MSE bowl"),
    ("polynomial_regression", "〰️", "Polynomial regression", "degree 1 → 12: watch it over-fit 30 districts"),
    ("ridge", "🪢", "Ridge", "a stronger L2 penalty: weights shrink smoothly"),
    ("lasso", "✂️", "Lasso", "a stronger L1 penalty: weights switch off (exactly 0)"),
    ("decision_tree", "🌳", "Decision tree", "one more best-first split, 1 → 14 leaves"),
    ("random_forest", "🌲", "Random forest", "more bootstrapped trees, averaged"),
    ("gradient_boosting", "🚀", "Gradient boosting", "trees fitted to the residuals"),
    ("knn", "📍", "k-nearest neighbours", "k from 1 to 1,024; validation picks k"),
    ("mlp", "🧠", "Neural network (MLP)", "real Adam training of a 1→8→8→1 ReLU network, all 97 weights"),
)

TITANIC_LESSONS_LIST = (
    ("logistic_regression", "📈", "Logistic regression", "full-batch gradient descent on the log loss"),
    ("perceptron", "⚡", "Perceptron", "online updates; a signed score, never called a probability"),
    ("linear_svm", "🛡️", "Linear SVM", "SGD epochs on hinge loss; dashed ±1 margins"),
    ("naive_bayes", "🔔", "Naive Bayes", "priors and per-class bell curves update"),
    ("decision_tree", "🌳", "Decision tree", "one more best-first split, 1 → 12 leaves"),
    ("random_forest", "🌲", "Random forest", "more bootstrapped trees, probabilities averaged"),
    ("gradient_boosting", "🚀", "Gradient boosting", "trees added in log-odds"),
    ("knn", "📍", "k-nearest neighbours", "k from 1 to 256; validation picks k"),
    ("mlp", "🧠", "Neural network (MLP)", "real Adam training of a 3→8→8→1 network"),
)

# title, blurb, video path
HOUSE_SCENES = (
    ("🏁 The Algorithm Race", "all 9 regressors train in lockstep; the leaderboard re-sorts, then a test-set podium",
     f"{HOUSE_DASH}/algorithm_race.mp4"),
    ("🗺️ Watch the model learn California", "4 models price the same typical house at every point in the state",
     f"{HOUSE_DASH}/california_map.mp4"),
)
TITANIC_SCENES = (
    ("🏁 The Algorithm Race", "9 classifiers race on validation error, then a test-set podium",
     f"{TITANIC_DASH}/algorithm_race.mp4"),
    ("🧭 Who survives?", "4 models paint P(survive) for every age, class and sex as they train",
     f"{TITANIC_DASH}/survival_landscape.mp4"),
)
