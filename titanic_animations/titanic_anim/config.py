"""Constants for the Titanic survival animations.

Canvas sizes and panel layout come from ``animkit.config``. Two ideas recur across the package:

- A classifier scores a passenger in one of two ways. Most give a probability, ``predict_proba``'s
  P(survive), and predict "survived" when it is above ``PROBABILITY_THRESHOLD`` (0.5). The SVM
  gives a signed margin, ``decision_function``'s f(x), and predicts "survived" when it is above
  ``MARGIN_THRESHOLD`` (0). ``SCORE_PROBABILITY`` and ``SCORE_MARGIN`` name the two kinds.
- The passengers are split three ways: ``TEST_FRACTION`` for the test set, ``VAL_FRACTION`` for
  validation, the rest for training. ``RANDOM_STATE`` fixes the split and every model's
  randomness, so a run can be repeated exactly.
"""

from pathlib import Path

RANDOM_STATE = 42

OPENML_NAME = "titanic"
OPENML_VERSION = 1
TARGET_COLUMN = "survived"

TEST_FRACTION = 0.2
VAL_FRACTION = 0.2
MIN_ROWS = 50

# How a model scores a passenger: predict_proba's P(survive), or decision_function's signed margin.
SCORE_PROBABILITY = "probability"
SCORE_MARGIN = "margin"

CLASSES = (0, 1)
CLASS_NAMES = ("did not survive", "survived")
# Probability models predict "survived" when P(survive) > 0.5; margin models when f(x) > 0.
PROBABILITY_THRESHOLD = 0.5
MARGIN_THRESHOLD = 0.0
# |f(x)| that shows as a fully saturated colour for margin models (the SVM).
MARGIN_COLOUR_LIMIT = 2.0

# Smallest training subset for the "more data" (learning-curve) strategy.
MIN_GROWTH_ROWS = 20

DEFAULT_INFERENCE_BATCH = 4
# The test set is small (262 passengers), so each inference frame is held a little longer.
INFERENCE_HOLD_FRAMES = 2
DEFAULT_OUTPUT_DIR = Path("outputs") / "titanic"
DEFAULT_MODEL = "random_forest"

SCORE_BIN_COUNT = 20
