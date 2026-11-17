"""Constants for the Titanic lesson videos: the three features, the map grid and the inference passenger.

Every lesson model reads three columns, in the order ``AGE``, ``LOG_FARE``, ``FEMALE``. Fare enters as
log(1 + fare), on the models and on the maps' fare axis alike: a log scale spreads out the many cheap
tickets instead of crowding them under a few very expensive ones. The map grid is the set of
``GRID_AGES`` × ``GRID_FARES`` (age, fare) points, one set per sex, on which each model's score is
computed to shade the maps.

Canvas, pacing and the encoder come from animkit.lessons.
"""

import math
from pathlib import Path

# Model inputs, in this column order: age in years, log(1 + fare), and 1 for women / 0 for men.
AGE, LOG_FARE, FEMALE = 0, 1, 2
FEATURE_NAMES = ("age", "fare", "female")

AGE_RANGE = (0.0, 80.0)
FARE_MAX = 520.0  # $, top of the fare axis (the highest fare paid was $512)
FARE_TICKS = (0, 5, 10, 25, 50, 100, 250, 500)
GRID_AGES = 120
GRID_FARES = 90
SEX_TITLES = ("Women", "Men")  # map order: female = 1 first

# The inference passenger: the test passenger closest to a 30-year-old woman on a $15 ticket.
QUERY_TARGET = (30.0, math.log1p(15.0), 1.0)

DEFAULT_OUTPUT_DIR = Path("outputs") / "titanic" / "lessons"
SUMMARY_FILE = "lessons_summary.csv"
SOURCE_NOTE = "Data: Titanic passenger list (OpenML 'titanic', version 1) · seed 42"
