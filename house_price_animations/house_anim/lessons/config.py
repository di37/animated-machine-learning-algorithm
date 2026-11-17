"""Constants for the regression lessons: the one-feature data view, the inference house and the credits.

Canvas, resolutions and pacing every lesson shares live in animkit.lessons.config; the ones house
modules use are re-exported here.
"""

from pathlib import Path

from animkit.lessons.config import DEFAULT_RESOLUTION, DURATION_SCALE_RANGE, RESOLUTIONS

__all__ = [
    "DEFAULT_OUTPUT_DIR", "DEFAULT_RESOLUTION", "DISPLAY_POINTS", "DURATION_SCALE_RANGE", "FEATURE", "GRID_POINTS",
    "INCOME_TO_K", "POLY_SAMPLE", "PRICE_CAP", "PRICE_TO_K", "QUERY_TARGET_INCOME", "RESIDUAL_POINTS",
    "RESOLUTIONS", "SOURCE_NOTE", "SUMMARY_FILE", "X_RANGE", "Y_RANGE_K",
]

# The single feature every lesson learns from, and the unit conversions for the axes.
FEATURE = "MedInc"
INCOME_TO_K = 10.0  # MedInc is in $10k
PRICE_TO_K = 100.0  # MedHouseVal is in $100k
PRICE_CAP = 5.0  # the census caps house values at $500k
X_RANGE = (0.4, 12.5)  # MedInc units shown on the x-axis ($4k to $125k)
Y_RANGE_K = (0.0, 540.0)
GRID_POINTS = 400

DISPLAY_POINTS = 900  # training districts drawn as dots (the model trains on all of them)
RESIDUAL_POINTS = 36  # of those, how many get a residual line
POLY_SAMPLE = 30  # polynomial, ridge and lasso fit this many districts so over-fitting is visible
QUERY_TARGET_INCOME = 6.0  # the inference house: the uncapped test district nearest $60k income

DEFAULT_OUTPUT_DIR = Path("outputs") / "california_housing" / "lessons"
SUMMARY_FILE = "lessons_summary.csv"

SOURCE_NOTE = "Data: California Housing, 1990 US census (scikit-learn) · seed 42"
