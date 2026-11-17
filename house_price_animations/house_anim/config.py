"""Central constants for the California Housing data splits, training schedules and price axes.

Rendering constants every dataset shares (frame rates, canvas sizes, panel layout) live in
animkit.config; the ones house modules use are re-exported here so they import a single config.

Units: the dataset stores each district's median house value in $100k (2.5 is $250k). The models
learn and predict in those units; the charts multiply by ``TARGET_TO_THOUSANDS`` to show $k.
"""

from pathlib import Path

from animkit.config import (
    ALGORITHM_PANEL_RECT,
    DASHBOARD_DPI_SCALE,
    DEFAULT_DPI,
    DEFAULT_FPS,
    DPI_RANGE,
    EQUATION_PANEL_RECT,
    FINAL_HOLD_SECONDS,
    FPS_RANGE,
    INFERENCE_ALGORITHM_RECT,
    INFERENCE_FIGSIZE,
    MAP_FIGSIZE,
    RACE_FIGSIZE,
    TRAINING_FIGSIZE,
    VIDEO_FORMATS,
)

__all__ = [
    "ALGORITHM_PANEL_RECT", "DASHBOARD_DPI_SCALE", "DEFAULT_DPI", "DEFAULT_FPS", "DEFAULT_INFERENCE_BATCH",
    "DEFAULT_MODEL", "DEFAULT_OUTPUT_DIR", "DPI_RANGE", "EQUATION_PANEL_RECT", "FINAL_HOLD_SECONDS",
    "FPS_RANGE", "GOOD_ENOUGH_K", "INFERENCE_ALGORITHM_RECT", "INFERENCE_FIGSIZE", "MAP_FIGSIZE", "MIN_ROWS",
    "PRICE_AXIS_MAX_K", "RACE_FIGSIZE", "RANDOM_STATE", "RESIDUAL_BIN_COUNT", "RESIDUAL_RANGE_K",
    "TARGET_COLUMN", "TARGET_TO_THOUSANDS", "TEST_FRACTION", "TRAINING_FIGSIZE", "TRAIN_EVAL_SAMPLE",
    "VAL_FRACTION", "VIDEO_FORMATS",
]

RANDOM_STATE = 42

# California Housing stores the target (median house value) in units of $100k.
TARGET_COLUMN = "MedHouseVal"
TARGET_TO_THOUSANDS = 100.0
PRICE_AXIS_MAX_K = 520.0

TEST_FRACTION = 0.2
VAL_FRACTION = 0.2
MIN_ROWS = 50

# Rows used to score training error each frame, so big ensembles stay responsive.
TRAIN_EVAL_SAMPLE = 3_000

DEFAULT_INFERENCE_BATCH = 40
DEFAULT_OUTPUT_DIR = Path("outputs") / "california_housing"
DEFAULT_MODEL = "random_forest"

# Residual histogram range and resolution, in $k.
RESIDUAL_RANGE_K = 300.0
RESIDUAL_BIN_COUNT = 40
# Tolerance used for the "predictions within ±$Xk" readout.
GOOD_ENOUGH_K = 50.0
