"""Rendering constants every dataset's dashboards share: frame rates, canvas sizes and panel layout.

Sizes are in inches, and a video's pixel size is its figure size times its dpi. The model dashboards
draw on a 20.48 × 11.52 inch canvas and record at ``DASHBOARD_DPI_SCALE`` times the video dpi, so at
the default 100 dpi they come out at 2560 × 1440. Panel rectangles are ``(left, bottom, width,
height)`` in figure fractions.
"""

DEFAULT_FPS = 12
DEFAULT_DPI = 100
VIDEO_FORMATS = ("mp4", "gif")
FPS_RANGE = (1, 60)
DPI_RANGE = (20, 300)

# Smallest training subset for the "more data" (learning-curve) strategy.
MIN_GROWTH_ROWS = 20

LIVE_PAUSE_S = 0.001
FINAL_HOLD_SECONDS = 2.0

# Model dashboards record at 2560x1440: a 20.48x11.52 in canvas at 1.25x the video dpi, so
# text is 25% larger relative to the frame than a plain 1600x900 layout scaled up.
TRAINING_FIGSIZE = (20.48, 11.52)
INFERENCE_FIGSIZE = (20.48, 11.52)
DASHBOARD_DPI_SCALE = 1.25
# Panel rectangles in figure fractions: (left, bottom, width, height).
ALGORITHM_PANEL_RECT = (0.045, 0.08, 0.265, 0.70)
INFERENCE_ALGORITHM_RECT = (0.045, 0.08, 0.235, 0.69)
EQUATION_PANEL_RECT = (0.73, 0.06, 0.255, 0.735)
# The live window opens at most this wide; recordings still use the full size.
LIVE_MAX_WIDTH_PX = 1500

RACE_FIGSIZE = (16.0, 9.0)
MAP_FIGSIZE = (16.0, 9.0)
