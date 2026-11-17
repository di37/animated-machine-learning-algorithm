"""Canvas, resolutions and pacing every lesson video shares.

The canvas is always 25.6 × 14.4 inches; a resolution only changes the dpi, so text and layout look
the same at every size. ``CHAPTER_SECONDS`` sets each chapter's length at a duration scale of 1.
``FPS`` and ``CRF`` (H.264's quality setting: lower is better, and bigger) go to the encoder.
"""

# 25.6 x 14.4 in at 100 dpi = 2560x1440 (16:9). Text keeps its point size, so the larger canvas is extra
# room (most of it for the algorithm diagram); the other resolutions only change the dpi.
FIGSIZE = (25.6, 14.4)
RESOLUTIONS = {"1440p": 100.0, "1080p": 1920.0 / 25.6, "720p": 1280.0 / 25.6, "preview": 384.0 / 25.6}
DEFAULT_RESOLUTION = "1440p"
FPS = 30
CRF = 19

# Seconds per chapter; the whole lesson runs about 108 s.
CHAPTER_SECONDS = {
    "title": 5, "data": 9, "initial": 6, "training": 26, "final": 8, "inference": 13,
    "exam": 15, "surface": 12, "features": 8, "takeaway": 6,
}
DURATION_SCALE_RANGE = (0.02, 3.0)

FOOTER = "IR · Isham Rashik · AI Engineer · Making AI simplified for all"
HEADER = "Engineering AI with Clarity"  # top right of every frame, title and takeaway cards included
