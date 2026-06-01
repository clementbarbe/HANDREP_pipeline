"""
Centralised default values and scientific constants.
"""

import numpy as np

# ── Finger / zone taxonomy ────────────────────────────────
FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]

FINGER_COLORS = {
    "thumb":  "#e41a1c",
    "index":  "#377eb8",
    "middle": "#4daf4a",
    "ring":   "#984ea3",
    "little": "#ff7f00",
}

ZONE_LABELS = {
    "Z1": "knuckle (base)",
    "Z2": "tip (bout)",
}

LANDMARK_ORDER = [
    (f, z) for f in FINGER_ORDER for z in ["Z1", "Z2"]
]

# ── Image geometry ────────────────────────────────────────
IMAGE_WIDTH  = 1920
IMAGE_HEIGHT = 1080
IMAGE_CENTER = np.array([IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2], dtype=float)

# ── Checkerboard ──────────────────────────────────────────
CHECKERBOARD_SIZE = (9, 6)   # internal corners (cols, rows)
SQUARE_SIZE_CM    = 2.4      # side length of one black square

# ── Mathematical correction fallback ──────────────────────
CAMERA_HEIGHT_CM     = 27
BOARD_ABOVE_HAND_CM  = 8
MATH_SCALE = CAMERA_HEIGHT_CM / (CAMERA_HEIGHT_CM + BOARD_ABOVE_HAND_CM)

# ── Neural network ────────────────────────────────────────
NN_IMG_SIZE     = 512
NN_HEATMAP_SIZE = 128
NN_CROP_SIZE    = 64

# ── Supported image extensions ────────────────────────────
IMAGE_GLOBS = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]


def default_config():
    """Return a mutable dict with all pipeline defaults."""
    return {
        "data_dir":          "data",
        "checkerboard":      True,
        "checkerboard_size": CHECKERBOARD_SIZE,
        "square_size_cm":    SQUARE_SIZE_CM,
        "force":             True,
        "subjects_filter":   None,
        "skip_figures":      False,
        "math_scale":        MATH_SCALE,
        "image_center":      IMAGE_CENTER,
        "image_width":       IMAGE_WIDTH,
        "image_height":      IMAGE_HEIGHT,
    }