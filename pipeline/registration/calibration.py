"""
Checkerboard-based homography + metric scale (px ↔ cm).

Uses two PNG images:
* ``table.png``   — checkerboard at hand level
* ``plateau.png`` — checkerboard at pointing-board level

The homography maps *plateau* → *table*.
The metric scale is derived from the **table** corners (hand plane).
"""

import logging
from pathlib import Path

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from pipeline.config.settings import SQUARE_SIZE_CM

logger = logging.getLogger(__name__)


def _require_cv2():
    if not _HAS_CV2:
        raise ImportError(
            "OpenCV is required for checkerboard calibration.  "
            "Install it:  pip install opencv-python"
        )


# ── Corner detection ──────────────────────────────────────

def find_checkerboard_corners(image_path, board_size=(9, 6)):
    """
    Detect internal corners with sub-pixel accuracy.

    Returns
    -------
    corners : ndarray (N, 2)
    """
    _require_cv2()
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not ret:
        flipped = (board_size[1], board_size[0])
        ret, corners = cv2.findChessboardCorners(gray, flipped, flags)
    if not ret:
        raise ValueError(
            f"Checkerboard {board_size} not found in {image_path}"
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


# ── Metric scale ──────────────────────────────────────────

def compute_px_per_cm(corners, square_size_cm=SQUARE_SIZE_CM):
    """
    Estimate the pixel-to-centimetre ratio from checkerboard corners.

    Strategy: for each corner, find its nearest neighbour (which lies
    one square away). The median of these distances, divided by the
    known square size, gives **px_per_cm**.

    Parameters
    ----------
    corners : ndarray (N, 2)
        Detected corners in pixel coordinates.
    square_size_cm : float
        Physical side length of one checkerboard square.

    Returns
    -------
    float
        Pixels per centimetre.
    """
    n = len(corners)
    # Pairwise distances (N, N)
    diffs = corners[:, np.newaxis, :] - corners[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)

    # Nearest-neighbour distance for each corner
    nn_dists = dists.min(axis=1)

    # Median is robust to boundary effects
    median_nn_px = float(np.median(nn_dists))
    px_per_cm = median_nn_px / square_size_cm

    logger.info(
        f"Metric scale: median NN = {median_nn_px:.1f} px → "
        f"{px_per_cm:.2f} px/cm  ({1.0 / px_per_cm:.4f} cm/px)"
    )
    return px_per_cm


# ── Homography from images ────────────────────────────────

def compute_homography_from_images(
    table_path, plateau_path, board_size=(9, 6), square_size_cm=SQUARE_SIZE_CM,
):
    """
    Compute the *plateau → table* homography **and** the metric scale.

    Returns
    -------
    dict
        ``homography_matrix``, ``camera_matrix`` (None),
        ``dist_coeffs`` (None), ``corners_table``, ``corners_plateau``,
        ``n_inliers``, ``n_total``, ``mean_reprojection_error``,
        ``px_per_cm``, ``cm_per_px``.
    """
    _require_cv2()

    logger.info("Detecting checkerboard in %s …", Path(table_path).name)
    ct = find_checkerboard_corners(table_path, board_size)
    logger.info("  → %d corners", len(ct))

    logger.info("Detecting checkerboard in %s …", Path(plateau_path).name)
    cp = find_checkerboard_corners(plateau_path, board_size)
    logger.info("  → %d corners", len(cp))

    if len(ct) != len(cp):
        raise ValueError(
            f"Corner count mismatch: table={len(ct)}, plateau={len(cp)}"
        )

    # Homography (RANSAC)
    H, mask = cv2.findHomography(
        cp.astype(np.float64), ct.astype(np.float64), cv2.RANSAC, 5.0,
    )
    if H is None:
        raise ValueError("Homography estimation failed")

    n_inliers = int(mask.sum()) if mask is not None else len(cp)

    # Reprojection error
    from pipeline.registration.correction import apply_homography
    reproj = apply_homography(cp, H)
    err = float(np.linalg.norm(reproj - ct, axis=1).mean())

    # Metric scale from table-level corners (= hand plane)
    px_cm = compute_px_per_cm(ct, square_size_cm)

    return {
        "homography_matrix":        H,
        "camera_matrix":            None,
        "dist_coeffs":              None,
        "corners_table":            ct,
        "corners_plateau":          cp,
        "n_inliers":                n_inliers,
        "n_total":                  len(cp),
        "mean_reprojection_error":  err,
        "px_per_cm":                px_cm,
        "cm_per_px":                1.0 / px_cm,
    }