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
    corners = np.asarray(corners, dtype=float)
    if square_size_cm <= 0 or not np.isfinite(square_size_cm):
        raise ValueError("Checkerboard square size must be positive and finite")
    if len(corners) < 4 or not np.isfinite(corners).all():
        raise ValueError("At least four finite calibration corners are required")
    n = len(corners)
    # Pairwise distances (N, N)
    diffs = corners[:, np.newaxis, :] - corners[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)

    # Nearest-neighbour distance for each corner
    nn_dists = dists.min(axis=1)

    # Median is robust to boundary effects
    median_nn_px = float(np.median(nn_dists))
    if median_nn_px <= 0:
        raise ValueError("Duplicate calibration corners")
    px_per_cm = median_nn_px / square_size_cm

    logger.info(
        f"Metric scale: median NN = {median_nn_px:.1f} px → "
        f"{px_per_cm:.2f} px/cm  ({1.0 / px_per_cm:.4f} cm/px)"
    )
    return px_per_cm


# ── Homography from images ────────────────────────────────

def compute_homography_from_images(
    table_path, plateau_path, board_size=(9, 6), square_size_cm=SQUARE_SIZE_CM,
    max_error_px=2.0, min_inlier_ratio=0.9, camera_path=None,
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

    # Lens parameters are optional, never inferred from only these two views.
    camera_matrix = dist_coeffs = None
    if camera_path is not None and Path(camera_path).is_file():
        with np.load(camera_path, allow_pickle=False) as camera:
            camera_matrix = camera["camera_matrix"]
            dist_coeffs = camera["dist_coeffs"]
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all() or not np.isfinite(dist_coeffs).all():
            raise ValueError("Invalid camera calibration parameters")
    from pipeline.registration.correction import undistort_points
    ct_raw, cp_raw = ct.copy(), cp.copy()
    ct = undistort_points(ct, camera_matrix, dist_coeffs).astype(float)
    cp = undistort_points(cp, camera_matrix, dist_coeffs).astype(float)

    table_image = cv2.imread(str(table_path))
    plateau_image = cv2.imread(str(plateau_path))
    if table_image.shape[:2] != plateau_image.shape[:2]:
        raise ValueError("Calibration images must have the same resolution")

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
    # A deterministic held-out subset detects poor fits beyond training residuals.
    holdout = np.arange(len(cp)) % 5 == 0
    H_train, _ = cv2.findHomography(cp[~holdout], ct[~holdout], 0)
    if H_train is None:
        raise ValueError("Held-out calibration fit failed")
    heldout_error = float(np.linalg.norm(apply_homography(cp[holdout], H_train) - ct[holdout], axis=1).mean())
    if n_inliers / len(cp) < min_inlier_ratio or max(err, heldout_error) > max_error_px:
        raise ValueError(f"Calibration rejected: fit {err:.2f}px, held-out {heldout_error:.2f}px, inliers {n_inliers}/{len(cp)}")

    # Metric scale from table-level corners (= hand plane)
    px_cm = compute_px_per_cm(ct, square_size_cm)
    # The nearest-neighbour scale is ONLY a display-unit choice. Physical metric
    # geometry comes from mapping every corner to its known 2-D grid position.
    cols, rows = board_size
    grid = np.array([(x, y) for y in range(rows) for x in range(cols)], dtype=float)
    if len(grid) != len(ct):
        raise ValueError("Checkerboard corner/grid count mismatch")
    canonical = grid * square_size_cm * px_cm + ct.min(axis=0)
    rectification, _ = cv2.findHomography(ct, canonical, 0)
    rect_train, _ = cv2.findHomography(ct[~holdout], canonical[~holdout], 0)
    if rectification is None or rect_train is None:
        raise ValueError("Metric rectification failed")
    metric_error_cm = float(np.linalg.norm(apply_homography(ct[holdout], rect_train) - canonical[holdout], axis=1).mean() / px_cm)
    if metric_error_cm * px_cm > max_error_px:
        raise ValueError(f"Metric rectification rejected: held-out error {metric_error_cm:.4f}cm")

    return {
        "homography_matrix":        H,
        "camera_matrix":            camera_matrix,
        "dist_coeffs":              dist_coeffs,
        "table_rectification_matrix": rectification,
        "plateau_rectification_matrix": rectification @ H,
        "corners_table_raw":        ct_raw,
        "corners_plateau_raw":      cp_raw,
        "corners_table":            ct,
        "corners_plateau":          cp,
        "n_inliers":                n_inliers,
        "n_total":                  len(cp),
        "mean_reprojection_error":  err,
        "heldout_error_px":         heldout_error,
        "heldout_metric_error_cm":  metric_error_cm,
        "image_width":              int(table_image.shape[1]),
        "image_height":             int(table_image.shape[0]),
        "coordinate_space":         "rectified_hand_plane_px",
        "lens_distortion_corrected": camera_matrix is not None,
        "validation_scope": "held_out_corners_same_images; independent physical validation still required",
        "px_per_cm":                px_cm,
        "cm_per_px":                1.0 / px_cm,
    }
