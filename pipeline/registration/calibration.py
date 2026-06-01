"""
Checkerboard-based homography estimation from two PNG images.

* ``table.png``   — checkerboard at hand level
* ``plateau.png`` — checkerboard at pointing-board level

The homography maps *plateau* coordinates → *table* coordinates,
thus correcting the parallax between the two planes.
"""

import logging
from pathlib import Path

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

logger = logging.getLogger(__name__)


def _require_cv2():
    if not _HAS_CV2:
        raise ImportError(
            "OpenCV is required for checkerboard calibration.  "
            "Install it: pip install opencv-python"
        )


def find_checkerboard_corners(image_path, board_size=(9, 6)):
    """
    Detect internal corners of a checkerboard with sub-pixel accuracy.

    Parameters
    ----------
    image_path : Path
    board_size : tuple (cols, rows) of internal corners

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


def compute_homography_from_images(table_path, plateau_path, board_size=(9, 6)):
    """
    Compute the *plateau → table* homography.

    Returns
    -------
    dict
        Keys: ``homography_matrix``, ``camera_matrix`` (None),
        ``dist_coeffs`` (None), ``corners_table``, ``corners_plateau``,
        ``n_inliers``, ``n_total``, ``mean_reprojection_error``.
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

    return {
        "homography_matrix": H,
        "camera_matrix": None,
        "dist_coeffs": None,
        "corners_table": ct,
        "corners_plateau": cp,
        "n_inliers": n_inliers,
        "n_total": len(cp),
        "mean_reprojection_error": err,
    }