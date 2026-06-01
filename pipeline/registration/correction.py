"""
Geometric correction applied to estimated landmark positions.

Two strategies:
1. Checkerboard homography (preferred)
2. Mathematical perspective scaling (fallback)
"""

import logging
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from pipeline.config.settings import IMAGE_CENTER, MATH_SCALE

logger = logging.getLogger(__name__)


# ── Low-level transforms ──────────────────────────────────

def undistort_points(points, camera_matrix=None, dist_coeffs=None):
    """Remove lens distortion. Pass-through if params are *None*."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0 or camera_matrix is None or dist_coeffs is None:
        return points.copy()
    if not _HAS_CV2:
        return points.copy()
    pts = points.reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, cameraMatrix=camera_matrix,
                              distCoeffs=dist_coeffs, P=camera_matrix)
    return out.reshape(-1, 2)


def apply_homography(points, H=None):
    """Apply a 3×3 homography to an (N, 2) array of points."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0 or H is None:
        return points.copy()
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    ph = np.hstack([points, ones])
    t = (H @ ph.T).T
    w = t[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return t[:, :2] / w


def calibrate_points(points, calib):
    """Undistort then apply homography."""
    pts = np.asarray(points, dtype=np.float64)
    pts = undistort_points(pts,
                           calib.get("camera_matrix"),
                           calib.get("dist_coeffs"))
    return apply_homography(pts, calib.get("homography_matrix"))


def mathematical_plane_correction(points, center=None, scale=None):
    """
    Simple radial scaling around image centre:
    ``corrected = centre + (point - centre) * scale``
    """
    center = center if center is not None else IMAGE_CENTER
    scale = scale if scale is not None else MATH_SCALE
    points = np.asarray(points, dtype=np.float64)
    return center + (points - center) * scale


# ── DataFrame-level interface ─────────────────────────────

def apply_geometry_correction(df, calib=None, cfg=None):
    """
    Add ``x_est_corr``, ``y_est_corr``, ``correction_method`` columns.

    Parameters
    ----------
    df : DataFrame
        Must contain ``x_est`` and ``y_est``.
    calib : dict or None
        Calibration dict with ``homography_matrix``.
    cfg : dict or None
        Pipeline config (``checkerboard``, ``math_scale``, ``image_center``).
    """
    cfg = cfg or {}
    out = df.copy()
    pts = out[["x_est", "y_est"]].to_numpy(dtype=float)

    use_cb = cfg.get("checkerboard", True) and calib is not None
    method = "mathematical_fallback"

    if use_cb:
        try:
            corrected = calibrate_points(pts, calib)
            method = "checkerboard_homography"
        except Exception as exc:
            logger.warning("Calibration failed (%s), falling back", exc)
            corrected = mathematical_plane_correction(
                pts,
                center=cfg.get("image_center"),
                scale=cfg.get("math_scale"),
            )
            method = "mathematical_fallback_after_error"
    else:
        corrected = mathematical_plane_correction(
            pts,
            center=cfg.get("image_center"),
            scale=cfg.get("math_scale"),
        )

    out["x_est_corr"] = corrected[:, 0]
    out["y_est_corr"] = corrected[:, 1]
    out["correction_method"] = method
    return out