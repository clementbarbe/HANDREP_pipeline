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
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("Homography projects points to infinity")
    result = t[:, :2] / w
    if not np.isfinite(result).all():
        raise ValueError("Homography produced nonfinite coordinates")
    return result


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
    out["x_real_image"] = out["x_real"]
    out["y_real_image"] = out["y_real"]

    use_cb = cfg.get("checkerboard", True) and calib is not None
    method = "mathematical_fallback"

    if use_cb:
        finite = np.isfinite(pts).all(axis=1)
        if "qc_valid" in out:
            finite &= out.qc_valid.to_numpy(bool)
        table_points = np.full_like(pts, np.nan)
        table_points[finite] = calibrate_points(pts[finite], calib)
        out[["x_est_table_px", "y_est_table_px"]] = table_points
        rect = calib["table_rectification_matrix"]
        corrected = np.full_like(pts, np.nan)
        corrected[finite] = apply_homography(table_points[finite], rect)
        ref = undistort_points(out[["x_real", "y_real"]].to_numpy(float),
                               calib.get("camera_matrix"), calib.get("dist_coeffs"))
        out[["x_real", "y_real"]] = apply_homography(ref, rect)
        # Raw pointing responses expressed in the same rectified frame, without
        # the correction between planes; useful for an interpretable sensitivity plot.
        raw_plane = undistort_points(pts[finite], calib.get("camera_matrix"), calib.get("dist_coeffs"))
        raw_rect = np.full_like(pts, np.nan)
        raw_rect[finite] = apply_homography(raw_plane, rect)
        out[["x_est_uncorrected_frame", "y_est_uncorrected_frame"]] = raw_rect
        method = "checkerboard_metric_rectification"
        out["coordinate_space"] = "rectified_hand_plane_px"
    else:
        if not cfg.get("allow_fallback", False):
            raise ValueError("No valid calibration; use --allow-fallback explicitly for exploratory pixel-only results")
        corrected = mathematical_plane_correction(
            pts,
            center=cfg.get("image_center"),
            scale=cfg.get("math_scale"),
        )
        out["coordinate_space"] = "image_px_approximate_plane_correction"
        out[["x_est_uncorrected_frame", "y_est_uncorrected_frame"]] = pts

    out["x_est_corr"] = corrected[:, 0]
    out["y_est_corr"] = corrected[:, 1]
    out["correction_method"] = method
    if "qc_valid" in out:
        out.loc[~out.qc_valid, ["x_est_corr", "y_est_corr"]] = np.nan
    return out
