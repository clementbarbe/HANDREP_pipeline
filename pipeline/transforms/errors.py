"""
Error computation, spatial normalisation, metric conversion.
"""

import numpy as np


def compute_errors(df):
    """Add ``error_x``, ``error_y``, ``error_eucl`` columns (pixels)."""
    out = df.copy()
    out["error_x"] = out["x_est_corr"] - out["x_real"]
    out["error_y"] = out["y_est_corr"] - out["y_real"]
    out["error_eucl"] = np.sqrt(out["error_x"] ** 2 + out["error_y"] ** 2)
    return out


def compute_reference_scale(df):
    """Diagonal of the bounding box of reference points (px)."""
    pts = df[["x_real", "y_real"]].dropna().to_numpy(dtype=float)
    if len(pts) < 2:
        return np.nan
    rx = pts[:, 0].max() - pts[:, 0].min()
    ry = pts[:, 1].max() - pts[:, 1].min()
    scale = np.sqrt(rx ** 2 + ry ** 2)
    return scale if scale > 0 else np.nan


def add_normalized_errors(df, scale):
    """Add ``error_*_norm`` and ``normalization_scale`` columns."""
    out = df.copy()
    if scale is None or np.isnan(scale) or scale == 0:
        out["error_eucl_norm"] = np.nan
        out["error_x_norm"] = np.nan
        out["error_y_norm"] = np.nan
    else:
        out["error_eucl_norm"] = out["error_eucl"] / scale
        out["error_x_norm"] = out["error_x"] / scale
        out["error_y_norm"] = out["error_y"] / scale
    out["normalization_scale"] = scale
    return out


def add_cm_errors(df, cm_per_px):
    """
    Add ``error_eucl_cm``, ``error_x_cm``, ``error_y_cm`` columns.

    If *cm_per_px* is None (no calibration), columns are filled with NaN.
    """
    out = df.copy()
    if cm_per_px is None or np.isnan(cm_per_px):
        out["error_eucl_cm"] = np.nan
        out["error_x_cm"] = np.nan
        out["error_y_cm"] = np.nan
    else:
        out["error_eucl_cm"] = out["error_eucl"] * cm_per_px
        out["error_x_cm"] = out["error_x"] * cm_per_px
        out["error_y_cm"] = out["error_y"] * cm_per_px
    out["cm_per_px"] = cm_per_px
    return out