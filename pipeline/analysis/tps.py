"""
Thin-Plate Spline (TPS) fitting and Procrustes alignment.

Used to build the proprioceptive deformation map.
"""

import numpy as np
from pipeline.config.settings import FINGER_ORDER, LANDMARK_ORDER


def tps_kernel(r):
    """TPS radial basis function: r² ln(r²)."""
    out = np.zeros_like(r)
    mask = r > 0
    out[mask] = r[mask] ** 2 * np.log(r[mask] ** 2)
    return out


def fit_tps(source, target, lambda_reg=1e-3):
    """
    Fit a 2-D thin-plate spline.

    Parameters
    ----------
    source, target : ndarray (N, 2)
    lambda_reg : float
        Regularisation strength.

    Returns
    -------
    callable
        ``transform(points)`` maps (M, 2) → (M, 2).
    """
    n = source.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = np.linalg.norm(source[i] - source[j])
            K[i, j] = tps_kernel(np.array([r]))[0]
    K += lambda_reg * np.eye(n)

    P = np.hstack([np.ones((n, 1)), source])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T

    Y = np.zeros((n + 3, 2))
    Y[:n, :] = target
    params = np.linalg.solve(L, Y)
    weights, affine = params[:n], params[n:]

    def transform(points):
        m = points.shape[0]
        K_new = np.zeros((m, n))
        for j in range(n):
            K_new[:, j] = tps_kernel(np.linalg.norm(points - source[j], axis=1))
        P_new = np.hstack([np.ones((m, 1)), points])
        return K_new @ weights + P_new @ affine

    return transform


def procrustes_align(source, target):
    """
    Align *target* onto *source* using Procrustes (rotation only,
    no scaling).

    Returns
    -------
    source_centered, target_aligned, rotation_matrix
    """
    sc = source.mean(axis=0)
    tc = target.mean(axis=0)
    src = source - sc
    tgt = target - tc
    U, _, Vt = np.linalg.svd(tgt.T @ src)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return src, tgt @ R.T, R


def compute_tps_analysis(real_positions, est_positions):
    """
    Run Procrustes + TPS on available landmarks.

    Returns
    -------
    dict or None
        Keys: ``source_proc``, ``target_proc``, ``transform``,
        ``valid_landmarks``, ``grid_bounds``.
    """
    valid = [lm for lm in LANDMARK_ORDER
             if lm in real_positions and lm in est_positions]

    if len(valid) < 5:
        return None

    src = np.array([real_positions[lm] for lm in valid], dtype=float)
    tgt = np.array([est_positions[lm] for lm in valid], dtype=float)

    src_p, tgt_p, _ = procrustes_align(src, tgt)
    transform = fit_tps(src_p, tgt_p, lambda_reg=1e-3)

    margin = 50
    bounds = {
        "x_min": src_p[:, 0].min() - margin,
        "x_max": src_p[:, 0].max() + margin,
        "y_min": src_p[:, 1].min() - margin,
        "y_max": src_p[:, 1].max() + margin,
    }

    return {
        "source_proc": src_p,
        "target_proc": tgt_p,
        "transform": transform,
        "valid_landmarks": valid,
        "grid_bounds": bounds,
    }