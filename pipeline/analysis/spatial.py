"""Signed anatomical errors and descriptive separation of bias and precision."""

import numpy as np
import pandas as pd
from pipeline.analysis.tps import procrustes_align


def add_anatomical_coordinates(df):
    out = df.copy()
    means = out.groupby(["finger", "zone"])[["x_real", "y_real"]].mean()
    required = [("middle", "Z1"), ("middle", "Z2"), ("index", "Z1"), ("little", "Z1")]
    if any(k not in means.index for k in required):
        raise ValueError("Anatomical axes require middle Z1/Z2, index Z1 and little Z1")
    distal = means.loc["middle", "Z2"].to_numpy() - means.loc["middle", "Z1"].to_numpy()
    length = np.linalg.norm(distal)
    if length <= 1e-9:
        raise ValueError("Degenerate middle finger reference")
    distal /= length
    radial = means.loc["index", "Z1"].to_numpy() - means.loc["little", "Z1"].to_numpy()
    radial -= np.dot(radial, distal) * distal
    width = np.linalg.norm(radial)
    if width <= 1e-9:
        raise ValueError("Degenerate reference hand width")
    radial /= width
    origin = means.xs("Z1", level="zone").mean().to_numpy()
    basis = np.column_stack([radial, distal])
    for source, prefix in ((["x_real", "y_real"], "target"), (["x_est_corr", "y_est_corr"], "response")):
        points = (out[source].to_numpy(float) - origin) @ basis
        out[prefix + "_ru"], out[prefix + "_pd"] = points[:, 0], points[:, 1]
    out["error_ru"] = out.response_ru - out.target_ru
    out["error_pd"] = out.response_pd - out.target_pd
    cm = out.cm_per_px.iloc[0] if "cm_per_px" in out else None
    if cm is not None and pd.notna(cm):
        for col in ("error_ru", "error_pd", "target_ru", "target_pd", "response_ru", "response_pd"):
            out[col + "_cm"] = out[col] * cm
    return out, dict(origin=origin.tolist(), radial=radial.tolist(), distal=distal.tolist(),
                     convention="positive RU towards thumb; positive PD towards fingertips")


def bias_precision_summary(df):
    rows = []
    for (finger, zone), group in df.groupby(["finger", "zone"]):
        errors = group[["error_ru", "error_pd"]].dropna().to_numpy(float)
        if not len(errors):
            continue
        mean = errors.mean(0)
        centered = errors - mean
        row = dict(finger=finger, zone=zone, n_trials=len(errors),
                   bias_ru=mean[0], bias_pd=mean[1], bias_magnitude=np.linalg.norm(mean),
                   rms_error=np.sqrt(np.mean(np.sum(errors**2, axis=1))),
                   variable_error=np.sqrt(np.mean(np.sum(centered**2, axis=1))),
                   median_error=np.median(np.linalg.norm(errors, axis=1)))
        # A dispersion ellipse, not a confidence ellipse of the mean.
        cov = np.cov(errors, rowvar=False, ddof=1) if len(errors) > 1 else np.full((2, 2), np.nan)
        row.update(sd_ru=np.sqrt(cov[0, 0]), sd_pd=np.sqrt(cov[1, 1]), cov_ru_pd=cov[0, 1])
        if len(errors) > 1:
            values, vectors = np.linalg.eigh(cov)
            row.update(ellipse_major_sd=np.sqrt(max(values[1], 0)), ellipse_minor_sd=np.sqrt(max(values[0], 0)),
                       ellipse_angle_degrees=np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])))
        cm = group.cm_per_px.iloc[0]
        if cm is not None and pd.notna(cm):
            for key in ("bias_ru", "bias_pd", "bias_magnitude", "rms_error", "variable_error", "median_error", "sd_ru", "sd_pd"):
                row[key + "_cm"] = row[key] * cm
        rows.append(row)
    return pd.DataFrame(rows)


def decompose_geometry(real, estimated):
    """Separate centroid shift, rigid rotation, uniform size and affine residual.

All landmarks are equally weighted. This is descriptive geometry, without
claims about neural mechanisms or independent landmark observations.
"""
    valid = [k for k in real if k in estimated and np.isfinite([real[k], estimated[k]]).all()]
    if len(valid) < 3:
        return pd.DataFrame()
    src = np.array([real[k] for k in valid])
    tgt = np.array([estimated[k] for k in valid])
    sc, aligned, rotation = procrustes_align(src, tgt)
    tc = tgt - tgt.mean(0)
    if np.linalg.matrix_rank(sc) < 2 or np.sum(sc**2) == 0:
        return pd.DataFrame()
    size = np.sum(sc * aligned) / np.sum(sc**2)
    affine = np.linalg.lstsq(sc, tc, rcond=None)[0]
    stretches = np.linalg.svd(affine, compute_uv=False)
    shift = tgt.mean(0) - src.mean(0)
    row = dict(n_landmarks=len(valid), translation_x=shift[0], translation_y=shift[1],
               translation_magnitude=np.linalg.norm(shift),
               rotation_degrees=np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])),
               uniform_scale=size, affine_stretch_max=stretches.max(), affine_stretch_min=stretches.min(),
               affine_determinant=np.linalg.det(affine),
               rigid_residual_rms=np.sqrt(np.mean(np.sum((aligned - sc)**2, axis=1))),
               similarity_residual_rms=np.sqrt(np.mean(np.sum((aligned - size * sc)**2, axis=1))),
               affine_residual_rms=np.sqrt(np.mean(np.sum((sc @ affine - tc)**2, axis=1))))
    return pd.DataFrame([row])
