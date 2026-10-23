"""Uncertainty for geometry of the mean map, conditional on fixed references."""

import numpy as np
import pandas as pd
from pipeline.analysis.uncertainty import landmark_mean_samples, percentile_interval


def add_morphometry_intervals(lengths, widths, shape, df, cm_per_px=None, n_boot=2000, seed=42, min_n=3):
    samples = landmark_mean_samples(df, n_boot, seed, min_n) if n_boot > 0 else {}
    lengths, widths = lengths.copy(), widths.copy()
    for table, kind in ((lengths, "length"), (widths, "width")):
        for idx, row in table.iterrows():
            if kind == "length":
                a, b = (row.finger, "Z1"), (row.finger, "Z2")
            else:
                f1, f2 = row.pair.split("-")
                a, b = (f1, "Z1"), (f2, "Z1")
            lo, hi = (np.nan, np.nan)
            if a in samples and b in samples:
                lo, hi = percentile_interval(np.linalg.norm(samples[b] - samples[a], axis=1))
            table.loc[idx, "ci_low"] = lo
            table.loc[idx, "ci_high"] = hi
            real = row[kind + "_real"]
            table.loc[idx, "pct_ci_low"] = 100 * (lo / real - 1) if real > 0 else np.nan
            table.loc[idx, "pct_ci_high"] = 100 * (hi / real - 1) if real > 0 else np.nan
            if cm_per_px is not None:
                table.loc[idx, "ci_low_cm"] = lo * cm_per_px
                table.loc[idx, "ci_high_cm"] = hi * cm_per_px
        table["ci_scope"] = "within_session_conditional_on_fixed_reference"
    required = [("index", "Z1"), ("little", "Z1"), ("middle", "Z1"), ("middle", "Z2")]
    if shape is not None:
        shape = dict(shape)
        shape["estimated_ci_low"], shape["estimated_ci_high"] = np.nan, np.nan
        if all(k in samples for k in required):
            a, b, c, d = (samples[k] for k in required)
            width = np.linalg.norm(a - b, axis=1)
            length = np.linalg.norm(c - d, axis=1)
            ratios = np.divide(100 * width, length, out=np.full_like(width, np.nan), where=length > 0)
            shape["estimated_ci_low"], shape["estimated_ci_high"] = percentile_interval(ratios)
    return lengths, widths, shape


def hand_width(df, real, estimated, cm_per_px=None):
    a, b = ("index", "Z1"), ("little", "Z1")
    if not all(k in real and k in estimated for k in (a, b)):
        return pd.DataFrame()
    wr = np.linalg.norm(np.asarray(real[a]) - real[b])
    we = np.linalg.norm(np.asarray(estimated[a]) - estimated[b])
    row = dict(pair="index-little", width_real=wr, width_est=we,
               pct_overestimation=100 * (we / wr - 1) if wr > 0 else np.nan)
    if cm_per_px is not None:
        row.update(width_real_cm=wr * cm_per_px, width_est_cm=we * cm_per_px)
    return pd.DataFrame([row])
