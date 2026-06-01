"""
Descriptive statistics and position extraction.
"""

import pandas as pd
import numpy as np
from pipeline.config.settings import FINGER_ORDER


def compute_summary(df):
    """Group-level summary per (subject, session, finger, zone)."""
    agg = dict(
        mean_error=("error_eucl", "mean"),
        std_error=("error_eucl", "std"),
        mean_error_x=("error_x", "mean"),
        mean_error_y=("error_y", "mean"),
        mean_error_norm=("error_eucl_norm", "mean"),
        std_error_norm=("error_eucl_norm", "std"),
        n_trials=("error_eucl", "count"),
    )
    if "error_eucl_cm" in df.columns and df["error_eucl_cm"].notna().any():
        agg["mean_error_cm"] = ("error_eucl_cm", "mean")
        agg["std_error_cm"]  = ("error_eucl_cm", "std")

    summary = (
        df.groupby(["subject", "session", "finger", "zone"], dropna=False)
        .agg(**agg)
        .reset_index()
    )
    summary["finger"] = pd.Categorical(
        summary["finger"], categories=FINGER_ORDER, ordered=True,
    )
    return summary.sort_values(
        ["subject", "session", "finger", "zone"]
    ).reset_index(drop=True)


def compute_per_landmark_stats(df):
    """Error statistics per (finger, zone)."""
    agg = dict(
        mean_error=("error_eucl", "mean"),
        std_error=("error_eucl", "std"),
        mean_error_x=("error_x", "mean"),
        mean_error_y=("error_y", "mean"),
        n_trials=("error_eucl", "count"),
    )
    if "error_eucl_cm" in df.columns and df["error_eucl_cm"].notna().any():
        agg["mean_error_cm"] = ("error_eucl_cm", "mean")
        agg["std_error_cm"]  = ("error_eucl_cm", "std")

    return (
        df.groupby(["finger", "zone"]).agg(**agg).reset_index()
    )


def compute_per_finger_stats(df):
    """Error statistics per finger."""
    agg = dict(
        mean_error=("error_eucl", "mean"),
        std_error=("error_eucl", "std"),
        n=("error_eucl", "count"),
    )
    if "error_eucl_cm" in df.columns and df["error_eucl_cm"].notna().any():
        agg["mean_error_cm"] = ("error_eucl_cm", "mean")
        agg["std_error_cm"]  = ("error_eucl_cm", "std")

    return df.groupby("finger").agg(**agg).reindex(FINGER_ORDER)


def extract_landmark_positions(df):
    """
    Compute mean positions for each landmark.

    Returns
    -------
    dict  ``{'real', 'estimated_corr', 'estimated_raw'}``
        each mapping ``(finger, zone) → (x, y)``.
    """
    real, est_corr, est_raw = {}, {}, {}
    for (finger, zone), grp in df.groupby(["finger", "zone"]):
        if grp["x_real"].notna().any():
            real[(finger, zone)] = (grp["x_real"].mean(), grp["y_real"].mean())
        est_corr[(finger, zone)] = (grp["x_est_corr"].mean(), grp["y_est_corr"].mean())
        est_raw[(finger, zone)] = (grp["x_est"].mean(), grp["y_est"].mean())
    return {"real": real, "estimated_corr": est_corr, "estimated_raw": est_raw}