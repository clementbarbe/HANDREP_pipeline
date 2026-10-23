"""
Schema and consistency checks on loaded DataFrames.
"""

import numpy as np
import pandas as pd
from pipeline.config.settings import FINGER_ORDER


def validate_subject_session(pred_df, ref_df, subject, session):
    """
    Assert that both DataFrames contain exactly the expected
    *subject* and *session* identifiers.

    Raises
    ------
    ValueError
        On any mismatch.
    """
    for label, df in [("Predictions", pred_df), ("Reference", ref_df)]:
        if df.empty:
            raise ValueError(f"{label} is empty")
        if df[["subject", "session", "finger", "zone"]].isna().any().any():
            raise ValueError(f"{label} contains missing identifiers")
        if not df["finger"].isin(FINGER_ORDER).all() or not df["zone"].isin(["Z1", "Z2"]).all():
            raise ValueError(f"{label} contains unknown fingers or zones")
        subjs = set(df["subject"].dropna().astype(str).unique())
        sesses = set(df["session"].dropna().astype(str).unique())
        if subjs != {subject}:
            raise ValueError(
                f"{label} subject mismatch: expected {{{subject}}}, got {subjs}"
            )
        if sesses != {session}:
            raise ValueError(
                f"{label} session mismatch: expected {{{session}}}, got {sesses}"
            )
    trials = pd.to_numeric(pred_df["trial"], errors="coerce")
    if not np.isfinite(trials).all() or (trials < 1).any() or (trials % 1 != 0).any():
        raise ValueError("Trial numbers must be positive integers")
    if trials.duplicated().any() or pred_df["filename"].duplicated().any():
        raise ValueError("Duplicate prediction trials or filenames")
    ref_keys = ["subject", "session", "image", "finger", "zone"]
    if ref_df.duplicated(ref_keys).any():
        raise ValueError("Duplicate reference landmarks within an image")
    for c in ("x_real", "y_real"):
        if not np.isfinite(pd.to_numeric(ref_df[c], errors="coerce")).all():
            raise ValueError("Reference coordinates must be numeric and finite")


def flag_prediction_quality(df, width, height):
    """Keep every trial and mark unusable coordinates; never silently trim outliers."""
    out = df.copy()
    for c in ("x_est", "y_est"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    finite = np.isfinite(out[["x_est", "y_est"]]).all(axis=1)
    bounds = out["x_est"].between(0, width - 1) & out["y_est"].between(0, height - 1)
    out["qc_valid"] = finite & bounds
    out["qc_reason"] = np.where(~finite, "nonfinite_prediction", np.where(~bounds, "outside_image", ""))
    return out
