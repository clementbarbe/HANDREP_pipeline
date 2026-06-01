"""
Low-level CSV writers for intermediate and final outputs.
"""

from pathlib import Path


_TRIAL_COLUMNS = [
    "filename", "subject", "session", "trial", "finger", "zone",
    "x_est", "y_est", "x_est_corr", "y_est_corr",
    "x_real", "y_real",
    "error_x", "error_y", "error_eucl",
    "error_x_norm", "error_y_norm", "error_eucl_norm",
    "normalization_scale", "correction_method",
]


def save_trial_csv(df, path):
    """Save per-trial results with a canonical column order."""
    cols = [c for c in _TRIAL_COLUMNS if c in df.columns]
    df[cols].to_csv(Path(path), index=False)


def save_summary_csv(df, path):
    df.to_csv(Path(path), index=False)


def save_biomech_csv(df, path):
    df.to_csv(Path(path), index=False)