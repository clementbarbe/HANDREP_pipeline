"""
Data loading with schema validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.transforms.normalization import normalize_subject, normalize_session


_PRED_COLUMNS = [
    "subject", "session", "trial", "finger", "zone",
    "x_est", "y_est", "filename",
]

_REF_COLUMNS = [
    "subject", "session", "finger", "zone", "landmark",
    "x_real", "y_real", "image",
]


def _check_columns(df, required, source):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def load_predictions(path):
    """Load and normalise a prediction CSV."""
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    _check_columns(df, _PRED_COLUMNS, path.name)
    df = df.copy()
    df["subject"] = df["subject"].apply(normalize_subject)
    df["session"] = df["session"].apply(normalize_session)
    return df


def load_reference(path):
    """Load and normalise a reference CSV."""
    path = Path(path)
    df = pd.read_csv(path, sep=None, engine="python")
    _check_columns(df, _REF_COLUMNS, path.name)
    df = df.copy()
    df["subject"] = df["subject"].apply(normalize_subject)
    df["session"] = df["session"].apply(normalize_session)
    for c in ("x_real", "y_real"):
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df


def load_trial_design(pred_df, design_path):
    """Optional acquisition metadata. Missing/partial designs are rejected."""
    if not Path(design_path).is_file():
        return pred_df
    design = pd.read_csv(design_path, sep=None, engine="python", dtype={"pair_id": str})
    _check_columns(design, ["trial", "condition", "pair_id"], str(design_path))
    design["trial"] = pd.to_numeric(design["trial"], errors="raise")
    if not np.isfinite(design.trial).all() or (design.trial < 1).any() or (design.trial % 1 != 0).any():
        raise ValueError("Design trial numbers must be positive integers")
    if design["trial"].duplicated().any():
        raise ValueError("Duplicate trials in acquisition design")
    if not design["condition"].isin(["coupled", "uncoupled"]).all():
        raise ValueError("Design condition must be coupled or uncoupled")
    if not set(pred_df["trial"].astype(int)).issubset(set(design["trial"])):
        raise ValueError("Acquisition design does not cover all prediction trials")
    if {"condition", "pair_id"}.intersection(pred_df.columns):
        raise ValueError("Condition metadata supplied both inline and in design file")
    columns = ["trial", "condition", "pair_id"] + (["block"] if "block" in design else [])
    if "block" in design and "block" in pred_df:
        raise ValueError("Block metadata supplied both inline and in design file")
    return pred_df.merge(design[columns], on="trial", validate="one_to_one")
