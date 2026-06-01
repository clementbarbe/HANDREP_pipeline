"""
Data loading with schema validation.
"""

import pandas as pd
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
    df = pd.read_csv(path)
    _check_columns(df, _PRED_COLUMNS, path.name)
    df = df.copy()
    df["subject"] = df["subject"].apply(normalize_subject)
    df["session"] = df["session"].apply(normalize_session)
    return df


def load_reference(path):
    """Load and normalise a reference CSV."""
    path = Path(path)
    df = pd.read_csv(path)
    _check_columns(df, _REF_COLUMNS, path.name)
    df = df.copy()
    df["subject"] = df["subject"].apply(normalize_subject)
    df["session"] = df["session"].apply(normalize_session)
    return df