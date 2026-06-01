"""
Identifier normalisation helpers.

Ensure consistent subject / session labels across heterogeneous inputs.
"""

import re
import numpy as np
import pandas as pd


def normalize_subject(value):
    """``'sub-S1'``, ``'s1'``, ``'1'`` → ``'S1'``."""
    if pd.isna(value):
        return np.nan
    m = re.search(r"(\d+)", str(value).strip())
    return f"S{int(m.group(1))}" if m else str(value).strip()


def normalize_session(value):
    """``'ses-1'``, ``'1'``, ``'session 2'`` → ``'ses-1'``."""
    if pd.isna(value):
        return np.nan
    m = re.search(r"(\d+)", str(value).strip().lower())
    return f"ses-{int(m.group(1))}" if m else str(value).strip()