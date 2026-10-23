"""Aggregate condition contrasts with participants as the independent units."""

import numpy as np
import pandas as pd
from pipeline.analysis.uncertainty import percentile_interval


def summarize_cohort(comparisons, n_boot=2000, seed=42):
    """Average repeated sessions within subject before resampling subjects.

Never pool calibration methods or recorded/inferred coupling definitions.
Small local datasets remain descriptive; there are no trial-level p-values.
"""
    if not comparisons:
        return pd.DataFrame(columns=["finger", "n_subjects", "mean_difference_pct_points", "ci_low", "ci_high", "inference"])
    data = pd.concat(comparisons, ignore_index=True)
    keys = ["finger", "coupling_source", "correction_method"]
    subjects = data.groupby(keys + ["subject"])["difference_pct_points"].mean().reset_index()
    rows = []
    rng = np.random.default_rng(seed)
    for key, group in subjects.groupby(keys):
        values = group.difference_pct_points.dropna().to_numpy(float)
        if not len(values):
            continue
        lo, hi = np.nan, np.nan
        if len(values) >= 5 and n_boot:
            boot = values[rng.integers(len(values), size=(n_boot, len(values)))].mean(axis=1)
            lo, hi = percentile_interval(boot)
        rows.append(dict(zip(keys, key), n_subjects=len(values),
                         mean_difference_pct_points=values.mean(),
                         sd_between_subjects=values.std(ddof=1) if len(values) > 1 else np.nan,
                         ci_low=lo, ci_high=hi,
                         inference="subject_bootstrap_exploratory" if len(values) >= 5 else "descriptive_fewer_than_5_subjects",
                         session_hand_protocol_comparability="must_be_verified_from_acquisition_metadata"))
    return pd.DataFrame(rows)
