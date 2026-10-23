"""Reproducible percentile intervals conditional on the recorded reference.

These within-session bootstraps do not include camera/annotation uncertainty
or account for arbitrary temporal autocorrelation. Paired endpoints are always
resampled together; unpaired endpoints are resampled within zone.
"""

import numpy as np


def percentile_interval(samples, confidence=0.95):
    samples = np.asarray(samples, float)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return np.nan, np.nan
    q = (1 - confidence) / 2
    return tuple(np.quantile(samples, [q, 1 - q]))


def centroid_distance_samples(z1, z2, n_boot=2000, seed=42, paired=False, min_n=3):
    z1, z2 = np.asarray(z1, float), np.asarray(z2, float)
    if min(len(z1), len(z2)) < min_n or n_boot < 1:
        return np.array([])
    rng = np.random.default_rng(seed)
    i1 = rng.integers(len(z1), size=(n_boot, len(z1)))
    if paired:
        if len(z1) != len(z2):
            raise ValueError("Paired bootstrap requires matching endpoints")
        i2 = i1
    else:
        i2 = rng.integers(len(z2), size=(n_boot, len(z2)))
    return np.linalg.norm(z2[i2].mean(axis=1) - z1[i1].mean(axis=1), axis=1)


def landmark_mean_samples(df, n_boot=2000, seed=42, min_n=3):
    """Stratify by landmark; preserve complete Z1/Z2 pairs within finger.

Other temporal dependence is not modelled. Incomplete pairs contribute their
valid endpoint to the overall map, but never to the coupled-length comparison.
"""
    rng = np.random.default_rng(seed)
    valid = df.dropna(subset=["x_est_corr", "y_est_corr"])
    sums, counts, consumed = {}, {}, set()
    if "couple_id" in valid:
        for finger, fg in valid[valid.is_coupled].groupby("finger"):
            pairs = []
            for _, pair in fg.groupby("couple_id"):
                if len(pair) == 2 and set(pair.zone) == {"Z1", "Z2"}:
                    pairs.append(np.array([pair.loc[pair.zone == z, ["x_est_corr", "y_est_corr"]].iloc[0] for z in ("Z1", "Z2")]))
                    consumed.update(pair.index)
            if pairs:
                pairs = np.asarray(pairs)
                draw = pairs[rng.integers(len(pairs), size=(n_boot, len(pairs)))].sum(axis=1)
                for j, z in enumerate(("Z1", "Z2")):
                    sums[finger, z] = draw[:, j]
                    counts[finger, z] = len(pairs)
    for key, group in valid.loc[~valid.index.isin(consumed)].groupby(["finger", "zone"]):
        pts = group[["x_est_corr", "y_est_corr"]].to_numpy(float)
        draw = pts[rng.integers(len(pts), size=(n_boot, len(pts)))].sum(axis=1)
        sums[key] = sums.get(key, 0) + draw
        counts[key] = counts.get(key, 0) + len(pts)
    return {key: value / counts[key] for key, value in sums.items() if counts[key] >= min_n}
