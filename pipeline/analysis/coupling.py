"""Comparable centroid geometry; recorded design takes priority over adjacency."""

import numpy as np
import pandas as pd

from pipeline.config.settings import FINGER_ORDER
from pipeline.analysis.uncertainty import centroid_distance_samples, percentile_interval


def detect_couples(df):
    """Retain invalid trials as sequence barriers; never bridge missing trials.

Optional condition/pair_id columns specify the planned acquisition design.
Otherwise greedy, non-overlapping adjacency is descriptive and exploratory.
"""
    out = df.copy()
    out["trial"] = pd.to_numeric(out["trial"], errors="raise")
    keys = [c for c in ("subject", "session") if c in out]
    sequence_keys = keys + (["block"] if "block" in out else [])
    if out.duplicated(keys + ["trial"]).any():
        raise ValueError("Duplicate trials cannot be paired")
    out = out.sort_values(keys + ["trial"]).reset_index(drop=True)
    out["couple_id"] = pd.Series([None] * len(out), dtype=object)
    out["is_coupled"] = False
    out["couple_direction"] = pd.Series([None] * len(out), dtype=object)
    out["pair_complete"] = False
    out["coupling_source"] = "inferred_adjacency_exploratory"
    recorded = {"condition", "pair_id"}.intersection(out.columns)
    if recorded and recorded != {"condition", "pair_id"}:
        raise ValueError("Recorded design requires condition and pair_id")
    groups = out.groupby(sequence_keys, sort=False, dropna=False) if sequence_keys else [(None, out)]
    cid = 0
    for _, group in groups:
        if recorded:
            if not group["condition"].isin(["coupled", "uncoupled"]).all():
                raise ValueError("Unknown recorded coupling condition")
            out.loc[group.index, "coupling_source"] = "recorded_design"
            coupled = group["condition"].eq("coupled")
            pair_ids = group["pair_id"].replace(r"^\s*$", np.nan, regex=True)
            if pair_ids[coupled].isna().any() or pair_ids[~coupled].notna().any():
                raise ValueError("pair_id must be present exactly for coupled trials")
            out.loc[group.index[coupled], "is_coupled"] = True
            for _, pair in group[coupled].groupby("pair_id", sort=False):
                if len(pair) > 2:
                    raise ValueError("A recorded pair contains more than two trials")
                cid += 1
                out.loc[pair.index, "couple_id"] = str(cid)
                if len(pair) == 1:
                    continue  # Planned coupled trial remains coupled, but unusable as a pair.
                a, b = pair.iloc[0], pair.iloc[1]
                if a.finger != b.finger or {a.zone, b.zone} != {"Z1", "Z2"} or b.trial - a.trial != 1:
                    raise ValueError("Recorded pair must be consecutive Z1/Z2 trials of the same finger")
                out.loc[pair.index, "pair_complete"] = True
                out.loc[pair.index, "couple_direction"] = f"{a.zone}→{b.zone}"
        else:
            indices = list(group.index)
            i = 0
            while i < len(indices) - 1:
                ia, ib = indices[i:i + 2]
                a, b = out.loc[ia], out.loc[ib]
                if (b.trial - a.trial == 1 and a.finger == b.finger
                        and {a.zone, b.zone} == {"Z1", "Z2"}):
                    cid += 1
                    out.loc[[ia, ib], "couple_id"] = str(cid)
                    out.loc[[ia, ib], "is_coupled"] = True
                    out.loc[[ia, ib], "pair_complete"] = True
                    out.loc[[ia, ib], "couple_direction"] = f"{a.zone}→{b.zone}"
                    i += 2
                else:
                    i += 1
    return out


def coupling_summary(df):
    result = {}
    for finger in FINGER_ORDER:
        fd = df[df.finger == finger]
        directions = fd.loc[fd.is_coupled, "couple_direction"]
        a, b = int(directions.eq("Z1→Z2").sum()) // 2, int(directions.eq("Z2→Z1").sum()) // 2
        result[finger] = dict(total=len(fd), coupled=int(fd.is_coupled.sum()),
                              uncoupled=int((~fd.is_coupled).sum()),
                              couples_Z1Z2=a, couples_Z2Z1=b, n_couples=a + b)
    return result


def _endpoints(df, finger, coupled, direction=None):
    sub = df[(df.finger == finger) & (df.is_coupled == coupled)].copy()
    if direction:
        sub = sub[sub.couple_direction == direction]
    if not coupled:
        return tuple(sub.loc[sub.zone == z, ["x_est_corr", "y_est_corr"]]
                     .dropna().to_numpy(float) for z in ("Z1", "Z2"))
    pairs = []
    for _, pair in sub.groupby("couple_id", sort=False):
        if len(pair) != 2 or set(pair.zone) != {"Z1", "Z2"}:
            continue
        pts = np.array([pair.loc[pair.zone == z, ["x_est_corr", "y_est_corr"]].iloc[0]
                        for z in ("Z1", "Z2")], float)
        if np.isfinite(pts).all():
            pairs.append(pts)
    if not pairs:
        return np.empty((0, 2)), np.empty((0, 2))
    points = np.asarray(pairs)
    return points[:, 0], points[:, 1]


def _lengths(df, real_pos, coupled, cm_per_px=None, n_boot=2000, seed=42, min_n=3, direction=None):
    rows = []
    for index, finger in enumerate(FINGER_ORDER):
        if any((finger, z) not in real_pos for z in ("Z1", "Z2")):
            continue
        z1, z2 = _endpoints(df, finger, coupled, direction)
        if not len(z1) or not len(z2):
            continue
        real = float(np.linalg.norm(np.asarray(real_pos[finger, "Z2"]) - real_pos[finger, "Z1"]))
        length = float(np.linalg.norm(z2.mean(axis=0) - z1.mean(axis=0)))
        samples = centroid_distance_samples(z1, z2, n_boot, seed + index, coupled, min_n)
        lo, hi = percentile_interval(samples)
        row = dict(finger=finger, condition="coupled" if coupled else "uncoupled",
                   direction=direction or "all", estimator="distance_between_centroids",
                   coupling_source=";".join(sorted(df.coupling_source.unique())),
                   length_est=length, length_real=real,
                   pct_overestimation=100 * (length / real - 1) if real > 0 else np.nan,
                   length_ci_low=lo, length_ci_high=hi,
                   pct_ci_low=100 * (lo / real - 1) if real > 0 else np.nan,
                   pct_ci_high=100 * (hi / real - 1) if real > 0 else np.nan,
                   n_z1=len(z1), n_z2=len(z2), n_pairs=len(z1) if coupled else 0,
                   ci_method="paired_percentile" if coupled else "zone_stratified_percentile",
                   ci_status="conditional_exploratory" if len(samples) else "insufficient_repeats")
        if coupled:
            distances = np.linalg.norm(z2 - z1, axis=1)
            row["pair_distance_mean"] = distances.mean()
            row["pair_distance_std"] = distances.std(ddof=1) if len(distances) > 1 else np.nan
        if cm_per_px is not None:
            for key in ("length_est", "length_real", "length_ci_low", "length_ci_high"):
                row[key + "_cm"] = row[key] * cm_per_px
        rows.append(row)
    return pd.DataFrame(rows)


def compute_coupled_finger_lengths(df, real_pos, cm_per_px=None, **kwargs):
    return _lengths(df, real_pos, True, cm_per_px, **kwargs)


def compute_uncoupled_finger_lengths(df, real_pos, cm_per_px=None, **kwargs):
    return _lengths(df, real_pos, False, cm_per_px, **kwargs)


def compare_coupling(df, real_pos, cm_per_px=None, n_boot=2000, seed=42, min_n=3):
    """Within-subject contrast; no trial-level population p-values."""
    rows = []
    for index, finger in enumerate(FINGER_ORDER):
        c1, c2 = _endpoints(df, finger, True)
        u1, u2 = _endpoints(df, finger, False)
        if any(len(p) == 0 for p in (c1, c2, u1, u2)):
            continue
        key1, key2 = (finger, "Z1"), (finger, "Z2")
        if key1 not in real_pos or key2 not in real_pos:
            continue
        real = np.linalg.norm(np.asarray(real_pos[key2]) - real_pos[key1])
        c = np.linalg.norm(c2.mean(0) - c1.mean(0))
        u = np.linalg.norm(u2.mean(0) - u1.mean(0))
        cb = centroid_distance_samples(c1, c2, n_boot, seed + index, True, min_n)
        ub = centroid_distance_samples(u1, u2, n_boot, seed + 100 + index, False, min_n)
        lo, hi = percentile_interval(cb - ub) if len(cb) and len(ub) else (np.nan, np.nan)
        row = dict(finger=finger, difference=c - u, ci_low=lo, ci_high=hi,
                   difference_pct_points=100 * (c - u) / real if real > 0 else np.nan,
                   ci_pct_low=100 * lo / real if real > 0 else np.nan,
                   ci_pct_high=100 * hi / real if real > 0 else np.nan,
                   n_pairs=len(c1), n_uncoupled_z1=len(u1), n_uncoupled_z2=len(u2),
                   coupling_source=";".join(sorted(df.coupling_source.unique())),
                   inference="within_session_exploratory_conditional_on_reference")
        if cm_per_px is not None:
            for k in ("difference", "ci_low", "ci_high"):
                row[k + "_cm"] = row[k] * cm_per_px
        rows.append(row)
    return pd.DataFrame(rows)


def individual_couple_distances(df, cm_per_px=None):
    rows = []
    for cid, pair in df[df.is_coupled].groupby("couple_id", sort=False):
        if len(pair) != 2 or set(pair.zone) != {"Z1", "Z2"}:
            continue
        pts = np.array([pair.loc[pair.zone == z, ["x_est_corr", "y_est_corr"]].iloc[0]
                        for z in ("Z1", "Z2")], float)
        if not np.isfinite(pts).all():
            continue
        row = dict(finger=pair.finger.iloc[0], couple_id=cid,
                   direction=pair.couple_direction.iloc[0], distance_px=np.linalg.norm(pts[1] - pts[0]),
                   first_trial=pair.trial.min(), second_trial=pair.trial.max(),
                   coupling_source=pair.coupling_source.iloc[0])
        if cm_per_px is not None:
            row["distance_cm"] = row["distance_px"] * cm_per_px
        rows.append(row)
    return pd.DataFrame(rows)
