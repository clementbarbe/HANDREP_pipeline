"""
Detection of consecutive Z1↔Z2 couples and split analysis.

Task design
-----------
For each finger (10 Z1 + 10 Z2 = 20 presentations):
  - 30 % → Z1→Z2 consecutive couples  (≈ 3 couples = 6 trials)
  - 30 % → Z2→Z1 consecutive couples  (≈ 3 couples = 6 trials)
  - 40 % → random, uncoupled            (≈ 8 trials)

This module detects which trials belong to a couple, then computes
finger-length estimates separately for coupled and uncoupled trials.
"""

import numpy as np
import pandas as pd

from pipeline.config.settings import FINGER_ORDER


# ── helpers ───────────────────────────────────────────────

def _dist(p1, p2):
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


# ── couple detection ──────────────────────────────────────

def detect_couples(df):
    """
    Scan the trial sequence and tag consecutive same-finger,
    different-zone pairs as *coupled*.

    Parameters
    ----------
    df : DataFrame
        Must contain ``trial``, ``finger``, ``zone``.

    Returns
    -------
    DataFrame
        Copy of *df* sorted by trial, with added columns:

        - ``couple_id``        : integer ID shared by the two rows of a
                                 couple (``NaN`` for uncoupled rows)
        - ``is_coupled``       : boolean
        - ``couple_direction`` : ``'Z1→Z2'``, ``'Z2→Z1'``, or ``None``
    """
    out = df.copy()
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out = out.sort_values("trial").reset_index(drop=True)

    n = len(out)
    cid   = np.full(n, np.nan)
    direc = np.array([None] * n, dtype=object)

    current_id = 1
    i = 0
    while i < n - 1:
        cur = out.iloc[i]
        nxt = out.iloc[i + 1]

        if cur["finger"] == nxt["finger"] and cur["zone"] != nxt["zone"]:
            cid[i]   = current_id
            cid[i+1] = current_id
            d = f"{cur['zone']}→{nxt['zone']}"
            direc[i]   = d
            direc[i+1] = d
            current_id += 1
            i += 2                       # skip past the pair
        else:
            i += 1

    out["couple_id"]        = cid
    out["is_coupled"]       = ~np.isnan(cid)
    out["couple_direction"] = direc
    return out


# ── summary counts ────────────────────────────────────────

def coupling_summary(df):
    """
    Per-finger counts of coupled / uncoupled trials.

    Returns
    -------
    dict[str, dict]
        ``{finger: {total, coupled, uncoupled,
                     couples_Z1Z2, couples_Z2Z1, n_couples}}``
    """
    result = {}
    for finger in FINGER_ORDER:
        fd = df[df["finger"] == finger]
        n_total    = len(fd)
        n_coupled  = int(fd["is_coupled"].sum())
        n_uncoupled = n_total - n_coupled

        coupled_only = fd[fd["is_coupled"]]
        dirs = coupled_only["couple_direction"].dropna()
        # each couple has 2 rows with the same direction string
        n_z1z2 = int((dirs == "Z1→Z2").sum()) // 2
        n_z2z1 = int((dirs == "Z2→Z1").sum()) // 2

        result[finger] = dict(
            total=n_total,
            coupled=n_coupled,
            uncoupled=n_uncoupled,
            couples_Z1Z2=n_z1z2,
            couples_Z2Z1=n_z2z1,
            n_couples=n_z1z2 + n_z2z1,
        )
    return result


# ── individual couple distances (for strip-plots) ────────

def individual_couple_distances(df, cm_per_px=None):
    """
    Compute the Z1–Z2 distance for every detected couple.

    Returns
    -------
    DataFrame
        One row per couple: ``finger, couple_id, direction,
        distance_px [, distance_cm]``.
    """
    coupled = df[df["is_coupled"]].copy()
    rows = []
    for finger in FINGER_ORDER:
        fc = coupled[coupled["finger"] == finger]
        for cid in fc["couple_id"].dropna().unique():
            pair = fc[fc["couple_id"] == cid]
            z1 = pair[pair["zone"] == "Z1"]
            z2 = pair[pair["zone"] == "Z2"]
            if len(z1) == 1 and len(z2) == 1:
                p1 = z1[["x_est_corr", "y_est_corr"]].values[0]
                p2 = z2[["x_est_corr", "y_est_corr"]].values[0]
                d = float(np.linalg.norm(p2 - p1))
                row = dict(
                    finger=finger,
                    couple_id=int(cid),
                    direction=pair["couple_direction"].iloc[0],
                    distance_px=d,
                )
                if cm_per_px is not None:
                    row["distance_cm"] = d * cm_per_px
                rows.append(row)
    return pd.DataFrame(rows)


# ── coupled finger lengths ────────────────────────────────

def compute_coupled_finger_lengths(df, real_pos, cm_per_px=None):
    """
    Finger-length estimates from **coupled** trials only.

    Each couple provides one Z1–Z2 distance.  The per-finger mean
    and std are computed across all couples of that finger.

    Returns
    -------
    DataFrame
        ``finger, condition, length_est, length_est_std, length_real,
        pct_overestimation, n_pairs [, _cm columns]``
    """
    coupled = df[df["is_coupled"]].copy()
    rows = []

    for finger in FINGER_ORDER:
        fc = coupled[coupled["finger"] == finger]
        couple_ids = fc["couple_id"].dropna().unique()

        dists = []
        for cid in couple_ids:
            pair = fc[fc["couple_id"] == cid]
            z1 = pair[pair["zone"] == "Z1"]
            z2 = pair[pair["zone"] == "Z2"]
            if len(z1) == 1 and len(z2) == 1:
                p1 = z1[["x_est_corr", "y_est_corr"]].values[0]
                p2 = z2[["x_est_corr", "y_est_corr"]].values[0]
                dists.append(float(np.linalg.norm(p2 - p1)))

        if not dists:
            continue
        k1, k2 = (finger, "Z1"), (finger, "Z2")
        if k1 not in real_pos or k2 not in real_pos:
            continue

        real_len  = _dist(real_pos[k1], real_pos[k2])
        mean_est  = float(np.mean(dists))
        std_est   = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
        pct       = 100 * (mean_est - real_len) / real_len if real_len else np.nan

        row = dict(
            finger=finger,
            condition="coupled",
            length_est=mean_est,
            length_est_std=std_est,
            length_real=real_len,
            pct_overestimation=pct,
            n_pairs=len(dists),
        )
        if cm_per_px is not None:
            row["length_est_cm"]     = mean_est * cm_per_px
            row["length_est_std_cm"] = std_est  * cm_per_px
            row["length_real_cm"]    = real_len * cm_per_px
        rows.append(row)

    return pd.DataFrame(rows)


# ── uncoupled finger lengths ─────────────────────────────

def compute_uncoupled_finger_lengths(df, real_pos, cm_per_px=None):
    """
    Finger-length estimates from **uncoupled** trials only.

    Mean Z1 and mean Z2 positions are computed from uncoupled trials;
    the distance between these two centroids is the finger-length
    estimate.

    Returns
    -------
    DataFrame
        ``finger, condition, length_est, length_real,
        pct_overestimation, n_z1, n_z2 [, _cm columns]``
    """
    uncoupled = df[~df["is_coupled"]].copy()
    rows = []

    for finger in FINGER_ORDER:
        fu = uncoupled[uncoupled["finger"] == finger]
        z1 = fu[fu["zone"] == "Z1"][["x_est_corr", "y_est_corr"]].values
        z2 = fu[fu["zone"] == "Z2"][["x_est_corr", "y_est_corr"]].values

        if len(z1) == 0 or len(z2) == 0:
            continue
        k1, k2 = (finger, "Z1"), (finger, "Z2")
        if k1 not in real_pos or k2 not in real_pos:
            continue

        mean_z1  = z1.mean(axis=0)
        mean_z2  = z2.mean(axis=0)
        est_len  = float(np.linalg.norm(mean_z2 - mean_z1))
        real_len = _dist(real_pos[k1], real_pos[k2])
        pct      = 100 * (est_len - real_len) / real_len if real_len else np.nan

        row = dict(
            finger=finger,
            condition="uncoupled",
            length_est=est_len,
            length_real=real_len,
            pct_overestimation=pct,
            n_z1=len(z1),
            n_z2=len(z2),
        )
        if cm_per_px is not None:
            row["length_est_cm"]  = est_len  * cm_per_px
            row["length_real_cm"] = real_len * cm_per_px
        rows.append(row)

    return pd.DataFrame(rows)