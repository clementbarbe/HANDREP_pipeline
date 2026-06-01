"""
Biomechanical metrics: finger lengths, inter-finger widths, Napier shape index.
"""

import numpy as np
import pandas as pd
from pipeline.config.settings import FINGER_ORDER


def _dist(p1, p2):
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


# ── DataFrame-level metrics (from all trials) ────────────

def compute_biomechanical_metrics(df):
    """
    Compute finger lengths and inter-finger distances from trial means.

    Returns a tidy DataFrame with columns:
    ``subject, session, metric_type, name, value_est, value_real,
    difference, percent_overestimation``.
    """
    mean_pts = (
        df.groupby(["subject", "session", "finger", "zone"])
        .agg(
            x_est_corr=("x_est_corr", "mean"),
            y_est_corr=("y_est_corr", "mean"),
            x_real=("x_real", "mean"),
            y_real=("y_real", "mean"),
        )
        .reset_index()
    )
    if mean_pts.empty:
        return pd.DataFrame(columns=[
            "subject", "session", "metric_type", "name",
            "value_est", "value_real", "difference",
            "percent_overestimation",
        ])

    subj = mean_pts["subject"].iloc[0]
    sess = mean_pts["session"].iloc[0]

    pt = {}
    for _, r in mean_pts.iterrows():
        pt[(r["finger"], r["zone"])] = {
            "est": (r["x_est_corr"], r["y_est_corr"]),
            "real": (r["x_real"], r["y_real"]),
        }

    rows = []

    # Finger lengths
    for finger in FINGER_ORDER:
        k1, k2 = (finger, "Z1"), (finger, "Z2")
        if k1 in pt and k2 in pt:
            le = _dist(pt[k1]["est"], pt[k2]["est"])
            lr = _dist(pt[k1]["real"], pt[k2]["real"])
            diff = le - lr
            pct = (diff / lr * 100) if lr else np.nan
            rows.append(dict(
                subject=subj, session=sess,
                metric_type="finger_length", name=finger,
                value_est=le, value_real=lr,
                difference=diff, percent_overestimation=pct,
            ))

    # Inter-finger distances
    for zone in ["Z1", "Z2"]:
        for i in range(len(FINGER_ORDER) - 1):
            f1, f2 = FINGER_ORDER[i], FINGER_ORDER[i + 1]
            k1, k2 = (f1, zone), (f2, zone)
            if k1 in pt and k2 in pt:
                de = _dist(pt[k1]["est"], pt[k2]["est"])
                dr = _dist(pt[k1]["real"], pt[k2]["real"])
                diff = de - dr
                pct = (diff / dr * 100) if dr else np.nan
                rows.append(dict(
                    subject=subj, session=sess,
                    metric_type=f"interfinger_{zone}", name=f"{f1}-{f2}",
                    value_est=de, value_real=dr,
                    difference=diff, percent_overestimation=pct,
                ))

    return pd.DataFrame(rows)


# ── From pre-computed position dicts ──────────────────────

def compute_finger_lengths(real_pos, est_pos):
    """Return a DataFrame comparing Z1–Z2 distances for each finger."""
    rows = []
    for finger in FINGER_ORDER:
        k1, k2 = (finger, "Z1"), (finger, "Z2")
        if k1 in real_pos and k2 in real_pos and k1 in est_pos and k2 in est_pos:
            lr = _dist(real_pos[k1], real_pos[k2])
            le = _dist(est_pos[k1], est_pos[k2])
            pct = 100 * (le - lr) / lr if lr else np.nan
            rows.append(dict(
                finger=finger, length_real=lr, length_est=le,
                pct_overestimation=pct,
            ))
    return pd.DataFrame(rows)


def compute_knuckle_widths(real_pos, est_pos):
    """Return a DataFrame comparing inter-knuckle Z1 distances."""
    rows = []
    for i in range(len(FINGER_ORDER) - 1):
        f1, f2 = FINGER_ORDER[i], FINGER_ORDER[i + 1]
        k1, k2 = (f1, "Z1"), (f2, "Z1")
        if k1 in real_pos and k2 in real_pos and k1 in est_pos and k2 in est_pos:
            wr = _dist(real_pos[k1], real_pos[k2])
            we = _dist(est_pos[k1], est_pos[k2])
            pct = 100 * (we - wr) / wr if wr else np.nan
            rows.append(dict(
                pair=f"{f1}-{f2}", width_real=wr, width_est=we,
                pct_overestimation=pct,
            ))
    return pd.DataFrame(rows)


def compute_shape_indices(real_pos, est_pos):
    """
    Napier shape index: ``100 × hand_width / middle_finger_length``.

    Returns ``{'real': float, 'estimated': float}`` or *None* if
    landmarks are insufficient.
    """
    required = [("index", "Z1"), ("little", "Z1"),
                ("middle", "Z1"), ("middle", "Z2")]
    if not all(k in real_pos for k in required):
        return None
    if not all(k in est_pos for k in required):
        return None

    def _si(pos):
        w = _dist(pos[("index", "Z1")], pos[("little", "Z1")])
        l = _dist(pos[("middle", "Z1")], pos[("middle", "Z2")])
        return 100 * w / l if l else np.nan

    return {"real": _si(real_pos), "estimated": _si(est_pos)}