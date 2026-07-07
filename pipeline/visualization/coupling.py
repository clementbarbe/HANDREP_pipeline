"""
Figure 09 — Finger length overestimation: coupled vs uncoupled trials.

Panel A: grouped bar chart of % overestimation per finger
Panel B: absolute finger lengths (real / coupled est. / uncoupled est.)
         with individual per-couple estimates as scatter dots
"""

import numpy as np
import matplotlib.pyplot as plt

from pipeline.config.settings import FINGER_ORDER

# Colour palette for the two conditions
_CLR_COUPLED   = "#e74c3c"
_CLR_UNCOUPLED = "#3498db"
_CLR_REAL      = "#2ecc71"


def plot_coupled_vs_uncoupled(analysis, output_path, cfg):
    """
    Generate Figure 09: coupled vs uncoupled finger-length estimates.

    Silently returns if data for either condition is missing.
    """
    df_c  = analysis.get("finger_lengths_coupled")
    df_u  = analysis.get("finger_lengths_uncoupled")
    indiv = analysis.get("couple_distances")
    cm    = analysis.get("cm_per_px")

    if df_c is None or df_c.empty or df_u is None or df_u.empty:
        return

    # Fingers present in both conditions
    fingers = [
        f for f in FINGER_ORDER
        if f in df_c["finger"].values and f in df_u["finger"].values
    ]
    if not fingers:
        return

    use_cm = cm is not None and "length_real_cm" in df_c.columns
    n = len(fingers)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── A) % overestimation ───────────────────────────────
    ax = axes[0]
    w = 0.30

    c_pct = [float(df_c.loc[df_c["finger"] == f, "pct_overestimation"].iloc[0])
             for f in fingers]
    u_pct = [float(df_u.loc[df_u["finger"] == f, "pct_overestimation"].iloc[0])
             for f in fingers]

    # Error bars on coupled (from per-couple std)
    c_err = None
    if "length_est_std" in df_c.columns:
        c_std_px = [float(df_c.loc[df_c["finger"] == f, "length_est_std"].iloc[0])
                    for f in fingers]
        c_real   = [float(df_c.loc[df_c["finger"] == f, "length_real"].iloc[0])
                    for f in fingers]
        # Convert std to % of real length for error bars
        c_err = [100 * s / r if r > 0 else 0 for s, r in zip(c_std_px, c_real)]

    ax.bar(x - w / 2, c_pct, w,
           yerr=c_err, capsize=4,
           label="Coupled (consecutive Z1↔Z2)",
           color=_CLR_COUPLED, edgecolor="black", alpha=0.85)
    ax.bar(x + w / 2, u_pct, w,
           label="Uncoupled (random order)",
           color=_CLR_UNCOUPLED, edgecolor="black", alpha=0.85)

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in fingers])
    ax.set_ylabel("% overestimation")
    ax.set_xlabel("Finger")
    ax.set_title("A) Finger-length overestimation\ncoupled vs uncoupled trials")
    ax.legend(fontsize=10)

    # ── B) Absolute lengths ───────────────────────────────
    ax = axes[1]
    w3 = 0.22

    if use_cm:
        r_vals = [float(df_c.loc[df_c["finger"] == f, "length_real_cm"].iloc[0])
                  for f in fingers]
        c_vals = [float(df_c.loc[df_c["finger"] == f, "length_est_cm"].iloc[0])
                  for f in fingers]
        u_vals = [float(df_u.loc[df_u["finger"] == f, "length_est_cm"].iloc[0])
                  for f in fingers]
        ind_col = "distance_cm"
        ylabel  = "Finger length (cm)"
    else:
        r_vals = [float(df_c.loc[df_c["finger"] == f, "length_real"].iloc[0])
                  for f in fingers]
        c_vals = [float(df_c.loc[df_c["finger"] == f, "length_est"].iloc[0])
                  for f in fingers]
        u_vals = [float(df_u.loc[df_u["finger"] == f, "length_est"].iloc[0])
                  for f in fingers]
        ind_col = "distance_px"
        ylabel  = "Finger length (px)"

    ax.bar(x - w3, r_vals, w3,
           label="Real", color=_CLR_REAL, edgecolor="black", alpha=0.85)
    ax.bar(x, c_vals, w3,
           label="Coupled est.", color=_CLR_COUPLED, edgecolor="black", alpha=0.85)
    ax.bar(x + w3, u_vals, w3,
           label="Uncoupled est.", color=_CLR_UNCOUPLED, edgecolor="black", alpha=0.85)

    # Overlay individual per-couple distances on the coupled bar
    if indiv is not None and not indiv.empty and ind_col in indiv.columns:
        rng = np.random.default_rng(42)
        for i, finger in enumerate(fingers):
            fi = indiv[indiv["finger"] == finger]
            if fi.empty:
                continue
            jitter = rng.normal(0, 0.04, len(fi))
            ax.scatter(
                i + jitter, fi[ind_col].values,
                c="white", edgecolors=_CLR_COUPLED, linewidths=1.2,
                s=35, alpha=0.85, zorder=5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in fingers])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Finger")
    ax.set_title("B) Absolute finger lengths\n(○ = individual coupled estimates)")
    ax.legend(fontsize=10, loc="upper right")

    # Value labels on top of bars (cm or px)
    for i in range(n):
        fmt = ".1f" if use_cm else ".0f"
        for val, xoff in [(r_vals[i], -w3), (c_vals[i], 0), (u_vals[i], w3)]:
            ax.text(
                x[i] + xoff, val + (max(r_vals) * 0.01),
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=7,
            )

    plt.suptitle(
        f"Finger-length estimation — {analysis['subject']} / {analysis['session']}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()