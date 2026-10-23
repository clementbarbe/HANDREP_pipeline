"""Figures 03 and 07 — finger lengths, knuckle widths, shape index.

When cm_per_px is available, bar labels show centimetre values.
"""

import matplotlib.pyplot as plt
from pipeline.config.settings import FINGER_COLORS


def plot_finger_lengths_widths(analysis, output_path, cfg):
    df_l = analysis["finger_lengths"]
    df_w = analysis["knuckle_widths"]
    cm   = analysis.get("cm_per_px")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── A) Finger length overestimation ───────────────────
    ax = axes[0]
    if not df_l.empty:
        colors = [FINGER_COLORS[f] for f in df_l["finger"]]
        bars = ax.bar(
            df_l["finger"], df_l["pct_overestimation"],
            color=colors, edgecolor="black", alpha=0.8,
        )
        if "pct_ci_low" in df_l:
            ax.vlines(range(len(df_l)), df_l.pct_ci_low, df_l.pct_ci_high, color="black", lw=1.5)
        # Annotate with cm values when available
        if cm is not None and "length_real_cm" in df_l.columns:
            for bar, (_, row) in zip(bars, df_l.iterrows()):
                label = (
                    f"real {row['length_real_cm']:.1f} cm\n"
                    f"est  {row['length_est_cm']:.1f} cm"
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5 if bar.get_height() >= 0
                    else bar.get_height() - 3,
                    label, ha="center", fontsize=7,
                )
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("% overestimation")
    ax.set_title("A) Finger length overestimation")

    # ── B) Knuckle spacing ────────────────────────────────
    ax = axes[1]
    if not df_w.empty:
        bars = ax.bar(
            df_w["pair"], df_w["pct_overestimation"],
            color="steelblue", edgecolor="black", alpha=0.8,
        )
        if "pct_ci_low" in df_w:
            ax.vlines(range(len(df_w)), df_w.pct_ci_low, df_w.pct_ci_high, color="black", lw=1.5)
        if cm is not None and "width_real_cm" in df_w.columns:
            for bar, (_, row) in zip(bars, df_w.iterrows()):
                label = (
                    f"real {row['width_real_cm']:.1f} cm\n"
                    f"est  {row['width_est_cm']:.1f} cm"
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5 if bar.get_height() >= 0
                    else bar.get_height() - 3,
                    label, ha="center", fontsize=7,
                )
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("% overestimation")
    ax.set_title("B) Knuckle spacing overestimation")

    plt.tight_layout()
    fig.text(.5, .005, "95% conditional bootstrap intervals; fixed reference/calibration", ha="center", fontsize=8)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shape_index(analysis, output_path, cfg):
    si = analysis["shape_indices"]
    if si is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [si["real"], si["estimated"]]
    bars = ax.bar(
        ["Actual hand", "Proprioceptive\nrepresentation"],
        vals, color=["#2c7bb6", "#d7191c"], edgecolor="black", width=0.5,
    )
    if "estimated_ci_low" in si:
        ax.vlines(1, si["estimated_ci_low"], si["estimated_ci_high"], color="black")
    ax.set_ylabel("Shape Index (100 × W / L)")
    ax.set_title("Napier Shape Index")
    ax.axhline(si["real"], ls="--", color="gray", alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{v:.1f}", ha="center", fontsize=12, fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
