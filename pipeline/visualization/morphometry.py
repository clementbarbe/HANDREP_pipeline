"""Figures 03 and 07 — finger lengths, knuckle widths, shape index."""

import matplotlib.pyplot as plt
from pipeline.config.settings import FINGER_COLORS


def plot_finger_lengths_widths(analysis, output_path, cfg):
    df_l = analysis["finger_lengths"]
    df_w = analysis["knuckle_widths"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if not df_l.empty:
        colors = [FINGER_COLORS[f] for f in df_l["finger"]]
        ax.bar(df_l["finger"], df_l["pct_overestimation"],
               color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("% overestimation")
    ax.set_title("A) Finger length overestimation")

    ax = axes[1]
    if not df_w.empty:
        ax.bar(df_w["pair"], df_w["pct_overestimation"],
               color="steelblue", edgecolor="black", alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("% overestimation")
    ax.set_title("B) Knuckle spacing overestimation")

    plt.tight_layout()
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
    ax.set_ylabel("Shape Index (100 × W / L)")
    ax.set_title("Napier Shape Index")
    ax.axhline(si["real"], ls="--", color="gray", alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()