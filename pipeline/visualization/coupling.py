"""Identical centroid estimators and conditional 95% intervals in both conditions."""

import numpy as np
import matplotlib.pyplot as plt
from pipeline.config.settings import FINGER_ORDER


def plot_coupled_vs_uncoupled(analysis, output_path, cfg):
    coupled, uncoupled = analysis["finger_lengths_coupled"], analysis["finger_lengths_uncoupled"]
    comparison = analysis["coupling_comparison"]
    if coupled.empty or uncoupled.empty or comparison.empty:
        return
    fingers = [f for f in FINGER_ORDER if f in set(coupled.finger) & set(uncoupled.finger)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    inferred = analysis["df"].coupling_source.eq("inferred_adjacency_exploratory").any()
    labels = ("Adjacent pairs", "Other trials") if inferred else ("Planned coupled", "Planned uncoupled")
    for table, offset, color, label in ((coupled, -.13, "#c0392b", labels[0]), (uncoupled, .13, "#2471a3", labels[1])):
        for i, finger in enumerate(fingers):
            row = table[table.finger == finger].iloc[0]
            axes[0].scatter(i + offset, row.pct_overestimation, color=color, label=label if i == 0 else None, s=65)
            if np.isfinite([row.pct_ci_low, row.pct_ci_high]).all():
                axes[0].vlines(i + offset, row.pct_ci_low, row.pct_ci_high, color=color, lw=2)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set(title="Distance between centroids in both conditions", ylabel="Finger length difference from reference (%)")
    axes[0].legend()
    for i, finger in enumerate(fingers):
        sub = comparison[comparison.finger == finger]
        if sub.empty:
            continue
        row = sub.iloc[0]
        axes[1].scatter(i, row.difference_pct_points, color="#5b2c6f", s=65)
        if np.isfinite([row.ci_pct_low, row.ci_pct_high]).all():
            axes[1].vlines(i, row.ci_pct_low, row.ci_pct_high, color="#5b2c6f", lw=2)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(title="Within-session condition contrast", ylabel=f"{labels[0]} minus {labels[1]} (percentage points)")
    for ax in axes:
        ax.set_xticks(range(len(fingers)), fingers)
    scope = "Exploratory adjacency; planned conditions unavailable" if inferred else "Recorded acquisition design"
    fig.suptitle(f"{analysis['subject']} / {analysis['session']} — {scope}")
    fig.text(.5, .015, "95% percentile bootstrap intervals; fixed reference/calibration; temporal dependence not modelled", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
