"""Anatomical bias and response dispersion, with explicit physical axes."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pipeline.config.settings import FINGER_COLORS


def plot_bias_precision(analysis, output_path, cfg):
    table = analysis["bias_precision"]
    if table.empty:
        return
    cm = analysis["cm_per_px"]
    scale, units = (cm, "cm") if cm is not None else (1., "px (approximate)")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for _, row in table.iterrows():
        color = FINGER_COLORS[row.finger]
        point = np.array([row.bias_ru, row.bias_pd]) * scale
        axes[0].scatter(*point, color=color, marker="o" if row.zone == "Z1" else "^", s=65)
        axes[0].annotate(f"{row.finger} {row.zone}", point, xytext=(3, 3), textcoords="offset points", fontsize=8)
        if "ellipse_major_sd" in row and np.isfinite(row.ellipse_major_sd):
            axes[0].add_patch(Ellipse(point, 2 * row.ellipse_major_sd * scale, 2 * row.ellipse_minor_sd * scale,
                                     angle=row.ellipse_angle_degrees, facecolor=color, edgecolor=color, alpha=.13))
    axes[0].axhline(0, color="gray", lw=.8)
    axes[0].axvline(0, color="gray", lw=.8)
    axes[0].set(xlabel=f"Ulnar ← error → radial ({units})", ylabel=f"Proximal ← error → distal ({units})",
                title="Mean bias; ellipses show 1 SD response dispersion")
    axes[0].set_aspect("equal", adjustable="datalim")
    x = np.arange(len(table))
    axes[1].bar(x - .18, table.bias_magnitude * scale, .36, label="Systematic bias")
    axes[1].bar(x + .18, table.variable_error * scale, .36, label="Variability around mean")
    axes[1].set_xticks(x, table.finger + " " + table.zone, rotation=60, ha="right")
    axes[1].set(ylabel=f"Magnitude ({units})", title="Bias and variability are separate components")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
