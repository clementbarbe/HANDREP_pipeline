"""Figure 08 — raw vs corrected positions."""

import matplotlib.pyplot as plt
from pipeline.config.settings import FINGER_ORDER, FINGER_COLORS


def plot_correction_effect(analysis, output_path, cfg):
    df = analysis["df"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # A) All points
    ax = axes[0]
    for finger in FINGER_ORDER:
        sub = df[df["finger"] == finger]
        if sub.empty:
            continue
        ax.scatter(sub["x_est"], sub["y_est"], c=FINGER_COLORS[finger],
                   alpha=0.35, s=50, marker="o")
        ax.scatter(sub["x_est_corr"], sub["y_est_corr"],
                   c=FINGER_COLORS[finger], alpha=0.85, s=50, marker="x")
    ax.set_title("A) Raw vs corrected positions")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.invert_yaxis(); ax.set_aspect("equal")

    # B) Mean + arrows
    ax = axes[1]
    mp = (df.groupby(["finger", "zone"])
          [["x_est", "y_est", "x_est_corr", "y_est_corr", "x_real", "y_real"]]
          .mean().reset_index())
    for _, r in mp.iterrows():
        c = FINGER_COLORS[r["finger"]]
        ax.scatter(r["x_real"], r["y_real"], c="black", s=70, marker="o", zorder=4)
        ax.scatter(r["x_est"], r["y_est"], c=c, s=90, marker="s", alpha=0.5, zorder=3)
        ax.scatter(r["x_est_corr"], r["y_est_corr"], c=c, s=90,
                   marker="x", linewidths=2, zorder=5)
        ax.annotate("", xy=(r["x_est_corr"], r["y_est_corr"]),
                    xytext=(r["x_est"], r["y_est"]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.8))
    ax.scatter([], [], c="gray", marker="s", label="Raw estimate")
    ax.scatter([], [], c="gray", marker="x", label="Corrected estimate")
    ax.scatter([], [], c="black", marker="o", label="Reference")
    ax.legend()
    ax.set_title("B) Correction effect (raw → corrected)")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.invert_yaxis(); ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()