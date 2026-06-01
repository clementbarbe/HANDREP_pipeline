"""Figure 01 — estimated landmark positions."""

import matplotlib.pyplot as plt
from pipeline.config.settings import FINGER_ORDER, FINGER_COLORS


def plot_estimated_positions(analysis, output_path, cfg):
    df = analysis["df"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: all observations
    ax = axes[0]
    for finger in FINGER_ORDER:
        sub = df[df["finger"] == finger]
        if sub.empty:
            continue
        ax.scatter(sub["x_est_corr"], sub["y_est_corr"],
                   c=FINGER_COLORS[finger], label=finger,
                   s=80, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("X (pixels)"); ax.set_ylabel("Y (pixels)")
    ax.set_title("Corrected estimated positions — all trials")
    ax.legend(title="Finger", fontsize=10)
    ax.invert_yaxis(); ax.set_aspect("equal")

    # Right: mean per landmark
    ax = axes[1]
    mean_est = (
        df.groupby(["finger", "zone"])[["x_est_corr", "y_est_corr"]]
        .mean().reset_index()
    )
    for finger in FINGER_ORDER:
        sub = mean_est[mean_est["finger"] == finger].sort_values("zone")
        if sub.empty:
            continue
        ax.scatter(sub["x_est_corr"], sub["y_est_corr"],
                   c=FINGER_COLORS[finger], s=150,
                   edgecolors="k", linewidths=1.5, zorder=5)
        if len(sub) == 2:
            ax.plot(sub["x_est_corr"].values, sub["y_est_corr"].values,
                    c=FINGER_COLORS[finger], lw=2, ls="--", alpha=0.7)
        for _, r in sub.iterrows():
            ax.annotate(f"{finger}\n{r['zone']}",
                        (r["x_est_corr"], r["y_est_corr"]),
                        textcoords="offset points", xytext=(10, 5),
                        fontsize=8, color=FINGER_COLORS[finger])
    ax.set_xlabel("X (pixels)"); ax.set_ylabel("Y (pixels)")
    ax.set_title("Mean estimated positions per landmark")
    ax.invert_yaxis(); ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()