"""Figure 02 — error bar charts and error vectors."""

import matplotlib.pyplot as plt
from pipeline.config.settings import FINGER_ORDER, FINGER_COLORS


def plot_error_analysis(analysis, output_path, cfg):
    df = analysis["df"]
    stats_lm = analysis["stats_landmark"].copy()
    stats_fg = analysis["stats_finger"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # A) Per finger
    ax = axes[0]
    colors = [FINGER_COLORS[f] for f in FINGER_ORDER if f in stats_fg.index]
    ax.bar(FINGER_ORDER, stats_fg.loc[FINGER_ORDER, "mean_error"],
           yerr=stats_fg.loc[FINGER_ORDER, "std_error"], capsize=5,
           color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Euclidean error (px)"); ax.set_title("A) Mean error per finger")

    # B) Per landmark
    ax = axes[1]
    stats_lm["label"] = stats_lm["finger"] + "\n" + stats_lm["zone"]
    idx_map = {f: i for i, f in enumerate(FINGER_ORDER)}
    stats_lm["_ord"] = stats_lm["finger"].map(idx_map)
    stats_lm = stats_lm.sort_values(["_ord", "zone"])
    xp = range(len(stats_lm))
    bc = [FINGER_COLORS[f] for f in stats_lm["finger"]]
    ax.bar(xp, stats_lm["mean_error"], yerr=stats_lm["std_error"],
           capsize=4, color=bc, edgecolor="black", alpha=0.8)
    ax.set_xticks(list(xp)); ax.set_xticklabels(stats_lm["label"], fontsize=9)
    ax.set_ylabel("Euclidean error (px)"); ax.set_title("B) Error per landmark")

    # C) Error vectors
    ax = axes[2]
    me = (df.groupby(["finger", "zone"])
          [["x_est_corr", "y_est_corr", "x_real", "y_real"]]
          .mean().reset_index())
    for _, r in me.iterrows():
        c = FINGER_COLORS[r["finger"]]
        ax.scatter(r["x_real"], r["y_real"], c="black", s=80, zorder=5)
        ax.scatter(r["x_est_corr"], r["y_est_corr"], c=c, s=80,
                   zorder=5, marker="x", linewidths=2)
        ax.annotate("", xy=(r["x_est_corr"], r["y_est_corr"]),
                    xytext=(r["x_real"], r["y_real"]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.scatter([], [], c="black", marker="o", label="Reference")
    ax.scatter([], [], c="gray", marker="x", linewidths=2, label="Estimated (mean)")
    ax.legend(fontsize=10)
    ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    ax.set_title("C) Error vectors (ref → est)")
    ax.invert_yaxis(); ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()