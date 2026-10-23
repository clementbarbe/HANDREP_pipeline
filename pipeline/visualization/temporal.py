"""Figure 04 — error evolution across trials."""

import matplotlib.pyplot as plt
from pipeline.config.settings import FINGER_ORDER, FINGER_COLORS, ZONE_LABELS


def plot_error_over_time(analysis, output_path, cfg):
    df = analysis["df"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, finger in enumerate(FINGER_ORDER):
        ax = axes[i]
        sub = df[df["finger"] == finger].sort_values("trial")
        for zone in ["Z1", "Z2"]:
            sz = sub[sub["zone"] == zone]
            if not sz.empty:
                ax.plot(sz["trial"], sz["error_eucl"], marker="o",
                        label=f"{zone} ({ZONE_LABELS[zone]})",
                        linewidth=1.5, markersize=6, alpha=0.8)
        ax.set_xlabel("Trial"); ax.set_ylabel("Error (px)")
        ax.set_title(finger.capitalize(), color=FINGER_COLORS[finger],
                     fontweight="bold", fontsize=14)
        ax.legend(fontsize=9); ax.set_ylim(bottom=0)

    # Combined panel
    ax = axes[5]
    for finger in FINGER_ORDER:
        sub = df[df["finger"] == finger].sort_values("trial")
        if not sub.empty:
            ax.scatter(sub["trial"], sub["error_eucl"],
                       c=FINGER_COLORS[finger], label=finger, s=50, alpha=0.7)
    ax.set_xlabel("Trial"); ax.set_ylabel("Error (px)")
    ax.set_title("All fingers", fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(bottom=0)

    plt.suptitle("Localisation error over time",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_signed_errors(analysis, output_path, cfg):
    df = analysis["df"]
    cm = analysis["cm_per_px"]
    factor, units = (cm, "cm") if cm is not None else (1., "approximate px")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, axis in zip(axes, ("ru", "pd")):
        for finger in FINGER_ORDER:
            for zone, marker in (("Z1", "o"), ("Z2", "^")):
                sub = df[(df.finger == finger) & (df.zone == zone)]
                ax.scatter(sub.trial, sub["error_" + axis] * factor, c=FINGER_COLORS[finger],
                           marker=marker, alpha=.7, s=35, label=f"{finger} {zone}")
        ax.axhline(0, color="black", lw=1)
        ax.set(xlabel="Acquisition trial", ylabel=f"Signed error ({units})",
               title="Radioulnar (+ radial)" if axis == "ru" else "Proximodistal (+ distal)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Signed anatomical errors over the acquisition sequence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
