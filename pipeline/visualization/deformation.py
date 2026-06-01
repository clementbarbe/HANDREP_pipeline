"""Figures 05 + 06 — TPS deformation maps and overlay."""

import numpy as np
import matplotlib.pyplot as plt
from pipeline.config.settings import FINGER_ORDER, FINGER_COLORS


def _draw_hand_skeleton(ax, pts_arr, valid_landmarks, ls="-"):
    """Draw finger segments and knuckle line."""
    for finger in FINGER_ORDER:
        fp = []
        for zone in ["Z1", "Z2"]:
            if (finger, zone) in valid_landmarks:
                idx = valid_landmarks.index((finger, zone))
                fp.append(pts_arr[idx])
        if len(fp) == 2:
            fp = np.array(fp)
            ax.plot(fp[:, 0], fp[:, 1], c=FINGER_COLORS[finger], lw=2, ls=ls)
    kp = []
    for finger in FINGER_ORDER:
        if (finger, "Z1") in valid_landmarks:
            idx = valid_landmarks.index((finger, "Z1"))
            kp.append(pts_arr[idx])
    if len(kp) >= 2:
        kp = np.array(kp)
        ax.plot(kp[:, 0], kp[:, 1], f"k{ls}", lw=2)


def plot_deformation_maps(analysis, output_dir, cfg):
    """Generate figures 05 and 06. Skipped if TPS data is unavailable."""
    tps = analysis.get("tps")
    if tps is None:
        return

    source = tps["source_proc"]
    target = tps["target_proc"]
    transform = tps["transform"]
    valid = tps["valid_landmarks"]
    b = tps["grid_bounds"]
    n_grid = 20
    gx = np.linspace(b["x_min"], b["x_max"], n_grid)
    gy = np.linspace(b["y_min"], b["y_max"], n_grid)

    # ── Figure 05: side-by-side ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax_idx, (pts, title, ls) in enumerate([
        (source, "Actual hand (reference grid)", "-"),
        (target, "Proprioceptive representation (deformed grid)", "--"),
    ]):
        ax = axes[ax_idx]; ax.set_title(title, fontsize=14)
        if ax_idx == 0:
            for x in gx:
                ax.plot([x, x], [b["y_min"], b["y_max"]], "gray", lw=0.5, alpha=0.5)
            for y in gy:
                ax.plot([b["x_min"], b["x_max"]], [y, y], "gray", lw=0.5, alpha=0.5)
        else:
            for x in gx:
                lp = np.column_stack([np.full(100, x), np.linspace(b["y_min"], b["y_max"], 100)])
                d = transform(lp); ax.plot(d[:, 0], d[:, 1], "gray", lw=0.5, alpha=0.6)
            for y in gy:
                lp = np.column_stack([np.linspace(b["x_min"], b["x_max"], 100), np.full(100, y)])
                d = transform(lp); ax.plot(d[:, 0], d[:, 1], "gray", lw=0.5, alpha=0.6)

        for i, (finger, zone) in enumerate(valid):
            c = FINGER_COLORS[finger]
            ax.scatter(pts[i, 0], pts[i, 1], c=c, s=120,
                       edgecolors="k", zorder=10, linewidths=1.5)
            ax.annotate(f"{finger[0].upper()}{zone[1]}", (pts[i, 0], pts[i, 1]),
                        textcoords="offset points", xytext=(8, 5), fontsize=9)
        _draw_hand_skeleton(ax, pts, valid, ls)
        ax.invert_yaxis(); ax.set_aspect("equal")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(output_dir / "05_deformation_map_TPS.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 06: overlay ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Overlay: actual (●) vs proprioceptive (○) with TPS grid", fontsize=14)

    for x in gx:
        lp = np.column_stack([np.full(100, x), np.linspace(b["y_min"], b["y_max"], 100)])
        d = transform(lp); ax.plot(d[:, 0], d[:, 1], color="lightcoral", lw=0.4, alpha=0.5)
    for y in gy:
        lp = np.column_stack([np.linspace(b["x_min"], b["x_max"], 100), np.full(100, y)])
        d = transform(lp); ax.plot(d[:, 0], d[:, 1], color="lightcoral", lw=0.4, alpha=0.5)
    for x in gx:
        ax.plot([x, x], [b["y_min"], b["y_max"]], color="lightblue", lw=0.3, alpha=0.5)
    for y in gy:
        ax.plot([b["x_min"], b["x_max"]], [y, y], color="lightblue", lw=0.3, alpha=0.5)

    for i, (finger, zone) in enumerate(valid):
        c = FINGER_COLORS[finger]
        ax.scatter(source[i, 0], source[i, 1], c=c, s=100,
                   edgecolors="k", zorder=10, linewidths=1.5, marker="o")
        ax.scatter(target[i, 0], target[i, 1], c="white", s=100,
                   edgecolors=c, zorder=10, linewidths=2, marker="o")
        ax.annotate("", xy=(target[i, 0], target[i, 1]),
                    xytext=(source[i, 0], source[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5, alpha=0.7))

    _draw_hand_skeleton(ax, source, valid, "-")
    _draw_hand_skeleton(ax, target, valid, "--")

    # Legend
    kp_s, kp_t = [], []
    for finger in FINGER_ORDER:
        if (finger, "Z1") in valid:
            idx = valid.index((finger, "Z1"))
            kp_s.append(source[idx]); kp_t.append(target[idx])
    if len(kp_s) >= 2:
        ax.plot([], [], "k-", lw=2, label="Actual hand")
    if len(kp_t) >= 2:
        ax.plot([], [], "k--", lw=2, label="Estimated hand")
    ax.legend(fontsize=12, loc="upper left")
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    plt.tight_layout()
    plt.savefig(output_dir / "06_deformation_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()