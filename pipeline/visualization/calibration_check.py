"""Calibration verification figure."""

import matplotlib.pyplot as plt
import cv2

from pipeline.registration.correction import apply_homography


def plot_calibration_check(calib, table_path, plateau_path, output_path):
    """
    Show detected corners on both images and reprojection quality.
    """
    timg = cv2.cvtColor(cv2.imread(str(table_path)), cv2.COLOR_BGR2RGB)
    pimg = cv2.cvtColor(cv2.imread(str(plateau_path)), cv2.COLOR_BGR2RGB)

    ct = calib["corners_table"]
    cp = calib["corners_plateau"]
    cp_repr = apply_homography(cp, calib["homography_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    ax.imshow(pimg)
    ax.scatter(cp[:, 0], cp[:, 1], c="lime", s=30, marker="+", linewidths=1.5,
               label="Detected corners")
    ax.set_title(f"plateau.png ({len(cp)} corners)"); ax.legend(); ax.axis("off")

    ax = axes[1]
    ax.imshow(timg)
    ax.scatter(ct[:, 0], ct[:, 1], c="lime", s=30, marker="+", linewidths=1.5,
               label="Detected (table)")
    ax.scatter(cp_repr[:, 0], cp_repr[:, 1], c="red", s=30, marker="x",
               linewidths=1.5, label="Reprojected (plateau)")
    ax.set_title(
        f"table.png — reprojection "
        f"(mean err = {calib['mean_reprojection_error']:.2f} px)"
    )
    ax.legend(); ax.axis("off")

    plt.suptitle("Checkerboard calibration verification",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()