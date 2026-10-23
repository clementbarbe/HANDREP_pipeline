"""Select a reference explicitly when multiple images have been annotated."""

import numpy as np
import pandas as pd
from pipeline.config.settings import LANDMARK_ORDER
from pipeline.registration.correction import apply_homography, undistort_points


def prepare_references(ref_df, raw_dir, calib=None, reference_image=None, max_motion_cm=None):
    images = sorted(ref_df.image.unique())
    if len(images) > 1 and reference_image is None:
        raise ValueError("Multiple reference images annotated: select --reference-image; references are not automatically averaged")
    selected_name = reference_image or images[0]
    if selected_name not in images:
        raise ValueError(f"Selected reference image not annotated: {selected_name}")
    selected = ref_df[ref_df.image == selected_name].copy()
    if set(zip(selected.finger, selected.zone)) != set(LANDMARK_ORDER):
        raise ValueError("Selected reference must contain all ten landmarks")
    baseline = selected.set_index(["finger", "zone"])[["x_real", "y_real"]]
    def metric(points):
        if calib is None:
            return np.asarray(points, float)
        pts = undistort_points(points, calib.get("camera_matrix"), calib.get("dist_coeffs"))
        return apply_homography(pts, calib["table_rectification_matrix"]) * calib["cm_per_px"]
    audit = []
    all_images = set(images)
    if raw_dir.exists():
        all_images.update(p.name for p in raw_dir.iterdir()
                          if p.is_file() and "ref" in p.stem.lower() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for name in sorted(all_images):
        annotated = ref_df[ref_df.image == name].set_index(["finger", "zone"])
        common = baseline.index.intersection(annotated.index)
        row = dict(image=name, selected=name == selected_name, n_landmarks=len(annotated),
                   motion_units="cm" if calib else "image_px", max_displacement=np.nan,
                   rms_displacement=np.nan, status="not_annotated")
        if len(common):
            delta = metric(annotated.loc[common, ["x_real", "y_real"]].to_numpy(float)) - metric(baseline.loc[common].to_numpy(float))
            distances = np.linalg.norm(delta, axis=1)
            row.update(max_displacement=float(distances.max()), rms_displacement=float(np.sqrt(np.mean(distances**2))),
                       status="selected_reference" if name == selected_name else "motion_measured_no_threshold")
            if name != selected_name and max_motion_cm is not None and calib is not None:
                row["status"] = "motion_exceeds_threshold" if distances.max() > max_motion_cm else "within_motion_threshold"
        audit.append(row)
    return selected, pd.DataFrame(audit)
