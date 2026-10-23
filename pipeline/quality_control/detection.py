"""Optional independent manual verification of pointer localization."""

import numpy as np
import pandas as pd


def evaluate_detection(predictions, annotations_path):
    """Manual CSV: filename,x_manual,y_manual (one consensus annotation/image).

This assesses detection only, in original image pixels. It does not validate
participant behaviour, physical plane registration, or train/test independence.
"""
    if not annotations_path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    annotations = pd.read_csv(annotations_path, sep=None, engine="python")
    if not {"filename", "x_manual", "y_manual"}.issubset(annotations.columns):
        raise ValueError("Detection validation requires filename,x_manual,y_manual")
    if annotations.filename.duplicated().any():
        raise ValueError("Duplicate manual pointer annotations")
    for c in ("x_manual", "y_manual"):
        annotations[c] = pd.to_numeric(annotations[c], errors="raise")
    if not np.isfinite(annotations[["x_manual", "y_manual"]]).all().all():
        raise ValueError("Manual pointer annotations must be finite")
    unknown = set(annotations.filename) - set(predictions.filename)
    if unknown:
        raise ValueError(f"Unknown filenames in pointer validation: {sorted(unknown)[:3]}")
    joined = predictions.merge(annotations, on="filename", validate="one_to_one")
    joined["detection_dx_px"] = joined.x_est - joined.x_manual
    joined["detection_dy_px"] = joined.y_est - joined.y_manual
    joined["detection_error_px"] = np.hypot(joined.detection_dx_px, joined.detection_dy_px)
    rows = []
    for key, g in joined.groupby(["finger", "zone"]):
        error = g.detection_error_px.dropna()
        rows.append(dict(finger=key[0], zone=key[1], n_annotated=len(g), n_valid_predictions=len(error),
                         median_error_px=error.median(), p95_error_px=error.quantile(.95),
                         bias_x_px=g.detection_dx_px.mean(), bias_y_px=g.detection_dy_px.mean(),
                         training_independence="not_verifiable_from_annotations"))
    return joined, pd.DataFrame(rows)
