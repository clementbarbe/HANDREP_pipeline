"""
Filesystem and data-integrity checks.

Each function returns ``(ok: bool, message: str)``.
"""

from pathlib import Path


def check_prediction_directory(nn_dir):
    """Verify that the predictions directory exists."""
    if not Path(nn_dir).exists():
        return False, f"Prediction directory not found: {nn_dir}"
    return True, "OK"


def check_reference_file(ref_path):
    """Verify that a specific reference CSV exists."""
    if not Path(ref_path).exists():
        return False, f"Reference file missing: {Path(ref_path).name}"
    return True, "OK"


def check_calibration_images(table_path, plateau_path):
    """Verify that both calibration images exist."""
    t, p = Path(table_path), Path(plateau_path)
    missing = []
    if not t.exists():
        missing.append(t.name)
    if not p.exists():
        missing.append(p.name)
    if missing:
        return False, f"Calibration image(s) missing: {', '.join(missing)}"
    return True, "OK"


def check_weights(weights_dir):
    """Verify that model weight files are present."""
    w = Path(weights_dir)
    coarse = w / "hrnet_weights.pth"
    refine = w / "refine_weights.pth"
    missing = []
    if not coarse.exists():
        missing.append(coarse.name)
    if not refine.exists():
        missing.append(refine.name)
    if missing:
        return False, f"Weight file(s) missing in {w}: {', '.join(missing)}"
    return True, "OK"


def report_missing_references(merged_df):
    """Return the number of rows with no reference match."""
    return int(merged_df["x_real"].isna().sum())