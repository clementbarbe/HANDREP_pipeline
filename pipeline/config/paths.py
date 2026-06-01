"""
Path resolution — every filesystem location derived from a single root.
"""

from pathlib import Path


def resolve_paths(data_dir="data"):
    """
    Build the full path dictionary from the data root.

    Returns
    -------
    dict[str, Path]
    """
    base = Path(data_dir).resolve()
    processed = base / "processed"

    return {
        "data_dir":      base,
        "raw_dir":       base / "raw",
        "ref_dir":       base / "ref",
        "calib_dir":     base / "calibration",
        "nn_dir":        base / "nn_outputs",
        "processed_dir": processed,
        "weights_dir":   Path("weights").resolve(),
        "table_image":   base / "calibration" / "table.png",
        "plateau_image": base / "calibration" / "plateau.png",
        # Helper returning a subject-level output dir
        "subject_dir_fn": lambda subj: processed / f"sub-{subj}",
    }


def trials_output_path(processed_dir, subject, session):
    """Canonical path for the per-trial CSV."""
    return processed_dir / f"sub-{subject}" / f"{subject}_{session}_trials.csv"


def summary_output_path(processed_dir, subject, session):
    return processed_dir / f"sub-{subject}" / f"{subject}_{session}_summary.csv"


def biomech_output_path(processed_dir, subject, session):
    return processed_dir / f"sub-{subject}" / f"{subject}_{session}_biomech.csv"


def figure_output_dir(processed_dir, subject, session):
    return processed_dir / f"sub-{subject}" / "figures" / f"{subject}_{session}"