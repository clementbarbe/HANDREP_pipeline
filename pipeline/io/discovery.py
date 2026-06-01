"""
Filesystem scanning and filename parsing.
"""

import re
from pathlib import Path

from pipeline.transforms.normalization import normalize_subject, normalize_session


def find_prediction_files(nn_dir):
    """Return a sorted list of ``*_predictions.csv`` paths."""
    return sorted(Path(nn_dir).glob("*_predictions.csv"))


def parse_pair_from_filename(filename):
    """
    Extract *(subject, session)* from a prediction filename.

    Expected pattern: ``S1_ses-1_predictions.csv``

    Returns
    -------
    tuple[str, str]
        ``(subject, session)`` normalised.

    Raises
    ------
    ValueError
        If the filename cannot be parsed.
    """
    name = Path(filename).stem
    name = name.replace("_predictions", "")
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse subject/session from '{filename}'")
    return normalize_subject(parts[0]), normalize_session(parts[1])


def parse_image_filename(name):
    """
    Parse an acquisition image filename.

    Expected: ``S1_T10_middle_Z1.jpg``

    Returns
    -------
    tuple
        ``(subject, trial, finger, zone)``
    """
    pattern = r"(S\d+)_T(\d+)_(\w+)_(Z\d)"
    m = re.match(pattern, name)
    if m:
        subject, trial, finger, zone = m.groups()
        return subject, int(trial), finger, zone
    return "UNKNOWN", -1, "UNK", "UNK"


def parse_subject_session_from_path(folder_path):
    """
    Extract subject / session from a BIDS-like directory tree.

    Looks for parts matching ``sub-*`` and ``ses-*``.
    """
    subject, session = None, None
    for part in Path(folder_path).parts:
        if part.startswith("sub-"):
            subject = part.replace("sub-", "")
        elif part.startswith("ses-"):
            session = part.replace("ses-", "")
    return normalize_subject(subject or "UNKNOWN"), normalize_session(session or "1")