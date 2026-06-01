"""
File + console logging setup.
"""

import logging
from pathlib import Path


_CONFIGURED = False


def setup_logging(processed_dir, level=logging.INFO):
    """
    Configure the root logger once.

    A ``pipeline.log`` file is created inside *processed_dir*.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    log_file = processed_dir / "pipeline.log"

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    # Silence noisy libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)