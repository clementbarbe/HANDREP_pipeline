#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — CLI entry point for the proprioception analysis pipeline.

Usage
-----
    python run.py
    python run.py --data-dir /path/to/data --force
    python run.py --no-checkerboard --subjects S1 S3
    python run.py --skip-figures
"""

import argparse
import sys
import time

from pipeline.config.settings import default_config, FINGER_ORDER
from pipeline.utils.console import header, success, warning, divider


def parse_args():
    """Parse command-line arguments and return a Namespace."""
    p = argparse.ArgumentParser(
        prog="propriohand",
        description="Automated hand proprioception analysis pipeline.",
    )
    p.add_argument(
        "--data-dir", default="data",
        help="Root data directory (default: data)",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Compatibility option; analyses are always recalculated",
    )
    p.add_argument(
        "--no-checkerboard", action="store_true", default=False,
        help="Skip checkerboard calibration, use mathematical correction",
    )
    p.add_argument(
        "--subjects", nargs="*", default=None,
        help="Process only these subjects (e.g. S1 S2)",
    )
    p.add_argument(
        "--skip-figures", action="store_true", default=False,
        help="Skip figure generation (faster processing)",
    )
    p.add_argument("--output-dir", help="Output directory; allows keeping earlier analyses")
    p.add_argument("--prediction-dir", help="Read prediction CSVs from this directory")
    p.add_argument("--allow-fallback", action="store_true", help="Explicitly allow approximate pixel-only analysis without calibration")
    p.add_argument("--bootstrap-samples", type=int, default=2000, help="Within-session bootstrap replicates (default: 2000)")
    p.add_argument("--seed", type=int, default=42, help="Reproducible bootstrap seed")
    p.add_argument("--square-size-cm", type=float, default=None, help="Measured checkerboard square side in cm")
    p.add_argument("--reference-image", help="Select an annotated reference when several are available")
    p.add_argument("--max-reference-motion-cm", type=float, help="Reject sessions exceeding this measured reference displacement")
    return p.parse_args()


def main():
    """Build configuration from CLI args and launch the pipeline."""
    args = parse_args()

    header("PROPRIOHAND — Automated Hand Proprioception Pipeline")

    cfg = default_config()
    cfg["data_dir"] = args.data_dir
    cfg["force"] = args.force
    cfg["checkerboard"] = not args.no_checkerboard
    cfg["subjects_filter"] = args.subjects
    cfg["skip_figures"] = args.skip_figures
    cfg["output_dir"] = args.output_dir
    cfg["prediction_dir"] = args.prediction_dir
    cfg["allow_fallback"] = args.allow_fallback
    cfg["bootstrap_samples"] = args.bootstrap_samples
    cfg["random_seed"] = args.seed
    cfg["reference_image"] = args.reference_image
    cfg["max_reference_motion_cm"] = args.max_reference_motion_cm
    if args.square_size_cm is not None:
        cfg["square_size_cm"] = args.square_size_cm

    # Lazy import so help text displays instantly
    from workflow import run_pipeline

    t0 = time.time()
    n_processed = run_pipeline(cfg)
    elapsed = time.time() - t0

    divider()
    if n_processed > 0:
        success(f"Pipeline complete — {n_processed} pair(s) processed in {elapsed:.1f}s")
    else:
        warning("Pipeline finished — nothing was processed")
    divider()


if __name__ == "__main__":
    main()
