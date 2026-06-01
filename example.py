#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example.py — Minimal usage example.

Runs the pipeline with default settings on the data/ directory.
"""

from pipeline.config.settings import default_config
from workflow import run_pipeline

if __name__ == "__main__":
    cfg = default_config()
    cfg["force"] = True          # reprocess everything
    cfg["checkerboard"] = True   # use checkerboard if available

    n = run_pipeline(cfg)
    print(f"\nDone — {n} pair(s) processed.")