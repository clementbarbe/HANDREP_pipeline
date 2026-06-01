#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workflow.py — Pipeline orchestration.

This module contains *no* scientific logic. It calls the specialised
modules in the correct order and manages inter-step dependencies.
"""

import logging
import traceback

from pipeline.config.settings import default_config
from pipeline.config.paths import resolve_paths, figure_output_dir
from pipeline.utils.log import setup_logging
from pipeline.utils.console import (
    step, info, success, warning, error as console_error, divider,
)
from pipeline.io.discovery import find_prediction_files, parse_pair_from_filename
from pipeline.io.loader import load_predictions, load_reference
from pipeline.io.writer import save_trial_csv, save_summary_csv, save_biomech_csv
from pipeline.preprocessing.validation import validate_subject_session
from pipeline.quality_control.checks import (
    check_prediction_directory,
    check_reference_file,
    check_calibration_images,
    report_missing_references,
)
from pipeline.registration.calibration import compute_homography_from_images
from pipeline.registration.correction import apply_geometry_correction
from pipeline.transforms.merging import merge_predictions_and_references
from pipeline.transforms.errors import (
    compute_errors,
    compute_reference_scale,
    add_normalized_errors,
)
from pipeline.analysis.summary import (
    compute_summary,
    compute_per_landmark_stats,
    compute_per_finger_stats,
    extract_landmark_positions,
)
from pipeline.analysis.biomechanics import (
    compute_biomechanical_metrics,
    compute_finger_lengths,
    compute_knuckle_widths,
    compute_shape_indices,
)
from pipeline.analysis.tps import compute_tps_analysis
from pipeline.export.reports import export_all_csvs


# ────────────────────────────────────────────────────────────
# Calibration (computed once for the whole run)
# ────────────────────────────────────────────────────────────

def _load_calibration(paths, cfg):
    """Compute or skip checkerboard calibration."""
    if not cfg["checkerboard"]:
        info("Checkerboard disabled → mathematical correction only")
        return None

    table = paths["table_image"]
    plateau = paths["plateau_image"]

    ok, msg = check_calibration_images(table, plateau)
    if not ok:
        warning(msg)
        info("Falling back to mathematical correction")
        return None

    try:
        calib = compute_homography_from_images(
            table, plateau, cfg["checkerboard_size"],
        )
        success(
            f"Homography computed  "
            f"({calib['n_inliers']}/{calib['n_total']} inliers, "
            f"reproj = {calib['mean_reprojection_error']:.2f} px)"
        )

        # Optional verification figure
        if not cfg["skip_figures"]:
            from pipeline.visualization.calibration_check import (
                plot_calibration_check,
            )
            fig_path = paths["processed_dir"] / "calibration_check.png"
            plot_calibration_check(calib, table, plateau, fig_path)
            info(f"Calibration figure → {fig_path.name}")

        return calib

    except Exception as exc:
        warning(f"Calibration failed: {exc}")
        info("Falling back to mathematical correction")
        return None


# ────────────────────────────────────────────────────────────
# Processing a single subject × session pair
# ────────────────────────────────────────────────────────────

def _process_pair(subject, session, pred_path, ref_path, calib, paths, cfg):
    """Run the full pipeline for one (subject, session) pair."""
    logger = logging.getLogger(__name__)

    # 1 — Load
    pred_df = load_predictions(pred_path)
    ref_df = load_reference(ref_path)
    info(f"Loaded {len(pred_df)} predictions, {len(ref_df)} reference landmarks")

    # 2 — Validate
    validate_subject_session(pred_df, ref_df, subject, session)

    # 3 — Merge
    merged = merge_predictions_and_references(pred_df, ref_df)
    n_missing = report_missing_references(merged)
    if n_missing:
        warning(f"{n_missing} row(s) without reference data")

    # 4 — Geometric correction
    merged = apply_geometry_correction(merged, calib, cfg)
    info(f"Correction method: {merged['correction_method'].iloc[0]}")

    # 5 — Errors
    merged = compute_errors(merged)
    scale = compute_reference_scale(merged)
    merged = add_normalized_errors(merged, scale)
    info(
        f"Mean Euclidean error: {merged['error_eucl'].mean():.1f} px "
        f"(±{merged['error_eucl'].std():.1f})"
    )

    # 6 — Analysis
    summary = compute_summary(merged)
    biomech = compute_biomechanical_metrics(merged)
    positions = extract_landmark_positions(merged)
    stats_lm = compute_per_landmark_stats(merged)
    stats_fg = compute_per_finger_stats(merged)
    finger_lengths = compute_finger_lengths(
        positions["real"], positions["estimated_corr"],
    )
    knuckle_widths = compute_knuckle_widths(
        positions["real"], positions["estimated_corr"],
    )
    shape_indices = compute_shape_indices(
        positions["real"], positions["estimated_corr"],
    )
    tps_results = compute_tps_analysis(
        positions["real"], positions["estimated_corr"],
    )

    # Package for downstream modules
    analysis = dict(
        df=merged,
        subject=subject,
        session=session,
        summary=summary,
        biomech=biomech,
        positions=positions,
        stats_landmark=stats_lm,
        stats_finger=stats_fg,
        finger_lengths=finger_lengths,
        knuckle_widths=knuckle_widths,
        shape_indices=shape_indices,
        tps=tps_results,
    )

    # 7 — Export
    out_dir = paths["subject_dir_fn"](subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_all_csvs(analysis, out_dir, subject, session)
    info("CSV exports saved")

    # 8 — Figures
    if not cfg["skip_figures"]:
        fig_dir = figure_output_dir(paths["processed_dir"], subject, session)
        fig_dir.mkdir(parents=True, exist_ok=True)
        from pipeline.visualization import generate_all_figures
        generate_all_figures(analysis, fig_dir, cfg)
        info(f"Figures → {fig_dir.relative_to(paths['processed_dir'])}")

    logger.info(
        f"DONE | {subject} {session} | "
        f"trials={len(merged)} | "
        f"method={merged['correction_method'].iloc[0]}"
    )


# ────────────────────────────────────────────────────────────
# Main pipeline loop
# ────────────────────────────────────────────────────────────

def run_pipeline(cfg=None):
    """
    Execute the full multi-subject pipeline.

    Parameters
    ----------
    cfg : dict or None
        Configuration dictionary. Uses ``default_config()`` if *None*.

    Returns
    -------
    int
        Number of successfully processed pairs.
    """
    cfg = cfg or default_config()
    paths = resolve_paths(cfg["data_dir"])
    setup_logging(paths["processed_dir"])
    logger = logging.getLogger(__name__)

    total_steps = 4
    n_processed = 0

    # ── Step 1 : discover ──────────────────────────────────
    step(1, total_steps, "Discovering prediction files…")
    ok, msg = check_prediction_directory(paths["nn_dir"])
    if not ok:
        console_error(msg)
        return 0

    pred_files = find_prediction_files(paths["nn_dir"])
    if not pred_files:
        warning("No prediction CSV files found")
        return 0
    info(f"Found {len(pred_files)} prediction file(s)")

    # ── Step 2 : calibration ───────────────────────────────
    step(2, total_steps, "Computing geometric calibration…")
    calib = _load_calibration(paths, cfg)

    # ── Step 3 : processing loop ───────────────────────────
    step(3, total_steps, "Processing subject × session pairs…")

    for pred_path in pred_files:
        try:
            subject, session = parse_pair_from_filename(pred_path.name)
        except ValueError as exc:
            warning(f"Skipping {pred_path.name}: {exc}")
            continue

        # Optional subject filter
        if cfg["subjects_filter"] and subject not in cfg["subjects_filter"]:
            info(f"Skipping {subject} (not in filter)")
            continue

        # Incremental: skip if output exists
        from pipeline.config.paths import trials_output_path
        out = trials_output_path(paths["processed_dir"], subject, session)
        if out.exists() and not cfg["force"]:
            info(f"SKIP {subject} / {session} — output already exists")
            continue

        # Reference file
        ref_path = paths["ref_dir"] / f"{subject}_{session}_ref.csv"
        ok, msg = check_reference_file(ref_path)
        if not ok:
            warning(f"{subject} / {session}: {msg}")
            continue

        divider(char="─", width=50)
        info(f"▶ {subject} / {session}")

        try:
            _process_pair(
                subject, session, pred_path, ref_path, calib, paths, cfg,
            )
            n_processed += 1
            success(f"{subject} / {session} — done")
        except Exception as exc:
            console_error(f"{subject} / {session}: {exc}")
            logger.error(traceback.format_exc())

    # ── Step 4 : summary ───────────────────────────────────
    step(4, total_steps, "Final summary")
    info(f"Pairs processed successfully: {n_processed}/{len(pred_files)}")

    return n_processed