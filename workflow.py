"""Orchestrate validated, reproducible within-session analyses."""

import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import cv2

from pipeline.config.settings import default_config
from pipeline.config.paths import resolve_paths, figure_output_dir
from pipeline.utils.log import setup_logging
from pipeline.utils.console import info, success, warning, error as console_error
from pipeline.io.discovery import find_prediction_files, parse_pair_from_filename
from pipeline.io.loader import load_predictions, load_reference, load_trial_design
from pipeline.preprocessing.validation import validate_subject_session, flag_prediction_quality
from pipeline.quality_control.references import prepare_references
from pipeline.quality_control.detection import evaluate_detection
from pipeline.quality_control.images import check_source_images
from pipeline.registration.calibration import compute_homography_from_images
from pipeline.registration.correction import apply_geometry_correction
from pipeline.transforms.merging import merge_predictions_and_references
from pipeline.transforms.errors import compute_errors, compute_reference_scale, add_normalized_errors, add_cm_errors
from pipeline.analysis.summary import compute_summary, compute_per_landmark_stats, compute_per_finger_stats, extract_landmark_positions
from pipeline.analysis.biomechanics import compute_biomechanical_metrics, compute_finger_lengths, compute_knuckle_widths, compute_shape_indices
from pipeline.analysis.tps import compute_tps_analysis
from pipeline.analysis.coupling import detect_couples, coupling_summary, compute_coupled_finger_lengths, compute_uncoupled_finger_lengths, individual_couple_distances, compare_coupling
from pipeline.analysis.spatial import add_anatomical_coordinates, bias_precision_summary, decompose_geometry
from pipeline.analysis.morphometry import add_morphometry_intervals, hand_width
from pipeline.analysis.temporal import compute_temporal_models
from pipeline.analysis.cohort import summarize_cohort
from pipeline.export.reports import export_all_csvs

ANALYSIS_VERSION = 2


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fingerprint(paths, cfg):
    root = Path(__file__).parent
    code = sorted((root / "pipeline").rglob("*.py")) + [root / "workflow.py", root / "run.py"]
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    source_hash = hashlib.sha256(b"".join(p.read_bytes() for p in code)).hexdigest()
    configuration = _jsonable({k: v for k, v in cfg.items() if k != "force"})
    payload = dict(inputs=inputs, source_sha256=source_hash, config=configuration)
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def _load_calibration(paths, cfg):
    if not cfg["checkerboard"]:
        if not cfg["allow_fallback"]:
            raise ValueError("--no-checkerboard requires --allow-fallback for exploratory pixel-only analysis")
        return None
    try:
        calib = compute_homography_from_images(
            paths["table_image"], paths["plateau_image"], cfg["checkerboard_size"], cfg["square_size_cm"],
            max_error_px=cfg["calibration_max_error_px"], min_inlier_ratio=cfg["calibration_min_inlier_ratio"],
            camera_path=paths["calib_dir"] / "camera.npz")
        calib["scope"] = paths.get("calibration_scope", "shared_across_sessions")
        success(f"Metric rectification: held-out {calib['heldout_error_px']:.3f} px; {calib['heldout_metric_error_cm']:.4f} cm")
        if not cfg["skip_figures"]:
            from pipeline.visualization.calibration_check import plot_calibration_check
            plot_calibration_check(calib, paths["table_image"], paths["plateau_image"], paths.get("calibration_figure", paths["processed_dir"] / "calibration_check.png"))
        return calib
    except (ValueError, FileNotFoundError, ImportError) as exc:
        if not cfg["allow_fallback"]:
            raise ValueError(f"Calibration unavailable or rejected: {exc}") from exc
        warning(f"Exploratory mathematical fallback; no cm conversion: {exc}")
        return None


def _process_pair(subject, session, pred_path, ref_path, calib, paths, cfg):
    cm_per_px = calib["cm_per_px"] if calib else None
    design_path = paths["design_dir"] / f"{subject}_{session}_design.csv"
    manual_path = paths["data_dir"] / "validation" / f"{subject}_{session}_pointer.csv"
    prediction_provenance = pred_path.with_name(pred_path.stem.replace("_predictions", "_prediction_provenance") + ".json")
    if prediction_provenance.is_file():
        provenance = json.loads(prediction_provenance.read_text(encoding="utf-8"))
        if provenance.get("prediction_csv_sha256") != hashlib.sha256(pred_path.read_bytes()).hexdigest():
            raise ValueError("Prediction CSV no longer matches its generation provenance")
    pred_df, ref_df = load_predictions(pred_path), load_reference(ref_path)
    validate_subject_session(pred_df, ref_df, subject, session)
    pred_df["trial"] = pd.to_numeric(pred_df.trial).astype(int)
    pred_df = load_trial_design(pred_df, design_path)
    width, height = (calib["image_width"], calib["image_height"]) if calib else (cfg["image_width"], cfg["image_height"])
    pred_df = flag_prediction_quality(pred_df, width, height)
    raw_dir = paths["raw_dir"] / f"sub-{subject}" / session
    pred_df = check_source_images(pred_df, raw_dir, width, height)
    # Pair on the complete acquisition sequence before any numerical exclusion.
    pred_df = detect_couples(pred_df)
    selected, reference_audit = prepare_references(
        ref_df, paths["raw_dir"] / f"sub-{subject}" / session, calib,
        cfg["reference_image"], cfg["max_reference_motion_cm"])
    if not (selected.x_real.between(0, width - 1) & selected.y_real.between(0, height - 1)).all():
        raise ValueError("Anatomical reference coordinates are outside the acquisition image")
    if reference_audit.status.eq("motion_exceeds_threshold").any():
        raise ValueError("Reference motion exceeds the configured threshold; inspect reference annotations")
    merged = merge_predictions_and_references(pred_df, selected)
    if merged[["x_real", "y_real"]].isna().any().any():
        raise ValueError("Missing anatomical reference; analysis stopped")
    merged = apply_geometry_correction(merged, calib, cfg)
    usable_counts = merged.qc_valid.groupby(merged.couple_id).transform("sum")
    merged["pair_usable"] = merged.is_coupled & merged.pair_complete & usable_counts.eq(2)
    merged = compute_errors(merged)
    merged = add_normalized_errors(merged, compute_reference_scale(merged))
    merged = add_cm_errors(merged, cm_per_px)
    merged, axes = add_anatomical_coordinates(merged)
    merged, temporal = compute_temporal_models(merged)
    if not merged.qc_valid.any():
        raise ValueError("No valid predictions remain")
    valid = merged[merged.qc_valid].copy()
    positions = extract_landmark_positions(valid)
    options = dict(n_boot=cfg["bootstrap_samples"], seed=cfg["random_seed"], min_n=cfg["min_trials_ci"])
    lengths = compute_finger_lengths(positions["real"], positions["estimated_corr"], cm_per_px)
    widths = compute_knuckle_widths(positions["real"], positions["estimated_corr"], cm_per_px)
    total_width = hand_width(valid, positions["real"], positions["estimated_corr"], cm_per_px)
    # Include overall width in the same bootstrap as adjacent knuckle widths.
    widths = pd.concat([widths, total_width], ignore_index=True)
    lengths, widths, shape = add_morphometry_intervals(
        lengths, widths, compute_shape_indices(positions["real"], positions["estimated_corr"]),
        merged, cm_per_px, **options)
    coupled = compute_coupled_finger_lengths(merged, positions["real"], cm_per_px, **options)
    uncoupled = compute_uncoupled_finger_lengths(merged, positions["real"], cm_per_px, **options)
    directions = pd.concat([compute_coupled_finger_lengths(merged, positions["real"], cm_per_px, direction=d, **options)
                            for d in ("Z1→Z2", "Z2→Z1")], ignore_index=True)
    comparison = compare_coupling(merged, positions["real"], cm_per_px, **options)
    for table in (comparison, coupled, uncoupled, directions):
        table["subject"], table["session"] = subject, session
        table["correction_method"] = merged.correction_method.iloc[0]
    pointer_trials, pointer_summary = evaluate_detection(pred_df, manual_path)
    warnings = []
    if calib is None:
        warnings.append("Geometry is approximate; metric units unavailable")
    elif not calib["lens_distortion_corrected"]:
        warnings.append("Lens distortion not independently calibrated")
    if calib and calib["scope"] == "shared_across_sessions":
        warnings.append("Calibration shared across sessions; camera/geometry invariance must be verified")
    if not merged.source_image_available.all():
        warnings.append("Some source images are unavailable; their resolution cannot be verified")
    missing_trials = sorted(set(range(1, int(merged.trial.max()) + 1)) - set(merged.trial))
    if missing_trials:
        warnings.append(f"Missing acquisition trial numbers before last observed trial: {missing_trials}")
    warnings.append("Physical plane alignment and checkerboard square size require independent verification")
    if reference_audit.status.eq("not_annotated").any():
        warnings.append("Additional reference images exist without anatomical annotations; hand motion unassessed")
    if reference_audit.selected.sum() == len(reference_audit):
        warnings.append("Only one reference available; hand motion unassessed")
    if not reference_audit.status.isin(["within_motion_threshold", "motion_exceeds_threshold"]).any():
        warnings.append("Reference motion acceptance threshold not assessed")
    if merged.coupling_source.eq("inferred_adjacency_exploratory").any():
        warnings.append("Coupling inferred from adjacency; planned experimental conditions unknown")
    if pointer_summary.empty:
        warnings.append("Automatic pointer detection not independently evaluated")
    if not prediction_provenance.is_file():
        warnings.append("Prediction-generation provenance unavailable")
    if (~merged.qc_valid).any():
        warnings.append(f"{int((~merged.qc_valid).sum())} invalid predictions retained in trial export, excluded from numerical analyses")
    warnings.append("Bootstrap CIs are exploratory, conditional on fixed references/calibration and do not model arbitrary temporal dependence")
    analysis = dict(df=merged, subject=subject, session=session, cm_per_px=cm_per_px,
                    summary=compute_summary(merged), stats_landmark=compute_per_landmark_stats(merged),
                    stats_finger=compute_per_finger_stats(merged), positions=positions,
                    biomech=compute_biomechanical_metrics(valid, cm_per_px), finger_lengths=lengths,
                    knuckle_widths=widths, shape_indices=shape,
                    tps=compute_tps_analysis(positions["real"], positions["estimated_corr"]),
                    finger_lengths_coupled=coupled, finger_lengths_uncoupled=uncoupled,
                    finger_lengths_by_direction=directions, coupling_comparison=comparison,
                    couple_distances=individual_couple_distances(merged, cm_per_px),
                    coupling_stats=coupling_summary(merged), bias_precision=bias_precision_summary(merged),
                    geometry_decomposition=decompose_geometry(positions["real"], positions["estimated_corr"]),
                    temporal_models=temporal, reference_audit=reference_audit,
                    pointer_validation_trials=pointer_trials, pointer_validation_summary=pointer_summary)
    source_images = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if raw_dir.is_dir() else []
    fingerprint = _fingerprint([pred_path, prediction_provenance, ref_path, design_path, manual_path, paths["table_image"],
                                paths["plateau_image"], paths["calib_dir"] / "camera.npz"] + source_images, cfg)
    manifest = dict(analysis_version=ANALYSIS_VERSION, status="in_progress", subject=subject, session=session,
                    **fingerprint, n_trials=len(merged), n_valid_trials=int(merged.qc_valid.sum()),
                    missing_trial_numbers=missing_trials,
                    units="rectified_px_and_cm" if calib else "approximate_image_px",
                    anatomical_axes=axes, calibration=calib, warnings=warnings,
                    uncertainty_scope="within-session sampling only; no calibration or annotation uncertainty",
                    versions=dict(python=platform.python_version(), opencv=cv2.__version__,
                                  **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")}))
    out_dir = paths["subject_dir_fn"](subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{subject}_{session}_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, allow_nan=False), encoding="utf-8")
    export_all_csvs(analysis, out_dir, subject, session)
    if not cfg["skip_figures"]:
        from pipeline.visualization import generate_all_figures
        fig_dir = figure_output_dir(paths["processed_dir"], subject, session)
        generate_all_figures(analysis, fig_dir, cfg)
    manifest["status"] = "complete"
    manifest["outputs"] = [str(p.relative_to(paths["processed_dir"])) for p in out_dir.rglob(f"{subject}_{session}*.csv")]
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / f"{subject}_{session}_quality_report.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")
    for message in warnings:
        warning(message)
    info(f"{subject}/{session}: {int(merged.qc_valid.sum())}/{len(merged)} valid trials")
    return analysis


def run_pipeline(cfg=None):
    config = default_config()
    config.update(cfg or {})
    cfg = config
    if cfg["bootstrap_samples"] < 0 or cfg["min_trials_ci"] < 2:
        raise ValueError("Invalid bootstrap configuration")
    if cfg["max_reference_motion_cm"] is not None and cfg["max_reference_motion_cm"] < 0:
        raise ValueError("Reference motion threshold cannot be negative")
    paths = resolve_paths(cfg["data_dir"], cfg["output_dir"], cfg["prediction_dir"])
    setup_logging(paths["processed_dir"])
    pred_files = find_prediction_files(paths["nn_dir"])
    if cfg["subjects_filter"]:
        pred_files = [p for p in pred_files if parse_pair_from_filename(p.name)[0] in cfg["subjects_filter"]]
    if not pred_files:
        warning("No prediction files selected")
        return 0
    calibration_cache = {}
    comparisons, processed, failures = [], 0, []
    for pred_path in pred_files:
        try:
            subject, session = parse_pair_from_filename(pred_path.name)
            session_paths = paths.copy()
            session_calibration_dir = paths["calib_dir"] / f"sub-{subject}" / session
            if session_calibration_dir.is_dir():
                session_paths.update(calib_dir=session_calibration_dir,
                                     table_image=session_calibration_dir / "table.png",
                                     plateau_image=session_calibration_dir / "plateau.png",
                                     calibration_scope="session_specific",
                                     calibration_figure=paths["processed_dir"] / f"calibration_{subject}_{session}.png")
            calibration_key = str(session_paths["calib_dir"])
            if calibration_key not in calibration_cache:
                calibration_cache[calibration_key] = _load_calibration(session_paths, cfg)
            calib = calibration_cache[calibration_key]
            ref_path = paths["ref_dir"] / f"{subject}_{session}_ref.csv"
            if not ref_path.is_file():
                raise ValueError(f"Missing reference: {ref_path}")
            # Recalculate by default: old outputs and partial runs cannot suppress corrections.
            # --force is retained as a compatible CLI option; no unsafe existence-only cache.
            analysis = _process_pair(subject, session, pred_path, ref_path, calib, session_paths, cfg)
            if not analysis["coupling_comparison"].empty:
                comparisons.append(analysis["coupling_comparison"])
            processed += 1
        except Exception as exc:
            console_error(f"{pred_path.name}: {exc}")
            logging.getLogger(__name__).error(traceback.format_exc())
            failures.append(dict(file=str(pred_path), error=str(exc)))
    cohort = summarize_cohort(comparisons, cfg["bootstrap_samples"], cfg["random_seed"])
    cohort.to_csv(paths["processed_dir"] / "cohort_coupling_comparison.csv", index=False)
    report = dict(analysis_version=ANALYSIS_VERSION, n_processed=processed, n_selected=len(pred_files), failures=failures)
    (paths["processed_dir"] / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} session(s) failed; inspect run_report.json. {processed} session(s) completed")
    return processed
