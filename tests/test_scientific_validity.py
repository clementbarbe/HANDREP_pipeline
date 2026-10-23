"""Counterexamples and invariance checks for scientific comparisons.

Run with: python -m unittest discover -s tests -v
"""

import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.analysis.tps import procrustes_align, fit_tps
from pipeline.analysis.coupling import detect_couples, compute_coupled_finger_lengths, compute_uncoupled_finger_lengths
from pipeline.analysis.uncertainty import centroid_distance_samples
from pipeline.analysis.spatial import add_anatomical_coordinates, bias_precision_summary
from pipeline.analysis.temporal import compute_temporal_models
from pipeline.analysis.cohort import summarize_cohort
from pipeline.preprocessing.validation import flag_prediction_quality, validate_subject_session
from pipeline.registration.calibration import compute_homography_from_images
from pipeline.registration.correction import apply_homography, apply_geometry_correction
from pipeline.transforms.errors import compute_errors, add_cm_errors, compute_reference_scale
from pipeline.quality_control.references import prepare_references
from pipeline.io.loader import load_trial_design
from pipeline.io.writer import save_trial_csv
from pipeline.quality_control.images import check_source_images


def sequence(trials, fingers=None, zones=None, subjects=None):
    n = len(trials)
    return pd.DataFrame(dict(trial=trials, subject=subjects or ["S1"] * n,
                             session=["ses-1"] * n, finger=fingers or ["index"] * n,
                             zone=zones or ["Z1", "Z2"] * (n // 2)))


def hand_data():
    rows = []
    for i, finger in enumerate(["thumb", "index", "middle", "ring", "little"]):
        for zone, y in [("Z1", 0.), ("Z2", 10.)]:
            for j in range(4):
                rows.append(dict(subject="S1", session="ses-1", finger=finger, zone=zone,
                                 trial=len(rows) + 1, filename=f"trial{len(rows)}.jpg", image="REF1.jpg",
                                 landmark=f"{finger}_{zone}", x_real=float(i), y_real=y,
                                 x_est_corr=i + 1. + (-1)**j, y_est_corr=y + 2., cm_per_px=.1))
    return pd.DataFrame(rows)


class AlignmentTests(unittest.TestCase):
    def test_rigid_transform_has_zero_residual(self):
        source = np.array([[0., 0.], [1., 0.], [2., 3.], [-1., 4.]])
        theta = np.radians(30)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        target = source @ rot.T + [15, -3]
        src, aligned, rotation = procrustes_align(source, target)
        np.testing.assert_allclose(aligned, src, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_alignment_preserves_size_difference(self):
        source = np.array([[0., 0.], [2., 0.], [0., 3.]])
        src, aligned, _ = procrustes_align(source, 2 * source + 7)
        np.testing.assert_allclose(aligned, 2 * src, atol=1e-12)

    def test_tps_reproduces_affine_mapping_and_rejects_collinearity(self):
        source = np.array([[0., 0.], [2., 0.], [0., 3.], [2., 3.]])
        target = source @ np.array([[2., .2], [0., .5]]) + [4., -7.]
        np.testing.assert_allclose(fit_tps(source, target)(source), target, atol=1e-10)
        with self.assertRaises(ValueError):
            fit_tps(np.array([[0., 0.], [1., 0.], [2., 0.]]), np.ones((3, 2)))


class CouplingTests(unittest.TestCase):
    def test_missing_trials_are_not_consecutive(self):
        paired = detect_couples(sequence([1, 3]))
        self.assertFalse(paired.is_coupled.any())

    def test_session_subject_boundaries_are_not_bridged(self):
        paired = detect_couples(sequence([1, 2], subjects=["S1", "S2"]))
        self.assertFalse(paired.is_coupled.any())

    def test_recorded_uncoupled_adjacency_stays_uncoupled(self):
        df = sequence([1, 2])
        df["condition"], df["pair_id"] = "uncoupled", np.nan
        self.assertFalse(detect_couples(df).is_coupled.any())

    def test_incomplete_planned_pair_does_not_become_uncoupled(self):
        df = sequence([3], zones=["Z1"])
        df["condition"], df["pair_id"] = "coupled", "pair2"
        result = detect_couples(df)
        self.assertTrue(result.is_coupled.iloc[0])
        self.assertFalse(result.pair_complete.iloc[0])

    def test_duplicate_and_nonconsecutive_recorded_pairs_rejected(self):
        with self.assertRaises(ValueError):
            detect_couples(sequence([1, 1]))
        df = sequence([1, 4])
        df["condition"], df["pair_id"] = "coupled", "pair1"
        with self.assertRaises(ValueError):
            detect_couples(df)

    def test_identical_endpoint_distributions_have_identical_length(self):
        df = sequence([1, 2, 3, 4, 5, 6, 7, 8])
        df["condition"] = ["coupled"] * 4 + ["uncoupled"] * 4
        df["pair_id"] = ["a", "a", "b", "b"] + [None] * 4
        df["x_est_corr"] = [0., 1., 0., 1.] * 2
        df["y_est_corr"] = [1., -1., -1., 1.] * 2
        df = detect_couples(df)
        reference = {("index", "Z1"): (0., 0.), ("index", "Z2"): (1., 0.)}
        c = compute_coupled_finger_lengths(df, reference)
        u = compute_uncoupled_finger_lengths(df, reference)
        self.assertAlmostEqual(c.length_est.iloc[0], 1.)
        self.assertAlmostEqual(u.length_est.iloc[0], 1.)
        self.assertGreater(c.pair_distance_mean.iloc[0], 2.)
        self.assertTrue(np.isnan(c.length_ci_low.iloc[0]))  # two pairs are insufficient

    def test_bootstrap_preserves_shared_pair_displacement(self):
        z1 = np.array([[0., 0.], [10., 5.], [-8., -5.], [3., 2.]])
        z2 = z1 + [0., 100.]
        samples = centroid_distance_samples(z1, z2, n_boot=100, paired=True)
        np.testing.assert_allclose(samples, 100.)


class MetricTests(unittest.TestCase):
    def test_metric_rectification_handles_perspective_and_anisotropic_scale(self):
        grid = np.array([(x, y) for y in range(6) for x in range(9)], float) * 3.
        table_projection = np.array([[40., 2., 300.], [1., 20., 200.], [.001, .0005, 1.]])
        board_projection = np.array([[50., 1., 250.], [0., 27., 160.], [.0008, .0003, 1.]])
        ct, cp = apply_homography(grid, table_projection), apply_homography(grid, board_projection)
        with patch("pipeline.registration.calibration.find_checkerboard_corners", side_effect=[ct, cp]), patch("pipeline.registration.calibration.cv2.imread", return_value=np.zeros((1080, 1920, 3), np.uint8)):
            calib = compute_homography_from_images("table.png", "plateau.png", square_size_cm=3.)
        real_world = np.array([[6., 6.], [6., 6.]])
        response_world = real_world + [[1., 0.], [0., 1.]]
        actual = apply_homography(real_world, table_projection)
        estimated = apply_homography(response_world, board_projection)
        df = pd.DataFrame(dict(x_est=estimated[:, 0], y_est=estimated[:, 1], x_real=actual[:, 0], y_real=actual[:, 1], qc_valid=[True, True]))
        out = add_cm_errors(compute_errors(apply_geometry_correction(df, calib)), calib["cm_per_px"])
        np.testing.assert_allclose(out.error_eucl_cm, [1., 1.], atol=1e-5)
        np.testing.assert_allclose(out.x_real_image, actual[:, 0])

    def test_calibration_failure_is_not_silently_replaced(self):
        df = pd.DataFrame(dict(x_est=[1.], y_est=[1.], x_real=[1.], y_real=[1.]))
        with self.assertRaises(ValueError):
            apply_geometry_correction(df)

    def test_projective_singularity_rejected(self):
        with self.assertRaises(ValueError):
            apply_homography([[0., 0.]], np.zeros((3, 3)))

    def test_normalization_invariant_to_rotation_and_trial_imbalance(self):
        df = hand_data()
        scale = compute_reference_scale(df)
        duplicated = pd.concat([df, df.iloc[:4]] * 2, ignore_index=True)
        self.assertAlmostEqual(scale, compute_reference_scale(duplicated))
        pts = df[["x_real", "y_real"]].to_numpy() @ np.array([[0., 1.], [-1., 0.]])
        df[["x_real", "y_real"]] = pts
        self.assertAlmostEqual(scale, compute_reference_scale(df))


class QualityAndAnalysisTests(unittest.TestCase):
    def test_source_resolution_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "trial.jpg"
            image.write_bytes(b"placeholder")
            with patch("pipeline.quality_control.images.cv2.imread", return_value=np.zeros((100, 100, 3), np.uint8)):
                with self.assertRaises(ValueError):
                    check_source_images(pd.DataFrame(dict(filename=["trial.jpg"])), Path(tmp), 1920, 1080)

    def test_invalid_predictions_kept_and_flagged(self):
        df = pd.DataFrame(dict(x_est=[10, -1, np.nan, 1920], y_est=[10] * 4))
        result = flag_prediction_quality(df, 1920, 1080)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.qc_valid.tolist(), [True, False, False, False])

    def test_bias_variability_decomposition_and_axis_rotation_invariance(self):
        df, _ = add_anatomical_coordinates(hand_data())
        summary = bias_precision_summary(df)
        np.testing.assert_allclose(summary.rms_error**2, summary.bias_magnitude**2 + summary.variable_error**2)
        rotated = hand_data()
        for columns in (["x_real", "y_real"], ["x_est_corr", "y_est_corr"]):
            rotated[columns] = rotated[columns].to_numpy() @ np.array([[0., -1.], [1., 0.]]) + [30., -7.]
        rotated, _ = add_anatomical_coordinates(rotated)
        np.testing.assert_allclose(rotated[["error_ru", "error_pd"]], df[["error_ru", "error_pd"]], atol=1e-12)

    def test_multiple_references_require_selection_and_motion_is_measured(self):
        ref = hand_data().drop_duplicates(["finger", "zone"])
        second = ref.copy()
        second["image"] = "REF2.jpg"
        second["x_real"] += 5
        both = pd.concat([ref, second])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare_references(both, Path(tmp))
            selected, audit = prepare_references(both, Path(tmp), reference_image="REF1.jpg")
        self.assertEqual(len(selected), 10)
        self.assertAlmostEqual(audit.loc[audit.image == "REF2.jpg", "max_displacement"].iloc[0], 5.)

    def test_population_uses_subjects_not_sessions_or_trials(self):
        rows = pd.DataFrame(dict(subject=["S1"] * 3 + ["S2"], finger=["index"] * 4,
                                 coupling_source=["recorded_design"] * 4, correction_method=["metric"] * 4,
                                 difference_pct_points=[0., 0., 0., 100.]))
        result = summarize_cohort([rows])
        self.assertEqual(result.n_subjects.iloc[0], 2)
        self.assertAlmostEqual(result.mean_difference_pct_points.iloc[0], 50.)
        self.assertTrue(np.isnan(result.ci_low.iloc[0]))

    def test_temporal_drift_not_confounded_by_landmark_intercepts(self):
        df = pd.concat([hand_data()] * 3, ignore_index=True)
        df["trial"] = np.arange(1, len(df) + 1)
        df["is_coupled"] = False
        df, _ = add_anatomical_coordinates(df)
        df["error_pd"] = df.trial * .25 + df.finger.map({"thumb": 10, "index": 2, "middle": -8, "ring": 9, "little": -1})
        _, models = compute_temporal_models(df)
        drift = models[(models.axis == "pd") & (models.model == "drift_adjusted_for_landmark") & (models.term == "trial_slope")]
        self.assertAlmostEqual(drift.coefficient.iloc[0], .025)

    def test_trial_export_keeps_quality_and_provenance(self):
        df = pd.DataFrame(dict(trial=[1], qc_valid=[False], coupling_source=["recorded_design"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.csv"
            save_trial_csv(df, path)
            self.assertEqual(list(pd.read_csv(path).columns), list(df.columns))

    def test_partial_acquisition_design_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "design.csv"
            pd.DataFrame(dict(trial=[1], condition=["uncoupled"], pair_id=[None])).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_trial_design(sequence([1, 2]), path)

    def test_end_to_end_invalid_trial_retained_without_metric_claims(self):
        from workflow import run_pipeline
        df = hand_data()
        df["x_est"], df["y_est"] = df.x_est_corr, df.y_est_corr
        df.loc[0, "x_est"] = -1.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nn_outputs").mkdir()
            (root / "ref").mkdir()
            df[["subject", "session", "trial", "finger", "zone", "x_est", "y_est", "filename"]].to_csv(root / "nn_outputs/S1_ses-1_predictions.csv", index=False)
            df.drop_duplicates(["finger", "zone"])[["subject", "session", "finger", "zone", "landmark", "x_real", "y_real", "image"]].to_csv(root / "ref/S1_ses-1_ref.csv", index=False)
            config = dict(data_dir=str(root), checkerboard=False, allow_fallback=True,
                          skip_figures=True, bootstrap_samples=30)
            self.assertEqual(run_pipeline(config), 1)
            table = pd.read_csv(root / "processed/sub-S1/S1_ses-1_trials.csv")
            self.assertEqual(len(table), len(df))
            self.assertEqual(int(table.qc_valid.sum()), len(df) - 1)
            self.assertTrue(table.error_eucl_cm.isna().all())
            self.assertTrue(table.loc[table.trial == 1, "error_eucl"].isna().all())
            manifest = json.loads((root / "processed/sub-S1/S1_ses-1_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["units"], "approximate_image_px")


if __name__ == "__main__":
    unittest.main()
