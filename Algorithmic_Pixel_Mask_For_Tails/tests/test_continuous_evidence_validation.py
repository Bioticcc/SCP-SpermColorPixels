import unittest

import cv2
import numpy as np

import algorithmic_tail_mask as atm
from tail_evidence import continuous_scp_evidence, sample_orientation_peaks


class ContinuousEvidenceValidation(unittest.TestCase):
    SETTINGS = {"sat_min": 20, "local_dark_min": 4, "value_max": 250}

    def test_local_dark_matches_production_gaussian_formula(self):
        rgb = np.zeros((31, 37, 3), np.uint8)
        rgb[:] = [230, 230, 230]
        rgb[12:19, 8:29] = [130, 100, 70]
        out = continuous_scp_evidence(rgb, self.SETTINGS)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sigma = max(25.0, min(rgb.shape[:2]) / 30.0)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        expected = np.clip(background.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)
        self.assertTrue(np.array_equal(out["local_dark"], expected))

    def test_production_raw_threshold_mask_equivalence(self):
        rgb = np.full((25, 25, 3), 235, np.uint8)
        rgb[10:15, 3:22] = [130, 90, 55]
        out = continuous_scp_evidence(rgb, self.SETTINGS)
        raw, cleaned, local_dark = atm.build_candidate_masks(rgb, {
            "sat_min": 20, "value_max": 250, "local_dark_min": 4, "min_area": 12,
        })
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat, value = cv2.split(hsv)
        expected = (((hue >= 6) & (hue <= 82) & (sat >= 20) &
                     (value <= 250) & (local_dark > 0)).astype(np.uint8) * 255)
        self.assertTrue(np.array_equal(local_dark, (out["local_dark"] >= 4).astype(np.uint8) * 255))
        expected = cv2.medianBlur(expected, 3)
        expected = cv2.morphologyEx(expected, cv2.MORPH_CLOSE, atm.scaled_kernel(3, 1.0), iterations=1)
        self.assertTrue(np.array_equal(raw, expected))
        self.assertEqual(out["local_dark"].dtype, np.uint8)
        self.assertEqual(cleaned.dtype, np.uint8)

    def test_faint_brown_below_hard_saturation_remains_soft_positive(self):
        hsv = np.full((31, 31, 3), [0, 0, 255], np.uint8)
        hsv[14:17, 5:26] = [20, 10, 240]
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        out = continuous_scp_evidence(rgb, self.SETTINGS)
        raw, _, _ = atm.build_candidate_masks(rgb, {
            "sat_min": 20, "value_max": 250, "local_dark_min": 4, "min_area": 12,
        })
        self.assertGreater(float(out["score"][15, 15]), 0.0)
        self.assertEqual(int(raw[15, 15]), 0)

    def test_x_cross_has_two_orthogonal_orientation_peaks(self):
        score = np.zeros((101, 101), np.float32)
        cv2.line(score, (15, 50), (85, 50), 1.0, 3)
        cv2.line(score, (50, 15), (50, 85), 1.0, 3)
        peaks = sample_orientation_peaks(score, [[50, 50]], [1, 2, 3], angles=12, max_peaks=2)[0]
        self.assertEqual(len(peaks), 2)
        delta = abs(peaks[0]["angle_radians_mod_pi"] - peaks[1]["angle_radians_mod_pi"])
        self.assertAlmostEqual(delta, np.pi / 2, delta=np.pi / 12 + 1e-6)

    def test_shared_radii_are_supported_for_two_points(self):
        score = np.ones((41, 41), np.float32)
        result = sample_orientation_peaks(score, [[10, 20], [30, 20]], [1, 2, 3], angles=8)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(peak["radius_px"] in {1.0, 2.0, 3.0} for row in result for peak in row))

    def test_include_terms_false_preserves_score_and_omits_terms(self):
        rgb = np.full((12, 12, 3), [180, 120, 80], np.uint8)
        with_terms = continuous_scp_evidence(rgb, self.SETTINGS, include_terms=True)
        without_terms = continuous_scp_evidence(rgb, self.SETTINGS, include_terms=False)
        self.assertTrue(np.array_equal(with_terms["score"], without_terms["score"]))
        self.assertNotIn("terms", without_terms)

    def test_malformed_inputs_and_scores_are_rejected(self):
        with self.assertRaises(ValueError): continuous_scp_evidence(np.zeros((5, 5), np.uint8), self.SETTINGS)
        with self.assertRaises(ValueError): continuous_scp_evidence(np.zeros((5, 5, 4), np.uint8), self.SETTINGS)
        with self.assertRaises(ValueError): continuous_scp_evidence(np.zeros((5, 5, 3), np.float32), self.SETTINGS)
        with self.assertRaises(ValueError): continuous_scp_evidence(np.zeros((5, 5, 3), np.uint8), {"sat_min": 20})
        for bad in (np.empty((0, 0), np.float32), np.full((4, 4), np.nan, np.float32), np.full((4, 4), 2, np.float32)):
            with self.assertRaises(ValueError): sample_orientation_peaks(bad, [[2, 2]], [1])
        with self.assertRaises(ValueError): sample_orientation_peaks(np.ones((4, 4), np.float32), [[2, 2]], [])
        with self.assertRaises(ValueError): sample_orientation_peaks(np.ones((4, 4), np.float32), [[2, 2]], [0])
        with self.assertRaises(ValueError): sample_orientation_peaks(np.ones((4, 4), np.float32), [[np.nan, 2]], [1])
        with self.assertRaises(ValueError): sample_orientation_peaks(np.ones((4, 4), np.float32), [[2, 2]], [np.nan])


if __name__ == "__main__":
    unittest.main()
