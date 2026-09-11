from __future__ import annotations

import unittest

import numpy as np

from mask_reconstruction.head_ownership import compose_head_and_tail


class R3HeadOwnershipTests(unittest.TestCase):
    def setUp(self):
        shape = (12, 16)
        self.head_a = np.zeros(shape, bool)
        self.head_b = np.zeros(shape, bool)
        self.head_a[5:8, 1:4] = True
        self.head_b[5:8, 10:13] = True

        self.line_a = np.zeros(shape, bool)
        self.line_b = np.zeros(shape, bool)
        self.line_a[6, 3:13] = True
        self.line_b[5:8, 2] = True

        self.tail_a = self.line_a.copy()
        self.tail_a[5:8, 9:13] = True  # growth into B's head around three pinned pixels
        self.tail_b = self.line_b.copy()
        self.tail_b[4:9, 2:4] = True

    def _compose_pair(self):
        a = compose_head_and_tail(self.tail_a, self.line_a, self.head_a, self.head_b, self.line_b)
        b = compose_head_and_tail(self.tail_b, self.line_b, self.head_b, self.head_a, self.line_a)
        return a, b

    def test_default_tail_priority_is_complementary_at_pinned_conflict(self):
        a, b = self._compose_pair()
        conflict = self.line_a & self.head_b
        self.assertEqual(int(conflict.sum()), 3)
        default_a = a["tail"] | a["head"]
        default_b = b["tail"] | b["head"]
        self.assertTrue(np.all(default_a[conflict]))
        self.assertFalse(np.any(default_b[conflict]))
        self.assertFalse(np.any(default_a & default_b))

    def test_head_priority_is_the_complementary_interpretation(self):
        a, b = self._compose_pair()
        conflict = self.line_a & self.head_b
        self.assertFalse(np.any(a["head_priority_instance"][conflict]))
        self.assertTrue(np.all(b["head_priority_instance"][conflict]))
        self.assertFalse(np.any(a["head_priority_instance"] & b["head_priority_instance"]))

    def test_raw_heads_centerlines_and_default_selected_tail_are_preserved(self):
        original_head_a = self.head_a.copy()
        original_head_b = self.head_b.copy()
        original_line_a = self.line_a.copy()
        original_tail_a = self.tail_a.copy()
        a, _ = self._compose_pair()
        self.assertTrue(np.array_equal(a["head_evidence"], original_head_a))
        self.assertTrue(np.array_equal(self.head_a, original_head_a))
        self.assertTrue(np.array_equal(self.head_b, original_head_b))
        self.assertTrue(np.array_equal(self.line_a, original_line_a))
        self.assertTrue(np.all(a["tail"][original_line_a]))
        self.assertTrue(np.array_equal(a["tail"] & original_line_a, original_tail_a & original_line_a))

    def test_uncertainty_is_visible_to_both_candidate_owners(self):
        a, b = self._compose_pair()
        conflict = self.line_a & self.head_b
        self.assertTrue(np.all(a["uncertainty"][conflict]))
        self.assertTrue(np.all(b["uncertainty"][conflict]))

    def test_foreign_growth_and_pinned_counts_are_separate(self):
        a, b = self._compose_pair()
        expected_growth = self.tail_a & self.head_b & ~self.line_a
        conflict = self.line_a & self.head_b
        self.assertEqual(a["growth_pixels_removed"], int(expected_growth.sum()))
        self.assertEqual(a["pinned_tail_pixels_in_foreign_head"], int(conflict.sum()))
        self.assertEqual(b["head_pixels_deferred"], int(conflict.sum()))
        self.assertFalse(np.any(a["tail"] & expected_growth))

    def test_rejects_shape_mismatch_and_non_2d_inputs(self):
        with self.assertRaisesRegex(ValueError, "one 2D shape"):
            compose_head_and_tail(self.tail_a, self.line_a[:-1], self.head_a, self.head_b, self.line_b)
        cube = np.zeros((2, 3, 4), bool)
        with self.assertRaisesRegex(ValueError, "one 2D shape"):
            compose_head_and_tail(cube, cube, cube, cube, cube)

    def test_rejects_centerline_missing_from_tail(self):
        incomplete = self.tail_a.copy()
        incomplete[6, 6] = False
        with self.assertRaisesRegex(ValueError, "centerline is absent"):
            compose_head_and_tail(incomplete, self.line_a, self.head_a, self.head_b, self.line_b)


if __name__ == "__main__":
    unittest.main()
