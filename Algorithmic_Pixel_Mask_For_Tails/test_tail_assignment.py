import unittest

import cv2
import numpy as np

import algorithmic_tail_mask as atm


def draw_head(labels: np.ndarray, head_id: int, center: tuple[int, int], radius: int) -> None:
    cv2.circle(labels, center, radius, head_id, thickness=-1)


class TailAssignmentTests(unittest.TestCase):
    def test_single_curved_tail_keeps_full_component(self) -> None:
        tail_pixels = np.zeros((100, 100), dtype=np.uint8)
        points = np.array([[15, 80], [30, 55], [55, 45], [75, 20]], dtype=np.int32)
        cv2.polylines(tail_pixels, [points], isClosed=False, color=1, thickness=5)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (14, 82), 8)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
        )

        self.assertEqual(plan["mode"], "single_head_component")
        self.assertEqual(len(plan["assignments"]), 1)
        np.testing.assert_array_equal(plan["assignments"][0]["mask"], tail)

    def test_crossing_tails_split_into_distinct_head_segments(self) -> None:
        tail_pixels = np.zeros((120, 120), dtype=np.uint8)
        cv2.line(tail_pixels, (20, 60), (100, 60), 1, thickness=5)
        cv2.line(tail_pixels, (60, 20), (60, 100), 1, thickness=5)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (14, 60), 7)
        draw_head(head_labels, 2, (106, 60), 7)
        draw_head(head_labels, 3, (60, 14), 7)
        draw_head(head_labels, 4, (60, 106), 7)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1, 2, 3, 4],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=7.0,
        )

        self.assertEqual(plan["mode"], "split_shared_tail")
        self.assertEqual(len(plan["assignments"]), 4)
        self.assertGreater(plan["directional_width"]["barrier_pixels"], 0)
        assignment_masks = [row["mask"] for row in plan["assignments"]]
        for mask in assignment_masks:
            self.assertGreater(np.count_nonzero(mask), 0)
            self.assertFalse(np.array_equal(mask, tail))

        unique_pairs = {
            (idx, other_idx)
            for idx, mask in enumerate(assignment_masks)
            for other_idx, other_mask in enumerate(assignment_masks)
            if idx < other_idx and np.array_equal(mask, other_mask)
        }
        self.assertEqual(unique_pairs, set())

    def test_weak_multi_head_tail_is_ambiguous_without_assignments(self) -> None:
        tail_pixels = np.zeros((80, 80), dtype=np.uint8)
        cv2.line(tail_pixels, (15, 40), (65, 40), 1, thickness=5)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (12, 40), 2)
        draw_head(head_labels, 2, (68, 40), 2)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1, 2],
            scale=1.0,
            contact_radius=8,
        )

        self.assertEqual(plan["mode"], "ambiguous")
        self.assertEqual(plan["assignments"], [])
        self.assertEqual(plan["ambiguous"]["reason"], "no_valid_head_candidates")

    def test_directional_width_profile_does_not_flag_curved_single_tail(self) -> None:
        tail_pixels = np.zeros((120, 120), dtype=np.uint8)
        points = np.array([[15, 95], [28, 75], [50, 60], [72, 42], [98, 24]], dtype=np.int32)
        cv2.polylines(tail_pixels, [points], isClosed=False, color=1, thickness=5)
        skeleton = atm.skeletonize_binary(tail_pixels)
        coords_yx, neighbors = atm.skeleton_graph(skeleton)
        anchor_idx, _, _ = atm.nearest_skeleton_anchor(
            coords_yx,
            cv2.circle(np.zeros(tail_pixels.shape, dtype=np.uint8), (14, 96), 7, 1, -1),
        )
        self.assertIsNotNone(anchor_idx)

        analysis = atm.directional_width_spike_analysis(
            tail_roi=tail_pixels,
            coords_yx=coords_yx,
            neighbors=neighbors,
            anchors={1: int(anchor_idx)},
            scale=1.0,
            average_tail_width_px=7.0,
        )

        self.assertEqual(analysis["barrier_pixels"], 0)

    def test_directional_width_profile_flags_perpendicular_crossing(self) -> None:
        tail_pixels = np.zeros((120, 120), dtype=np.uint8)
        cv2.line(tail_pixels, (20, 60), (100, 60), 1, thickness=5)
        cv2.line(tail_pixels, (60, 20), (60, 100), 1, thickness=5)
        skeleton = atm.skeletonize_binary(tail_pixels)
        coords_yx, neighbors = atm.skeleton_graph(skeleton)
        head_roi = cv2.circle(np.zeros(tail_pixels.shape, dtype=np.uint8), (14, 60), 7, 1, -1)
        anchor_idx, _, _ = atm.nearest_skeleton_anchor(coords_yx, head_roi)
        self.assertIsNotNone(anchor_idx)

        analysis = atm.directional_width_spike_analysis(
            tail_roi=tail_pixels,
            coords_yx=coords_yx,
            neighbors=neighbors,
            anchors={1: int(anchor_idx)},
            scale=1.0,
            average_tail_width_px=7.0,
        )

        self.assertGreater(analysis["barrier_pixels"], 0)


class TailPathV2Tests(unittest.TestCase):
    def test_head_outline_pixels_are_rejected_as_false_tail(self) -> None:
        tail_pixels = np.zeros((90, 90), dtype=np.uint8)
        cv2.circle(tail_pixels, (40, 40), 11, 1, thickness=2)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (40, 40), 8)
        rgb = np.full((90, 90, 3), 245, dtype=np.uint8)
        rgb[tail] = (130, 90, 45)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=3.0,
            rgb=rgb,
            use_path_scoring=True,
        )

        self.assertEqual(plan["mode"], "single_head_component")
        self.assertEqual(len(plan["assignments"]), 1)
        self.assertFalse(plan["assignments"][0]["valid_tail_continuation"])
        self.assertEqual(plan["assignments"][0]["path_rejection_reason"], "head_outline_false_tail")

    def test_true_narrow_neck_tail_passes_head_collar_gate(self) -> None:
        tail_pixels = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(tail_pixels, (49, 40), (92, 42), 1, thickness=2)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (40, 40), 8)
        rgb = np.full((100, 100, 3), 245, dtype=np.uint8)
        rgb[tail] = (135, 85, 35)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=3.0,
            rgb=rgb,
            use_path_scoring=True,
        )

        self.assertEqual(len(plan["assignments"]), 1)
        assignment = plan["assignments"][0]
        self.assertTrue(assignment["valid_tail_continuation"])
        self.assertIsNone(assignment["path_rejection_reason"])
        self.assertGreater(assignment["head_exit_path_length_px"], 20.0)

    def test_crossing_tails_with_different_colors_select_same_color_path(self) -> None:
        tail_pixels = np.zeros((120, 120), dtype=np.uint8)
        cv2.line(tail_pixels, (20, 60), (105, 60), 1, thickness=5)
        cv2.line(tail_pixels, (60, 20), (60, 105), 1, thickness=5)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (14, 60), 7)
        rgb = np.full((120, 120, 3), 245, dtype=np.uint8)
        cv2.line(rgb, (20, 60), (105, 60), (135, 80, 35), thickness=5)
        cv2.line(rgb, (60, 20), (60, 105), (35, 120, 165), thickness=5)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=7.0,
            rgb=rgb,
            use_path_scoring=True,
        )

        assignment = plan["assignments"][0]
        self.assertGreater(assignment["path_endpoint_xy"][0], 95)
        self.assertIsNone(assignment["path_ambiguity_reason"])
        self.assertGreater(assignment["path_color_continuity_score"], 80.0)

    def test_crossing_tails_with_indistinguishable_colors_become_ambiguous(self) -> None:
        tail_pixels = np.zeros((120, 120), dtype=np.uint8)
        cv2.line(tail_pixels, (20, 60), (105, 60), 1, thickness=5)
        cv2.line(tail_pixels, (60, 20), (60, 105), 1, thickness=5)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (14, 60), 7)
        rgb = np.full((120, 120, 3), 245, dtype=np.uint8)
        rgb[tail] = (135, 80, 35)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=7.0,
            rgb=rgb,
            use_path_scoring=True,
        )

        assignment = plan["assignments"][0]
        self.assertEqual(assignment["path_ambiguity_reason"], "ambiguous_path_color_margin")

    def test_extra_branch_mask_is_trimmed_from_final_path_mask(self) -> None:
        tail_pixels = np.zeros((120, 120), dtype=np.uint8)
        cv2.line(tail_pixels, (20, 70), (105, 70), 1, thickness=5)
        cv2.line(tail_pixels, (60, 70), (60, 25), 1, thickness=5)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (14, 70), 7)
        rgb = np.full((120, 120, 3), 245, dtype=np.uint8)
        cv2.line(rgb, (20, 70), (105, 70), (135, 80, 35), thickness=5)
        cv2.line(rgb, (60, 70), (60, 25), (40, 125, 165), thickness=5)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=7.0,
            rgb=rgb,
            use_path_scoring=True,
        )

        assignment = plan["assignments"][0]
        final_mask = assignment["mask"]
        branch_pixels_remaining = np.count_nonzero(final_mask[:45, 56:65])
        self.assertGreater(assignment["path_trimmed_branch_pixels"], 20)
        self.assertLess(branch_pixels_remaining, 12)

    def test_normal_single_curved_tail_remains_valid(self) -> None:
        tail_pixels = np.zeros((120, 120), dtype=np.uint8)
        points = np.array([[18, 96], [32, 75], [55, 62], [77, 43], [101, 28]], dtype=np.int32)
        cv2.polylines(tail_pixels, [points], isClosed=False, color=1, thickness=5)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (14, 99), 8)
        rgb = np.full((120, 120, 3), 245, dtype=np.uint8)
        rgb[tail] = (135, 80, 35)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=7.0,
            rgb=rgb,
            use_path_scoring=True,
        )

        assignment = plan["assignments"][0]
        self.assertTrue(assignment["valid_tail_continuation"])
        self.assertIsNone(assignment["path_rejection_reason"])
        self.assertIsNone(assignment["path_ambiguity_reason"])

    def test_path_ablation_scores_are_recorded(self) -> None:
        tail_pixels = np.zeros((120, 120), dtype=np.uint8)
        cv2.line(tail_pixels, (20, 60), (105, 60), 1, thickness=5)
        cv2.line(tail_pixels, (60, 20), (60, 105), 1, thickness=5)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (14, 60), 7)
        rgb = np.full((120, 120, 3), 245, dtype=np.uint8)
        cv2.line(rgb, (20, 60), (105, 60), (135, 80, 35), thickness=5)
        cv2.line(rgb, (60, 20), (60, 105), (35, 120, 165), thickness=5)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=7.0,
            rgb=rgb,
            use_path_scoring=True,
            path_mode="hybrid",
        )

        assignment = plan["assignments"][0]
        self.assertIn("path_score_geometry", assignment)
        self.assertIn("path_score_color", assignment)
        self.assertIn("path_score_hybrid", assignment)
        self.assertGreater(assignment["path_score_hybrid"], 0.0)

    def test_local_background_normalization_records_raw_and_normalized_color(self) -> None:
        tail_pixels = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(tail_pixels, (18, 50), (85, 50), 1, thickness=4)
        tail = tail_pixels > 0

        head_labels = np.zeros(tail.shape, dtype=np.int32)
        draw_head(head_labels, 1, (12, 50), 7)
        gradient = np.tile(np.linspace(210, 250, 100, dtype=np.uint8), (100, 1))
        rgb = np.dstack([gradient, gradient, gradient])
        rgb[tail] = (135, 80, 35)

        plan = atm.build_tail_assignment_plan(
            tail_component=tail,
            head_labels=head_labels,
            touching_head_ids=[1],
            scale=1.0,
            contact_radius=8,
            average_tail_width_px=5.0,
            rgb=rgb,
            use_path_scoring=True,
            normalization="local-background",
        )

        assignment = plan["assignments"][0]
        self.assertEqual(assignment["path_color_normalization"], "local-background")
        self.assertIsNotNone(assignment["color_score_raw"])
        self.assertIsNotNone(assignment["color_score_normalized"])

    def test_crop_boundary_distance_helpers(self) -> None:
        bbox = (10, 20, 80, 100)
        self.assertEqual(atm.point_to_bbox_edge_distance([10, 50], bbox), 0.0)
        self.assertEqual(atm.crop_boundary_risk_from_distance(3.0, 1.0), "high")
        self.assertEqual(atm.crop_boundary_risk_from_distance(9.0, 1.0), "medium")
        self.assertEqual(atm.crop_boundary_risk_from_distance(30.0, 1.0), "low")


def candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "image_stem": "synthetic",
        "crop_id": "synthetic__tail_001__head_001",
        "tail_id": 1,
        "head_label_id": 1,
        "assigned_tail_pixels": 5000,
        "assigned_skeleton_pixels": 500,
        "foreign_content_score": 0.0,
        "touching_head_count": 1,
        "tail_branchpoint_count": 0,
        "tail_endpoint_count": 2,
        "tail_overlap_reasons": [],
        "tail_has_overlap": False,
        "width_spike_split": False,
        "tail_assignment_mode": "single_head_component",
        "source_bbox_touches_image_border": False,
        "crop_bbox_touches_image_border": False,
    }
    row.update(overrides)
    return row


class CropCandidateArbitrationTests(unittest.TestCase):
    def test_isolated_single_head_candidate_remains_accepted(self) -> None:
        candidates = [candidate()]

        result = atm.arbitrate_crop_candidates(candidates, scale=1.0)

        self.assertEqual(result["duplicate_rejections"], 0)
        self.assertEqual(candidates[0]["candidate_status"], "accepted")
        self.assertIsNone(candidates[0]["rejection_reason"])
        self.assertIsNone(candidates[0]["ambiguity_reason"])

    def test_duplicate_head_keeps_stronger_candidate(self) -> None:
        strong = candidate(crop_id="synthetic__tail_001__head_001")
        weak = candidate(
            crop_id="synthetic__tail_002__head_001",
            assigned_tail_pixels=1200,
            assigned_skeleton_pixels=110,
            foreign_content_score=0.05,
        )
        candidates = [weak, strong]

        result = atm.arbitrate_crop_candidates(candidates, scale=1.0)

        self.assertEqual(result["duplicate_rejections"], 1)
        self.assertEqual(strong["candidate_status"], "accepted")
        self.assertEqual(weak["candidate_status"], "rejected")
        self.assertEqual(weak["rejection_reason"], "duplicate_lower_scoring_candidate")
        self.assertEqual(weak["duplicate_of_crop_id"], strong["crop_id"])

    def test_tiny_single_head_fragment_is_rejected(self) -> None:
        candidates = [
            candidate(
                assigned_tail_pixels=100,
                assigned_skeleton_pixels=5,
            )
        ]

        atm.arbitrate_crop_candidates(candidates, scale=1.0)

        self.assertEqual(candidates[0]["candidate_status"], "rejected")
        self.assertEqual(candidates[0]["rejection_reason"], "tiny_tail_fragment")

    def test_broad_multi_head_overlap_is_marked_ambiguous(self) -> None:
        candidates = [
            candidate(
                touching_head_count=3,
                tail_has_overlap=True,
                tail_overlap_reasons=["local_width_spike"],
                tail_assignment_mode="split_shared_tail",
                foreign_content_score=0.4,
                width_spike_split=False,
            )
        ]

        atm.arbitrate_crop_candidates(candidates, scale=1.0)

        self.assertEqual(candidates[0]["candidate_status"], "ambiguous")
        self.assertEqual(candidates[0]["ambiguity_reason"], "unresolved_shared_tail_overlap")

    def test_crop_with_heavy_foreign_content_is_marked_ambiguous(self) -> None:
        candidates = [candidate(foreign_content_score=0.9)]

        atm.arbitrate_crop_candidates(candidates, scale=1.0)

        self.assertEqual(candidates[0]["candidate_status"], "ambiguous")
        self.assertEqual(candidates[0]["ambiguity_reason"], "foreign_content_in_crop")

    def test_duplicate_tail_path_is_measured_and_suppressed(self) -> None:
        shared_path = np.zeros((40, 40), dtype=bool)
        shared_path[10:30, 20] = True
        first = candidate(
            crop_id="synthetic__tail_001__head_001",
            tail_id=1,
            head_label_id=1,
            tail_path_skeleton_mask=shared_path,
        )
        second = candidate(
            crop_id="synthetic__tail_001__head_002",
            tail_id=1,
            head_label_id=2,
            tail_path_skeleton_mask=shared_path.copy(),
            assigned_tail_pixels=4800,
        )

        result = atm.arbitrate_crop_candidates([first, second], scale=1.0)

        statuses = {row["crop_id"]: row["candidate_status"] for row in (first, second)}
        self.assertIn("ambiguous", statuses.values())
        self.assertEqual(result["tail_duplicate_ambiguous"], 1)
        self.assertGreaterEqual(float(second["shared_path_fraction"]), 0.65)
        self.assertIn("tail_reuse_group_id", first)

    def test_balanced_review_can_accept_high_quality_mask_review_case(self) -> None:
        row = candidate(
            path_v2=True,
            path_confidence_score=85.0,
            path_color_continuity_score=80.0,
            path_trimmed_branch_fraction=0.9,
            path_low_margin=True,
            valid_tail_continuation=True,
            tail_has_overlap=True,
            tail_branchpoint_count=8,
        )

        atm.arbitrate_crop_candidates([row], scale=1.0, acceptance_profile="balanced-review")

        self.assertEqual(row["candidate_status"], "accepted")
        self.assertIn("accepted_with_mask_review", row["candidate_risk_flags"])


if __name__ == "__main__":
    unittest.main()
