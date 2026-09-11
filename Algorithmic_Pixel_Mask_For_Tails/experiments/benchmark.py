"""Record the unchanged baseline's behavior on known constructed geometry."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import algorithmic_tail_mask as atm
from tests.fixtures import case_names, generate_case


def centerline_measurements(predicted: np.ndarray, points: list, tolerance: float = 2.0) -> dict:
    """Pixel-length coverage of a rasterized truth polyline, not vertex accuracy."""
    truth = np.zeros(predicted.shape, np.uint8)
    cv2.polylines(truth, [np.array(points, np.int32)], False, 1, 1)
    pred = predicted.astype(bool)
    gt = truth.astype(bool)
    if not pred.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tolerance_px": tolerance}
    dist_to_truth = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 5)
    precision = float(np.mean(dist_to_truth[pred] <= tolerance))
    recall = float(np.mean(dist_to_pred[gt] <= tolerance))
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12), "tolerance_px": tolerance}


def run_fixture_baseline(output_root: Path, baseline_args, seed: int = 0) -> dict:
    root = output_root / "fixtures"
    root.mkdir()
    cases = []
    palette = [(0, 170, 220), (235, 80, 120), (80, 170, 40), (170, 70, 230)]
    for name in case_names():
        case = generate_case(name, seed=seed)
        directory = root / name
        directory.mkdir()
        image_path = directory / "image.png"
        Image.fromarray(case.rgb).save(image_path)
        Image.fromarray(case.tail_mask.astype(np.uint8) * 255).save(directory / "tail_evidence.png")
        case.export_truth(directory / "truth.json")
        labels = np.zeros(case.tail_mask.shape, np.int32)
        head_numbers = {}
        for index, head in enumerate(case.truth["heads"], 1):
            head_numbers[head["id"]] = index
            cv2.circle(labels, tuple(head["center_xy"]), head["radius_px"], index, -1)
        # Conditioning on constructed masks/heads isolates assignment defects;
        # a separate RGB run below measures upstream generation as well.
        plan = atm.build_path_scored_tail_assignment_plan(
            case.tail_mask, labels, list(head_numbers.values()), scale=1.0,
            contact_radius=8, average_tail_width_px=5.0, rgb=case.rgb,
            path_mode=baseline_args.path_mode, normalization=baseline_args.normalization)
        assignments = {row["head_id"]: row for row in plan["assignments"]}
        truth_paths = {row["instance_id"]: row for row in case.truth["true_edge_paths"]}
        rows = []
        truth_preview, predicted_preview = case.rgb.copy(), case.rgb.copy()
        determinate = case.truth["expected_behavior"]["determinate"]
        for index, instance in enumerate(case.truth["instances"]):
            color = palette[index % len(palette)]
            points = instance["centerline_xy"]
            cv2.polylines(truth_preview, [np.asarray(points, np.int32)], False, color, 2)
            head_name = truth_paths[instance["id"]]["node_ids"][0]
            assignment = assignments.get(head_numbers[head_name])
            skeleton = assignment["path_skeleton_mask"] if assignment else np.zeros(case.tail_mask.shape, bool)
            predicted_preview[skeleton > 0] = color
            row = {"instance_id": instance["id"], "head_id": head_name,
                   "expected_status": 'partial' if instance['id'] in case.truth['expected_behavior'].get('partial_instance_ids', []) else 'full',
                   "path_available": assignment is not None,
                   "ambiguity_reason": assignment.get("path_ambiguity_reason") if assignment else plan.get("ambiguous"),
                   "rejection_reason": assignment.get("path_rejection_reason") if assignment else None,
                   "endpoint_xy": assignment.get("path_endpoint_xy") if assignment else None}
            # Do not turn an arbitrary latent identity into an accuracy target.
            row["centerline"] = centerline_measurements(skeleton, points) if determinate else None
            if assignment:
                Image.fromarray(skeleton.astype(np.uint8)*255).save(directory / f"baseline_head_{head_numbers[head_name]}.png")
            rows.append(row)
        args = copy.copy(baseline_args)
        args.save_intermediates = False
        pipeline_root = directory / "pipeline"
        atm.prepare_output(pipeline_root, False)
        pipeline = atm.process_image(image_path, directory, pipeline_root,
                                     atm.mode_settings(args.mode, args.min_component_area), args)
        measurements = {"case": name, "seed": seed, "expected_behavior": case.truth["expected_behavior"],
                        "conditional_assignment": rows, "rejected_heads": plan["rejected_heads"],
                        "rgb_pipeline_summary": pipeline,
                        "scope": "Constructed cases only. Assignment metrics condition on constructed masks and heads; RGB results use the full unchanged pipeline."}
        (directory / "baseline.json").write_text(json.dumps(measurements, indent=2)+"\n")
        panels = [('Image', case.rgb), ('Truth (option 1)' if not determinate else 'Truth', truth_preview)]
        if not determinate:
            edges = {edge['id']: edge for edge in case.truth['graph']['edges']}
            option = case.rgb.copy()
            for index, path in enumerate(case.truth['identity_solutions'][1]['paths']):
                for edge_id in path['edge_ids']:
                    cv2.polylines(option, [np.array(edges[edge_id]['points'], np.int32)], False, palette[index % len(palette)], 2)
            panels.append(('Truth (option 2)', option))
        panels.append(('Baseline', predicted_preview))
        width, height = case.rgb.shape[1], case.rgb.shape[0]
        preview = Image.new('RGB', (width * len(panels), height + 28), 'white')
        draw = ImageDraw.Draw(preview)
        for index, (label, pixels) in enumerate(panels):
            draw.text((index * width + 3, 5), label, fill='black')
            preview.paste(Image.fromarray(pixels), (index * width, 28))
        preview.save(directory / "comparison.png")
        full_rows = [row for row in rows if row['expected_status'] == 'full']
        good = sum(row['centerline'] is not None and row['centerline']['recall'] >= .95 and row['centerline']['precision'] >= .95 for row in full_rows)
        summary = (f"{good}/{len(full_rows)} constructed full tracks have at least 95% centerline precision and recall at 2 px tolerance. {len(rows)-len(full_rows)} partial tracks measured separately."
                   if determinate else 'Identity is indeterminate by construction; alternate ownerships are retained in truth, and no identity accuracy is computed.')
        cases.append({"name": name, "summary": summary, "determinate": determinate,
                      "conditional_assignment": rows, "rgb_pipeline_candidates": pipeline['crop_candidates']})
    report = {"schema_version": "r0.fixture-baseline.v1", "seed": seed,
              "claims": "Synthetic engineering measurements only; no real-image accuracy claim.", "cases": cases}
    (output_root / "fixture_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
