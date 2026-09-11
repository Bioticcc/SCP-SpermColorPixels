import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiments.provenance import compare_runs, effective_args, git_provenance, sha256_file, write_metadata_atomic


def candidate(tail=1, head=2, status="accepted", score=10.0, length=20.0):
    return {"tail_id": tail, "head_label_id": head, "candidate_status": status, "tail_assignment_mode": "path_v2", "path_length_px": length, "score": score, "assigned_tail_pixels": 11, "candidate_risk_flags": ["boundary"], "path_v2": True}


def image_record(image="a.tif", candidates=None, counts=None, mode="high-recall", parameters=None):
    candidates = [candidate()] if candidates is None else candidates
    return {"relative_image": image, "mode": mode, "parameters": parameters or {"path_v2": True, "path_mode": "hybrid"}, "counts": counts or {"crop_candidates": len(candidates), "tail_pixels": 10}, "head_connected_postprocessing": {"crop_candidates_detail": candidates}}


class ProvenanceTests(unittest.TestCase):
    def make_run(self, root, records, totals=None):
        (root / "json").mkdir(parents=True)
        (root / "summary.json").write_text(json.dumps({"totals": totals or {"crop_candidates": len(records)}}))
        for index, record in enumerate(records):
            (root / "json" / f"{index}.json").write_text(json.dumps(record))

    def comparison(self, old_record, new_record):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.make_run(root / "old", [old_record])
        self.make_run(root / "new", [new_record])
        return compare_runs(root / "old", root / "new")["images"]["a.tif"]

    def test_count_change_is_reported(self):
        result = self.comparison(image_record(counts={"tail_pixels": 10}), image_record(counts={"tail_pixels": 12}))
        self.assertEqual(result["counts_changed"]["tail_pixels"], {"canonical": 10, "fresh": 12})

    def test_status_assignment_and_metric_changes_are_individual(self):
        changed = candidate(status="rejected", score=7.0, length=24.0)
        changed["tail_assignment_mode"] = "split_shared_tail"
        detail = self.comparison(image_record(), image_record(candidates=[changed]))["candidate_differences"]["tail:1|head:2|occurrence:1"]
        self.assertEqual(detail["status_changed"]["fresh"], "rejected")
        self.assertIn("tail_assignment_mode", detail["assignment_changed"])
        self.assertIn("score", detail["metrics_changed"])
        self.assertIn("path_length_px", detail["metrics_changed"])

    def test_head_identity_change_is_a_candidate_assignment_difference(self):
        result = self.comparison(image_record(), image_record(candidates=[candidate(1, 9)]))
        self.assertEqual(result["missing_candidates_in_fresh"], ["tail:1|head:2|occurrence:1"])
        self.assertEqual(result["extra_candidates_in_fresh"], ["tail:1|head:9|occurrence:1"])

    def test_numeric_to_null_bool_and_identity_metadata_changes_are_reported(self):
        changed = candidate(length=None)
        changed["path_v2"] = False
        changed["candidate_risk_flags"] = ["overlap"]
        result = self.comparison(image_record(), image_record(candidates=[changed], parameters={"path_v2": False, "path_mode": "legacy"}))
        detail = result["candidate_differences"]["tail:1|head:2|occurrence:1"]
        self.assertEqual(detail["all_shared_numeric_fields_changed"]["path_length_px"]["fresh"], None)
        self.assertIn("path_v2", detail["assignment_changed"])
        self.assertIn("candidate_risk_flags", detail["identity_metadata_changed"])
        self.assertIn("parameters", result["identity_metadata_changed"])

    def test_missing_and_extra_images_and_candidates_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root / "old", [image_record("a.tif", [candidate(), candidate(3, 4)]), image_record("missing.tif", [])])
            self.make_run(root / "new", [image_record("a.tif", [candidate(), candidate(5, 6)]), image_record("extra.tif", [])])
            comparison = compare_runs(root / "old", root / "new")
            self.assertEqual(comparison["missing_images_in_fresh"], ["missing.tif"])
            self.assertEqual(comparison["extra_images_in_fresh"], ["extra.tif"])
            self.assertEqual(len(comparison["images"]["a.tif"]["missing_candidates_in_fresh"]), 1)
            self.assertEqual(len(comparison["images"]["a.tif"]["extra_candidates_in_fresh"]), 1)

    def test_candidate_order_does_not_create_difference(self):
        original, reordered = image_record(candidates=[candidate(1, 2), candidate(3, 4)]), image_record(candidates=[candidate(3, 4), candidate(1, 2)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root / "old", [original])
            self.make_run(root / "new", [reordered])
            self.assertFalse(compare_runs(root / "old", root / "new")["has_differences"])

    def test_additive_and_missing_candidate_schema_are_distinguished(self):
        additive = candidate()
        additive["new_diagnostic"] = {"ridge_score": 0.8}
        additive_comparison = self.comparison(image_record(), image_record(candidates=[additive]))
        additive_detail = additive_comparison["candidate_differences"]["tail:1|head:2|occurrence:1"]
        self.assertIn("new_diagnostic", additive_detail["new_additive_fields_in_fresh"])
        self.assertFalse(additive_comparison["has_prediction_difference"])
        missing = candidate()
        del missing["path_length_px"]
        missing_detail = self.comparison(image_record(), image_record(candidates=[missing]))["candidate_differences"]["tail:1|head:2|occurrence:1"]
        self.assertIn("path_length_px", missing_detail["missing_fields_in_fresh"])

    def test_additive_only_fields_do_not_set_aggregate_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fresh = candidate()
            fresh["new_diagnostic"] = 1
            self.make_run(root / "old", [image_record()])
            self.make_run(root / "new", [image_record(candidates=[fresh])])
            comparison = compare_runs(root / "old", root / "new")
            self.assertFalse(comparison["has_differences"])
            self.assertTrue(comparison["has_additive_schema_fields"])

    def test_truncated_list_and_missing_summary_total_are_schema_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = candidate()
            original["geometry"] = [1, 2]
            fresh = candidate()
            fresh["geometry"] = [1]
            self.make_run(root / "old", [image_record(candidates=[original])], totals={"crop_candidates": 1, "tail_pixels": 10})
            self.make_run(root / "new", [image_record(candidates=[fresh])], totals={"crop_candidates": 1})
            comparison = compare_runs(root / "old", root / "new")
            detail = comparison["images"]["a.tif"]["candidate_differences"]["tail:1|head:2|occurrence:1"]
            self.assertIn("geometry[1]", detail["missing_fields_in_fresh"])
            self.assertIn("totals.tail_pixels", comparison["summary"]["missing_fields_in_fresh"])
            self.assertTrue(comparison["has_schema_regressions"])
            self.assertTrue(comparison["has_differences"])

    def test_effective_args_atomic_write_and_untracked_code_hash(self):
        before = sys.argv[:]
        def parser():
            return Namespace(option=sys.argv[1], output=Path("out"))
        self.assertEqual(effective_args(parser, argv=["chosen"]), {"option": "chosen", "output": "out"})
        self.assertEqual(sys.argv, before)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "metadata.json"
            write_metadata_atomic(destination, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_metadata_atomic(destination, {"value": 2})
            self.assertEqual(json.loads(destination.read_text())["value"], 1)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "untracked_source.py"
            source.write_text("value = 1\n")
            provenance = git_provenance(root)
            self.assertTrue(provenance["available"])
            self.assertIn({"path": "untracked_source.py", "sha256": sha256_file(source)}, provenance["untracked_code_files"])
