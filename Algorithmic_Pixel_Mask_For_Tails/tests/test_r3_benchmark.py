from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from experiments.r3_benchmark import run_r3_fixtures
R2_ROOT=Path(__file__).resolve().parents[1]/"outputs/experiments/r2_joint_2026_09_09_reviewed"

class R3BenchmarkTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(); cls.root=Path(cls.tmp.name)/"r3"
  cls.report=run_r3_fixtures(cls.root,{"fixture_seed":0,"r2_run":str(R2_ROOT)})
 @classmethod
 def tearDownClass(cls): cls.tmp.cleanup()
 def test_complete_inventory_and_width_sweeps(self):
  self.assertEqual(len(self.report["cases"]),20)
  names={r["name"] for r in self.report["cases"]}
  self.assertTrue({f"isolated_width_width_{w}" for w in (1,3,5,7)}<=names)
  self.assertTrue({"unequal_crossing_width_1_7","unequal_crossing_width_3_5","extra_branch_width_5","loop_width_5","boundary_width_5"}<=names)
 def test_determinate_truth_maps_head_to_instance(self):
  row=next(r for r in self.report["cases"] if r["name"]=="same_color_x_rendered")
  self.assertEqual({x["id"] for x in row["instances"]},{"sperm_horizontal","sperm_vertical"})
  self.assertTrue(all(x["tail_truth_recall"] is not None for x in row["instances"]))
  self.assertTrue(any(x["iou"]<1 for x in row["instances"]))
 def test_fixed_r2_provenance_exact(self):
  for row in self.report["cases"]:
   if row["mode"]!="fixed_r2_selected_rendered_assignment": continue
   payload=json.loads((R2_ROOT/"fixtures"/row["name"].removesuffix("_rendered")/"rendered"/"assignment.json").read_text()); hyps={x["id"]:x for x in payload["hypotheses"]}
   for instance in row["instances"]:
    p=instance["provenance"]
    if p is None: continue
    route=hyps[p["selected_hypothesis_id"]]["metadata"]["r1_route"]
    digest=hashlib.sha256(json.dumps(route["points_xy"],separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()
    self.assertEqual(p["r1_route_points_sha256"],digest); self.assertEqual(p["r1_route_node_ids"],route["node_ids"])
 def test_null_selection_primary_unscored(self):
  row=next(r for r in self.report["cases"] if r["name"]=="indeterminate_identity_rendered")
  self.assertFalse(row["determinate"]); self.assertEqual(row["fixed_r2_missing_selection_count"],1)
  missing=next(x for x in row["instances"] if x.get("reason") == "fixed_r2_null_or_missing_route")
  self.assertEqual(missing["predicted_pixels"],0); self.assertIsNone(missing["tail_truth_recall"])
  self.assertEqual(len(row["indeterminate_identity_options"]),2)
  for instance in row["instances"]:
   for key in ("tail_truth_recall", "iou", "missing_truth_pixels", "extra_vs_truth_pixels", "truth_pixels", "whole_component_baseline"):
    self.assertIsNone(instance[key])
  self.assertFalse(any(x["case"] == row["name"] for x in self.report["acceptance"]["failures"]))
 def test_geometry_and_ownership_metrics(self):
  self.assertEqual(self.report["acceptance"]["ordinary_duplicate_pixels"],0)
  for row in self.report["cases"]:
   self.assertEqual(row["diagnostics"]["ordinary_duplicate_pixels"],0)
   for instance in row["instances"]:
    self.assertEqual(instance["supported_centerline_missing_pixels"],0)
    if instance["predicted_pixels"]: self.assertTrue((self.root/row["name"]/f'{instance["id"]}_mask.png').is_file())
  extra=next(r for r in self.report["cases"] if r["name"]=="extra_branch_width_5")
  self.assertEqual(extra["instances"][0]["extra_vs_truth_pixels"],0)
  # Frozen R2 partial/wrong routes remain visible as failures; reconstruction
  # must not turn centerline retention alone into a passing benchmark.
  self.assertGreater(self.report["acceptance"]["failure_count"],0)
  self.assertIn("fixed_r2_determinate_tail_truth_recall_min",{x["invariant"] for x in self.report["acceptance"]["failures"]})
  self.assertTrue(all(x["whole_component_baseline"] is not None for r in self.report["cases"] for x in r["instances"] if r["determinate"]))
 def test_adapter_crossing_policy_and_metadata_are_used(self):
  fixed=[r for r in self.report["cases"] if r["mode"]=="fixed_r2_selected_rendered_assignment"]
  self.assertTrue(any(region["track_ids"] and len(region["track_ids"])==1 for r in fixed for region in r["crossing_regions"]))
  for row in fixed:
   for region in row["crossing_regions"]:
    self.assertTrue((self.root/region["mask_file"]).is_file())
    self.assertGreaterEqual(region["radius_px"],1)
 def test_width_recovery_numeric(self):
  rows=[r for r in self.report["cases"] if r["name"].startswith(("isolated_width","unequal_crossing"))]
  for row in rows:
   for instance in row["instances"]:
    self.assertGreaterEqual(instance["tail_truth_recall"],.90)
    self.assertEqual(instance["extra_vs_truth_pixels"],0)
 def test_panels_and_output_safety(self):
  self.assertTrue(all((self.root/r["name"]/"comparison.png").is_file() for r in self.report["cases"]))
  with self.assertRaises(ValueError): run_r3_fixtures(self.root,{"r2_run":str(R2_ROOT)})

if __name__=="__main__": unittest.main()
