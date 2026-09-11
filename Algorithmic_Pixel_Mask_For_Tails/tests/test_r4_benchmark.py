from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from experiments.r4_benchmark import run_r4a_fixtures

class R4aBenchmarkTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(); cls.root=Path(cls.tmp.name)/"r4"
  cls.report=run_r4a_fixtures(cls.root,{})
 @classmethod
 def tearDownClass(cls): cls.tmp.cleanup()
 def test_fixed_inventory_and_outputs(self):
  self.assertEqual(len(self.report["cases"]),16)
  rendered=[x for x in self.report["cases"] if x["mode"]=="rendered"]
  self.assertEqual({x["width_px"] for x in rendered},{1,3,5})
  self.assertEqual(len(rendered),15)
  self.assertTrue(all((self.root/x["comparison_file"]).is_file() for x in self.report["cases"]))
 def test_old_routes_retained_and_passthrough_available_across_sweep(self):
  self.assertTrue(all(x["baseline_routes_retained_exactly"] for x in self.report["cases"]))
  self.assertTrue(all(x["h1_full_passthrough_available"] for x in self.report["cases"]))
  self.assertEqual(self.report["acceptance"]["failure_count"],0)
 def test_explicit_pairing_improves_only_after_extension(self):
  row=next(x for x in self.report["cases"] if x["name"]=="explicit_mechanism")
  old={x["head_id"]:x for x in row["old_selected_metrics"]}; new={x["head_id"]:x for x in row["new_selected_metrics"]}
  self.assertLess(old["h1"]["centerline"]["recall"],.6)
  self.assertEqual(new["h1"]["centerline"]["recall"],1.0)
  self.assertEqual(new["h2"]["centerline"]["recall"],1.0)
  chosen={x["id"]:x for x in row["new_hypotheses"]}[row["new_selected_by_head"]["h1"]]
  route=chosen["metadata"]["r1_route"]
  self.assertEqual(route["segment_ids"],["left_b","b_right"])
  self.assertEqual(route["metadata"]["passed_foreign_head_ids"],["h2"])
 def test_metrics_include_all_heads_and_missing_is_zero_scored(self):
  for row in self.report["cases"]:
   self.assertEqual({x["head_id"] for x in row["old_selected_metrics"]},{"h1","h2"})
   self.assertEqual({x["head_id"] for x in row["new_selected_metrics"]},{"h1","h2"})
   self.assertEqual({x["head_id"] for x in row["expanded_pool_oracle_metrics"]},{"h1","h2"})
   for selection in (row["old_selected_metrics"],row["new_selected_metrics"]):
    for item in selection:
     if item["selected_null_or_missing"]: self.assertEqual(item["centerline"]["recall"],0.0)
   if row["mode"]=="rendered":
    new={x["head_id"]:x for x in row["new_selected_metrics"]}
    oracle={x["head_id"]:x for x in row["expanded_pool_oracle_metrics"]}
    self.assertGreaterEqual(new["h1"]["centerline"]["recall"],.98)
    self.assertTrue(all(x["centerline"]["recall"]>=.98 for x in oracle.values()))
 def test_output_refuses_overwrite(self):
  with self.assertRaises(FileExistsError): run_r4a_fixtures(self.root,{})

if __name__=="__main__": unittest.main()
