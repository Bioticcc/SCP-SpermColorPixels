"""Small read-only artifact checks for the R2 audit."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from experiments.audit_r2 import audit
from experiments.provenance import sha256_file
from experiments.r2_assignment import assign_pool
from experiments.r2_candidates import build_candidate_pool
from graph_construction import from_explicit_graph


def _record():
    graph = from_explicit_graph({"nodes": [
        {"id": "h", "xy": [0, 0], "kind": "head_anchor"}, {"id": "e", "xy": [10, 0], "kind": "endpoint"},
    ], "edges": [{"id": "s", "start": "h", "end": "e", "points": [[0, 0], [10, 0]]}]})
    route = {"id": "h:h1", "head_id": "h", "segment_ids": ["s"], "directions": [True], "node_ids": ["h", "e"],
             "points_xy": [[0, 0], [10, 0]], "termination": "endpoint", "costs": {"direction": 0., "curvature": 0., "evidence": 0.}, "score": 0.}
    return {"relative_image": "source.tif", "components": [{"tail_id": 1, "roi_xyxy": [0, 0, 12, 12],
            "graph": graph.to_dict(), "searches": {"h": {"hypotheses": [route]}}, "baseline": []}]}


class R2AuditTests(unittest.TestCase):
    def _write_run(self, root: Path):
        r1_dir = root / "frozen-r1" / "images" / "one"; r1_dir.mkdir(parents=True)
        record = _record(); routes = r1_dir / "routes.json"; routes.write_text(json.dumps(record))
        config = {"expected_image_count": 1, "null_cost": 1.5, "partial_cost": .85, "coverage_weight": 1.}
        pool = build_candidate_pool(record, config); assignment = assign_pool(pool["hypotheses"])
        run = root / "r2"; image = run / "images" / "one"; image.mkdir(parents=True)
        payload = {"relative_image": "source.tif", "hypotheses": [asdict(item) for item in pool["hypotheses"]],
                   "assignment": assignment, "r1_record_sha256": sha256_file(routes),
                   "baseline_changes": [{"head_id": "h", "selected_id": assignment["selected_by_head"]["h"], "changed_from_unary_independent": False}]}
        (image / "assignment.json").write_text(json.dumps(payload))
        (run / "config.json").write_text(json.dumps(config)); (run / "completion.json").write_text(json.dumps({"full_inventory_run": True}))
        (run / "integrity.json").write_text(json.dumps({"passed": True}))
        (run / "provenance.json").write_text(json.dumps({"frozen_r1_sha256": {str(routes): sha256_file(routes)}, "input_manifest": {"images": [{"relative_image": "source.tif"}]}}))
        fixture = run / "fixture_report.json"; fixture.write_text('{}')
        (run / "fixture_completion.json").write_text(json.dumps({"fixture_report.json": sha256_file(fixture)}))
        (run / "index.html").write_text('<a href="images/one/assignment.json">ok</a>')
        (image / "image_completion.json").write_text(json.dumps({"summary": {"relative_image": "source.tif"}, "sha256": {"assignment.json": sha256_file(image / "assignment.json")}}))
        return run, image

    def test_accepts_deterministic_feasible_pool_and_rejects_changed_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, image = self._write_run(Path(temporary))
            self.assertTrue(audit(run)["passed"])
            payload = json.loads((image / "assignment.json").read_text())
            payload["assignment"]["selected_by_head"] = {"h": "missing"}
            (image / "assignment.json").write_text(json.dumps(payload))
            result = audit(run)
            self.assertFalse(result["passed"])
            self.assertIn("selected hypothesis/head mismatch", " ".join(result["errors"]))

    def test_reports_missing_local_html_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = self._write_run(Path(temporary))
            (run / "index.html").write_text('<img src="missing.svg">')
            result = audit(run)
            self.assertFalse(result["passed"])
            self.assertIn("missing local viewer target", " ".join(result["errors"]))

    def test_requires_checkpoint_and_baseline_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, image = self._write_run(Path(temporary))
            (image / "image_completion.json").unlink()
            payload = json.loads((image / "assignment.json").read_text()); payload.pop("baseline_changes")
            (image / "assignment.json").write_text(json.dumps(payload))
            result = audit(run)
            self.assertFalse(result["passed"])
            self.assertIn("missing baseline comparisons", " ".join(result["errors"]))
            self.assertIn("image checkpoint", " ".join(result["errors"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
