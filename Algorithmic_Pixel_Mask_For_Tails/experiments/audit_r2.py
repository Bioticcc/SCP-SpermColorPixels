"""Read-only invariant audit for a completed finite-pool R2 release."""
from __future__ import annotations

from dataclasses import asdict
import argparse
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

SCP_ROOT = Path(__file__).resolve().parents[1]
if str(SCP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCP_ROOT))

from experiments.provenance import sha256_file
from experiments.r2_candidates import build_candidate_pool


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        self.links.extend(values[key] for key in ("href", "src") if values.get(key))


def _fail(errors, message):
    errors.append(message)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _compatible(selected, by_id, errors, label):
    segments, endpoints = set(), set()
    for head, item_id in selected.items():
        item = by_id.get(item_id)
        if item is None or item.get("head_id") != head:
            _fail(errors, f"{label}: selected hypothesis/head mismatch")
            continue
        for segment in item.get("exclusive_segments", ()):
            if segment in segments: _fail(errors, f"{label}: repeated ordinary resource")
            segments.add(segment)
        endpoint = item.get("endpoint_id")
        if endpoint is not None:
            if endpoint in endpoints: _fail(errors, f"{label}: repeated endpoint resource")
            endpoints.add(endpoint)


def audit(root: Path) -> dict:
    root = Path(root).resolve(); errors = []; counts = {"images": 0, "hypotheses": 0, "local_links": 0}
    try:
        config = json.loads((root / "config.json").read_text()); provenance = json.loads((root / "provenance.json").read_text())
        completion = json.loads((root / "completion.json").read_text()); integrity = json.loads((root / "integrity.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [f"required R2 metadata unavailable: {exc}"], "counts": counts,
                "scope": "Artifact invariants only; no real-image accuracy claim."}
    if not completion.get("full_inventory_run") or not integrity.get("passed"):
        _fail(errors, "incomplete inventory or failed integrity")
    for name, digest in provenance.get("frozen_r1_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != digest: _fail(errors, f"frozen R1 asset changed: {name}")
    fixture_checkpoint = root / "fixture_completion.json"
    if not fixture_checkpoint.is_file():
        _fail(errors, "missing fixture checkpoint")
    else:
        for relative, digest in json.loads(fixture_checkpoint.read_text()).items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != digest: _fail(errors, f"fixture checkpoint changed: {relative}")
    frozen_routes = {Path(name).parent.name: Path(name) for name in provenance.get("frozen_r1_sha256", {}) if Path(name).name == "routes.json"}
    identities = set()
    for directory in sorted((root / "images").iterdir() if (root / "images").exists() else []):
        assignment_path = directory / "assignment.json"
        if not assignment_path.is_file(): _fail(errors, f"missing assignment: {directory.name}"); continue
        payload = json.loads(assignment_path.read_text()); counts["images"] += 1
        identity = payload.get("relative_image")
        if identity in identities: _fail(errors, "duplicate source identity")
        identities.add(identity)
        r1_path = frozen_routes.get(directory.name)
        if r1_path is None: _fail(errors, f"missing frozen R1 routes: {directory.name}"); continue
        if payload.get("r1_record_sha256") != sha256_file(r1_path): _fail(errors, f"R1 route hash mismatch: {directory.name}")
        record = json.loads(r1_path.read_text())
        if payload.get("relative_image") != record.get("relative_image"):
            _fail(errors, f"assignment/source identity mismatch: {directory.name}")
        try:
            from experiments.run_r2 import load_checkpoint
            if load_checkpoint(directory, record["relative_image"]) is None: _fail(errors, f"missing or changed image checkpoint: {directory.name}")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            _fail(errors, f"invalid image checkpoint: {directory.name}: {exc}")
        recreated = [asdict(item) for item in build_candidate_pool(record, config)["hypotheses"]]
        if json.dumps(payload.get("hypotheses"), sort_keys=True) != json.dumps(recreated, sort_keys=True): _fail(errors, f"non-deterministic or changed pool: {directory.name}")
        hypotheses = payload.get("hypotheses", []); counts["hypotheses"] += len(hypotheses)
        by_id = {item.get("id"): item for item in hypotheses}
        if len(by_id) != len(hypotheses): _fail(errors, f"duplicate hypothesis IDs: {directory.name}")
        heads = {item.get("head_id") for item in hypotheses}
        assignment = payload.get("assignment", {}); selected = assignment.get("selected_by_head", {})
        if set(selected) != heads: _fail(errors, f"not exactly one selected choice per head: {directory.name}")
        _compatible(selected, by_id, errors, directory.name)
        cost = sum(by_id[item_id]["cost"] for item_id in selected.values() if item_id in by_id)
        if not _finite(assignment.get("cost")) or abs(cost - assignment["cost"]) > 1e-8: _fail(errors, f"assignment cost mismatch: {directory.name}")
        for bound in ("lower_bound", "upper_bound"):
            if assignment.get(bound) is not None and not _finite(assignment[bound]): _fail(errors, f"non-finite assignment bound: {directory.name}")
        if assignment.get("lower_bound") is not None and assignment["lower_bound"] > assignment["cost"] + 1e-8: _fail(errors, f"invalid lower bound: {directory.name}")
        if assignment.get("upper_bound") is not None and assignment["upper_bound"] + 1e-8 < assignment["cost"]: _fail(errors, f"invalid upper bound: {directory.name}")
        comparisons = payload.get("baseline_changes")
        if not isinstance(comparisons, list):
            _fail(errors, f"missing baseline comparisons: {directory.name}")
        else:
            compared = {row.get("head_id"): row for row in comparisons}
            if set(compared) != set(selected): _fail(errors, f"baseline comparison coverage mismatch: {directory.name}")
            for head, item_id in selected.items():
                row = compared.get(head, {})
                if row.get("selected_id") != item_id or bool(row.get("changed_from_unary_independent")) != (head in assignment.get("changed_head_ids", [])):
                    _fail(errors, f"baseline comparison selection mismatch: {directory.name}")
        for item in hypotheses:
            if item.get("metadata", {}).get("kind") == "baseline_control" and selected.get(item.get("head_id")) != item.get("id"):
                _fail(errors, "isolated baseline control was not selected")
        runner_up = assignment.get("runner_up")
        if runner_up is not None:
            runner_selected = runner_up.get("selected_by_head", {})
            if set(runner_selected) != heads or runner_selected == selected: _fail(errors, f"invalid global runner-up: {directory.name}")
            _compatible(runner_selected, by_id, errors, "global runner-up")
            runner_cost = sum(by_id[item_id]["cost"] for item_id in runner_selected.values() if item_id in by_id)
            if not _finite(runner_up.get("cost")) or abs(runner_cost - runner_up["cost"]) > 1e-8 or runner_up["cost"] + 1e-8 < assignment["cost"]:
                _fail(errors, f"global runner-up cost mismatch: {directory.name}")
        for group in assignment.get("groups", []):
            local = group.get("assignment", {}); group_heads = set(group.get("head_ids", [])); base = (local.get("solutions") or [{}])[0]
            for head, entry in local.get("head_alternatives", {}).items():
                alternative = entry.get("alternative")
                if alternative:
                    if set(alternative.get("selected_by_head", {})) != group_heads: _fail(errors, "per-head alternative is not group-complete")
                    if alternative["selected_by_head"].get(head) == base.get("selected_by_head", {}).get(head): _fail(errors, "per-head alternative does not change constrained head")
                    _compatible(alternative["selected_by_head"], by_id, errors, "group alternative")
                    merged = dict(selected); merged.update(alternative["selected_by_head"]); _compatible(merged, by_id, errors, "merged group alternative")
                    if entry.get("score_margin") is not None and local.get("status") == "optimal" and entry.get("status") == "optimal" and base:
                        expected = alternative["cost"] - base.get("cost", 0.0)
                        if abs(entry["score_margin"] - expected) > 1e-8: _fail(errors, "inconsistent exact margin")
                    elif entry.get("score_margin") is not None:
                        _fail(errors, "exact margin reported without proven base and constrained solutions")
        for item in hypotheses:
            if item.get("metadata", {}).get("kind") != "baseline_control": continue
            meta = item["metadata"]; found = [row for component in record["components"] for row in component.get("baseline", []) if row.get("crop_id") == meta.get("crop_id")]
            if len(found) != 1 or meta.get("pixels_xy_unordered") != found[0].get("pixels_xy") or not any(ref.get("status") == found[0].get("status") for ref in meta.get("baseline_comparison_refs", [])):
                _fail(errors, "isolated baseline control changed raster or status reference")
    expected = config.get("expected_image_count")
    if expected is not None and len(identities) != expected: _fail(errors, "source identity count differs from configuration")
    manifest_identities = {row.get("relative_image") for row in provenance.get("input_manifest", {}).get("images", [])}
    if not manifest_identities or identities != manifest_identities: _fail(errors, "source identities differ from provenance input manifest")
    for path in root.rglob("*.html"):
        parser = _Links(); parser.feed(path.read_text(encoding="utf-8"))
        for link in parser.links:
            split = urlsplit(link)
            if not split.scheme and split.path:
                counts["local_links"] += 1
                if not (path.parent / unquote(split.path)).is_file(): _fail(errors, f"missing local viewer target: {path.name}: {link}")
    return {"passed": not errors, "errors": sorted(set(errors)), "counts": counts,
            "scope": "Checks frozen inputs, deterministic finite pools, assignment constraints, controls, bounds, and local assets; no real-image accuracy claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
