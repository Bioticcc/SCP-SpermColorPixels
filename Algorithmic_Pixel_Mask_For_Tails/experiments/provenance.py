"""Reproducibility records and structural SCP baseline comparisons.

These helpers deliberately compare machine-readable prediction content rather
than rendered files.  Output paths are run-root dependent and PNG byte identity
is neither expected nor asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = "scp-r0-provenance-v1"
_CODE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".json"}
_CANDIDATE_METRICS = (
    "assigned_tail_pixels", "assigned_skeleton_pixels", "path_length_px",
    "path_confidence_score", "path_score_margin", "score", "crop_quality_score",
    "mask_quality_score", "foreign_content_score", "tail_endpoint_count",
    "tail_branchpoint_count", "head_exit_count",
)
_ASSIGNMENT_FIELDS = (
    "tail_id", "head_label_id", "tail_assignment_mode", "touching_head_ids",
    "touching_head_count", "path_endpoint_xy", "path_v2", "valid_tail_continuation",
    "path_rejection_reason", "path_ambiguity_reason", "rejection_reason", "ambiguity_reason",
)
_IMAGE_IDENTITY_FIELDS = ("mode", "parameters")
_CANDIDATE_IDENTITY_FIELDS = (
    "candidate_status", "candidate_risk_flags", "tail_overlap_reasons",
    "duplicate_group_id", "duplicate_of_crop_id", "path_rejection_reason",
    "path_ambiguity_reason", "rejection_reason", "ambiguity_reason",
    "valid_tail_continuation", "path_low_margin", "width_spike_split",
)
_OUTPUT_ONLY_FIELDS = {
    "candidate_diagnostics_written", "diagnostics", "diagnostic_limit",
    "tail_debug_overlays", "tail_debug_limit", "tail_assignment_debug_overlays",
}


def _is_excluded_path(path: str) -> bool:
    """Diagnostic artifacts do not change a prediction or its assignment."""
    return any(
        segment in _OUTPUT_ONLY_FIELDS or segment.startswith("diagnostic_")
        for segment in path.replace("[", ".[").split(".")
    )


def _prediction_parameters(record: dict[str, Any]) -> dict[str, Any]:
    parameters = record.get("parameters")
    if not isinstance(parameters, dict):
        return {"parameters": parameters}
    return {"parameters": {key: value for key, value in parameters.items() if key not in _OUTPUT_ONLY_FIELDS}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return {key: _json_value(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def effective_args(
    parse_args: Callable[[], argparse.Namespace] | None = None,
    *,
    namespace: argparse.Namespace | None = None,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Return supplied effective settings, or call a no-argument parser safely.

    ``argv=None`` preserves the caller's command line.  A supplied ``argv`` is
    parsed using a temporary argv and is restored even if argparse raises.
    """
    if namespace is not None:
        return _json_value(namespace)
    if parse_args is None:
        return {}
    if argv is None:
        return _json_value(parse_args())
    previous = sys.argv[:]
    try:
        sys.argv = [previous[0] if previous else "scp"] + list(argv)
        return _json_value(parse_args())
    finally:
        sys.argv = previous


def collect_input_manifest(input_root: Path, limit: int | None = None) -> dict[str, Any]:
    """Hash actual SCP input images using the pipeline's sidecar filtering."""
    # Importing the SCP module gives this record the exact production filter
    # (including ``._*`` and ``__MACOSX`` exclusions), without running its CLI.
    from algorithmic_tail_mask import collect_images, valid_image_count

    root = Path(input_root).resolve()
    images = collect_images(root, limit)
    return {
        "input_root": str(root),
        "available_real_images": valid_image_count(root),
        "selected_real_images": len(images),
        "limit": limit,
        "sidecar_filter": "algorithmic_tail_mask.is_real_image",
        "images": [
            {
                "relative_image": image.relative_to(root).as_posix(),
                "size_bytes": image.stat().st_size,
                "sha256": sha256_file(image),
            }
            for image in images
        ],
    }


def dependency_metadata() -> dict[str, Any]:
    """Read installed distribution versions without importing optional ML code."""
    versions: dict[str, str | None] = {}
    for distribution in ("numpy", "opencv-python", "opencv-contrib-python", "Pillow", "ultralytics"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    try:
        import cv2  # SCP itself already requires OpenCV.
        thinning_backend = "cv2.ximgproc.thinning" if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning") else "zhang-suen-fallback"
        opencv_runtime = cv2.__version__
    except ImportError:
        thinning_backend, opencv_runtime = "unavailable", None
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "opencv_runtime": opencv_runtime,
        "thinning_backend": thinning_backend,
        "distributions": versions,
    }


def _run_git(project_root: Path, *args: str) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        return None, str(exc)
    if result.returncode:
        return None, result.stderr.strip() or f"git exited {result.returncode}"
    return result.stdout, None


def _status_paths(status: str) -> list[dict[str, str]]:
    records = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        code, filename = line[:2], line[3:]
        # Porcelain uses "old -> new" for renames; retain the current name too.
        records.append({"status": code, "path": filename.split(" -> ")[-1]})
    return records


def git_provenance(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    repository, repository_error = _run_git(root, "rev-parse", "--is-inside-work-tree")
    if repository_error or (repository or "").strip() != "true":
        return {"available": False, "revision": None, "dirty": None, "error": repository_error,
                "changed_paths": [], "untracked_code_files": []}
    revision, revision_error = _run_git(root, "rev-parse", "HEAD")
    status, status_error = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    entries = _status_paths(status or "")
    untracked = []
    for entry in entries:
        path = root / entry["path"]
        if entry["status"] == "??" and path.suffix.lower() in _CODE_SUFFIXES and path.is_file():
            untracked.append({"path": entry["path"], "sha256": sha256_file(path)})
    return {
        "available": True,
        "revision": (revision or "").strip() or None,
        "dirty": bool(entries),
        "error": status_error or revision_error,
        "changed_paths": entries,
        "untracked_code_files": untracked,
    }


def source_metadata(project_root: Path, paths: Iterable[Path | str]) -> list[dict[str, str]]:
    root = Path(project_root).resolve()
    records = []
    for item in paths:
        path = Path(item).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Provenance source/config file does not exist: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = str(path)
        records.append({"path": relative, "sha256": sha256_file(path)})
    return sorted(records, key=lambda row: row["path"])


def capture_environment(
    repo_root: Path,
    code_paths: Iterable[Path | str],
    config_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    """Capture code/config hashes, installed environment, and complete Git state.

    This is intentionally separate from an input manifest so a caller can save
    it before or after a subprocess run without touching source data.
    """
    return {
        "source_files": source_metadata(repo_root, code_paths),
        "config_files": source_metadata(repo_root, config_paths),
        "dependencies": dependency_metadata(),
        "git": git_provenance(repo_root),
    }


def build_provenance(
    *, input_root: Path, project_root: Path, code_paths: Iterable[Path | str],
    config_paths: Iterable[Path | str] = (), parse_args: Callable[[], argparse.Namespace] | None = None,
    namespace: argparse.Namespace | None = None, argv: list[str] | None = None,
    command: list[str] | None = None, limit: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if command is None else command),
        "effective_args": effective_args(parse_args, namespace=namespace, argv=argv),
        "input_manifest": collect_input_manifest(input_root, limit=limit),
        **capture_environment(project_root, code_paths, config_paths),
    }


def write_metadata_atomic(path: Path, metadata: dict[str, Any], *, overwrite: bool = False) -> Path:
    """Atomically create metadata; protect an existing run record by default."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing run metadata: {destination}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_value(metadata), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            # link() is an atomic create operation: a concurrent writer cannot
            # slip between the earlier exists() check and publication.
            try:
                os.link(temporary_name, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite existing run metadata: {destination}"
                ) from exc
            os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _per_image_run(run_root: Path) -> dict[str, dict[str, Any]]:
    json_dir = Path(run_root) / "json"
    if not json_dir.is_dir():
        raise FileNotFoundError(f"Expected per-image JSON directory: {json_dir}")
    images: dict[str, dict[str, Any]] = {}
    for path in sorted(json_dir.glob("*.json")):
        record = _read_json(path)
        image_id = record.get("relative_image")
        if not isinstance(image_id, str):
            raise ValueError(f"{path} lacks string relative_image")
        if image_id in images:
            raise ValueError(f"Duplicate relative_image in run: {image_id}")
        images[image_id] = record
    return images


def _candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    post = record.get("head_connected_postprocessing", {})
    values = post.get("crop_candidates_detail", post.get("crops", []))
    return [value for value in values if isinstance(value, dict)]


def _candidate_id(candidate: dict[str, Any], occurrence: int) -> str:
    if candidate.get("tail_id") is not None and candidate.get("head_label_id") is not None:
        return f"tail:{candidate['tail_id']}|head:{candidate['head_label_id']}|occurrence:{occurrence}"
    return f"crop:{candidate.get('crop_id', '<missing>')}|occurrence:{occurrence}"


def _candidate_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    counts: dict[str, int] = {}
    result = {}
    for candidate in _candidates(record):
        base = f"{candidate.get('tail_id')}|{candidate.get('head_label_id')}"
        counts[base] = counts.get(base, 0) + 1
        result[_candidate_id(candidate, counts[base])] = candidate
    return result


def _changed_fields(left: dict[str, Any], right: dict[str, Any], fields: Iterable[str]) -> tuple[dict[str, Any], list[str]]:
    changes, absent = {}, []
    for field in fields:
        if _is_excluded_path(field):
            continue
        if field not in left or field not in right:
            absent.append(field)
        elif left[field] != right[field]:
            changes[field] = {"canonical": left[field], "fresh": right[field]}
    return changes, absent


def _shared_numeric_changes(
    canonical: Any, fresh: Any, path: str = ""
) -> tuple[dict[str, Any], list[str]]:
    """Compare shared numeric/bool leaves, including a numeric-to-null change."""
    changes: dict[str, Any] = {}
    unavailable: list[str] = []
    if _is_excluded_path(path):
        return changes, unavailable
    canonical_number = isinstance(canonical, (int, float)) and not isinstance(canonical, bool)
    fresh_number = isinstance(fresh, (int, float)) and not isinstance(fresh, bool)
    canonical_scalar = canonical_number or isinstance(canonical, bool)
    fresh_scalar = fresh_number or isinstance(fresh, bool)
    if canonical_scalar or fresh_scalar:
        if canonical != fresh or type(canonical) is not type(fresh):
            changes[path] = {"canonical": canonical, "fresh": fresh}
        return changes, unavailable
    # A null in place of a previously numeric/bool prediction is a value
    # regression, not a harmless absent diagnostic field.
    if (canonical is None and fresh is not None) or (fresh is None and canonical is not None):
        if isinstance(canonical, (int, float, bool)) or isinstance(fresh, (int, float, bool)):
            changes[path] = {"canonical": canonical, "fresh": fresh}
        return changes, unavailable
    if isinstance(canonical, dict) and isinstance(fresh, dict):
        for key in sorted(set(canonical) | set(fresh)):
            child = f"{path}.{key}" if path else key
            if _is_excluded_path(child):
                continue
            if key not in canonical or key not in fresh:
                unavailable.append(child)
                continue
            sub_changes, sub_unavailable = _shared_numeric_changes(canonical[key], fresh[key], child)
            changes.update(sub_changes)
            unavailable.extend(sub_unavailable)
    elif isinstance(canonical, list) and isinstance(fresh, list):
        for index in range(max(len(canonical), len(fresh))):
            child = f"{path}[{index}]"
            if index >= len(canonical) or index >= len(fresh):
                unavailable.append(child)
                continue
            sub_changes, sub_unavailable = _shared_numeric_changes(canonical[index], fresh[index], child)
            changes.update(sub_changes)
            unavailable.extend(sub_unavailable)
    return changes, unavailable


def _field_availability(canonical: Any, fresh: Any, path: str = "") -> tuple[list[str], list[str]]:
    """Return (missing in fresh, additive in fresh) paths for dictionary schemas."""
    missing, additive = [], []
    if _is_excluded_path(path):
        return missing, additive
    if isinstance(canonical, dict) and isinstance(fresh, dict):
        for key in sorted(set(canonical) | set(fresh)):
            child = f"{path}.{key}" if path else key
            if _is_excluded_path(child):
                continue
            if key not in fresh:
                missing.append(child)
            elif key not in canonical:
                additive.append(child)
            else:
                child_missing, child_additive = _field_availability(canonical[key], fresh[key], child)
                missing.extend(child_missing)
                additive.extend(child_additive)
    elif isinstance(canonical, list) and isinstance(fresh, list):
        for index in range(max(len(canonical), len(fresh))):
            child = f"{path}[{index}]"
            if index >= len(fresh):
                missing.append(child)
            elif index >= len(canonical):
                additive.append(child)
            else:
                child_missing, child_additive = _field_availability(
                    canonical[index], fresh[index], child
                )
                missing.extend(child_missing)
                additive.extend(child_additive)
    return missing, additive


def compare_runs(canonical_root: Path, fresh_root: Path) -> dict[str, Any]:
    """Compare stable summaries and candidate records; never assert byte identity."""
    canonical_root, fresh_root = Path(canonical_root), Path(fresh_root)
    canonical_summary, fresh_summary = _read_json(canonical_root / "summary.json"), _read_json(fresh_root / "summary.json")
    old_totals, new_totals = canonical_summary.get("totals", {}), fresh_summary.get("totals", {})
    total_changes, unavailable_totals = _changed_fields(old_totals, new_totals, sorted(set(old_totals) | set(new_totals)))
    summary_missing, summary_additive = _field_availability(old_totals, new_totals, "totals")
    old_images, new_images = _per_image_run(canonical_root), _per_image_run(fresh_root)
    result: dict[str, Any] = {
        "comparison_kind": "structural_prediction_comparison",
        "byte_equivalence_asserted": False,
        "documented_exclusions": [
            "absolute output-root-dependent artifact paths (image, outputs, crop/overlay paths)",
            "additive diagnostic fields that have no historical comparator",
        ],
        "summary": {
            "totals_changed": total_changes,
            "missing_fields_in_fresh": sorted(set(summary_missing)),
            "new_additive_fields_in_fresh": sorted(set(summary_additive)),
            "schema_fields_lacking_historical_comparator": unavailable_totals,
        },
        "missing_images_in_fresh": sorted(set(old_images) - set(new_images)),
        "extra_images_in_fresh": sorted(set(new_images) - set(old_images)),
        "images": {},
    }
    for image_id in sorted(set(old_images) & set(new_images)):
        old, new = old_images[image_id], new_images[image_id]
        count_changes, count_absent = _changed_fields(old.get("counts", {}), new.get("counts", {}), sorted(set(old.get("counts", {})) | set(new.get("counts", {}))))
        count_missing, count_additive = _field_availability(old.get("counts", {}), new.get("counts", {}), "counts")
        old_identity = {"mode": old.get("mode"), **_prediction_parameters(old)}
        new_identity = {"mode": new.get("mode"), **_prediction_parameters(new)}
        image_identity_changes, image_identity_absent = _changed_fields(
            old_identity, new_identity, _IMAGE_IDENTITY_FIELDS
        )
        # Candidate lists are keyed below, so remove them here: changing their
        # order must not be reported as a false per-image numeric regression.
        old_numeric = {key: value for key, value in old.items() if key not in {"image", "outputs"}}
        new_numeric = {key: value for key, value in new.items() if key not in {"image", "outputs"}}
        for record in (old_numeric, new_numeric):
            parameters = record.get("parameters")
            if isinstance(parameters, dict):
                record["parameters"] = {
                    key: value for key, value in parameters.items()
                    if key not in _OUTPUT_ONLY_FIELDS
                }
            post = record.get("head_connected_postprocessing")
            if isinstance(post, dict):
                post = dict(post)
                post.pop("crops", None)
                post.pop("crop_candidates_detail", None)
                record["head_connected_postprocessing"] = post
        numeric_changes, numeric_absent = _shared_numeric_changes(old_numeric, new_numeric)
        image_missing, image_additive = _field_availability(old_numeric, new_numeric)
        old_candidates, new_candidates = _candidate_map(old), _candidate_map(new)
        candidate_details = {}
        for candidate_id in sorted(set(old_candidates) & set(new_candidates)):
            old_candidate, new_candidate = old_candidates[candidate_id], new_candidates[candidate_id]
            assignment_changes, assignment_absent = _changed_fields(old_candidate, new_candidate, _ASSIGNMENT_FIELDS)
            metric_changes, metric_absent = _changed_fields(old_candidate, new_candidate, _CANDIDATE_METRICS)
            identity_changes, identity_absent = _changed_fields(old_candidate, new_candidate, _CANDIDATE_IDENTITY_FIELDS)
            all_numeric_changes, all_numeric_absent = _shared_numeric_changes(old_candidate, new_candidate)
            candidate_missing, candidate_additive = _field_availability(old_candidate, new_candidate)
            status = None
            if old_candidate.get("candidate_status") != new_candidate.get("candidate_status"):
                status = {"canonical": old_candidate.get("candidate_status"), "fresh": new_candidate.get("candidate_status")}
            prediction_difference = bool(
                assignment_changes or metric_changes or identity_changes or all_numeric_changes
                or status or candidate_missing
            )
            if prediction_difference or candidate_additive:
                candidate_details[candidate_id] = {
                    "has_prediction_difference": prediction_difference,
                    "status_changed": status, "assignment_changed": assignment_changes,
                    "metrics_changed": metric_changes,
                    "identity_metadata_changed": identity_changes,
                    "all_shared_numeric_fields_changed": all_numeric_changes,
                    "missing_fields_in_fresh": candidate_missing,
                    "new_additive_fields_in_fresh": candidate_additive,
                    "schema_fields_lacking_historical_comparator": sorted(set(assignment_absent + metric_absent + identity_absent + all_numeric_absent)),
                }
        image_prediction_difference = bool(
            count_changes or image_identity_changes or numeric_changes or image_missing
            or set(old_candidates) != set(new_candidates)
            or any(detail["has_prediction_difference"] for detail in candidate_details.values())
        )
        result["images"][image_id] = {
            "has_prediction_difference": image_prediction_difference,
            "counts_changed": count_changes,
            "identity_metadata_changed": image_identity_changes,
            "all_shared_numeric_fields_changed": numeric_changes,
            "missing_fields_in_fresh": sorted(set(count_missing + image_missing)),
            "new_additive_fields_in_fresh": sorted(set(count_additive + image_additive)),
            "schema_fields_lacking_historical_comparator": sorted(set(count_absent + image_identity_absent + numeric_absent)),
            "missing_candidates_in_fresh": sorted(set(old_candidates) - set(new_candidates)),
            "extra_candidates_in_fresh": sorted(set(new_candidates) - set(old_candidates)),
            "candidate_differences": candidate_details,
        }
    result["has_prediction_differences"] = bool(
        result["summary"]["totals_changed"] or result["missing_images_in_fresh"]
        or result["extra_images_in_fresh"]
        or any(item["has_prediction_difference"] for item in result["images"].values())
    )
    result["has_schema_regressions"] = bool(
        result["summary"]["missing_fields_in_fresh"]
        or any(item["missing_fields_in_fresh"] or any(
            detail["missing_fields_in_fresh"] for detail in item["candidate_differences"].values()
        ) for item in result["images"].values())
    )
    result["has_additive_schema_fields"] = bool(
        result["summary"]["new_additive_fields_in_fresh"]
        or any(item["new_additive_fields_in_fresh"] or any(
            detail["new_additive_fields_in_fresh"] for detail in item["candidate_differences"].values()
        ) for item in result["images"].values())
    )
    # Backward-compatible aggregate: a fresh-only diagnostic is auditable but
    # does not make prediction output differ from a historical run.
    result["has_differences"] = (
        result["has_prediction_differences"] or result["has_schema_regressions"]
    )
    return result
