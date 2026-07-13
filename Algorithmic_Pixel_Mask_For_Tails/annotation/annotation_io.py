#!/usr/bin/env python3
"""Image discovery and safe JSON I/O for Ward annotations."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .annotation_models import DEFAULT_DATASET_ID, SCHEMA_VERSION, ImageAnnotation, utc_now_iso


SCP_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_INPUT_ROOT = Path("Raw_Ward_Data")
DEFAULT_ANNOTATION_ROOT = Path("annotations/ward_gold_v1")


@dataclass
class LoadResult:
    annotation: ImageAnnotation
    path: Path
    loaded_mtime_ns: int | None
    recovered_from_backup: Path | None = None
    warnings: list[str] | None = None


@dataclass
class SaveResult:
    path: Path
    mtime_ns: int | None
    backup_path: Path | None = None
    conflict_path: Path | None = None


def resolve_scp_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SCP_ROOT / path).resolve()


def is_real_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
        return False
    if path.name.startswith("._"):
        return False
    return "__MACOSX" not in path.parts


def collect_source_images(input_root: Path, limit: int | None = None) -> list[Path]:
    resolved = resolve_scp_path(input_root)
    images = sorted(path for path in resolved.rglob("*") if is_real_image(path))
    if limit is not None:
        images = images[:limit]
    return images


def safe_image_id(image_path: Path, input_root: Path) -> str:
    image_path = image_path.resolve()
    input_root = resolve_scp_path(input_root)
    rel = image_path.relative_to(input_root)
    parts = [part.replace(" ", "_").replace(".", "_") for part in rel.parts]
    return "__".join(parts)


def source_path_for_image(image_path: Path) -> str:
    image_path = image_path.resolve()
    try:
        return image_path.relative_to(SCP_ROOT).as_posix()
    except ValueError:
        return image_path.as_posix()


def magnification_from_path(path: Path) -> str:
    joined = " ".join(path.parts).lower()
    if "40x" in joined:
        return "40x"
    if "100x" in joined:
        return "100x"
    return "unknown"


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Could not read image dimensions: {path}") from exc


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_blank_annotation(
    image_path: Path,
    input_root: Path,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> ImageAnnotation:
    width, height = image_size(image_path)
    now = utc_now_iso()
    return ImageAnnotation(
        schema_version=SCHEMA_VERSION,
        dataset_id=dataset_id,
        image_id=safe_image_id(image_path, input_root),
        source_path=source_path_for_image(image_path),
        source_sha256=file_sha256(image_path),
        width=width,
        height=height,
        magnification=magnification_from_path(image_path),
        annotation_status="not_started",
        created_at=now,
        updated_at=now,
    )


def annotation_json_path(annotation_root: Path, image_id: str) -> Path:
    return resolve_scp_path(annotation_root) / "json" / f"{image_id}.json"


def backup_dir(annotation_root: Path, image_id: str) -> Path:
    return resolve_scp_path(annotation_root) / "backups" / image_id


def conflict_dir(annotation_root: Path, image_id: str) -> Path:
    return resolve_scp_path(annotation_root) / "conflicts" / image_id


def _timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read_annotation_file(path: Path) -> ImageAnnotation:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ImageAnnotation.from_dict(data)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    json.loads(tmp_path.read_text(encoding="utf-8"))
    os.replace(tmp_path, path)


def _write_conflict_copy(annotation_root: Path, annotation: ImageAnnotation) -> Path:
    out_dir = conflict_dir(annotation_root, annotation.image_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{annotation.image_id}__conflict_{_timestamp_for_filename()}.json"
    _write_json_atomic(path, annotation.to_dict())
    return path


def _rotate_backups(directory: Path, keep: int) -> None:
    backups = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns)
    for old_path in backups[:-keep]:
        old_path.unlink()


def _copy_current_to_backup(path: Path, annotation_root: Path, image_id: str, keep_backups: int) -> Path | None:
    if not path.exists():
        return None
    directory = backup_dir(annotation_root, image_id)
    directory.mkdir(parents=True, exist_ok=True)
    backup_path = directory / f"{image_id}__{_timestamp_for_filename()}.json"
    shutil.copy2(path, backup_path)
    _rotate_backups(directory, keep=keep_backups)
    return backup_path


def save_annotation(
    annotation: ImageAnnotation,
    annotation_root: Path,
    loaded_mtime_ns: int | None = None,
    *,
    keep_backups: int = 5,
    allow_locked: bool = False,
) -> SaveResult:
    path = annotation_json_path(annotation_root, annotation.image_id)
    if annotation.annotation_status == "locked" and not allow_locked:
        raise PermissionError("Refusing to save a locked annotation without allow_locked=True")

    if path.exists():
        current_mtime = path.stat().st_mtime_ns
        if loaded_mtime_ns is None or current_mtime != loaded_mtime_ns:
            conflict_path = _write_conflict_copy(annotation_root, annotation)
            return SaveResult(path=path, mtime_ns=current_mtime, conflict_path=conflict_path)

    backup_path = _copy_current_to_backup(path, annotation_root, annotation.image_id, keep_backups)
    annotation.updated_at = utc_now_iso()
    _write_json_atomic(path, annotation.to_dict())
    return SaveResult(path=path, mtime_ns=path.stat().st_mtime_ns, backup_path=backup_path)


def _valid_backup_paths(annotation_root: Path, image_id: str) -> list[Path]:
    directory = backup_dir(annotation_root, image_id)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)


def latest_valid_backup(annotation_root: Path, image_id: str) -> tuple[ImageAnnotation, Path] | None:
    for path in _valid_backup_paths(annotation_root, image_id):
        try:
            return read_annotation_file(path), path
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return None


def load_annotation_for_image(
    image_path: Path,
    input_root: Path,
    annotation_root: Path,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> LoadResult:
    image_id = safe_image_id(image_path, input_root)
    path = annotation_json_path(annotation_root, image_id)
    warnings: list[str] = []

    if path.exists():
        try:
            annotation = read_annotation_file(path)
            return LoadResult(
                annotation=annotation,
                path=path,
                loaded_mtime_ns=path.stat().st_mtime_ns,
                warnings=warnings,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"Canonical annotation could not be loaded: {exc}")
            recovered = latest_valid_backup(annotation_root, image_id)
            if recovered is not None:
                annotation, backup_path = recovered
                warnings.append(f"Recovered from backup: {backup_path}")
                return LoadResult(
                    annotation=annotation,
                    path=path,
                    loaded_mtime_ns=None,
                    recovered_from_backup=backup_path,
                    warnings=warnings,
                )
            raise

    annotation = create_blank_annotation(image_path, input_root, dataset_id=dataset_id)
    return LoadResult(annotation=annotation, path=path, loaded_mtime_ns=None, warnings=warnings)


def load_all_annotations(
    input_root: Path,
    annotation_root: Path,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> list[LoadResult]:
    return [
        load_annotation_for_image(path, input_root, annotation_root, dataset_id=dataset_id)
        for path in collect_source_images(input_root)
    ]
