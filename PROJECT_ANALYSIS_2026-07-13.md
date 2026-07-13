# Sperm Morphology Recognition - Project Analysis

Date: 2026-07-13
Workspace: `/home/kyle/Desktop/Adam/Sperm_Morphology_Recognition`

## Document Purpose

This document captures the current state of the sperm morphology recognition
project as it exists in this workspace. It covers:

- current repository structure,
- code practices and style,
- algorithms and workflows in use,
- current output runs and metrics,
- observed project progression,
- the current algorithmic wall and likely next analysis checkpoints.

Comment: this report is based on the files, scripts, summaries, CSVs, and
output artifacts currently present in the local workspace. The root `.git/`
directory is empty, so normal `git status` and `git log` do not work. Progression
is therefore inferred from documentation, output folder names, generated
summaries, review bundles, and code evolution visible in the files.

## Executive Summary

The project has moved away from a model-first YOLO approach and is currently
centered on a deterministic image-processing method named **Sperm Color Pixels
(SCP)**. SCP lives in `Algorithmic_Pixel_Mask_For_Tails/` and is the active
full-sperm crop generation path for Ward mouse sperm microscopy images.

The current bottleneck is not simply "detect sperm-colored pixels." The system
can find large numbers of plausible head and tail components. The harder wall is
turning crowded, crossing, looping, or partially cropped components into one
clean full-sperm crop per sperm without selecting the wrong branch, including
extra tails, cutting tails short, or mistaking head outlines/thick tail sections
for tails or heads.

The latest path-v2 SCP output is deliberately conservative:

- 24 Ward images processed.
- 393 tail components detected.
- 325 head candidates detected.
- 209 crop candidates generated.
- 22 candidates accepted.
- 45 candidates rejected.
- 142 candidates marked ambiguous.

Comment: the drop from the older 97 accepted crops to the current 22 accepted
crops is not just a regression; it reflects a stricter scoring layer reacting to
manual review evidence that many older accepted/reviewed candidates were partial,
cut off, or incorrectly assigned. The current state is better thought of as a
high-precision/review-heavy checkpoint, not a solved production cropper.

## Checkpoint 1 - Current Project Structure

### Top-Level Layout

```text
Sperm_Morphology_Recognition/
  README.md
  ProjectContext.md
  AGENT.md
  PROJECT_ANALYSIS_2026-07-13.md
  .gitignore
  .venv/
  yolo26n.pt
  yolov8n.pt
  yolov8n-seg.pt

  Algorithmic_Pixel_Mask_For_Tails/
    README.md
    .gitignore
    algorithmic_tail_mask.py
    build_candidate_review_csv.py
    analyze_candidate_review_baseline.py
    review_candidate_gui.py
    test_tail_assignment.py
    Raw_Ward_Data/
    outputs/

  scripts/
    prepare_yolo_sperm_seg_dataset.py
    train_yolov8_sperm_seg.py
    run_yolo_inference_pilot.py
    yolo_seg_review_gui.py
    prepare_yolo_full_sperm_dataset.py
    train_yolov8_full_sperm.py

  Training_Data/
    Human_Positives/
    Mouse_Positives_Cropped_Tail/
    Raw_Yan_Data/

  datasets/
    yolo_sperm_seg_human_pseudo/

  runs/
    segment/

  review_bundles/
    next_steps_review_bundle_2026-06-21/
```

### Size and Artifact Notes

Local disk usage observed:

| Path | Approx size | Role |
| --- | ---: | --- |
| `Algorithmic_Pixel_Mask_For_Tails/` | 21 GB | SCP code, Ward raw images, generated outputs |
| `Training_Data/` | 4.5 GB | Human positives, NMA mouse positives, Yan raw data |
| `datasets/` | 1.8 GB | YOLO pseudo-mask dataset |
| `review_bundles/` | 628 MB | Shared comparison/review bundle |
| `runs/` | 35 MB | YOLO training/evaluation outputs |
| `scripts/` | 216 KB | YOLO and review utility scripts |

Comment: this is an active local research/artifact workspace, not a clean
minimal Python package. Large generated outputs and source datasets are present
locally but are ignored by `.gitignore`.

### Raw Ward Data

Ward source images live under:

```text
Algorithmic_Pixel_Mask_For_Tails/Raw_Ward_Data/
  Sperm 40x Raw Photos/
  Sperm 100x Raw Photos/
```

The current SCP summary reports:

- 24 total real Ward images available.
- macOS sidecars and `__MACOSX` folders skipped.

The raw Ward set includes 15 40x TIFF images and 9 100x TIFF images, with
sidecar artifacts also present.

### Git and Reproducibility State

The root `.git/` directory exists but is empty. As a result:

- `git status` fails.
- `git log` fails.
- there is no commit history available from this workspace.

There is also no root `requirements.txt`, `pyproject.toml`, `setup.py`, or
environment YAML. Dependencies are therefore implicit in imports and in the
existing `.venv`.

Comment: for scientific reproducibility, the two biggest repo-organization gaps
are the missing version history and missing dependency manifest.

## Checkpoint 2 - Current Direction

The project objective is to build a mouse sperm morphology pipeline that can:

- detect individual full sperm in microscopy images,
- isolate one crop per full sperm,
- preserve head, neck, midpiece, and full tail,
- avoid duplicate, debris, partial-sperm, and merged multi-sperm crops,
- support later anatomy segmentation and morphology classification.

The current primary method is **SCP: Sperm Color Pixels**.

SCP is deterministic image processing, not neural network inference. It uses the
brown/olive sperm signal in Ward RGB TIFF images to create high-recall masks,
detect head and tail components, associate tails to heads, score final crop
candidates, and export accepted/rejected/ambiguous review artifacts.

Supporting methods are still present:

- YOLOv8 segmentation trained on human pseudo-masks.
- YOLO weak full-sperm detector scripts.
- Ben Skinner NMA outputs for nucleus/head-oriented analysis.
- Manual review CSVs and label summaries.

Comment: current documentation and output evidence agree that YOLO and NMA are
supporting signals, not replacements for SCP as the full mouse-sperm cropper.

## Checkpoint 3 - Code Practices and Code Style

### General Python Style

The codebase is script-oriented Python. Common traits:

- `argparse` CLIs are used for runnable tools.
- `pathlib.Path` is used heavily for path handling.
- JSON and CSV files are used as the main structured output formats.
- Type hints are common.
- `dataclasses` are used for structured records where helpful.
- OpenCV, NumPy, PIL, Ultralytics, and Tkinter are the main non-stdlib imports.
- Most scripts are executable with `#!/usr/bin/env python3`.
- SCP relative paths resolve inside `Algorithmic_Pixel_Mask_For_Tails/`.

The largest file is:

```text
Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py
```

It is about 3,868 lines and contains the main deterministic algorithm.

Comment: the code is practical and explicit. It favors inspectable intermediate
artifacts and deterministic thresholds over hidden model behavior. The tradeoff
is that the main SCP script has become very large, with many thresholds and
responsibilities in one file.

### Dependency Pattern

Main dependencies inferred from imports:

- Standard library: `argparse`, `csv`, `json`, `random`, `shutil`, `heapq`,
  `collections`, `dataclasses`, `pathlib`, `typing`, `unittest`, `os`.
- Scientific/image stack: `cv2`, `numpy`, `PIL`.
- ML stack: `ultralytics.YOLO`.
- GUI stack: `tkinter`, `PIL.ImageTk`.

There is no dependency lockfile or package manifest.

### Artifact and Output Practices

The project strongly favors saving visual and structured artifacts:

- binary masks,
- RGB overlays,
- highlighted crops,
- skeleton overlays,
- per-image JSON metadata,
- run-level `summary.csv` and `summary.json`,
- candidate review CSVs,
- diagnostic overlays for path decisions.

This is a strong practice for image-analysis debugging because failure modes are
visual. The current output folders are large, but they make the algorithm easier
to audit.

### Testing Practices

There is one unit test file:

```text
Algorithmic_Pixel_Mask_For_Tails/test_tail_assignment.py
```

It covers synthetic cases such as:

- single curved tail assignment,
- crossing tail splitting,
- weak multi-head ambiguity,
- directional width spike detection,
- head-outline false-tail rejection,
- true narrow-neck tail acceptance,
- color-continuity path selection,
- ambiguous indistinguishable-color crossings,
- extra branch trimming,
- crop boundary helpers,
- duplicate candidate arbitration,
- tiny fragment rejection,
- heavy foreign-content ambiguity,
- balanced-review acceptance behavior.

Comment: the synthetic unit tests are valuable because they isolate geometric
logic. The missing layer is a small, stable, manually labeled real-image gold
set that can report precision/recall-like metrics across SCP versions.

### Code Organization Strengths

- Clear separation between SCP and older YOLO/NMA workflows.
- Rich output metadata for later review and analysis.
- Good use of CSV review queues for human-in-the-loop tuning.
- CLI flags expose important ablation dimensions such as path mode,
  normalization, diagnostics, and acceptance profile.
- Tests cover several algorithmic edge cases that caused prior failures.

### Code Organization Weaknesses

- The main SCP script is large and mixes masking, component analysis, path
  scoring, crop writing, diagnostics, and summarization.
- No formal package layout.
- No dependency manifest.
- No functioning Git metadata in this workspace.
- Many thresholds are embedded in code and CLI presets rather than collected in
  a versioned config file.
- Metrics are mostly candidate-level and review-level. Missed visible sperm are
  not yet measured in a systematic gold set.
- Some docs are slightly out of sync: root-level context still highlights
  `updated_outputs/`, while the SCP script default and SCP README point to
  `updated_outputs_path_v2/` as the current path-v2 output direction.

## Checkpoint 4 - Algorithms and Workflows

## 4.1 SCP - Sperm Color Pixels

Main file:

```text
Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py
```

SCP is a deterministic full-sperm crop candidate generator for Ward RGB images.

### SCP Inputs

- Input root: `Algorithmic_Pixel_Mask_For_Tails/Raw_Ward_Data/`
- Accepted image extensions: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`
- Sidecar/invalid image handling skips `__MACOSX` and `._*` artifacts.

### SCP Threshold Presets

The default preset is `high-recall`:

| Setting | Value |
| --- | ---: |
| `sat_min` | 20 |
| `value_max` | 250 |
| `local_dark_min` | 4 |
| `head_value_max` | 130 |
| `min_area` | 12 |
| `tail_min_area` | 70 |
| `tail_min_aspect` | 2.0 |
| `tail_max_fill` | 0.55 |

Other modes are `balanced` and `high-precision`.

Comment: default SCP intentionally over-detects faint sperm-colored pixels, then
relies on downstream scoring and review buckets to filter risky candidates.

### SCP Algorithm Stages

1. **Load image**

   Images are loaded with PIL and converted to RGB NumPy arrays.

2. **Build sperm-color candidate masks**

   The image is converted to HSV and grayscale. The algorithm estimates local
   background brightness using Gaussian blur, then computes a local-darkness
   mask. Candidate sperm pixels must satisfy:

   - hue in the brown/olive range,
   - sufficient saturation,
   - not too bright,
   - darker than local background.

   Morphological close and small-component removal clean the mask.

3. **Detect head candidates**

   Head seeds are darker compact regions inside the cleaned sperm-color mask.
   Components are filtered by:

   - area,
   - aspect ratio,
   - fill ratio,
   - contour circularity.

4. **Detect tail components**

   Tail seeds are produced by subtracting head pixels from the cleaned mask.
   Tail candidates are connected components filtered as:

   - line-like elongated components,
   - sparse branch-like components,
   - large irregular low-fill components.

5. **Skeletonize tails**

   The code uses `cv2.ximgproc.thinning` when available, otherwise a local
   Zhang-Suen thinning implementation. Skeletons are used to compute endpoints,
   branchpoints, and path graphs.

6. **Flag overlap-suspicion regions**

   Tail components are flagged for possible overlaps using:

   - wide branchpoints,
   - local width spikes,
   - multiple endpoints near wide regions,
   - wide sparse components.

7. **Associate tails with heads**

   Tail components are dilated by a scaled contact radius. Any detected heads
   within that dilated region become touching head candidates.

8. **Build tail assignment plans**

   There are two assignment families:

   - legacy/full-component assignment,
   - path-v2 skeleton path scoring.

   Clean isolated simple single-head tails may use fast legacy assignment when
   `--path-fast-isolated` is enabled.

9. **Legacy shared-tail splitting**

   For multi-head shared tails, legacy logic builds a skeleton graph, finds
   nearest head anchors, detects compact width-spike barriers, assigns skeleton
   ownership from multiple sources, then floods tail pixels back to owners.

10. **Path-v2 scoring**

    Path-v2 scores candidate skeleton paths from head anchor to distal endpoint.
    Available path modes:

    - `geometry-only`,
    - `color-only`,
    - `hybrid`,
    - `legacy`.

    Path scoring uses:

    - path length,
    - head-collar exit distance,
    - outside-head-collar tail pixels,
    - curvature,
    - branchpoint penalties,
    - width spike penalties,
    - Lab/HSV color continuity,
    - path score margins against competing paths.

    Important path-v2 rejection/ambiguity concepts:

    - `head_outline_false_tail`,
    - `tail_path_too_short`,
    - `low_color_continuity`,
    - `ambiguous_path_color_margin`,
    - `ambiguous_path_conflict`.

11. **Reconstruct final tail mask**

    Path-v2 reconstructs a final mask from the selected path, trimming branches
    rather than blindly accepting the whole connected component.

12. **Create crop candidates**

    Each accepted tail/head assignment becomes a candidate crop. Crop metadata
    includes source and padded bounding boxes, head/tail IDs, assigned pixels,
    skeleton pixels, foreign content, boundary risk, path metrics, and output
    artifact paths.

13. **Score and arbitrate candidates**

    Candidates receive:

    - `score`,
    - `crop_quality_score`,
    - `mask_quality_score`,
    - `path_confidence_score`,
    - `candidate_risk_flags`,
    - `candidate_status`.

    Candidate statuses are:

    - `accepted`,
    - `rejected`,
    - `ambiguous`.

14. **Write outputs**

    SCP writes masks, overlays, highlighted crops, skeleton overlays, per-image
    JSON, `summary.csv`, and `summary.json`.

### SCP Candidate Scoring Rules

Candidates score better when they have:

- many assigned tail pixels,
- many skeleton pixels,
- simple single-head assignment,
- successful width-spike split,
- high path confidence,
- clean crop and mask quality.

Candidates score worse or become review-only when they have:

- too few tail/skeleton pixels,
- reused head assignments,
- shared tail components,
- overlap flags,
- low path confidence,
- weak color continuity,
- low path score margin,
- many branchpoints/endpoints,
- foreign head/tail pixels in the crop,
- source or crop boundary contact.

Hard status rules include:

- tiny fragments become `rejected`,
- head-outline-only false tails become `rejected`,
- lower-scoring duplicate accepted candidates for a head become `rejected`,
- heavy foreign content or unresolved crossings become `ambiguous`,
- low color continuity and ambiguous competing paths become `ambiguous`.

Comment: current SCP is conservative about ambiguity. That is appropriate for
creating high-confidence crops, but it means the accepted set is currently small.

## 4.2 Candidate Review CSV Workflow

Main files:

```text
Algorithmic_Pixel_Mask_For_Tails/build_candidate_review_csv.py
Algorithmic_Pixel_Mask_For_Tails/analyze_candidate_review_baseline.py
Algorithmic_Pixel_Mask_For_Tails/review_candidate_gui.py
```

The CSV builder reads per-image SCP JSON files and creates:

- `candidate_review_all_candidates.csv`,
- `candidate_review_queue.csv`,
- `candidate_review_label_key.md`.

The default review queue includes:

- all ambiguous candidates,
- the 30 lowest-scoring accepted candidates,
- the 20 highest-scoring rejected candidates.

Human review columns include:

- `crop_label`,
- `mask_label`,
- `review_notes`,
- `reviewer`,
- `reviewed_at`.

Review labels include crop-level categories such as:

- `good_full_sperm`,
- `partial_or_cut_off`,
- `merged_multi_sperm`,
- `duplicate`,
- `debris_or_false_positive`,
- `unclear`.

Mask labels include:

- `mask_good`,
- `tail_mask_partial`,
- `head_mask_wrong`,
- `wrong_tail_assigned`,
- `extra_tail_assigned`,
- `mask_unclear`.

The baseline analyzer summarizes completed review CSVs by label, status, score
band, assignment reason, and normalized notes.

## 4.3 YOLOv8 Human Pseudo-Mask Segmentation

Main files:

```text
scripts/prepare_yolo_sperm_seg_dataset.py
scripts/train_yolov8_sperm_seg.py
scripts/run_yolo_inference_pilot.py
scripts/yolo_seg_review_gui.py
```

Dataset preparation creates weak pseudo-masks from cropped human sperm positives.
The human images do not include hand-authored masks, so the script extracts the
largest dominant foreground contour and writes a YOLO segmentation polygon.

Important detail: these masks are weak labels, not ground truth.

Training uses Ultralytics YOLOv8 segmentation with `yolov8n-seg.pt`.

Comment: YOLO metrics are high on the generated human pseudo-mask dataset, but
that does not prove transfer to mouse Ward full-sperm cropping. The June review
bundle shows transfer is limited, especially on 40x Ward images.

## 4.4 YOLO Weak Full-Sperm Detector

Main files:

```text
scripts/prepare_yolo_full_sperm_dataset.py
scripts/train_yolov8_full_sperm.py
```

This older/supporting path builds weak YOLO detection labels by assigning one
near-full-image bounding box per positive crop. It can include:

- human positive cropped sperm,
- mouse positive images exported from NMA single-cell outputs.

No generated `datasets/yolo_full_sperm_no_annotations/` directory or full-sperm
training summary was present in the current workspace scan, so this appears to
be a prepared experiment path rather than the current active run.

## 4.5 NMA Workflow

NMA is present through training data and review-bundle outputs. Its realistic
role is nucleus/head localization or morphology support. It does not solve
full-tail cropping.

The June review bundle used the existing NMA 2.4.0 standalone jar and found 52
nuclei across 12 PNG-compatible images.

Comment: NMA is useful as a head/nucleus signal, but not as a full-sperm cropper.

## Checkpoint 5 - Current Outputs and Metrics

## 5.1 Current SCP Output Layout

Canonical current path-v2 output:

```text
Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs_path_v2/
```

Important subfolders:

```text
intermediates/
masks/
overlays/
head_connected_overlays/
sperm_crops/
highlighted_sperm_crops/
tail_skeletons/
candidate_review/rejected/
candidate_review/ambiguous/
tail_assignment_debug/
json/
summary.csv
summary.json
```

Current path-v2 artifact counts:

- 24 per-image JSON files.
- 22 accepted highlighted crop images.
- 45 rejected highlighted candidate images.
- 142 ambiguous highlighted candidate images.
- 209 total candidate rows in `candidate_review_all_candidates.csv`.
- 184 review rows in `candidate_review_queue.csv` plus header.

## 5.2 SCP Run Comparison

| Output run | Images | Tail comps | Head comps | Overlap comps | Candidates | Accepted | Rejected | Ambiguous | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `legacy_outputs/tail_mask_run` | 24 | 393 | 325 | 180 | n/a | n/a | n/a | n/a | Early mask/component run before candidate buckets |
| `legacy_outputs/tail_mask_run_width_spike` | 24 | 393 | 325 | 180 | n/a | n/a | n/a | n/a | Width-spike/shared-tail debug generation |
| `legacy_outputs/tail_mask_run_shared_tail_fix` | 24 | 393 | 325 | 180 | n/a | n/a | n/a | n/a | Shared-tail splitting iteration |
| `updated_outputs` | 24 | 393 | 325 | 180 | 213 | 97 | 35 | 81 | Older v1 candidate buckets; manually reviewed |
| `updated_outputs_path_v2` | 24 | 393 | 325 | 180 | 209 | 22 | 45 | 142 | Current conservative path-v2 run |
| `updated_outputs_path_v2_smoke` | 1 | 59 | 10 | 22 | 4 | 1 | 0 | 3 | Smoke run |
| `updated_outputs_revamp_full_2026_07_02` | 24 | 393 | 325 | 180 | 209 | 22 | 41 | 146 | Full revamp run with 209 diagnostics |
| `updated_outputs_revamp_full_2026_07_02_cached` | 24 | 393 | 325 | 180 | 209 | 22 | 41 | 146 | Cached copy of revamp full run |
| `ablation_2026_07_03_geometry_only` | 24 | 393 | 325 | 180 | 209 | 29 | 32 | 148 | Geometry-only path scoring comparison |
| `ablation_2026_07_03_color_only` | 24 | 393 | 325 | 180 | 209 | 24 | 45 | 140 | Color-only path scoring comparison |
| `ablation_2026_07_03_hybrid` | 24 | 393 | 325 | 180 | 209 | 19 | 44 | 146 | Hybrid ablation comparison |

Comment: all full SCP runs see the same base component counts because the
high-recall mask/head/tail detection stage is stable. The major changes are in
assignment, path scoring, candidate scoring, and status arbitration.

## 5.3 Current Path-V2 Candidate Status Metrics

Run:

```text
Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs_path_v2/
```

Summary:

| Status | Count |
| --- | ---: |
| accepted | 22 |
| ambiguous | 142 |
| rejected | 45 |

Rejected reasons:

| Reason | Count |
| --- | ---: |
| `tiny_tail_fragment` | 27 |
| `head_outline_false_tail` | 18 |

Ambiguity reasons:

| Reason | Count |
| --- | ---: |
| `mask_branch_extra_tail` | 83 |
| `ambiguous_path_conflict` | 25 |
| `low_color_continuity` | 21 |
| `ambiguous_path_color_margin` | 9 |
| `mask_tail_partial` | 2 |
| `foreign_content_in_crop` | 1 |
| `unresolved_shared_tail_overlap` | 1 |

Top risk flags:

| Risk flag | Count |
| --- | ---: |
| `path_low_margin` | 171 |
| `path_trimmed_extra_branch` | 171 |
| `tail_overlap_component` | 149 |
| `mask_branch_extra_tail` | 141 |
| `foreign_content_in_crop` | 108 |
| `shared_tail_component` | 101 |
| `low_color_continuity` | 41 |
| `ambiguous_path_conflict` | 31 |
| `crop_bbox_touches_image_border` | 21 |
| `head_outline_false_tail` | 18 |
| `source_bbox_touches_image_border` | 15 |
| `ambiguous_path_color_margin` | 13 |

Comment: path-v2 is hitting the exact difficult cases: branch-heavy masks,
low-margin paths, shared components, foreign content, and low color continuity.
The high number of risk flags on 209 candidates explains why 142 went to
ambiguous review.

## 5.4 Current Path-V2 Score Averages

| Status | Count | Score avg | Score min | Score max | Crop quality avg | Mask quality avg | Path confidence avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accepted | 22 | 66.61 | 38.33 | 81.72 | 79.32 | 59.28 | 83.10 |
| ambiguous | 142 | 30.20 | -24.60 | 62.69 | 63.71 | 46.84 | 80.60 |
| rejected | 45 | -31.39 | -71.51 | 5.35 | 60.30 | 35.05 | 53.54 |

Accepted assignment modes:

| Assignment mode | Accepted count |
| --- | ---: |
| `single_head_component` | 21 |
| `split_shared_tail` | 1 |

Comment: many ambiguous candidates still have high path confidence. They are
often ambiguous because of low margins, extra branches, shared components, or
color/path conflicts rather than simply because no path exists.

## 5.5 Manual Review Baseline for Older `updated_outputs`

Completed review CSV:

```text
Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs/candidate_review/candidate_review_queue.csv
```

Baseline analysis:

```text
Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs/candidate_review/baseline_analysis/
```

Review totals:

- 131 total rows.
- 131 labeled rows.
- 0 unlabeled rows.

Crop labels:

| Crop label | Count | Percent |
| --- | ---: | ---: |
| `partial_or_cut_off` | 92 | 70.23 |
| `good_full_sperm` | 30 | 22.90 |
| `debris_or_false_positive` | 6 | 4.58 |
| `merged_multi_sperm` | 2 | 1.53 |
| `unclear` | 1 | 0.76 |

Mask labels:

| Mask label | Count | Percent |
| --- | ---: | ---: |
| `tail_mask_partial` | 66 | 50.38 |
| `wrong_tail_assigned` | 20 | 15.27 |
| `mask_good` | 12 | 9.16 |
| `mask_unclear` | 12 | 9.16 |
| `extra_tail_assigned` | 11 | 8.40 |
| `head_mask_wrong` | 10 | 7.63 |

Status labels in that reviewed queue:

| Current status | Count | Percent |
| --- | ---: | ---: |
| `ambiguous` | 81 | 61.83 |
| `accepted` | 30 | 22.90 |
| `rejected` | 20 | 15.27 |

Top assignment/status reasons:

| Status | Assignment reason | Count | Percent |
| --- | --- | ---: | ---: |
| ambiguous | `unresolved_shared_tail_overlap` | 61 | 46.56 |
| accepted | `single_head_component` | 21 | 16.03 |
| ambiguous | `foreign_content_in_crop` | 20 | 15.27 |
| rejected | `tiny_tail_fragment` | 17 | 12.98 |
| accepted | `split_shared_tail` | 7 | 5.34 |
| rejected | `duplicate_lower_scoring_candidate` | 3 | 2.29 |
| accepted | `filtered_single_shared_tail` | 2 | 1.53 |

Comment: this manual review explains the path-v2 pivot. The older run accepted
more crops, but many reviewed candidates were partial/cut-off or had partial,
wrong, or extra tail masks.

## 5.6 YOLO Human Pseudo-Mask Dataset

Dataset:

```text
datasets/yolo_sperm_seg_human_pseudo/
```

Dataset summary:

- 2,609 human images.
- 2,087 train images.
- 260 validation images.
- 262 test images.
- All masks generated by `foreground_contour`.
- One class: `full_sperm`.

Human category distribution:

| Category | Count |
| --- | ---: |
| `Partial Agreement/Abnormal` | 1,201 |
| `Disagreement` | 562 |
| `Partial Agreement/Normal` | 343 |
| `Full Agreement/Abnormal` | 262 |
| `Notl abelled` | 148 |
| `Full Agreement/Normal` | 93 |

Comment: category names come from the source human dataset. The typo
`Notl abelled` is present in the dataset summary and appears to be inherited
from the extracted directory/category naming.

## 5.7 YOLOv8 Segmentation Training Metrics

Run:

```text
runs/segment/runs/segment/human_pseudo_yolov8nseg_gpu/
```

Training config:

| Setting | Value |
| --- | --- |
| model | `yolov8n-seg.pt` |
| data | `datasets/yolo_sperm_seg_human_pseudo/data.yaml` |
| epochs | 50 |
| image size | 640 |
| batch | 16 |
| device | `0` |
| workers | 4 |
| patience | 12 |
| seed | 42 |

Validation metrics:

| Metric group | mAP50 | mAP50-95 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| box | 0.9915 | 0.8562 | 0.9733 | 0.9803 | 0.9768 |
| mask | 0.9918 | 0.6155 | 0.9734 | 0.9834 | 0.9783 |

Test metrics:

| Metric group | mAP50 | mAP50-95 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| box | 0.9887 | 0.8520 | 0.9644 | 0.9695 | 0.9669 |
| mask | 0.9900 | 0.6271 | 0.9644 | 0.9695 | 0.9669 |

Comment: these are strong internal metrics on weak pseudo-labels. They should
not be interpreted as true full-sperm mouse cropping performance.

## 5.8 June 2026 Review Bundle

Bundle:

```text
review_bundles/next_steps_review_bundle_2026-06-21/
```

Purpose: compare SCP, YOLO, and NMA on a shared 12-image set.

Selected images:

- 6 Ward images.
- 6 raw Yan pilot images.
- Four Ward images also have copied SCP outputs.

YOLO human-only review-bundle results:

- 12 images.
- 7 images with detections.
- 5 images without detections.
- 12 total detections.
- Mean detections per image: 1.0.
- Mean detection confidence: 0.4289.
- All three listed 40x Ward images in the named inference CSV had 0 detections.

NMA review-bundle results:

- 12 PNG-compatible images analyzed.
- 52 nuclei detected.
- 52 single-cell images.
- 52 single-cell annotations.

Comment: the shared review bundle supports the current project direction. YOLO
does not transfer strongly enough to replace SCP, and NMA detects nuclei/heads,
not full tails.

## Checkpoint 6 - Project Progression

Because Git history is unavailable, this progression is reconstructed from code,
docs, and output artifacts.

### Stage 1 - Data Collection and Model-First Baselines

The project started with:

- human positive cropped sperm images,
- mouse positive crops from NMA outputs,
- raw Yan pilot data,
- YOLOv8 checkpoint files,
- scripts for weak full-sperm detection and segmentation.

The model-first idea was reasonable: build a detector/segmenter from available
positive crops, then use it on raw images.

### Stage 2 - Human Pseudo-Mask YOLO Segmentation

The project created `datasets/yolo_sperm_seg_human_pseudo/` by extracting weak
foreground-contour masks from 2,609 human sperm crops.

YOLOv8 segmentation trained successfully and achieved high internal metrics on
validation/test splits.

The problem: those metrics measure performance against generated pseudo-labels
on cropped human data. They did not translate into reliable full mouse sperm
cropping from raw Ward images.

### Stage 3 - YOLO/NMA Shared Review Bundle

The June 21 review bundle compared:

- YOLO human-only inference,
- NMA head/nucleus analysis,
- SCP color-mask examples.

YOLO detected only 12 objects across 7 of 12 images, with low-to-moderate
confidence. NMA found nuclei but did not solve tail detection or full-sperm
cropping.

This pushed the project toward SCP as the main cropper.

### Stage 4 - SCP V1 High-Recall Candidate Generation

SCP v1 generated:

- 213 crop candidates,
- 97 accepted crops,
- 35 rejected candidates,
- 81 ambiguous candidates.

This was a major improvement because it produced full reviewable crop candidates
from real Ward images without relying on trained models.

Manual review then showed that many candidates still had problems:

- partial or cut-off sperm,
- tail masks cut off early,
- wrong tail branch assigned,
- extra tail assigned,
- thick tail sections treated as heads,
- head outlines treated as tail signal,
- intersection/overlap confusion.

### Stage 5 - Shared-Tail Fixes and Width-Spike Logic

Legacy outputs show iterations focused on:

- width-spike detection,
- shared-tail splitting,
- directional width barriers,
- multi-head tail assignment.

These changes targeted crossings and crowded components.

### Stage 6 - Path-V2 Revamp

Path-v2 added a more explicit head-to-endpoint skeleton path selection layer.
It scores paths using geometry, color continuity, head-collar exit behavior,
branch penalties, width penalties, and path score margins.

This led to:

- fewer accepted crops,
- many more ambiguous candidates,
- explicit `head_outline_false_tail` rejections,
- better diagnostic metadata for why candidates are risky.

Current path-v2 run:

- 209 candidates.
- 22 accepted.
- 45 rejected.
- 142 ambiguous.

Comment: path-v2 changed the project from "generate many plausible crops" to
"only trust crops with stronger path evidence." That is a useful checkpoint, but
it creates the current throughput wall.

### Stage 7 - Review Tooling and Baseline Analysis

The project added:

- a candidate review queue builder,
- label key definitions,
- a Tkinter review GUI,
- baseline analysis summaries.

This is important because future progress now depends on measured review
outcomes, not only visual impressions.

## Checkpoint 7 - Current Wall and Failure Modes

The current wall is a precision/coverage tradeoff.

SCP can detect many candidate structures, but the final accepted set is small
because the system is trying to avoid known failure modes:

- selecting a wrong branch at intersections,
- trimming away real tail loops,
- including extra tail branches,
- cutting off tails near image or crop boundaries,
- mistaking head outlines for short tails,
- mistaking thick tail segments for heads,
- handling shared-tail components with multiple nearby heads,
- dealing with low color contrast or indistinguishable crossing tails,
- handling loops where skeleton topology and true biology diverge.

The strongest current numeric signal:

- path-v2 produced 142 ambiguous candidates out of 209.
- 83 ambiguous candidates were `mask_branch_extra_tail`.
- 25 were `ambiguous_path_conflict`.
- 21 were `low_color_continuity`.
- 18 rejected candidates were `head_outline_false_tail`.

Comment: path-v2 is identifying the hard cases instead of silently accepting
them. That is progress, but the project now needs a better way to separate
"ambiguous but good enough" from "ambiguous and truly wrong."

## Checkpoint 8 - Recommended Analysis Next

These are analysis checkpoints rather than broad rewrites.

### 1. Label the Current Path-V2 Review Queue

The current v2 queue has 184 review candidates plus header. Label it using the
same crop and mask labels used for the v1 baseline.

Goal: compare v1 and v2 by human labels, not just accepted/rejected counts.

### 2. Build a Small Gold Image Set With Missed-Sperm Labels

The current candidate review measures generated candidates. It does not measure
visible sperm that SCP missed entirely.

A minimal gold set should label, per source image:

- visible full sperm count,
- good crop candidates,
- duplicates,
- missed full sperm,
- partial sperm,
- merged multi-sperm crops,
- debris/false positives.

Goal: start measuring recall, not only candidate quality.

### 3. Compare Conservative vs Balanced Review Profiles

The code already has `--acceptance-profile conservative|balanced-review`.

Run both profiles on the same images and compare:

- accepted count,
- good full sperm rate,
- partial/cut-off rate,
- wrong-tail rate,
- extra-tail rate,
- false-positive rate.

Goal: determine whether the current accepted set is too conservative.

### 4. Audit Ambiguous Buckets Separately

The largest current ambiguous bucket is `mask_branch_extra_tail`. This should be
split into visually meaningful subtypes:

- true extra branch included,
- true tail loop trimmed,
- wrong branch selected,
- correct skeleton but mask too broad,
- crop boundary problem,
- visually impossible intersection.

Goal: avoid tuning one threshold against several different failure mechanisms.

### 5. Turn Review Results Into Threshold Changes

Likely threshold families to tune after v2 review:

- path score margin threshold,
- mask quality cutoff,
- allowed trimmed branch fraction,
- color continuity cutoff,
- foreign content thresholds,
- head-collar exit requirements,
- boundary-risk handling.

Goal: make threshold changes traceable to review evidence.

### 6. Preserve YOLO and NMA as Support Signals

YOLO and NMA should not be the primary cropper right now, but they may help with:

- rough ROI proposals,
- head candidate validation,
- ambiguity flags,
- false-positive filtering,
- hard-negative mining.

Goal: use them where they match their strengths rather than asking them to solve
full-tail crop generation alone.

## Checkpoint 9 - Common Commands

Run current SCP path-v2 full batch:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py \
  --output-root outputs/updated_outputs_path_v2 \
  --overwrite
```

Run SCP smoke test:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py \
  --output-root outputs/updated_outputs_path_v2_smoke \
  --limit 2 \
  --overwrite
```

Build candidate review CSV:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/build_candidate_review_csv.py \
  --output-root outputs/updated_outputs_path_v2
```

Analyze completed review CSV:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/analyze_candidate_review_baseline.py \
  --review-csv outputs/updated_outputs_path_v2/candidate_review/candidate_review_queue.csv
```

Run synthetic unit tests:

```bash
cd Algorithmic_Pixel_Mask_For_Tails
../.venv/bin/python -m unittest test_tail_assignment.py
```

Train YOLOv8 human pseudo-mask segmentation:

```bash
.venv/bin/python scripts/train_yolov8_sperm_seg.py \
  --data datasets/yolo_sperm_seg_human_pseudo/data.yaml \
  --model yolov8n-seg.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 4 \
  --patience 12 \
  --name human_pseudo_yolov8nseg_gpu
```

## Final Assessment

The project has made real progress. It has:

- a clear primary method,
- reproducible SCP outputs,
- structured candidate metadata,
- manual review tooling,
- concrete failure labels,
- synthetic tests for hard assignment cases,
- evidence that YOLO/NMA should remain supporting tools.

The wall is now measurable: path-v2 greatly reduces risky acceptances but leaves
most candidates ambiguous. The next meaningful step is not a new algorithm from
scratch. It is a labeled comparison of v1 and v2, followed by focused threshold
and subtype analysis on the ambiguous buckets.

Comment: SCP is close enough to be worth measuring carefully, but not yet stable
enough to trust as an automatic full-sperm cropper without review.
