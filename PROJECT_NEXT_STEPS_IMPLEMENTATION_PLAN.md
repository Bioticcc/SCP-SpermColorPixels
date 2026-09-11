# Sperm Morphology Recognition
## Project Next Steps and Detailed Implementation Plan

**Prepared:** July 13, 2026
**Primary project:** `Sperm_Morphology_Recognition`
**Primary active method:** SCP (Sperm Color Pixels)
**Source analysis:** `PROJECT_ANALYSIS_2026-07-13.md`
**Intended use:** Execute one work package at a time with Codex. Do not provide the entire plan to an implementation agent as one undifferentiated task.

**Active execution update — September 7, 2026:** The user has directed that new
lab images and human annotations be treated as optional, because their arrival
is uncertain and demonstrable algorithm progress is needed first. Follow
[`ANNOTATION_OPTIONAL_EXECUTION_PLAN_2026-09-07.md`](ANNOTATION_OPTIONAL_EXECUTION_PLAN_2026-09-07.md)
for the current R0–R6 development sequence. It supersedes the gold-annotation
prerequisites and execution order below for those work packages. Use synthetic
geometry tests, existing-image regressions, and visual demonstrations as the
development gates. Preserve the gold-data/evaluator specifications below for
optional future validation; their accuracy criteria remain unverified until
appropriate labels exist. Continue to implement only the explicitly requested
bounded work package, with SCP as the primary method and no default model
training.

---

## Document Purpose

This document converts the current project analysis into an executable development roadmap. It is designed to answer four practical questions:

1. What must be built next?
2. What must be annotated manually, and in what format?
3. How should overlapping tails be resolved without requiring a large training dataset?
4. What exact, bounded work package should be given to Codex at each stage?

The immediate objective is not morphology classification. The immediate objective is to produce one defensible, complete sperm instance per visible sperm whenever the image contains enough evidence, while identifying truly indeterminate cases instead of fabricating a confident assignment.

---

## Executive Decision

The project should continue with SCP as the primary image-processing foundation. The current bottleneck is no longer broad sperm-pixel detection. It is instance decomposition: assigning a biologically plausible tail track to each head when multiple tails touch, cross, loop, or form one connected mask.

The next implementation should therefore use this strategy:

1. Build a manually annotated gold dataset from the 24 Ward images.
2. Freeze and measure the current path-v2 baseline against that gold dataset.
3. Replace independent path selection with a direction-aware, multi-hypothesis, globally optimized tail-assignment system.
4. Reconstruct each tail mask from its selected centerline rather than from the entire connected component.
5. Audit the remaining failures.
6. Introduce a narrowly scoped learned component only if the residual failures demonstrate a specific visual signal that deterministic processing cannot extract reliably.

The 24 Ward images are sufficient for annotation, evaluation, algorithm development, threshold calibration, and a limited auxiliary-model feasibility test. They are not sufficient by themselves for a robust end-to-end full-frame instance-segmentation model across diverse future imaging conditions.

---

# 1. Current State and Why the Plan Changes the Core Assignment Logic

## 1.1 Current strengths

The project already has several strong foundations:

- A deterministic high-recall sperm-color mask for Ward RGB images.
- Head and tail component detection.
- Tail skeletonization and graph construction.
- Endpoint, branchpoint, width-spike, overlap, and boundary diagnostics.
- Path-v2 candidate-path scoring using geometry and color continuity.
- Accepted, rejected, and ambiguous candidate buckets.
- Rich visual artifacts and structured JSON/CSV output.
- A manual candidate-review GUI and review labels.
- Synthetic unit tests for several tail-assignment edge cases.

These assets should be preserved and reused.

## 1.2 Current evidence

The current path-v2 run processes 24 Ward images and generates 209 candidate crops, of which 22 are accepted, 45 rejected, and 142 ambiguous. The older run accepted many more crops, but manual review found that partial or cut-off sperm and partial or incorrectly assigned tails were common.

Path-v2 is therefore valuable because it recognizes risk, but it has not yet resolved the difficult cases. The fact that almost all accepted crops are simple single-head components shows that shared-tail decomposition remains unsolved.

## 1.3 Root cause

The binary mask and skeleton encode image connectivity, not biological identity. At an X-shaped crossing, a skeleton graph sees one branchpoint with multiple possible continuations. A biological interpretation requires deciding which pair of branches forms one smooth tail and which pair forms the other.

Independent path selection is insufficient because a locally plausible path for one head may conflict with the best path for another head. The assignment must be solved jointly across all heads and candidate paths in the connected component.

## 1.4 Central architectural change

The overlap core should change from:

```text
connected tail component
    -> skeleton graph
    -> choose one path independently for each head
    -> trim branches
    -> score candidate
```

To:

```text
continuous tail evidence + local orientation
    -> direction-aware skeleton state graph
    -> top-K path hypotheses per head
    -> global multi-head path assignment
    -> centerline-based instance-mask reconstruction
    -> calibrated local, global, and completeness confidence
```

---

# 2. Project Principles and Constraints

## 2.1 Preserve what already works

Do not redesign the high-recall mask, isolated-sperm path, output artifacts, or review workflow without evidence of a specific failure. The first structural changes should be isolated to graph representation, path generation, global assignment, and reconstruction.

## 2.2 Make every change measurable

Every algorithmic phase must be evaluated against the same locked gold dataset and the same frozen baseline configuration. Candidate counts alone are not sufficient. The project must measure missed sperm, wrong associations, duplicates, merged crops, and tail completeness.

## 2.3 Keep ambiguous cases explicit

Some crossings cannot be resolved from one static RGB image. The correct output for those cases is an ambiguity state with alternate hypotheses, not a forced assignment.

## 2.4 Avoid a large annotation burden

The first annotation set should use heads, neck points, centerlines, endpoints, crossings, and instance identities. It should not require perfect full-width pixel polygons around every tail. SCP-derived width and masks can later convert centerlines into approximate full masks.

## 2.5 Avoid premature model training

Do not train an end-to-end full-sperm instance model during Phase 1 or Phase 2. Introduce learning only after the deterministic resolver has been evaluated and its remaining errors have been categorized.

## 2.6 Prevent data leakage

All annotations and derived patches from one source image must remain in the same dataset partition. Never split sperm crops from one image across training and validation/test sets.

## 2.7 Require inspectable outputs

Every new stage must save diagnostics sufficient to explain why a path or assignment was selected, rejected, or marked ambiguous.

---

# 3. Target Repository Structure

The current main script is large and combines many responsibilities. Refactoring should be incremental and behavior-preserving.

Recommended eventual structure:

```text
Sperm_Morphology_Recognition/
  pyproject.toml
  README.md
  ProjectContext.md
  AGENT.md

  configs/
    ward_high_recall.yaml
    path_v2_baseline.yaml
    crossing_graph_v1.yaml
    annotation_gui.yaml

  Algorithmic_Pixel_Mask_For_Tails/
    algorithmic_tail_mask.py          # temporary compatibility entry point
    scp/
      __init__.py
      config.py
      models.py
      preprocessing.py
      tail_evidence.py
      head_detection.py
      head_anchors.py
      skeletonization.py
      graph_construction.py
      junction_transitions.py
      path_hypotheses.py
      global_assignment.py
      mask_reconstruction.py
      candidate_scoring.py
      evaluation.py
      artifacts.py

    annotation/
      annotation_gui.py
      annotation_models.py
      annotation_io.py
      annotation_validation.py
      annotation_export.py

    tests/
      test_tail_assignment.py
      test_annotation_io.py
      test_annotation_validation.py
      test_junction_transitions.py
      test_global_assignment.py
      test_mask_reconstruction.py
      fixtures/

  annotations/
    ward_gold_v1/
      manifest.json
      images/                       # references or symlinks, not duplicate originals if avoidable
      json/
      previews/
      exports/
      reports/

  outputs/
    baselines/
    experiments/
```

The refactor must not be performed all at once. First create the annotation subsystem, then freeze a baseline, and only afterward extract modules needed by the new algorithm.

---

# 4. Annotation Dataset Design

## 4.1 Purpose of the annotations

The 24 Ward images should become a gold dataset used for:

- Measuring visible-sperm recall.
- Measuring one-to-one instance accuracy.
- Evaluating tail completeness.
- Evaluating head-to-tail association.
- Evaluating crossing continuation decisions.
- Comparing path-v2 with the new resolver.
- Identifying residual failures that may justify a learned component.

The annotations are primarily an evaluation and algorithm-development dataset. They are not initially a large neural-network training dataset.

## 4.2 Annotation unit

The primary annotation unit is one visible sperm instance in one source image.

Each instance should include:

- A stable sperm instance ID within the image.
- A head annotation.
- A neck attachment point.
- A tail centerline.
- A distal endpoint when visible.
- Visibility/completeness status.
- Boundary status.
- Crossing participation.
- Reviewer certainty.
- Optional notes.

## 4.3 Minimal required annotation fields

### Image-level fields

```json
{
  "schema_version": "1.0.0",
  "dataset_id": "ward_gold_v1",
  "image_id": "stable-relative-path-or-hash",
  "source_path": "relative/path/to/image.tif",
  "width": 0,
  "height": 0,
  "magnification": "40x_or_100x_or_unknown",
  "annotation_status": "not_started|in_progress|review_needed|complete|locked",
  "annotator": "",
  "reviewer": "",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "notes": "",
  "sperm_instances": [],
  "crossings": []
}
```

### Sperm instance fields

```json
{
  "instance_id": "sperm_001",
  "head": {
    "type": "ellipse|polygon|point_only|not_visible",
    "center": [0.0, 0.0],
    "axes": [0.0, 0.0],
    "angle_degrees": 0.0,
    "polygon": []
  },
  "neck_point": [0.0, 0.0],
  "tail_centerline": [[0.0, 0.0], [0.0, 0.0]],
  "distal_endpoint": [0.0, 0.0],
  "distal_endpoint_visible": true,
  "instance_status": "full|partial|boundary_truncated|occluded|uncertain",
  "head_visibility": "full|partial|not_visible|uncertain",
  "tail_visibility": "full|partial|uncertain",
  "certainty": "high|medium|low|indeterminate",
  "crossing_ids": [],
  "notes": ""
}
```

### Crossing fields

```json
{
  "crossing_id": "crossing_001",
  "center": [0.0, 0.0],
  "radius_pixels": 0.0,
  "involved_instance_ids": ["sperm_001", "sperm_002"],
  "determinacy": "determinable|indeterminate|uncertain",
  "continuations": [
    {
      "instance_id": "sperm_001",
      "incoming_point": [0.0, 0.0],
      "outgoing_point": [0.0, 0.0]
    }
  ],
  "notes": ""
}
```

## 4.4 Annotation conventions

### Head

Use an ellipse when the head boundary is reasonably visible. Use a polygon only when the shape cannot be represented adequately by an ellipse. Use `point_only` only for a severely obscured head that is still identifiable.

### Neck point

Place the neck point at the estimated head-tail attachment. It should be one point, not a region. This point defines the origin and initial direction of the tail track.

### Tail centerline

Draw a polyline through the center of the visible tail. Use enough points to preserve curvature, but do not place a point at every pixel. Add points before and after crossings so the intended continuation is explicit.

### Distal endpoint

Mark the visible end of the tail. If the tail leaves the image, is cut off, or cannot be followed reliably, set `distal_endpoint_visible` to false and classify the instance appropriately.

### Crossings

Create a crossing object whenever two or more logical tail tracks touch or overlap in the image. Associate the crossing with all involved instance IDs. If the continuation is visually clear, record it. If the image does not contain enough evidence, mark the crossing indeterminate.

### Partial sperm

Annotate partial and boundary-truncated sperm rather than ignoring them. They are needed to evaluate whether the algorithm correctly rejects or classifies incomplete instances.

### Debris and non-sperm structures

Do not annotate debris as sperm. A later optional negative-region annotation can be added for recurring false positives, but this is not required in the first pass.

## 4.5 Annotation quality process

Each image should pass through these states:

```text
not_started -> in_progress -> review_needed -> complete -> locked
```

Recommended self-review process for a sole annotator:

1. Annotate all heads and assign instance IDs.
2. Draw neck points and centerlines.
3. Add crossings and continuation decisions.
4. Mark completeness and certainty.
5. Hide the source overlay and review annotations alone.
6. Re-enable the image and inspect every crossing at high zoom.
7. Run automatic validation.
8. Mark the image complete.
9. Revisit difficult images after completing several other images, reducing immediate confirmation bias.
10. Lock the gold set only after the baseline evaluator can consume every file without warnings.

---

# 5. Manual Annotation GUI Specification

## 5.1 Technology choice

Use Python with Tkinter and Pillow because the repository already uses Tkinter and PIL for review tooling. This minimizes dependencies and allows reuse of existing image-loading, path-resolution, and GUI conventions.

The annotation data model and I/O logic must be independent of Tkinter so that another frontend could replace the GUI later without changing the annotation format.

## 5.2 Main window layout

```text
+--------------------------------------------------------------------------------+
| File / Image controls | Tool controls | View toggles | Save / Validate / Export |
+----------------------------------------------+---------------------------------+
|                                              | Image metadata                  |
|                                              | Annotation object tree          |
|                  Image canvas                | Selected object properties      |
|             zoomable and pannable            | Validation messages             |
|                                              | Review status and notes         |
|                                              |                                 |
+----------------------------------------------+---------------------------------+
| Image 3/24 | zoom 240% | selected sperm_004 | autosaved | 2 warnings           |
+--------------------------------------------------------------------------------+
```

## 5.3 Required tools

1. **Select/Edit** - select, move, and edit annotation points.
2. **Pan** - move the viewport without changing annotations.
3. **Head ellipse** - click-drag an ellipse and rotate/resize it.
4. **Head polygon** - optional polygon tool for irregular heads.
5. **Neck point** - place or move the attachment point.
6. **Tail polyline** - click to add centerline points; double-click or Enter to finish.
7. **Distal endpoint** - normally derived from the final centerline point, with explicit visibility toggle.
8. **Crossing region** - click-drag a circle or place a center and adjust radius.
9. **Crossing continuation** - choose an instance and mark incoming/outgoing points through the crossing.
10. **Erase/Delete** - delete the selected object with confirmation for whole-instance deletion.
11. **Split/Join centerline** - correct centerline editing without redrawing the entire track.
12. **SCP overlay toggle** - show existing masks, skeletons, head candidates, or path-v2 suggestions as optional aids.

## 5.4 Interaction requirements

- Smooth mouse-wheel zoom centered on the cursor.
- Middle-button or Space-drag panning.
- Coordinates stored in original image space, never screen space.
- Zoom must not change annotation coordinates.
- Undo and redo for all annotation changes.
- Autosave after every completed edit using an atomic temporary-file replacement.
- Manual save command.
- Previous/next image navigation.
- Filter navigation to incomplete images, warning images, 40x images, 100x images, or images containing crossings.
- Visible current tool and selected instance.
- Color-coded instance annotations with a deterministic palette.
- Adjustable line width and point size for high-resolution TIFF images.
- Ability to hide all annotations except the selected sperm.
- Ability to dim unselected instances.
- Ability to hide the image and inspect centerline topology alone.

## 5.5 Recommended keyboard shortcuts

| Action | Shortcut |
|---|---|
| Select/Edit | `V` |
| Pan | `H` or hold `Space` |
| New sperm instance | `N` |
| Head ellipse | `E` |
| Head polygon | `G` |
| Neck point | `K` |
| Tail polyline | `T` |
| Crossing | `C` |
| Finish current drawing | `Enter` |
| Cancel current drawing | `Esc` |
| Delete selected object | `Delete` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` or `Ctrl+Shift+Z` |
| Save | `Ctrl+S` |
| Validate | `Ctrl+Shift+V` |
| Previous image | `PageUp` |
| Next image | `PageDown` |
| Toggle SCP overlay | `O` |
| Show selected instance only | `I` |
| Fit image to window | `F` |
| Zoom 100% | `1` |

## 5.6 Object tree and properties panel

The right-side tree should show:

```text
Image
  sperm_001
    head
    neck
    tail centerline (37 points)
    endpoint
    crossing_002
  sperm_002
  crossing_001
  crossing_002
```

Selecting an item should expose editable properties without requiring direct canvas manipulation. Important properties include status, certainty, visibility, notes, magnification, and crossing determinacy.

## 5.7 Validation rules

Validation should report errors and warnings separately.

### Errors

- Duplicate instance IDs.
- Invalid or missing schema version.
- Annotation coordinates outside image bounds.
- Tail centerline with fewer than two points when a tail is declared visible.
- Crossing references to nonexistent sperm IDs.
- Invalid JSON or incomplete required fields.
- A locked file modified without explicit unlock action.

### Warnings

- Missing head on an instance not marked `head_visibility=not_visible`.
- Missing neck point.
- Neck point unusually far from the head boundary.
- First centerline point unusually far from the neck.
- Distal endpoint differs substantially from the final centerline point.
- A centerline intersects another centerline but no crossing object is present nearby.
- A crossing contains fewer than two involved instances.
- A full sperm centerline reaches the image boundary.
- A boundary-truncated sperm does not approach an image boundary.
- An instance is marked high certainty but participates in an indeterminate crossing.

Thresholds used by validation must be configurable and recorded in the validation report.

## 5.8 Autosave and failure safety

Saving must follow this sequence:

1. Serialize to a temporary file in the same directory.
2. Validate JSON serialization.
3. Flush and close the file.
4. Atomically replace the prior annotation file.
5. Keep a bounded backup history, such as the most recent five versions per image.

The GUI should recover from a crash by loading the latest valid autosave or backup. It must never silently overwrite a newer annotation file.

## 5.9 Optional SCP-assisted annotation

The GUI may load the following existing artifacts when available:

- Cleaned sperm-color mask.
- Head candidate overlay.
- Tail skeleton.
- Path-v2 selected path.
- Accepted/rejected/ambiguous crop boxes.

These are visual aids only. They must not modify gold annotations unless the user explicitly chooses an import command.

A later convenience feature may initialize centerline points from a selected SCP skeleton path. Imported annotations must be marked with provenance and require manual confirmation.

## 5.10 Exports

The annotation subsystem should support:

- Canonical per-image JSON.
- Dataset manifest JSON.
- Human-readable CSV summary.
- Preview PNG with instance IDs.
- Head mask raster.
- Centerline raster with per-instance IDs.
- Endpoint heatmap.
- Neck-point heatmap.
- Crossing mask.
- Local orientation map derived from centerlines.
- Optional COCO-compatible head polygons.

The canonical JSON remains the source of truth. Derived exports can be regenerated.

---

# 6. Evaluation Framework

## 6.1 Baseline freezing

Before changing algorithm behavior:

- Record the exact current CLI command.
- Copy the active configuration into a versioned file.
- Store dependency versions.
- Store the input image manifest and hashes.
- Run path-v2 on all 24 images.
- Preserve outputs under a baseline-specific directory.
- Record a baseline identifier in all future comparisons.

Suggested baseline directory:

```text
outputs/baselines/path_v2_2026_07_13/
```

## 6.2 Matching predictions to gold instances

Predicted sperm candidates should be matched one-to-one to gold instances using a configurable composite similarity based on:

- Head overlap or head-center distance.
- Neck-point distance.
- Tail centerline overlap/coverage.
- Distal endpoint distance where applicable.
- Instance completeness compatibility.

Use a global bipartite assignment for matching predictions to gold instances. Do not greedily match candidates because duplicates and merged crops can distort the result.

## 6.3 Required metrics

### Detection and instance metrics

- Visible sperm count.
- Proposed sperm count.
- Matched true positives.
- Missed sperm.
- False-positive sperm candidates.
- Instance precision.
- Instance recall.
- Duplicate-crop rate.
- Merged-multi-sperm rate.

### Head-tail association metrics

- Correct head-to-tail association rate.
- Wrong-tail association rate.
- Correct crossing continuation rate.
- Indeterminate-crossing abstention rate.
- Forced-error rate on gold-indeterminate crossings.

### Tail quality metrics

- Centerline precision.
- Centerline recall.
- Centerline F-score.
- Fraction of gold tail length recovered.
- Fraction of predicted tail outside the gold track tolerance.
- Distal endpoint distance when visible.
- Premature tail termination rate.
- Extra-branch inclusion rate.

### Crop metrics

- Good full-sperm crop rate.
- Partial or cut-off crop rate.
- Boundary-truncated crop correctly classified.
- Foreign-content rate.
- One-to-one complete crop success rate.

## 6.4 Stratified reporting

Report every major metric separately for:

- Isolated sperm.
- Shared connected components.
- X-crossings.
- T-junction-like contacts.
- Self-loops.
- Low-contrast tails.
- Boundary-truncated sperm.
- 40x images.
- 100x images.

A change must not be accepted based only on aggregate performance if it regresses isolated sperm or one magnification group substantially.

## 6.5 Experiment output

Every experiment should create:

```text
experiment_root/
  config.yaml
  environment.json
  input_manifest.json
  metrics_summary.json
  metrics_by_image.csv
  metrics_by_subtype.csv
  prediction_matches.csv
  failures/
    missed/
    wrong_tail/
    duplicate/
    merged/
    extra_branch/
    premature_end/
    indeterminate_forced/
  diagnostics/
```

---

# 7. Phase-by-Phase Implementation Roadmap

## Phase 0 - Repository stabilization and baseline freeze

### Objective

Create a reproducible starting point before implementing annotation or changing SCP behavior.

### Scope

- Restore or initialize valid Git metadata if appropriate for the working repository.
- Add a dependency manifest.
- Add versioned configuration support without changing runtime behavior.
- Freeze the current path-v2 command and outputs.
- Record source image hashes and run metadata.

### Deliverables

- `pyproject.toml` or `requirements.txt` with documented Python version.
- `configs/path_v2_baseline.yaml` containing all effective settings.
- Baseline run directory and manifest.
- Reproducibility metadata writer.
- Documentation describing how to reproduce the baseline.

### Exclusions

- No tail-assignment changes.
- No threshold tuning.
- No output-status changes.
- No large file movement unless necessary.

### Exit criteria

- A clean environment can reproduce the same candidate counts and statuses, allowing only explicitly documented nondeterministic differences.
- The baseline contains all 24 source images in its manifest.
- Every future experiment can record configuration and environment metadata.

---

## Phase 1A - Annotation data model, I/O, validation, and GUI

### Objective

Build a reliable manual annotation application and canonical schema before annotating the Ward images.

### Scope

- Implement annotation dataclasses or typed models.
- Implement schema-versioned JSON load/save.
- Implement atomic autosave and backup rotation.
- Implement validation.
- Implement the Tkinter GUI specified in Section 5.
- Implement optional read-only SCP overlays.
- Add tests for I/O, coordinate transforms, undo/redo, and validation.

### Deliverables

- `annotation_gui.py` runnable from the command line.
- Canonical annotation models and JSON schema documentation.
- Dataset manifest generation.
- Preview export.
- Automated tests.
- User guide with shortcuts and annotation conventions.

### Exit criteria

- The GUI can open every Ward TIFF image.
- Zoom and pan preserve exact image-space coordinates.
- A complete sperm annotation can be created, edited, saved, closed, and reloaded without coordinate drift.
- Undo/redo covers all core annotation operations.
- Autosave recovery is demonstrated in a test or controlled manual check.
- Validation catches broken references and invalid coordinates.

---

## Phase 1B - Annotation pilot and schema revision

### Objective

Test the annotation workflow on a small but diverse subset before committing to all 24 images.

### Pilot selection

Select at least:

- One relatively simple isolated-sperm image.
- One crowded image with several crossings.
- One image from the other magnification group.

### Tasks

- Fully annotate the pilot images.
- Record friction points and ambiguous conventions.
- Revise the schema or GUI only when the pilot demonstrates a concrete need.
- Test exports and validation reports.
- Confirm that crossing continuations can be represented without awkward workarounds.

### Exit criteria

- The pilot annotations represent every encountered structure.
- No required biological decision is stored only in free-text notes.
- The schema reaches version `1.0.0` and is frozen for the full annotation pass, except for backward-compatible additions.

---

## Phase 1C - Annotate all 24 Ward images and lock `ward_gold_v1`

### Objective

Create the complete gold dataset.

### Annotation order

1. Images with mostly isolated sperm.
2. Images with simple two-tail crossings.
3. Crowded multi-head shared components.
4. Self-loops, low-contrast cases, and boundary-truncated sperm.

### Required quality controls

- Validate every image before marking complete.
- Review all low-certainty and indeterminate annotations after the first pass.
- Generate preview PNGs for all images.
- Generate a dataset-level summary of counts and subtypes.
- Lock files after completion.

### Deliverables

- 24 canonical annotation JSON files.
- Dataset manifest.
- Dataset summary.
- Preview images.
- Validation report with zero errors.
- Annotation conventions document updated with any clarified rules.

### Exit criteria

- All source images are represented exactly once.
- Every visible sperm is either annotated or explicitly recorded as unresolvable/not annotatable with a reason.
- Every centerline intersection has either a crossing annotation or a documented reason it is not a biological crossing.
- Gold files are locked and checksummed.

---

## Phase 1D - Gold evaluator and path-v2 baseline report

### Objective

Measure the current algorithm before changing it.

### Scope

- Implement gold/prediction matching.
- Implement the metrics from Section 6.
- Evaluate path-v2.
- Produce per-image and per-subtype reports.
- Generate a visual failure bundle.

### Deliverables

- Evaluation CLI.
- Machine-readable metrics.
- Human-readable baseline report.
- Failure galleries grouped by error type.
- Baseline thresholds and match tolerances in configuration.

### Exit criteria

- Every prediction and gold sperm is accounted for as matched, missed, false positive, duplicate, or merge-related.
- Results can be reproduced from the locked gold set and frozen baseline.
- The report identifies the exact crossing subset that Phase 2 must improve.

---

## Phase 2A - Behavior-preserving SCP modularization

### Objective

Extract the modules needed for overlap work without changing outputs.

### Scope

- Extract configuration, typed records, skeleton graph, artifact writing, and existing path-v2 logic.
- Keep the existing CLI entry point working.
- Add regression tests comparing baseline metadata and candidate statuses.

### Exit criteria

- The refactored code reproduces the frozen baseline.
- Any differences are documented and approved before continuing.
- No overlap algorithm improvements are mixed into the refactor.

---

## Phase 2B - Continuous tail evidence and head-neck anchors

### Objective

Provide better local evidence before changing global assignment.

### Scope

- Preserve a continuous tail-likelihood score instead of relying only on a binary mask.
- Estimate local tail orientation using image structure and/or skeleton neighborhoods.
- Estimate local width from a distance transform.
- Fit a head ellipse where possible.
- Predict a neck attachment point and outward direction for each head.
- Save diagnostic maps and overlays.

### Important representation

For orientation, use an undirected representation such as `cos(2*theta)` and `sin(2*theta)` where appropriate, avoiding discontinuity between equivalent directions separated by 180 degrees.

### Exit criteria

- Orientation diagnostics follow isolated tails visually.
- Neck anchors align with manual neck annotations on the gold set at a measurable accuracy.
- The existing isolated-sperm assignment does not regress materially.

---

## Phase 2C - Direction-aware junction state graph

### Objective

Represent crossing continuation explicitly.

### Scope

- Detect junction neighborhoods.
- Split skeletons into branch segments between endpoints and junctions.
- Estimate tangent direction for each segment near each junction.
- Define transition costs for incoming-to-outgoing segment pairs.
- Represent path state as `(junction_or_node, incoming_segment)` rather than node alone.
- Allow a logical track to traverse a crossing without forcing exclusive ownership of the crossing-center pixels.

### Transition-cost inputs

- Turning angle.
- Curvature change.
- Color continuity.
- Width continuity.
- Local orientation agreement.
- Ridge/tail likelihood.
- Head-anchor direction compatibility.
- Gap length and unsupported-pixel penalties.

### Exit criteria

- Synthetic X-crossings select straight/smooth continuations.
- Synthetic curved crossings select curvature-consistent continuations.
- Self-loop traversal does not fail merely because a spatial node is revisited under a different incoming state.
- All transition terms are logged separately.

---

## Phase 2D - Top-K path hypotheses per head

### Objective

Stop committing to one path before considering competing sperm.

### Scope

- Generate the best K candidate tracks from each head anchor.
- Include a null, partial, or boundary-truncated hypothesis where appropriate.
- Deduplicate near-identical paths.
- Record decomposed scores and path differences.
- Save path hypothesis overlays.

### Exit criteria

- The gold-correct path appears in the top K for a high fraction of determinable crossing cases.
- The evaluator reports top-1, top-3, top-5, and top-K oracle coverage.
- If the correct path is absent, diagnostics identify whether evidence, graph construction, endpoint generation, or pruning caused the miss.

---

## Phase 2E - Global multi-head assignment

### Objective

Choose a mutually compatible set of tracks for all heads in a connected component.

### Scope

- Formulate a global objective over head-path hypotheses.
- Enforce at most one selected track per head.
- Penalize or constrain incompatible reuse of ordinary branch segments.
- Permit shared ownership in confirmed or high-probability crossing neighborhoods.
- Penalize duplicate endpoint usage where biologically inappropriate.
- Permit unassigned heads and unresolved components.
- Compare exact optimization with bounded search if needed.

### Recommended initial implementation

Begin with an integer linear programming formulation if a suitable lightweight solver is available. Otherwise implement a deterministic branch-and-bound or beam-search solver with explicit optimality/termination diagnostics.

### Exit criteria

- Global assignment improves correct head-tail association on the gold crossing subset over independent path selection.
- Isolated-sperm performance is unchanged or improved.
- The solver can abstain when the best and second-best global solutions are too similar.
- Runtime and solver status are recorded for every component.

---

## Phase 2F - Centerline-based instance-mask reconstruction

### Objective

Create clean instance masks without reintroducing unselected branches.

### Scope

- Use the selected track as the instance centerline.
- Estimate radius along the centerline from the original tail mask or distance transform.
- Reconstruct local-width masks around the selected centerline.
- Restrict reconstructed pixels by tail likelihood, orientation compatibility, and nearest-track evidence.
- Allow two instance masks to share pixels in true crossing centers.
- Keep selected and unselected branches separate.

### Exit criteria

- Extra-branch inclusion decreases on the gold set.
- Tail completeness does not decrease substantially.
- Crossing-center sharing does not create broad duplicate masks outside the crossing neighborhood.
- Reconstruction diagnostics show centerline, local radius, accepted pixels, rejected nearby pixels, and shared pixels.

---

## Phase 2G - Confidence calibration and acceptance policy

### Objective

Replace one composite confidence with interpretable confidence components.

### Required confidence outputs

1. **Local path confidence** - support along the selected track.
2. **Global assignment confidence** - margin between the best and next-best complete assignment.
3. **Completeness confidence** - evidence that the head, neck, full visible tail, and endpoint are represented appropriately.
4. **Boundary/partial confidence** - probability that the instance is intentionally incomplete due to image boundaries or visibility.

### Scope

- Derive features from the selected paths and assignments.
- Fit simple calibration models only if the gold sample supports them.
- Otherwise use empirical score bins and transparent rules.
- Maintain conservative and balanced-review profiles.

### Exit criteria

- Confidence bins correspond monotonically to observed correctness.
- Ambiguity is driven primarily by global solution uncertainty and completeness evidence, not by an arbitrary total score.
- The selected operating profile has a documented precision/recall tradeoff.

---

## Phase 3 - Residual failure audit and model decision gate

### Objective

Determine whether any remaining errors require machine learning.

### Procedure

Group remaining errors into:

- Missing/faint tail evidence.
- Incorrect local orientation.
- Incorrect neck attachment.
- Tail-versus-debris confusion.
- Head-versus-thick-tail confusion.
- Graph topology error.
- Global optimization error.
- Mask reconstruction error.
- Truly indeterminate image evidence.

For each category, answer:

1. Is the correct information visible in the source image?
2. Is it already represented in current features?
3. Can a deterministic feature reasonably extract it?
4. Would a learned local predictor address this exact problem?
5. Is there enough independent source-image diversity to validate it?

### Decision rule

Proceed to Phase 4 only for a narrowly defined output with a measurable residual-error target. Do not proceed with a generic goal such as “train AI to detect sperm better.”

### Exit criteria

- A written model/no-model decision.
- A prioritized residual-error table.
- A specific proposed model output, input, loss, and evaluation metric if learning is justified.

---

## Phase 4 - Optional auxiliary learned evidence model

### Objective

Train a small model to predict only the visual evidence the deterministic resolver lacks.

### Candidate outputs

Choose one initial task:

- Tail probability.
- Local tail orientation.
- Junction probability.
- Endpoint probability.
- Neck attachment probability.
- Head-versus-thick-tail probability.

Do not begin with complete sperm instance masks.

### Data strategy

- Use the 24 gold images for grouped cross-validation or a fixed source-image holdout.
- Use SCP to generate weak labels only in high-confidence easy regions.
- Use manual centerlines and crossings as authoritative labels.
- Generate synthetic crossings from isolated tails.
- Use patch sampling while keeping all patches from one image in the same partition.
- Freeze most of a pretrained encoder initially.
- Use strong geometric and photometric augmentation, but preserve biologically valid tail geometry.

### Required learning-curve experiment

Train with progressively larger subsets of source images and evaluate on an unchanged holdout. The purpose is to determine whether performance is plateauing or remains strongly data-limited.

### Exit criteria

- The model improves the targeted residual-error metric on held-out source images.
- Improvement persists when integrated into the graph resolver.
- No data leakage exists.
- The learned component can be disabled for deterministic ablation.
- Failure diagnostics remain inspectable.

---

## Phase 5 - Production hardening and morphology-ready output

### Objective

Turn the research cropper into a stable upstream component for future morphology analysis.

### Scope

- Stable configuration and model/version identifiers.
- Batch resume and failure recovery.
- Deterministic output naming.
- Formal instance output schema.
- Full provenance for masks, paths, confidence, and ambiguity.
- Performance profiling.
- Regression suite covering the locked gold set.
- Export of aligned head, neck, midpiece, and tail measurements.

### Recommended instance output

Each sperm output should include:

- Source image ID.
- Instance ID.
- Crop box and transformation.
- Head mask and fitted geometry.
- Neck point.
- Tail centerline.
- Tail mask.
- Distal endpoint status.
- Crossing participation.
- Local, global, completeness, and boundary confidence.
- Accepted, review, rejected, or indeterminate status.
- All risk flags.
- Algorithm/configuration version.

### Exit criteria

- A single command reproduces the gold-set evaluation.
- Regression tests fail when key metrics degrade beyond configured tolerances.
- Downstream morphology code can consume instance outputs without reading SCP-internal artifacts.

---

# 8. Codex Work Packages

## How to use these prompts

- Provide one prompt at a time.
- Use Plan mode for phases that change architecture or data formats.
- Require Codex to inspect the current repository and referenced documentation before proposing modifications.
- Do not allow later-phase work to leak into the current phase.
- Review the plan before allowing implementation.
- After implementation, require tests, commands run, outputs changed, and known limitations.

The prompts below are intentionally bounded. Paths should be adjusted only if the repository changes.

---

## Codex Prompt 0 - Stabilize repository and freeze baseline

```text
You are working in the current Sperm_Morphology_Recognition repository state.

Read the repository documentation completely before making changes, especially README.md, ProjectContext.md, AGENT.md, PROJECT_ANALYSIS_2026-07-13.md, the SCP README, and the current algorithmic_tail_mask.py CLI behavior.

Goal:
Create a reproducible baseline for the current path-v2 SCP pipeline without changing its image-processing, assignment, scoring, or candidate-status behavior.

Required work:
1. Inspect the current Python environment and imports.
2. Add an appropriate dependency manifest and document the supported Python version.
3. Add a versioned configuration representation for all effective path-v2 settings. Preserve the existing CLI and defaults.
4. Add run metadata that records configuration, dependency versions, input image paths/hashes, command-line arguments, timestamps, and available repository revision information.
5. Freeze the current full 24-image path-v2 output under a clearly named baseline directory.
6. Add a reproducibility command and documentation.
7. Add checks that compare the reproduced baseline candidate counts, statuses, and key metadata with the frozen run.

Constraints:
- Do not change mask thresholds, head/tail detection, path selection, scoring, acceptance rules, or output semantics.
- Do not perform the planned modular refactor yet.
- Do not delete or move source data.
- Keep large generated outputs ignored by Git.
- If Git metadata is invalid or absent, report the condition and make the smallest safe repository-level correction only if appropriate.

Before implementation:
Return a concise plan identifying files to create or modify, the exact baseline command, and how behavior equivalence will be tested.

After implementation:
Run the relevant tests and baseline command. Report all commands, candidate-count comparisons, changed files, and any unavoidable differences.
```

---

## Codex Prompt 1A - Build annotation schema and GUI

```text
You are working in the current Sperm_Morphology_Recognition repository after the baseline-freeze phase has been completed.

Read all relevant repository documentation and the approved Project Next Steps and Detailed Implementation Plan before making changes.

Goal:
Build a robust manual annotation subsystem for the 24 Ward images. The canonical annotations must support sperm heads, neck attachment points, tail centerlines, distal endpoints, partial/boundary status, crossings, correct crossing continuations, certainty, and notes.

Required architecture:
- Keep annotation models, I/O, validation, exports, and Tkinter UI in separate modules.
- Use schema-versioned per-image JSON as the canonical source of truth.
- Store all coordinates in original image pixel coordinates.
- Use atomic autosave and bounded backup history.
- Include undo/redo.
- Keep SCP overlays read-only and optional.

Required GUI features:
- Open and navigate all Ward TIFF images.
- Cursor-centered zoom and panning.
- Select/edit tool.
- Head ellipse and optional head polygon tools.
- Neck-point tool.
- Tail-centerline polyline tool with point insertion, movement, and deletion.
- Distal-endpoint visibility control.
- Crossing-region tool.
- Crossing-continuation annotation for each involved sperm.
- New/delete sperm instance.
- Deterministic per-instance colors.
- Object tree and editable property panel.
- Completion/review status and notes.
- Optional display of existing SCP masks, skeletons, heads, and path outputs.
- Previous/next and filtered navigation.
- Keyboard shortcuts documented in the user guide.

Required validation:
Implement the errors and warnings defined in the plan, including invalid coordinates, broken instance references, missing required geometry, neck/centerline distance warnings, unannotated centerline intersections, and boundary/status inconsistencies.

Required exports:
- Canonical JSON.
- Dataset manifest.
- Preview PNG.
- CSV summary.
- Regenerable head, centerline, endpoint, neck, crossing, and orientation-map exports.

Testing requirements:
- Round-trip JSON serialization.
- Schema-version handling.
- Coordinate transform correctness across zoom/pan.
- Atomic-save recovery behavior.
- Undo/redo for core actions.
- Validation of intentionally invalid fixtures.
- Load/save of at least one real Ward image annotation.

Constraints:
- Do not modify SCP algorithm behavior.
- Do not begin automatic overlap resolution.
- Do not require dense tail polygons.
- Do not make Tkinter classes the data model.
- Do not silently import SCP predictions into gold annotations.

Before implementation:
Inspect the existing review_candidate_gui.py and reuse suitable conventions without coupling the new annotation schema to the old candidate-review CSV format. Return a detailed implementation plan and proposed file structure.

After implementation:
Run tests, launch a smoke test on representative 40x and 100x Ward images, and report commands, results, limitations, and annotation-file examples.
```

---

## Codex Prompt 1B - Pilot annotation workflow

```text
Review the completed annotation subsystem and the approved annotation conventions.

Goal:
Prepare and validate a three-image annotation pilot before the full 24-image annotation pass.

Required work:
1. Select three representative Ward images: one mostly isolated, one crowded/crossing-heavy, and one from the other magnification group.
2. Create a pilot dataset manifest.
3. Verify that the GUI and schema can represent all encountered heads, necks, centerlines, partial sperm, boundaries, crossings, and continuation decisions.
4. Add only changes justified by observed pilot friction or missing representation.
5. Keep schema changes backward compatible where possible.
6. Finalize the annotation conventions and freeze schema version 1.0.0.
7. Add a pilot validation and export command.

Do not fabricate biological annotations. The developer will perform the manual drawing. Your role is to prepare the pilot, expose any software limitations, and implement approved GUI/schema corrections.

Before implementation:
Report the selected image criteria, expected pilot artifacts, and any risks.

After implementation:
Report schema/UI changes, tests, validation results, and the exact command to open the pilot set.
```

---

## Codex Prompt 1D - Build gold evaluator and baseline report

```text
You are working after the ward_gold_v1 annotations have been completed and locked.

Goal:
Build an evaluation system that matches SCP predictions to gold sperm instances and produces a complete path-v2 baseline report.

Required work:
1. Read the canonical annotation schema and locked dataset manifest.
2. Implement one-to-one prediction/gold matching using a global bipartite assignment.
3. Use configurable head overlap/distance, neck distance, centerline coverage, endpoint distance, and completeness compatibility.
4. Account for every gold instance and prediction as matched, missed, false positive, duplicate, merged, partial, or indeterminate-related.
5. Implement all required detection, association, centerline, endpoint, crop, and crossing metrics.
6. Report results by image, magnification, and failure subtype.
7. Generate visual failure bundles for missed sperm, wrong tails, duplicates, merges, extra branches, premature ends, and forced decisions on indeterminate crossings.
8. Preserve all matching tolerances and metric settings in configuration.

Constraints:
- Do not change SCP predictions in this phase.
- Do not tune path-v2 thresholds while building the evaluator.
- Do not use greedy matching.
- Keep gold annotations read-only.

Before implementation:
Return the proposed matching formulation, tie-breaking rules, metric definitions, and file outputs.

After implementation:
Evaluate the frozen path-v2 baseline. Report commands, dataset counts, headline metrics, stratified metrics, and the dominant error categories that Phase 2 must target.
```

---

## Codex Prompt 2A - Behavior-preserving SCP modularization

```text
You are working after the gold evaluator and baseline report are complete.

Goal:
Modularize the SCP code needed for overlap-resolution development while preserving frozen path-v2 behavior.

Required work:
- Extract typed records/configuration.
- Extract existing skeleton and graph utilities.
- Extract current path-v2 scoring and assignment code.
- Extract artifact writing and run summarization.
- Keep algorithmic_tail_mask.py as a compatible CLI entry point.
- Add regression tests comparing candidate counts, statuses, assignments, and key metrics with the frozen baseline.

Constraints:
- No new overlap algorithm.
- No threshold changes.
- No cleanup that changes numeric behavior unless separately documented and approved.
- Avoid a large rewrite. Move cohesive code incrementally.

Before implementation:
Provide a dependency map of the current large script, proposed module boundaries, and the regression checks that will prove equivalence.

After implementation:
Run the full frozen baseline and regression suite. Report all differences, even if visually minor.
```

---

## Codex Prompt 2B - Tail evidence and head-neck anchors

```text
Goal:
Add continuous tail evidence, local orientation/width estimation, and improved head-neck anchors without replacing the current global assignment yet.

Required work:
1. Preserve continuous tail likelihood before binary thresholding.
2. Estimate local orientation from image and skeleton evidence.
3. Estimate local width using the tail mask distance transform.
4. Fit head geometry and estimate a neck attachment point plus outward direction.
5. Compare predicted neck points to gold annotations.
6. Save probability, orientation, width, head-axis, and neck-anchor diagnostics.
7. Make all new features available to later graph stages behind configuration flags.

Constraints:
- Keep current path-v2 assignment available as a baseline.
- Do not implement global optimization yet.
- Do not train a model.
- Do not silently change accepted/rejected/ambiguous rules.

Before implementation:
Propose algorithms for continuous evidence, orientation, width, and neck anchoring, with expected failure modes and tests.

After implementation:
Report neck-point accuracy, orientation diagnostics, isolated-sperm regression metrics, and all configuration additions.
```

---

## Codex Prompt 2C - Direction-aware junction graph

```text
Goal:
Build a direction-aware skeleton state graph that represents biologically plausible continuation through crossings.

Required work:
- Segment skeletons into branches between endpoints and junction neighborhoods.
- Estimate incoming/outgoing tangent directions at junctions.
- Define decomposed transition costs for turn angle, curvature, color, width, orientation, tail likelihood, head direction, and unsupported gaps.
- Represent traversal state using the incoming branch/segment.
- Support self-crossing paths without invalid rejection caused only by revisiting a spatial junction in another directional state.
- Permit shared logical use of crossing-center pixels while preserving ordinary branch exclusivity for later optimization.
- Save junction and transition-cost diagnostics.

Testing:
- Straight X-crossing.
- Curved crossing.
- T-contact.
- Self-loop.
- Width spike.
- Indistinguishable crossing.

Before implementation:
Describe the graph data model, state transitions, cost equations, and synthetic tests.

After implementation:
Report test results and visual diagnostics, but do not yet replace production assignment with this graph.
```

---

## Codex Prompt 2D - Top-K path hypotheses

```text
Goal:
Generate multiple plausible tail-track hypotheses per head using the new direction-aware graph.

Required work:
- Generate configurable top-K paths from each head/neck anchor.
- Include null, partial, and boundary-truncated hypotheses when applicable.
- Deduplicate geometrically near-identical paths.
- Record decomposed scores, endpoints, branch usage, crossings, and disagreement regions.
- Add evaluator metrics for top-1/top-3/top-5/top-K oracle coverage of the gold path.
- Generate hypothesis overlays for review.

Constraints:
- Do not select paths globally yet.
- Do not change final candidate statuses.
- Avoid pruning that cannot be diagnosed.

Before implementation:
Explain the K-shortest-path or equivalent method, cycle handling, deduplication, and pruning strategy.

After implementation:
Report gold-path oracle coverage by subtype and identify why correct paths are absent where failures remain.
```

---

## Codex Prompt 2E - Global multi-head assignment

```text
Goal:
Select a mutually compatible set of tail paths for all heads in each connected component.

Required work:
- Define binary selection variables or equivalent hypothesis-selection state.
- Select at most one hypothesis per head.
- Penalize or prohibit incompatible reuse of ordinary branch segments.
- Permit explicitly modeled shared use within crossing neighborhoods.
- Handle endpoint competition, duplicate tracks, unassigned heads, partial paths, and unresolved cases.
- Compute the best and second-best complete assignments or a defensible approximation.
- Expose a global solution margin and solver status.
- Keep independent path-v2 assignment available for ablation.

Evaluation:
Compare head-tail association, crossing continuation, wrong-tail rate, missed sperm, and isolated-sperm behavior against the frozen baseline.

Before implementation:
Propose the optimization objective, constraints, solver/dependency choice, fallback behavior, and complexity limits.

After implementation:
Report full and stratified metrics, solver runtimes/statuses, unresolved components, and examples where global reasoning changed the result.
```

---

## Codex Prompt 2F - Centerline-based mask reconstruction

```text
Goal:
Reconstruct clean per-instance tail masks from selected logical centerlines without adding unselected branches.

Required work:
- Estimate local radius from the original tail evidence/mask.
- Grow masks around selected centerlines with local width.
- Restrict pixels using likelihood, orientation, and track compatibility.
- Permit overlapping instance masks only inside modeled crossing neighborhoods.
- Prevent broad duplicate masks outside crossings.
- Generate reconstruction diagnostics and gold-set metrics for completeness and extra-branch inclusion.

Constraints:
- Do not use the full connected component as the final instance mask.
- Preserve selected centerlines exactly in the output mask where supported.
- Keep reconstruction configurable and independently testable.

Before implementation:
Describe the reconstruction algorithm, ownership rules, crossing sharing, and safeguards.

After implementation:
Report extra-branch, tail-completeness, foreign-content, and crossing-sharing metrics compared with the prior reconstruction.
```

---

## Codex Prompt 2G - Confidence calibration and acceptance profiles

```text
Goal:
Replace the current composite confidence behavior with separate local-path, global-assignment, completeness, and boundary/partial confidence.

Required work:
- Define interpretable feature sets for each confidence component.
- Evaluate empirical correctness by score bin.
- Use a simple calibration method only if supported by sample size; otherwise use transparent empirical bins/rules.
- Define conservative and balanced-review operating profiles.
- Base ambiguity primarily on global assignment margin, crossing determinacy, and completeness evidence.
- Preserve explicit indeterminate output.
- Produce calibration tables and reliability plots.

Constraints:
- Do not overfit thresholds to aggregate accepted count.
- Do not hide poor calibration in one total score.
- Keep all rules/configuration versioned.

Before implementation:
Describe the proposed confidence components, calibration method, sample-size safeguards, and operating-point selection.

After implementation:
Report calibration quality, precision/recall by profile, ambiguous-case composition, and regressions by subtype.
```

---

## Codex Prompt 3 - Residual-error audit and model decision

```text
Goal:
Audit all remaining gold-set failures after the deterministic overlap resolver and decide whether a learned auxiliary component is justified.

Required work:
- Categorize each remaining failure as evidence, orientation, neck anchor, debris confusion, head/thick-tail confusion, graph topology, optimization, reconstruction, boundary handling, or truly indeterminate.
- Produce counts and representative visual examples.
- Determine whether the correct information is visible and whether current features encode it.
- Recommend deterministic corrections where appropriate.
- If learning is justified, specify exactly one initial prediction target, input representation, annotation source, training partition, loss, evaluation metric, and integration point.
- Include a no-model recommendation if the evidence does not justify training.

Constraints:
- Do not train a model in this phase.
- Do not recommend an end-to-end full-sperm detector without gold-set evidence.
- Separate algorithmic defects from information-theoretic ambiguity.

Deliverable:
A decision report that can be reviewed before authorizing Phase 4.
```

---

## Codex Prompt 4 - Optional narrow auxiliary model

```text
This phase is authorized only if the residual-error audit identifies a specific learned prediction target.

Goal:
Train and integrate one narrowly scoped auxiliary evidence model using the approved target and evaluation protocol.

Required work:
- Use source-image-level train/validation/test grouping.
- Prevent patch leakage between partitions.
- Use the locked gold annotations as authoritative labels.
- Use SCP pseudo-labels only in approved high-confidence regions and track their provenance.
- Add synthetic crossings if appropriate, while preserving a real-image-only held-out evaluation.
- Begin with transfer learning and a mostly frozen encoder.
- Run a source-image learning curve.
- Compare deterministic-only and deterministic-plus-model pipelines.
- Keep model use optional through configuration.

Before implementation:
Return the dataset construction plan, partition list, architecture, losses, augmentations, and stopping/selection criteria.

After implementation:
Report held-out metrics, learning curve, integration effects, failure cases, and whether more independent source images are required.
```

---

# 9. Testing Strategy

## 9.1 Annotation tests

- Serialization round trip.
- Backward-compatible schema loading.
- Image-coordinate invariance under zoom/pan.
- Undo/redo command stack.
- Atomic save and backup recovery.
- Validation rule coverage.
- Crossing-reference integrity.
- Export reproducibility.

## 9.2 Synthetic graph tests

- One isolated curved tail.
- Two clean crossing tails.
- Three-tail crossing.
- T-shaped contact without true continuation.
- Parallel touching tails.
- Self-loop.
- Head outline near a tail.
- Thick tail mistaken for head.
- Broken faint segment requiring a short gap.
- Boundary-truncated tail.
- Indistinguishable symmetric crossing.

Each synthetic fixture should have an explicit expected outcome or expected ambiguity.

## 9.3 Gold-set regression tests

Gold-set regression should check:

- No unexpected change in source-image count.
- No missing prediction files.
- Instance recall floor.
- Isolated-sperm correctness floor.
- Crossing-association floor.
- Wrong-tail ceiling.
- Duplicate and merge ceilings.
- Extra-branch ceiling.
- Runtime and solver-failure ceilings.

Exact numeric tolerances should be set after the baseline and first improved resolver are measured.

---

# 10. Data Management and Reproducibility

## 10.1 Keep original images immutable

Gold annotations should refer to source images by stable relative path and file hash. Do not alter or resave the original TIFF files.

## 10.2 Version annotations

Use:

```text
ward_gold_v1
ward_gold_v1.1   # only for documented corrections or backward-compatible additions
ward_gold_v2     # material reinterpretation or schema-breaking change
```

Every dataset release should include a manifest and checksums.

## 10.3 Separate source, gold, derived, and predictions

- Source images: immutable microscopy input.
- Gold JSON: human-authored source of truth.
- Derived exports: regenerated from gold JSON.
- Predictions: generated by a named algorithm/configuration version.
- Evaluation outputs: generated from a named gold version and prediction version.

## 10.4 Record provenance

Every imported or suggested annotation should record whether it was:

- Manually created.
- Initialized from SCP.
- Generated synthetically.
- Predicted by a learned model.
- Manually corrected.

Gold status should require human confirmation.

---

# 11. Risks and Mitigations

## Risk: The annotation task becomes too slow

**Mitigation:** Use centerlines and ellipses rather than dense tail polygons, autosave continuously, provide editable SCP overlays, and annotate easy images before crowded images.

## Risk: The gold set is biased by current SCP predictions

**Mitigation:** SCP overlays are optional and read-only; review annotations with overlays hidden; annotate missed sperm explicitly.

## Risk: Only one annotator is available

**Mitigation:** Use a two-pass self-review, lock files only after validation, revisit uncertain images after temporal separation, and record certainty/indeterminate states rather than forcing consistency.

## Risk: The new graph resolver becomes too complex

**Mitigation:** Implement direction-aware transitions, top-K generation, global assignment, and reconstruction as separate ablatable modules. Preserve the baseline at every stage.

## Risk: Global optimization is slow or fails

**Mitigation:** Solve per connected component, cap hypotheses, log solver state, provide deterministic fallback, and retain ambiguity rather than using uncontrolled approximations.

## Risk: Confidence is overfit to 24 images

**Mitigation:** Use simple models or empirical bins, grouped validation, transparent features, and no claims of broad generalization.

## Risk: A learned model appears strong due to leakage

**Mitigation:** Partition at source-image level, keep all derived patches from one image together, and preserve a real-image-only held-out set.

## Risk: Future imaging conditions differ from Ward images

**Mitigation:** Treat Phase 4 results as Ward-domain evidence only. Add new independent images before claiming cross-domain robustness.

---

# 12. Items Explicitly Deferred

The following should not be included in the immediate work unless a later decision gate authorizes them:

- End-to-end full-sperm instance segmentation.
- Training on the existing human pseudo-mask dataset as the primary solution.
- Morphology classification.
- Head, neck, midpiece, and tail abnormality labels.
- Large-scale data augmentation before the gold evaluator exists.
- Replacing SCP with YOLO, SAM, or another general segmentation model.
- A complete GUI redesign unrelated to annotation efficiency.
- Aggressive threshold tuning based only on accepted-candidate counts.

---

# 13. Immediate Next Actions

Execute in this order:

1. Run Codex Prompt 0 to freeze the baseline and stabilize reproducibility.
2. Run Codex Prompt 1A to build the annotation data model and GUI.
3. Manually complete the three-image pilot supported by Codex Prompt 1B.
4. Freeze annotation schema version 1.0.0.
5. Manually annotate and review all 24 Ward images.
6. Run Codex Prompt 1D to build the evaluator and baseline report.
7. Continue with Phase 2 one bounded work package at a time.

The first manual annotation should not begin until the GUI can round-trip annotations safely, validate them, and recover from autosave backups.

---

# 14. Phase Completion Checklist

## Phase 0

- [ ] Dependency manifest exists.
- [ ] Baseline configuration is versioned.
- [ ] Baseline run is frozen.
- [ ] Input hashes and environment metadata are recorded.
- [ ] Reproduction checks pass.

## Phase 1A

- [ ] Canonical annotation schema implemented.
- [ ] GUI supports all required objects.
- [ ] Zoom/pan coordinates verified.
- [ ] Undo/redo verified.
- [ ] Atomic autosave and recovery verified.
- [ ] Validation and exports verified.

## Phase 1B/1C

- [ ] Pilot complete.
- [ ] Schema 1.0.0 frozen.
- [ ] All 24 images annotated.
- [ ] All validation errors resolved.
- [ ] Uncertain and indeterminate cases reviewed.
- [ ] Gold manifest locked and checksummed.

## Phase 1D

- [ ] One-to-one matching implemented.
- [ ] Missed sperm measured.
- [ ] Baseline report generated.
- [ ] Failure galleries generated.
- [ ] Crossing subset identified.

## Phase 2

- [ ] Refactor reproduces baseline.
- [ ] Neck anchors and orientation measured.
- [ ] Junction state graph tested.
- [ ] Top-K oracle coverage measured.
- [ ] Global assignment improves crossing association.
- [ ] Reconstruction reduces extra branches.
- [ ] Confidence calibrated or empirically binned.

## Phase 3/4

- [ ] Residual errors categorized.
- [ ] Model decision documented.
- [ ] Any model target is narrow and measurable.
- [ ] Source-image partitions prevent leakage.
- [ ] Held-out improvement demonstrated before adoption.

---

# 15. Final Success Definition

The detection and cropping stage is ready to support later morphology analysis when it can:

- Identify the large majority of visible sperm in the Ward gold set.
- Produce one matched instance per sperm without excessive duplicates or merges.
- Preserve the complete visible tail when determinable.
- Correctly associate heads and tails through common crossing types.
- Avoid adding unrelated tail branches.
- Correctly classify partial and boundary-truncated sperm.
- Mark genuinely indeterminate crossings for review.
- Produce stable, versioned, auditable instance outputs.
- Reproduce evaluation metrics through one documented command.

The project does not need to eliminate every ambiguous case. It needs to resolve evidence-supported cases accurately and represent irreducible uncertainty honestly.

---

## Source Basis

This plan is based on the project state documented in `PROJECT_ANALYSIS_2026-07-13.md`, including the current SCP architecture, path-v2 outputs, review results, known overlap failure modes, test coverage, and repository/reproducibility gaps.
