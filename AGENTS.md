# Repository Instructions

## Required Context

Before modifying this repository, read the following files completely:

1. `PROJECT_ANALYSIS_2026-07-13.md`
2. `PROJECT_NEXT_STEPS_IMPLEMENTATION_PLAN.md`
3. The README and implementation files directly relevant to the requested task

Inspect the current repository state before making changes. The repository code and artifacts are the source of truth when documentation and implementation disagree.

The implementation plan is the authoritative project roadmap, but only the phase or subphase explicitly requested in the current task may be implemented.

Do not begin later phases early.

## Current Project Direction

The active method is **Sperm Color Pixels (SCP)** under:

```text
Algorithmic_Pixel_Mask_For_Tails/
```

SCP is the primary full-sperm candidate-generation pipeline. It uses deterministic image processing to:

* detect high-recall sperm-colored pixels;
* detect head and tail candidates;
* skeletonize tail structures;
* associate heads with tail paths;
* reconstruct candidate sperm masks;
* create accepted, rejected, and ambiguous review outputs.

The current bottleneck is not basic sperm-pixel detection. It is correctly separating individual sperm when tails touch, cross, overlap, loop, or form shared connected components.

The project is therefore proceeding through:

1. manual gold-dataset annotation;
2. baseline evaluation;
3. direction-aware crossing analysis;
4. multi-hypothesis tail-path generation;
5. global multi-head assignment;
6. centerline-based instance-mask reconstruction;
7. targeted learned components only if residual errors justify them.

Do not convert the project back to an end-to-end model-first approach without explicit authorization and new evidence.

## Supporting Methods

YOLO and NMA remain supporting methods:

* **YOLO** may be used for comparisons, ROI proposals, hard-negative mining, or auxiliary confidence signals.
* **NMA** may be used for nucleus or head-localization support.
* Neither should replace SCP as the full-tail cropper unless the requested phase explicitly calls for such an experiment.

Do not begin model training merely because the existing deterministic pipeline reports ambiguous candidates.

## Important Paths

* Main SCP implementation:

  ```text
  Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py
  ```

* SCP documentation:

  ```text
  Algorithmic_Pixel_Mask_For_Tails/README.md
  ```

* Ward source images:

  ```text
  Algorithmic_Pixel_Mask_For_Tails/Raw_Ward_Data/
  ```

* Current path-v2 outputs:

  ```text
  Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs_path_v2/
  ```

* Historical outputs:

  ```text
  Algorithmic_Pixel_Mask_For_Tails/outputs/legacy_outputs/
  Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs/
  ```

* Current project analysis:

  ```text
  PROJECT_ANALYSIS_2026-07-13.md
  ```

* Phased implementation roadmap:

  ```text
  PROJECT_NEXT_STEPS_IMPLEMENTATION_PLAN.md
  ```

`Training_Data/Raw_Ward_Data` was intentionally moved into the SCP directory. Do not create another copy under `Training_Data/` unless explicitly requested.

## Phase Control

Each Codex task must be treated as a bounded work package.

Before implementing:

1. identify the requested phase or subphase;
2. inspect the relevant current code;
3. identify reusable existing components;
4. verify that the requested work does not depend on an unfinished earlier phase;
5. define the files expected to change;
6. preserve existing behavior outside the requested scope.

After implementing:

1. run relevant tests;
2. perform the required smoke test;
3. compare behavior with the phase acceptance criteria;
4. report any deviations from the implementation plan;
5. stop without beginning the next phase.

## Annotation Data Rules

Manual annotations are scientific source data.

Do not:

* alter raw Ward images;
* overwrite existing annotations without an explicit migration or backup;
* silently change annotation coordinates;
* store viewport coordinates instead of original-image coordinates;
* discard uncertainty or indeterminate labels;
* regenerate human-reviewed annotations from algorithm predictions;
* use test or evaluation annotations as training data without explicit authorization.

Annotation files must use a versioned schema and support round-trip loading and editing.

Atomic writes, backups, validation, and recovery behavior are required for annotation tools.

## Algorithm Development Rules

Preserve the existing high-recall SCP masking pipeline unless the requested phase specifically targets it.

For overlap-resolution work:

* distinguish pixel connectivity from biological instance identity;
* preserve multiple plausible paths until global assignment;
* use direction-aware junction transitions;
* avoid independently assigning conflicting paths to different heads;
* permit shared logical ownership of pixels in genuine crossing regions;
* reconstruct masks from selected centerlines rather than taking entire connected components;
* retain explicit ambiguity when the source image does not contain enough information.

Do not resolve difficult cases merely by lowering acceptance thresholds.

## Code Organization

The current SCP implementation is large and script-oriented. New major functionality should be placed in focused modules where practical rather than continuing to expand one monolithic file.

Prefer modules with explicit responsibilities such as:

```text
annotation/
evaluation/
graph_construction/
junction_transitions/
path_hypotheses/
global_assignment/
mask_reconstruction/
```

Do not perform a broad refactor unrelated to the requested phase.

Preserve existing CLI and output compatibility where practical.

## Dependencies

Avoid adding dependencies unless they materially simplify or improve the requested implementation.

The current core stack includes:

* Python standard library;
* OpenCV;
* NumPy;
* Pillow;
* Tkinter;
* Ultralytics in model-related workflows.

Any new dependency must be:

* justified in the completion report;
* added to the project dependency manifest;
* compatible with the current environment;
* used only where necessary.

## Testing Requirements

Add or update tests for changed behavior.

Depending on the phase, tests should cover relevant areas such as:

* annotation serialization and round-tripping;
* image-to-viewport coordinate transforms;
* validation rules;
* safe autosave and recovery;
* skeleton graph construction;
* direction-aware junction transitions;
* path ranking;
* global assignment constraints;
* instance-mask reconstruction;
* duplicate prevention;
* boundary handling;
* deterministic output.

Run the relevant existing tests before completing a task.

Current SCP tests can be run with:

```bash
cd Algorithmic_Pixel_Mask_For_Tails
../.venv/bin/python -m py_compile algorithmic_tail_mask.py test_tail_assignment.py
../.venv/bin/python -m unittest test_tail_assignment.py
```

Additional phase-specific commands should be documented alongside the implementation that introduces them.

## Output and Artifact Safety

Generated outputs must remain inside the appropriate project output or temporary test directory.

Be careful with `--overwrite`, because it deletes and recreates the selected output root.

Do not overwrite:

* raw source images;
* completed human annotations;
* existing reviewed CSV files;
* canonical baseline outputs;

unless the current task explicitly authorizes it.

Smoke tests should use temporary or clearly named test output directories.

## Git and Large Files

The workspace may not currently have functioning Git metadata. Do not assume that `git status`, `git diff`, or `git log` will work without verifying them.

Large data and generated artifacts should not be committed to ordinary Git history, including:

* `Training_Data/`;
* raw microscopy datasets;
* generated SCP output folders;
* model checkpoints;
* training runs;
* derived datasets.

Do not change ignore rules in a way that exposes these files without explicit authorization.

## Completion Report

At the end of each task, report:

* the requested phase or subphase;
* files added;
* files changed;
* principal implementation decisions;
* tests added or updated;
* exact commands run;
* test and smoke-test results;
* generated artifacts;
* deviations from the implementation plan;
* unresolved limitations;
* whether every phase exit criterion was satisfied.

Stop after the requested phase is complete.
