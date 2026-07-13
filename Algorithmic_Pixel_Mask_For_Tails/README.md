# Sperm Color Pixels

Sperm Color Pixels (SCP) is the current primary method for detecting and
cropping full mouse sperm from the high-contrast Ward RGB TIFF images.

This workspace is intentionally separate from the older YOLO/NMA workflow. SCP
uses deterministic color/pixel masking and image geometry, not model inference,
as the main crop-generation path.

## Data

`Raw_Ward_Data/` contains the higher-quality RGB Ward TIFF images. The script
recursively ignores `__MACOSX` folders and `._*` sidecar files, so the original
data export can stay intact.

## Run

From the project root:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py \
  --output-root outputs/updated_outputs_path_v2 \
  --overwrite
```

Or from this folder:

```bash
../.venv/bin/python algorithmic_tail_mask.py \
  --output-root outputs/updated_outputs_path_v2 \
  --overwrite
```

For a quick smoke test:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py \
  --output-root outputs/updated_outputs_path_v2_smoke \
  --limit 2 \
  --overwrite
```

Relative `--output-root` paths resolve inside this SCP folder.
Path-v2 scoring is enabled by default. Use `--no-path-v2` only when comparing
against the legacy full-component/flood-split behavior.

Useful revamp/debug flags:

```bash
../.venv/bin/python algorithmic_tail_mask.py \
  --output-root outputs/updated_outputs_revamp_debug \
  --path-mode hybrid \
  --normalization raw \
  --acceptance-profile conservative \
  --diagnostics \
  --diagnostic-limit 25
```

- `--path-mode legacy|geometry-only|color-only|hybrid` selects the active
  head-to-endpoint path ranking mode. `legacy` disables path-v2 scoring.
- `--normalization raw|local-background|white-balance` controls color
  normalization before Lab color-continuity scoring. `raw` is the default; run
  local-background/white-balance only for targeted color-ablation checks.
- `--acceptance-profile conservative|balanced-review` controls whether good
  crops with mask/path risk flags can remain accepted for review.
- `--diagnostics` writes per-candidate overlays for head anchors, path ablation,
  crop boundaries, duplicate reuse, and cutoff-stage review. Pair it with
  `--diagnostic-limit` on full batches.
- `--tail-debug-overlays` writes expensive full-frame tail assignment overlays.
  Leave it off for routine full runs, or pair it with `--tail-debug-limit`.
- `--record-ablation-scores` records inactive geometry/color/hybrid score
  columns. Leave it off when running one active path mode at a time.
- `--path-fast-isolated` is enabled by default and keeps legacy full-component
  assignment for clean isolated single-head tails.

## Candidate Review CSV

Build the manual review queue from the current SCP JSON outputs:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/build_candidate_review_csv.py \
  --output-root outputs/updated_outputs_path_v2
```

This writes:

- `<output-root>/candidate_review/candidate_review_queue.csv`: targeted manual
  review queue with blank human-label columns.
- `<output-root>/candidate_review/candidate_review_all_candidates.csv`: all
  current crop candidates for reference.
- `<output-root>/candidate_review/candidate_review_label_key.md`: label
  definitions for `crop_label`, `mask_label`, and notes.

The default queue includes all ambiguous candidates, the 30 lowest-scoring
accepted candidates, and the 20 highest-scoring rejected candidates.

Summarize a completed labeled queue for before/after benchmarking:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/analyze_candidate_review_baseline.py \
  --review-csv outputs/updated_outputs/candidate_review/candidate_review_queue.csv
```

This writes reusable tables for `crop_label`, `mask_label`, current status,
assignment/failure reason, score bands, and review notes under
`candidate_review/baseline_analysis/`.

Launch the simple review GUI:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/review_candidate_gui.py
```

The GUI opens `candidate_review_queue.csv` by default, shows the candidate image
and stats, and saves your `crop_label`, `mask_label`, and `review_notes` back
into the CSV. Use `Save & Next` for the fastest review loop. The first save also
creates a `.bak` copy of the CSV.

## Ward Gold Annotation GUI

Phase 1A adds a source-image annotation tool for building the Ward gold dataset.
It is separate from candidate CSV review and stores all coordinates in original
image pixels.

Launch from this folder:

```bash
../.venv/bin/python annotation_gui.py \
  --input-root Raw_Ward_Data \
  --annotation-root annotations/ward_gold_v1 \
  --scp-output-root outputs/updated_outputs_path_v2
```

Useful non-GUI smoke check:

```bash
../.venv/bin/python annotation_gui.py \
  --input-root Raw_Ward_Data \
  --annotation-root /tmp/scp_annotation_smoke \
  --scp-output-root outputs/updated_outputs_path_v2 \
  --check-only
```

Regenerate derived manifest, previews, CSV summary, rasters, and orientation
maps:

```bash
../.venv/bin/python annotation/annotation_export.py \
  --input-root Raw_Ward_Data \
  --annotation-root annotations/ward_gold_v1 \
  --export-root annotations/ward_gold_v1/exports
```

Core shortcuts: `N` new sperm, `E` head ellipse, `G` head polygon, `K` neck,
`T` tail centerline, `C` crossing, `V` select/edit, `H`/Space pan, `Enter`
finish drawing, `Esc` cancel, `Ctrl+Z` undo, `Ctrl+Y` redo, `Ctrl+S` save,
`Ctrl+Shift+V` validate, `O` SCP aid overlay, `I` selected-only view, `F` fit,
and `1` for 100% zoom.

## Output Layout

Current outputs are organized under:

```text
outputs/
```

- `legacy_outputs/`: older SCP runs preserved for comparison.
- `updated_outputs/`: latest SCP run with candidate scoring and review buckets.
- `updated_outputs_path_v2/`: path-scored SCP v2 runs for comparison against the
  completed v1 review labels.

Important folders inside each SCP output run:

- `overlays/`: RGB review PNGs with tails in cyan, heads in magenta, and overlap candidates in yellow.
- `masks/`: binary sperm-color, tail, head, and overlap masks.
- `json/`: per-image detections, candidate scores, candidate statuses, and parameters.
- `head_connected_overlays/`: stricter overlays after removing tail components without head contact.
- `sperm_crops/`: accepted original-image crops and crop masks.
- `highlighted_sperm_crops/`: accepted crops with tail/head/overlap highlights and crop-frame border.
- `tail_skeletons/`: one-line tail skeleton masks and skeleton overlays for accepted crops.
- `candidate_review/rejected/`: tiny fragments and lower-scoring duplicate candidates.
- `candidate_review/ambiguous/`: unresolved overlaps, crossings, loops, or crowded crop boxes.
- `tail_assignment_debug/`: optional full-frame overlays written only with
  `--tail-debug-overlays`.
- `diagnostics/head_anchors/`: candidate overlays with head anchors, head exits,
  distal endpoints, and selected/competing path lines.
- `diagnostics/path_ablation/`: overlays with geometry/color/hybrid path scores.
- `diagnostics/crop_boundaries/`: overlays with source crop boxes and boundary
  risk/cutoff-stage metadata.
- `diagnostics/duplicate_reuse/`: overlays with repeated head/tail reuse counts.
- `diagnostics/cutoff_stage/`: overlays focused on inferred cutoff cause.
- `summary.csv` and `summary.json`: run-level counts for tuning.

Existing v1 full-run totals in `updated_outputs/`:

- 24 real Ward images processed.
- 213 crop candidates.
- 97 accepted crops.
- 35 rejected candidates.
- 81 ambiguous candidates.
- 0 accepted reused-head duplicates in the verification run.

Current v2 full-run totals in `updated_outputs_path_v2/`:

- 24 real Ward images processed.
- 209 crop candidates.
- 22 accepted crops.
- 45 rejected candidates.
- 142 ambiguous candidates.
- 18 `head_outline_false_tail` rejections.

## Algorithm Summary

SCP v2 currently:

1. Builds high-recall sperm-color masks from Ward RGB images using HSV,
   saturation/value thresholds, and local-darkness contrast.
2. Detects compact dark head candidates.
3. Detects elongated or sparse tail components after subtracting head pixels.
4. Skeletonizes tail components and records endpoints, branchpoints, and
   overlap-suspicion regions.
5. Associates tail components with touching or nearby heads.
6. Scores head-to-endpoint skeleton paths using length, head-collar exit,
   curvature, branch/width penalties, and Lab color continuity.
7. Reconstructs the final tail mask from the selected path, trimming extra
   branches instead of blindly accepting the whole connected tail component.
8. Builds crop candidates from tail/head assignments.
9. Scores candidates and assigns one of three statuses:
   - `accepted`
   - `rejected`
   - `ambiguous`
10. Writes accepted crops as the current final crop set and preserves rejected or
   ambiguous candidates for review.

## Candidate Scoring

Candidate scoring is deterministic. It does not currently use YOLO, NMA, or
manual annotation labels.

The legacy `score` field remains for compatibility. SCP v2 also writes:

- `crop_quality_score`
- `mask_quality_score`
- `path_confidence_score`
- `candidate_risk_flags`
- head-collar/path metrics such as `tail_pixels_outside_head_collar`,
  `head_exit_path_length_px`, `path_color_continuity_score`, and
  `path_trimmed_branch_fraction`

Candidates score better when they have:

- more assigned tail pixels,
- more assigned skeleton pixels,
- a simpler single-head assignment,
- a successful compact width-spike split.

Candidates score worse or become review-only when they have:

- too few tail or skeleton pixels,
- reused head assignments,
- many touching heads,
- overlap flags such as width spikes or wide sparse components,
- low path confidence or weak color continuity,
- no convincing tail path leaving the head collar,
- many skeleton branchpoints or endpoints,
- lots of foreign head/tail pixels inside the crop box,
- source or crop boxes touching the image border.

Hard status rules:

- tiny tail/skeleton candidates become `rejected`.
- head-outline-only path artifacts become `rejected` with
  `head_outline_false_tail`.
- duplicate accepted candidates for the same detected head keep only the best
  scoring candidate.
- heavy foreign content becomes `ambiguous`.
- unresolved multi-head overlap/crossing regions become `ambiguous`, including
  `ambiguous_path_color_margin` and `ambiguous_path_conflict`.

## Notes

The default `high-recall` mode intentionally prefers finding faint tail signal
before the candidate scoring layer filters or flags candidates. If review
overlays are too noisy, rerun with `--mode balanced` or `--mode high-precision`,
but compare against missed faint tails.

The base `overlays/` folder visualizes the main mask pipeline; it does not
discard every tail-like component that lacks a head contact. The accepted crop
folders are the stricter downstream result.

Ambiguous candidates are not necessarily bad detections. They are cases where
the current SCP evidence is not strong enough to trust the crop as a clean
single full sperm.

## Supporting Methods

YOLOv8, NMA, and manual labels remain useful, but they are supporting resources:

- YOLO may later provide rough regions, comparison signals, or ambiguity flags.
- NMA may later help validate head/nucleus locations.
- Manual labels should guide evaluation and future rejection experiments.

Do not replace SCP with a model-first workflow unless future review evidence
shows that the supporting method improves full-sperm crop quality.
