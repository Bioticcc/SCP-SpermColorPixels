# SCP experiments without new annotations

The active execution sequence is described in
[`ANNOTATION_OPTIONAL_EXECUTION_PLAN_2026-09-07.md`](../../ANNOTATION_OPTIONAL_EXECUTION_PLAN_2026-09-07.md).
R0 freezes current behavior and records constructed failure cases. It does not
enable an improved resolver or train a model.

Run from the project root:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_baseline.py \
  --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r0_baseline_2026_09_07
```

The directory must be new. This wrapper resolves a relative `--output-root`
against the current working directory; the original SCP CLI continues to
resolve relative paths against the SCP folder. No `--overwrite` is available
in the experiment wrapper.

The historical `updated_outputs_path_v2` reference predates the current code.
The first R0 full run returned exit status 2 to flag real historical differences:
209 candidates, 19 accepted, 44 rejected, and 146 ambiguous, versus the older
22/45/142 statuses. The fresh run matches the saved
`ablation_2026_07_03_hybrid` predictions structurally across all 24 images after
documented output-diagnostic exclusions. The main SCP source was unchanged.
Use the R0 frozen predictions as the baseline for subsequent resolver work;
the historical mismatch is not an algorithm improvement.

After reviewing reporting changes, rebuild diagnostics without repeating the
expensive full-image predictions:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/refresh_report.py \
  --run-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r0_baseline_2026_09_07 \
  --report-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r0_baseline_2026_09_07/review
```

The report directory must also be new. This command verifies the original
predictor fingerprint, configuration, complete input inventory, dependency
versions, and historical reference metadata before reusing the full-image
predictions. It records separate reporting provenance and checks integrity at
the end. The current reviewed viewer is `review/index.html`; the independently
verified hybrid comparison is `review/comparison_current_hybrid.json`.

The wrapper invokes the unchanged original CLI with every setting explicitly
recorded in `configs/overlap_demo_baseline.json`. It processes all 24 existing
Ward images and refuses a partial baseline. It writes:

- `provenance.json`: effective command/settings, threshold preset, dependency
  versions, thinning backend, Git state, and source/config/reference hashes.
- `input_manifest.json`: source-relative image IDs, sizes, and SHA-256 hashes.
- `config.json` and `demo_manifest.json`: copies of the executed configuration
  and six-image panel chosen before algorithm changes.
- `predictions/`: the new full SCP run; existing canonical predictions remain
  read-only.
- `prediction_stdout.log`: subprocess progress and final counts.
- `comparison.json`: comparison by source image and candidate identity,
  including status, assignment, numerical fields, and schema availability.
- `fixtures/` and `fixture_report.json`: constructed RGB/masks, independent
  graph/identity truth, conditional assignment measurements, full RGB pipeline
  outputs, and comparison panels.
- `index.html`: local viewer with original/saved/reproduced image toggles.
- `completion.json`: completion and comparison status. A generated report is
  not itself proof of matching predictions; inspect the comparison result.
- `integrity.json`: final source/configuration and input-inventory checks.

Open `index.html` directly in a browser; no server or external assets are
required. This is a fixed qualitative real-image panel and a synthetic
engineering benchmark, not a real-image accuracy report.

Tests, from the SCP directory:

```bash
../.venv/bin/python -m unittest discover -p 'test_*.py'
../.venv/bin/python -m py_compile algorithmic_tail_mask.py experiments/*.py tests/fixtures/*.py
```

Fixtures expose `tests.fixtures.case_names()` and
`tests.fixtures.generate_case(name, seed=0)`. Truth includes an explicit segment
graph, logical edge paths, head associations, and behavior expectations. The
loop revisits a junction without repeating ordinary edges. The indeterminate
case provides two identity solutions with identical rendered evidence; identity
accuracy is deliberately undefined for that case. The same-color X has a
documented smooth-continuation assumption rather than an unqualified biological
determinacy claim.

Conditional baseline metrics supply the constructed head and tail masks to
path-v2, isolating assignment behavior. The separately saved RGB run exercises
the production masking and head/tail stages. Centerline measurements use a
rasterized polyline and a 2-pixel tolerance, so adding polyline vertices does
not inflate coverage. The panel's 95% precision/recall threshold is a diagnostic
description of constructed tracks, not an acceptance threshold in SCP.

No new annotations, dependencies, or learned models are required. Existing
human annotations and review CSVs are never regenerated from predictions.

## R1: direction-aware route alternatives

R1 adds `graph_construction/`, `junction_transitions/`, and `path_hypotheses/`.
The original SCP CLI and its accepted/rejected/ambiguous decisions remain the
frozen baseline. The new experimental entry point generates alternatives from
the existing saved tail/head masks and original RGB evidence:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_r1.py \
  --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r1_routes_2026_09_09_verified
```

Run from the project root. The output directory must be new and inside
`Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/`. Settings are in
`configs/overlap_r1.json`. A development-only `--limit 1` is available; it does
not satisfy the full 24-image exit check. A stopped run with unchanged code,
configuration, inputs, and frozen evidence can continue using the same command
with `--resume`. Completed image checkpoints are verified before reuse. A
directory with partially written, unverified image artifacts requires inspection
or a new run directory; the runner does not delete those artifacts.

Open the new `index.html` locally. It links every processed image, places the
fixed six-image panel first, and provides original/baseline/independent-route
toggles. Individual-head controls show each retained route, its termination,
cost decomposition, branch usage, and junction pairings. `routes.json` contains
both K=5 and diagnostic K=10 results. Paths use original-image XY coordinates
with explicit component-ROI offsets; the JPEG backdrop is resized for display.

The graph removes redundant diagonal triangle connections, clusters adjacent
junction pixels, preserves boundary ports and degree-two loops, and splits
segments at snapped baseline anchors. It records both pixel and edge accounting.
Transitions use radius-scaled arclength tangents outside junctions, signed
curvature consistency, and existing segment evidence. Pixel bridges through
junctions must remain supported by the skeleton. Unsupported connections are
reported, not drawn as invented chords.

Search permits revisiting a junction through unused segments, but never repeats
an ordinary segment. It retains different routes to the same endpoint and
explicit partial, boundary, and null possibilities. K limits non-null routes;
null remains separate. For K>1, a partial route reserves one slot when present.
The soft partial-termination cost is 0.25; the curvature weight is 0.35. These
are inspectable engineering preferences, not calibrated probabilities or
morphology rules. Search has a deterministic expansion limit and a bounded
frontier. Diagnostics distinguish complete enumeration from limited search;
returned alternatives are ranked among explored candidates, with the documented
partial reservation. There is no global multi-head selection in R1.

`fixture_report.json` separates explicit-graph route availability from rendered
mask recovery. The rendered tests supply constructed tail and head masks; they
do not establish full RGB detection accuracy. R0's separate full RGB fixture
outputs remain available. The indeterminate example preserves both logical
identity solutions and receives no arbitrary identity-accuracy score. Flip,
180-degree rotation, and padded-translation diagnostics report raster/thinning
sensitivity rather than hiding failures. Synthetic cases are development
fixtures, not untouched real validation data.

Verify a completed report from the project root:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/audit_r1.py \
  Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r1_routes_2026_09_09_verified
```

The audit checks the complete source-image inventory, frozen evidence hashes,
image checkpoints, retained baseline coordinates/statuses, graph accounting,
route support and continuity, edge-use and K limits, score decomposition,
constructed graph-route coverage, identity alternatives, and local viewer links.
This is an engineering audit, not a real-image correctness claim.

Tests and compilation, from the SCP directory:

```bash
../.venv/bin/python -m unittest discover -p 'test_*.py'
../.venv/bin/python -m py_compile graph_construction/*.py junction_transitions/*.py path_hypotheses/*.py experiments/*.py tests/test_r1*.py
```

## R2: joint assignment of frozen route proposals

From the project root, using a new experiment directory:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_r2.py \
  --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r2_joint_new
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/audit_r2.py \
  Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r2_joint_new
```

Configuration is `configs/overlap_r2.json`; the frozen input release is identified
by `experiments/r1_manifest.json`. `--limit` creates a development-only partial
run. `--resume` verifies unchanged code, configuration, inputs and evidence, then
reuses only hash-verified complete image checkpoints. New runs and partial image
recovery never delete existing artifacts.

The reviewed release is `outputs/experiments/r2_joint_2026_09_09_reviewed/`.
Open `index.html` or `fixed_panel_contact_sheet.png`. Image pages compare original,
frozen baseline, R1 rank 1, R2 unary-independent, joint, runner-up and complete
per-head counterfactual assignments. `assignment.json` records the finite pool,
source offsets, decomposed costs, resource ownership, solver bounds and baseline
comparisons. The two strict isolated controls retain their baseline rasters.
Other baseline rasters remain comparison references, not fabricated ordered
paths. Ordinary branches and endpoints are exclusive; crossing-node pixels can
have shared logical ownership. Extended connectors/corridors remain unmodeled.

The solver uses dependency-free deterministic branch-and-bound. There is one
choice, including explicit null, per source-image head even when several tail
components touch that head. Independent conflict groups can be solved separately.
Score margins are uncalibrated, and an optimal assignment is optimal only over
the supplied route pool. R1 route-search limits remain visible independently.

The all-image run has 183 heads, 1,158 choices and 21 changes from unary choices.
All ordinary/endpoint conflicts disappear, but this is not measured biological
accuracy. Rendered full-instance F1 is 0.9063 for unary independent and 0.7982 for
joint choices on the development fixtures. The shared-connector failure is
explicitly preserved for later work. Null predictions count as zero recall/F1
in primary synthetic means. See [the full R2 report](../../R2_IMPLEMENTATION_REPORT.md)
for measurements, provenance, tests and exit-criterion details.

Verification from the SCP directory:

```bash
../.venv/bin/python -m unittest discover -p 'test_*.py'
../.venv/bin/python -m py_compile global_assignment/*.py experiments/r2*.py experiments/run_r2.py experiments/audit_r2.py tests/test_r2*.py tests/test_global_assignment.py
```


## R3: conditional logical instance masks

R3 reconstructs frozen R2 selections using local width and explicit pixel
ownership. It preserves raw RGB, full detected-head evidence, selected supported
centerlines, and uncertainty. No new training, annotations or dependency is
required. Review [the R3 report](../../R3_IMPLEMENTATION_REPORT.md) and
[released all-image viewer](../outputs/experiments/r3_masks_2026_09_09_release/index.html).
The manifest is `r3_manifest.json`.

From the project root, use a new output directory:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_r3.py --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r3_my_run
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/audit_r3.py Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r3_my_run
```

The default configuration is `configs/overlap_r3.json`. `--resume` requires exact
verified checkpoints and unchanged frozen inputs/code/configuration. A partial
image directory is not silently overwritten. The runner verifies R2 and its
frozen R1/R0 evidence before reconstruction.

`reconstruction.json` schema `scp.r3.reconstruction.v2` records original-image
ROIs, nine per-instance exports, null choices, component crossing masks, and
explicit head/tail evidence conflicts. Default logical masks prioritize supported
tail centerlines; `head_priority_instance_mask` supplies the alternate policy.
`head_evidence_mask` preserves the complete original detected head. Neither
ownership policy establishes biological identity or occlusion. The standalone
audit checks both, preserving raw evidence conflicts as a separate count.

The released 24-image run exports 170 non-null instances and 13 null choices.
Both interpretations have zero forbidden logical overlap; three unresolved
head/tail evidence pixels remain recorded. Twenty synthetic cases expose 16
remaining determinate completeness/leakage failures inherited from, or
conditional on, fixed selections. Indeterminate identities are unscored and
abstention is allowed. These development fixtures do not measure real accuracy.

Tests, from the SCP directory:

```bash
../.venv/bin/python -m unittest discover -s . -p 'test*.py'
```

Final verification: 176 tests, full 24-image run, ownership/source audit, and 25
page-script syntax checks passed. Browser interaction remains unverified.


## R4a: optional foreign attachment continuations

The reviewed release is in `r4a_manifest.json`; see
[results and limitations](../../R4A_IMPLEMENTATION_REPORT.md) and the
[all-image viewer](../outputs/experiments/r4a_attachments_2026_09_09_complete/index.html).
The wrapper preserves all old R1 routes and graph geometry, then adds bounded
pass-through alternatives before unchanged R2 joint assignment. It is an
experimental path proposal mode. No R3 masks, production decisions or raw images
are changed.

From the project root, require a fresh output root:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_r4a.py --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r4a_my_run
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/audit_r4a.py Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r4a_my_run
```

Default config: `configs/overlap_r4a.json`. `--limit` makes an explicitly partial
smoke run; `--resume` requires unchanged verified checkpoints. Completed or
unverified image directories are not overwritten. The runner verifies the R3
release and frozen R1/R2 source evidence. Assignment determinism excludes only
wall-clock diagnostics, retaining comparisons of choices, scores, bounds and
search counts.

The final run processed 24 images in 51.42 seconds, added 60 routes, and changed
12 head selections. Selected nulls increased from 13 to 17; this is not an
accuracy claim. Sixteen mechanism fixtures pass their declared invariants.
Sixteen existing canonical graph/rendered cases remain unchanged. All 201 tests,
the full artifact audit, and 25 page-script syntax checks pass; browser interaction
remains unverified. Continuous image evidence, physical neck-anchor proposals,
gaps, shared corridors and final consolidation remain later bounded work.
