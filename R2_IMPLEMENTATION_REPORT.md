# R2 joint assignment — September 9, 2026

R2 is complete as a bounded finite-pool assignment work package under
`ANNOTATION_OPTIONAL_EXECUTION_PLAN_2026-09-07.md`. It adds a deterministic
optimizer and a reviewable all-image experiment. It does **not** establish
better real-image accuracy. R3–R6 remain outstanding; the overall implementation
goal is not complete.

The reviewed release is
[`r2_joint_2026_09_09_reviewed/index.html`](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r2_joint_2026_09_09_reviewed/index.html).
Its [fixed six-image contact sheet](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r2_joint_2026_09_09_reviewed/fixed_panel_contact_sheet.png)
shows original pixels, the frozen baseline overlay, independent unary choices,
and joint proposals. The comparison includes visibly truncated paths.

## Results and interpretation

The full 24-image assignment/report run took **10.31 seconds**, including
provenance checks, fixture generation, and report writing; image processing and
export took **6.48 seconds**. It uses frozen R1 graph routes and R0 evidence,
not a new execution of full RGB segmentation.

| Recorded quantity | Result |
| --- | ---: |
| Source images | 24 |
| Unique anchored image/head identities | 183 |
| Pool choices, including nulls and controls | 1,158 |
| Independent conflict groups | 157 |
| Largest conflict group | 3 heads |
| Joint changes from the same-cost independent selection | 21 heads in 12 images |
| Duplicate ordinary-segment/endpoint resource claims before / after assignment | 68 / 0 |
| Solver-limited groups / per-head constrained solves | 0 / 0 |
| Incomplete upstream R1 route searches | 5 |
| Selected endpoints / boundaries / partials / nulls / baseline controls | 162 / 5 / 1 / 13 / 2 |
| Approximate missing-route component/head diagnostics | 9 |

The 209 R1 component/head searches refer to 183 unique image/head identities.
Twenty-three heads occur in multiple tail components; their extra occurrences
account for the remaining 26 searches. The optimizer consequently groups by
actual head identity and branch/endpoint conflicts across components, rather
than allowing each component to assign the same head independently.

The two strict isolated controls retain their exact baseline rasters:
40x Prssly E, tail 1/head 2, and 40x Teyorf1 B, tail 5/head 5. Nine further
non-junction single-anchor baseline rows are excluded from this fixed-control
policy because the same head is anchored in another component. They remain
visible as baseline comparisons and participate through R1 alternatives.

All 209 original baseline candidates, coordinates and statuses remain in the
unchanged R1/R0 evidence. Ordinary legacy baseline rasters are comparison
references; only the two eligible isolated rasters are selectable preservation
controls. Unordered or branch-contaminated rasters are not silently converted
to ordered paths with invented costs. This leaves a known pool-coverage limit:
the longest K=5 route is less than half the baseline raster pixel count for nine
component/head observations. Arclength and raster pixel count are only an
approximate diagnostic, and the baseline is not biological truth.

## Constructed geometry measurements

There are 22 records: the eight canonical R0 cases and a new competing-endpoint
case at widths 1, 3 and 5, each evaluated as an explicit graph and as a rendered
mask. The rendered stage receives constructed head and tail masks; it does not
measure full RGB detector accuracy. These are development fixtures, not an
untouched validation set.

On the explicit new two-head case, both independent choices take the right
endpoint. Joint assignment selects upper-to-right and lower-to-top, matching
the declared compatible geometry. The test asserts the exact segment pairing,
not merely disappearance of conflicts. The indeterminate explicit fixture has
a tied runner-up (score gap zero), retains both declared identity options, and
receives no arbitrary identity-accuracy score.

Across the 16 declared full rendered instances, with a two-pixel centerline
tolerance:

| Selection | Mean precision | Mean recall | Mean F1 |
| --- | ---: | ---: | ---: |
| R1 rank 1 | 1.0000 | 0.6684 | 0.7629 |
| R2 unary independent | 0.9152 | 0.8987 | 0.9063 |
| R2 joint | 0.8729 | 0.7627 | 0.7982 |

These results distinguish changes to unary preferences from the effect of
joint constraints. **Joint selection regresses relative to the new independent
choices on these rendered full tracks.** On the separate declared partial
T-contact instance, F1 is 1.0000 / 0.6769 / 1.0000 respectively. All declared
instances are included in primary means; a null or missing prediction contributes
zero recall and F1. This run has no null rendered-fixture selections.

In the width-3 rendered competing case, thinning produces two junctions joined
by `segment_005`. Both complete crossing routes require that connector. R2
treats it as an exclusive ordinary segment, so the solution instead retains an
upper partial route and sends the lower route right. The per-head F1 values are
0.6176 and 0.5312. The required representation for shared connectors/corridors
remains R5 work; the optimizer cannot repair a missing capacity model. This is
visible in the comparison panels and machine-readable fixture records.

## Implementation decisions

- Branch-and-bound uses negative-safe suffix lower bounds, deterministic tie
  ordering, two incumbents, a bounded frontier, and a deterministic expansion
  limit. Discarded frontier states remain represented in bounds. No external
  solver or training dependency was added.
- Exactly one explicit choice, including null, is selected per head. Ordinary
  segments and terminal endpoints are exclusive. Crossing-node pixels are not
  exclusive resources. Headless fragments and unused evidence are permitted.
- Disconnected conflict groups are solved independently; their best and second
  solutions compose into image-wide assignments. Per-head counterfactuals forbid
  that head's selected hypothesis and can change other heads in its group.
  Viewer alternatives merge the complete changed group into the full image.
- Costs separate mean transition geometry, existing image evidence, relative
  route coverage, partial termination and null choices. Coverage is normalized
  against the longest available route for the **same head across image
  components**. Defaults are coverage weight 1, partial cost 0.85 and null cost
  1.5. These are uncalibrated engineering preferences, not sperm-length rules.
- Exact score margins are emitted only when base and constrained solves are
  proven optimal over the supplied pool. Limited cases expose bounds instead.
  A proven pool optimum does not mean the R1 search found every useful route.
- New output roots are required. Image and fixture checkpoints, raw-input and
  frozen-evidence hashes, configuration and code provenance permit verified
  resumption. Existing partial image artifacts are not deleted or overwritten.
- SVG paths use original-image coordinates with ROI offsets. Saved baseline
  rasters are scatter geometry. Every frozen baseline remains accessible. The
  viewer exposes all changed selections, null/partial states, costs, margins,
  missing-route diagnostics and upstream/solver limits separately.

## Files

Added under `Algorithmic_Pixel_Mask_For_Tails/`:

- `global_assignment/__init__.py`, `global_assignment/solver.py`.
- `experiments/r2_candidates.py`, `r2_assignment.py`, `r2_benchmark.py`,
  `r2_viewer.py`, `r2_contact_sheet.py`, `run_r2.py`, `audit_r2.py`,
  `r2_manifest.json`.
- `tests/fixtures/r2_competing.py` and tests `test_global_assignment.py`,
  `test_r2_assignment.py`, `test_r2_candidates.py`, `test_r2_competing_fixture.py`,
  `test_r2_benchmark.py`, `test_r2_runner.py`, `test_r2_audit.py`,
  `test_r2_viewer.py`.

Also added `configs/overlap_r2.json` and this report. Updated root README, SCP
README and experiments README to document the release. No production predictor,
threshold, raw image, annotation, reviewed CSV or historical prediction changed.
The predictor SHA-256 remains
`cd10122dba110b7cffa55e2e72dadb96f66b6e2d6f9e884d40a5bf6f3a65ac51`.

## Verification

Final full suite: **132 tests passed**. Meaningful new checks include exhaustive
small-instance solver comparison, negative costs and ties, valid limited-search
bounds, top-two pruning, interleaved conflict-group composition, exact competing
pairing, abstention-safe metrics, repeated-head coverage normalization, ROI
coordinate preservation, checkpoint tampering, complete constrained overlays,
baseline-control display and consistent truth/prediction head colors.

From the SCP directory:

```bash
../.venv/bin/python -m unittest discover -p 'test_*.py'
../.venv/bin/python -m py_compile algorithmic_tail_mask.py test_tail_assignment.py global_assignment/*.py experiments/*.py tests/test_r2*.py tests/test_global_assignment.py tests/fixtures/r2_competing.py
../.venv/bin/python -m unittest tests.test_r2_benchmark tests.test_r2_viewer
```

From the project root:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_r2.py --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r2_joint_2026_09_09_reviewed
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/audit_r2.py Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r2_joint_2026_09_09_reviewed
git diff --check
```

The full run completed successfully with final integrity passing. The audit
passed for 24 images, 1,158 hypotheses and 147 local links. Each of 25 generated
page scripts passed `node --check /tmp/scp_r2_viewer_controls.js` after extraction.
The fixed-panel PNG and explicit/rendered competing panels were visually
inspected. Browser interaction remains unverified because the available
headless Firefox runtime failed during R1 verification; syntax and asset checks
are not presented as browser execution.

An initial complete run at `r2_joint_2026_09_09_verified` is superseded: visual
review caught inconsistent head-color order between synthetic truth and
prediction panels. The corrected reviewed run above preserves the same numeric
metrics. The earlier tree has `review_status.json`; no artifacts were overwritten.
Development checks briefly encountered incomplete viewer imports while its
agent was editing; final compilation and tests passed after integration.

## Exit-criterion audit and remaining work

| R2 criterion | Outcome |
| --- | --- |
| Agreement with exhaustive small solutions | Passed |
| Ordinary branch and endpoint ownership constraints | Passed, all 24 images and fixtures |
| Competing-head improvement over independent choices | Demonstrated on explicit geometry; rendered regression remains visible |
| Tied evidence remains ambiguous | Passed explicit indeterminate/tie checks; shared corridor identities remain unresolved |
| Isolated baseline preservation | Passed for both eligible controls; other nine broad candidates have reused heads |
| All changed real assignments reported | Passed, per-head records and complete alternatives in all-image viewer |

R2 satisfies the bounded optimizer milestone. It does not satisfy an image-level
claim that overlap separation is solved. Extending assignment across components
that reuse a head and adding inspectable unary costs are necessary integration
details within R2; neither changes the production pipeline. Shared connectors,
missing routes, anchor errors and weak real-image completeness remain limitations
for the later authorized packages. R3 can now reconstruct masks around selected
tracks while preserving these assignment limitations explicitly.
