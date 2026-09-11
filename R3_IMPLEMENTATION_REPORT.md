# R3 conditional instance-mask reconstruction — September 9, 2026

R3 implements the bounded reconstruction package in the annotation-optional plan.
It reconstructs logical masks around frozen R2 selections. It does not establish
that those selections identify the correct complete sperm. R4–R6 remain
outstanding; no training or new annotations were required.

The release viewer is
[`r3_masks_2026_09_09_release/index.html`](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r3_masks_2026_09_09_release/index.html).
It covers all 24 images and the fixed six-image panel, with separate centerline,
logical-mask, original-image, frozen-baseline, and per-instance views. Original
RGB crops retain other visible sperm; logical masks express selected ownership.

## Results

| Engineering measurement | Result |
| --- | ---: |
| Existing source images | 24 |
| Frozen selected non-null instances / null choices | 170 / 13 |
| Supported selected centerline pixels lost | 0 |
| Forbidden full-instance duplicate pixels, default / head-priority alternative | 0 / 0 |
| Foreign-head growth pixels removed | 24 |
| Unresolved pinned head/tail evidence pixels | 3 |
| Synthetic cases / declared instance records | 20 / 30 |
| Unit tests | 176 passing |
| Synthetic determinate completeness/leakage threshold failures | 16, retained in report |

The three ambiguous pixels occur in 100x WT B6 Exemplary B, at original image
coordinates `(931,369)`, `(932,369)`, `(933,370)`. A frozen supported tail path
crosses another selected instance's detected head boundary. The default logical
representation keeps those tail pins and defers those pixels from the other
logical head. A separate detected-head-priority mask expresses the other
interpretation. Both have no forbidden duplicate ownership. The entire original
detected head remains unchanged in `head_evidence_mask`, and both candidate
owners are recorded in uncertainty metadata. This is a representation policy,
not evidence of biological occlusion. Use head evidence, not the refined logical
head, for detected-head measurements; inspect its uncertainty.

On direct constructed width tests, intended-mask recall is 98.73–100% with zero
extra pixels at drawing widths 1, 3, 5 and 7. Unequal-width X fixtures retain
98.73–100% of each track with zero extra pixels. These OpenCV drawing-width
parameters are not calibrated physical diameters. In the width-5 extra-branch
case, whole-component extraction includes 319 unwanted branch pixels; R3
includes zero, with the same 98.73% tail recall as its isolated width-5 control.
The direct loop and boundary cases retain 94.49% and 99.40% respectively.

The benchmark also reconstructs the exact frozen selections from all 11 R2
rendered fixture records, with selected IDs and route-point hashes. It exposes
16 failures against the stricter determinate recall/leakage checks, including
wrong curved-crossing assignments, partial competing routes, and an isolated
rendered route with 88.87% mask recall. All failures remain visible. Keeping a
selected centerline is not evidence that it is the right or complete centerline.
The indeterminate fixture retains both declared identities, reports its null
choice, and receives no arbitrary per-identity truth score.

These are development fixtures inspected during algorithm work, not an untouched
validation set. Real-image precision, recall and biological identity accuracy
have not been measured.

## Implementation

- Radius estimates use a half-pixel boundary convention for the evidence
  distance transform. Samples near modeled crossings or truncated image borders
  are excluded and interpolated along ordered track arclength. With no usable
  width samples, the conservative radius floor is used and uncertainty recorded.
- Supported centerline pins are retained. Unsupported points do not seed mask
  growth. Coarse path segments use the same OpenCV LINE_8 raster convention as
  frozen-route verification; a slope-one-half regression catches missed pins.
- Local tubes follow supplied track geometry. Ordinary evidence pixels go to
  the nearest normalized tube. Sharing requires explicit pairwise crossing
  permission with every existing owner. Overlapping A/B and B/C permissions do
  not silently authorize A/C ownership. No independent RGB orientation estimate
  is claimed in this package.
- Head/tail composition excludes foreign selected-head growth, preserves pins,
  and exports typed evidence disagreements. No hidden texture is synthesized.
- Each instance has nine exports: original RGB crop, tail mask, logical head
  mask, logical instance mask, supported centerline, highlighted crop, original
  head evidence, ownership uncertainty, and head-priority alternative mask.
  Coordinates remain in the source-image system with explicit crop ROIs.
- The runner freezes input, source-code, configuration, dependency, R0/R1/R2
  evidence, and fixture hashes. It requires a new output root or verified
  checkpoints. No historical outputs are overwritten.
- The release audit independently checks exact raw crops, full head evidence,
  frozen supported centerlines, tail evidence subsets, both composition policies,
  exact uncertainty regions, image inventory, source hashes, and viewer links.

The predictor remains unchanged, with SHA-256
`cd10122dba110b7cffa55e2e72dadb96f66b6e2d6f9e884d40a5bf6f3a65ac51`.
Raw images, annotations, reviewed CSVs, production thresholds/statuses and frozen
R0/R1/R2 outputs were preserved. No dependency was added.

## Files and verification

Added `mask_reconstruction/{__init__,reconstruction,head_ownership}.py`;
`experiments/{r3_adapter,r3_instance_checks,r3_benchmark,r3_viewer,r3_contact_sheet,run_r3,audit_r3}.py`;
`experiments/r3_manifest.json`; `configs/overlap_r3.json`; this report;
`tests/test_mask_reconstruction.py` and the seven `tests/test_r3_*.py` files.
Updated root, SCP and experiments READMEs. Tests cover width recovery, boundaries,
unsupported samples, deterministic ownership, rasterization, pairwise sharing,
head/tail conflict composition, exact exports, tampering, frozen route identity,
indeterminate metric scope, viewer assets and checkpoint validation.

Exact final verification commands from the project root:

```bash
.venv/bin/python -m py_compile Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py Algorithmic_Pixel_Mask_For_Tails/test_tail_assignment.py Algorithmic_Pixel_Mask_For_Tails/mask_reconstruction/*.py Algorithmic_Pixel_Mask_For_Tails/experiments/*.py Algorithmic_Pixel_Mask_For_Tails/tests/test_r3*.py
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_r3.py --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r3_masks_2026_09_09_release
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/audit_r3.py Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r3_masks_2026_09_09_release
git diff --check
```

Full suite, from `Algorithmic_Pixel_Mask_For_Tails`:

```bash
../.venv/bin/python -m unittest discover -s . -p 'test*.py'
```

Runtime, final audit and JavaScript syntax results are recorded in the release
manifest. Synthetic unequal-width and extra-branch panels, the fixed real panel,
and enlarged three-pixel ownership interpretations were visually inspected.
Browser interaction remains unverified: the available headless Firefox runtime
failed during R1; static asset and script checks are not browser execution.

Two earlier complete runs remain preserved with `review_status.json`:
`r3_masks_2026_09_09_reviewed` failed the full-instance ownership audit with 27
pixels (24 removable growth pixels and three pinned conflicts);
`r3_masks_2026_09_09_ownership_reviewed` passed ownership but exposed an erroneous
indeterminate truth score and an invalid abstention penalty in the benchmark.
Correcting that metric scope changes the reported failure count from 17 to 16,
without changing masks or determinate geometric results. The final release
regenerates everything with the corrected code and provenance.

## Exit criteria and limitations

| R3 criterion | Outcome |
| --- | --- |
| Retain selected supported centerlines | Passed, fixtures and all 24 images |
| Retain intended widths | Passed declared isolated/unequal-width sweeps |
| Remove extra-branch contamination without additional completeness loss | Passed versus same-width isolated reconstruction control; 319 to zero unwanted pixels |
| No broad ownership duplication outside declared regions | Passed both logical interpretations; source conflict remains explicit |
| View masks separately and superimposed | Exported and checked; browser interaction unverified |

The bounded reconstruction criteria are satisfied at the artifact and tested
algorithm level. There is no claim that every rendered fixture passes the
stricter end-to-end mask targets. Head-evidence alternatives are a necessary R3
integration detail, not a change to head detection. Wrong/missing paths,
shared-junction connectors, unresolved corridors and unknown widths remain
limitations for subsequent authorized work. R3 does not implement R4 or R5.
