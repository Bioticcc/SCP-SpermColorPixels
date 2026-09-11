# R1 direction-aware route alternatives — September 9, 2026

R1 is complete as a route-generation and demonstration work package. R2–R6
remain outstanding; the overall annotation-optional implementation goal is not
complete. The original SCP CLI and its status decisions remain unchanged.

## Results

The complete new graph/search run processed all 24 existing Ward images and
393 tail components in 58.40 seconds (57.22 seconds in image processing). It
retained all 209 baseline candidates, including their exact source-coordinate
skeleton pixels, head identities, and statuses. There are 209 anchored searches
and 977 non-null K=5 alternatives. Five searches reached an expansion/frontier
limit and are explicitly reported as incomplete searches. They are not assigned
confidence based on an assumed exhaustive search.

Synthetic results use the eight R0 development fixtures. Full tracks and the
deliberately partial T-contact branch are reported separately:

| Check | R0 baseline | R1 |
| --- | --- | --- |
| Declared determinate graph routes retained in K=5 | Not represented as top-K alternatives | 11/11, including one partial branch |
| Declared graph routes retained in diagnostic K=10 | Not represented as top-K alternatives | 11/11 |
| Rendered full tracks with ≥95% centerline precision and recall at 2 px | 5/10 | 6/10 at rank 1; 9/10 available in K=5 |
| Rendered T-contact partial branch with the same spatial criterion | 0/1 | 1/1 |
| Indeterminate identity solutions retained | No identity accuracy assigned | Both solutions' constituent routes available |
| Explicit-graph flip/rotation/translation rank and score checks | Not measured in this package | 32/32 case-transform checks pass |

The rendered isolated curve still has 91.01% recall. The missing distal evidence
is inherited from skeletonization; R1's best curve and the R0 baseline have the
same measured recall. Both curved-crossing routes and the self-loop are now
available with complete constructed centerline coverage, but rank 1 still
prefers partial stops in these cases. The self-loop rank-1 recall is worse than
R0 (14.43% versus 34.83%), while the correct loop exists at rank 2. This is a
representation improvement, not evidence that local ranking is solved.

On real images, many first alternatives are short prefixes. The fixed contact
sheet makes this visible. These are independent experimental proposals and do
not replace the baseline. Joint selection and completeness evidence are still
needed; no real-image accuracy gain is claimed.

## Implementation

Added under `Algorithmic_Pixel_Mask_For_Tails/`:

- `graph_construction/__init__.py`, `segment_graph.py`: immutable records,
  explicit geometry import with validation, redundant diagonal-edge removal,
  clustered junction pixels and boundary ports, interior anchor splits,
  degree-two cycles, local radii, and pixel/edge conservation diagnostics.
- `junction_transitions/__init__.py`, `scoring.py`: correctly oriented incoming
  ports, multiple arclength scales based on local radius, direction costs and
  signed curvature consistency. Junction bridges use supported pixels.
- `path_hypotheses/__init__.py`, `search.py`: deterministic bounded directed-edge
  search, repeated-edge prevention, spatial junction revisits, multiple routes
  to the same endpoint, partial/boundary/null possibilities, K=5/K=10, explicit
  pruning diagnostics, and decomposed uncalibrated costs. Cached geometry and
  lightweight terminal records avoid reconstructing discarded pixel paths.
- `experiments/r1_adapter.py`, `r1_benchmark.py`, `r1_viewer.py`, `run_r1.py`,
  `audit_r1.py`: verified frozen-mask reuse, baseline-coordinate retention,
  explicit/rendered fixture measurements, local HTML/SVG comparisons, all-image
  execution, safe checkpoints, and artifact-invariant verification.
- `tests/test_segment_graph.py`, `test_path_hypotheses.py`,
  `test_search_performance.py`, `test_r1_adapter.py`, `test_r1_benchmark.py`,
  `test_r1_checkpoints.py`.

Also added `configs/overlap_r1.json` and this report. Updated
`experiments/README.md`, the SCP README, and the root README with R1 usage and
status. No dependency was added. No source TIFF, human annotation, reviewed CSV,
canonical output, production threshold, or main SCP implementation was changed.

K counts non-null routes; null is separate. For K>1 a partial possibility
reserves one slot when available. The soft partial-termination preference is
0.25 and curvature weight is 0.35. Evidence is a length-weighted average of
segment support costs, so dividing one uniformly supported branch into more
segments does not change its evidence score. These settings are engineering
preferences rather than calibrated probabilities. The main implementation and
all executed helper/configuration sources are fingerprinted in run provenance.

## Verification and artifacts

From the SCP directory:

```bash
../.venv/bin/python -m unittest discover -p 'test_*.py'
../.venv/bin/python -m py_compile graph_construction/*.py junction_transitions/*.py path_hypotheses/*.py experiments/*.py tests/test_r1*.py
```

96 tests passed; compilation passed. This includes the original production
behavior tests, graph/route geometry checks, all named graph fixtures, strict
partial-stop semantics, both indeterminate solutions, supported junction joins,
boundary/foreign-head handling, score decomposition, subdivision invariance,
deterministic limits, and source-coordinate/checkpoint safeguards. The large
branching regression enumerates 1,997 terminal candidates while materializing
only five retained polylines. Frozen route/score regressions cover all eight
fixture cases across the search performance change.

From the project root:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_r1.py --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r1_routes_2026_09_09_verified
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/audit_r1.py Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r1_routes_2026_09_09_verified
git diff --check
```

The full run and final integrity checks passed. The artifact audit passed with
24 images, 393 components, 209 preserved baseline candidates, 977 K=5
alternatives, 109 local viewer links, and 1,169 controls. It verifies route
support/continuity and edge-use limits, baseline coordinates/statuses, source
and candidate inventories, frozen-evidence hashes, checkpoint hashes, cost
decomposition, graph coverage, and identity alternatives.

The release artifact root is:

```text
Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r1_routes_2026_09_09_verified/
```

Open `index.html`. It links all 24 image viewers; each offers individual-route
controls, junction pairings, costs, and K=10 JSON diagnostics. Other important
artifacts are `fixture_report.json`, `fixtures/*/comparison.png`,
`images/*/routes.json`, `images/*/image_completion.json`, `provenance.json`,
`integrity.json`, `completion.json`, `audit.json`, and
`fixed_panel_contact_sheet.png`. The latter is an additional overview drawn
from the frozen report's coordinates and previews. Real and synthetic panels
were visually inspected. The embedded viewer JavaScript passes `node --check`.

A headless Firefox screenshot attempt failed with graphics/X initialization
errors and exit 139 despite software-rendering settings. Browser-executed
interaction is not certified. The standalone PNG panels and local asset/code
checks provide the verified visual artifacts for this milestone.

The earlier `r1_routes_2026_09_09/` attempt stopped after two images; its process
handle disappeared and it has no final completion/integrity record. Its cause
of termination is not established. `attempt_status.json` marks it incomplete.
It is not the release. Smaller `r1_*development*` artifacts remain separate.

## Exit criteria and limitations

- Intended routes survive in top-K on every named determinate graph fixture:
  satisfied. The rendered upstream recovery gaps remain separately visible.
- Indeterminate alternatives survive: satisfied; no hidden identity is scored
  as uniquely correct, and extended-corridor global ownership is deferred to R5.
- Self-loop traversal is bounded: satisfied by distinct-edge state, deterministic
  limits, tests, and saved search diagnostics.
- Existing non-junction baseline behavior is preserved: the unchanged default
  implementation, tests, frozen evidence hashes, and exact baseline-coordinate
  audit establish this. Experimental first alternatives can differ and are not
  adopted as production decisions.
- Fixed six-image viewer/contact sheet and full 24-image inventory: satisfied.
  Browser interaction remains an environment-limited verification gap as noted
  above; static visual artifacts are inspected and available.

Four of 32 rendered case-transform checks change node/segment counts: isolated
curve under vertical flip/180 rotation, and self-loop under horizontal/vertical
flip. Investigation confirms differences in the unchanged thinning output
(34, 34, 73, and 53 skeleton pixels respectively in the inverse-mapped symmetric
difference). The isolated anchor also shifts from (22,94) to (20,96) inside the
constructed head. Explicit graph transformations pass, so these are recorded
upstream raster/anchor sensitivities for R4, not unexplained graph-search
equivalence claims. Equal node counts alone are not proof of topology identity.

R1's scope and engineering exit criteria are satisfied, with the displayed
limitations. There is no deviation toward training or annotation prerequisites.
R2 is the next bounded package: compatible joint assignment, including an
explicit treatment of partial/null choices and preservation of isolated
baseline routes. R1 scores alone must not drive acceptance or be mistaken for
complete biological identities.
