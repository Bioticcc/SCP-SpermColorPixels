# R4a optional head-attachment continuations — September 9, 2026

R4a is complete as a bounded attachment-route work package in the
annotation-optional plan. It treats a projected foreign head attachment as a
possible stopping point and also retains supported continuation alternatives.
The baseline still stops there. No new image evidence, physical neck locations,
gap edges, detector thresholds or training were introduced.

Open the [all-24-image R4a viewer](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r4a_attachments_2026_09_09_complete/index.html).
A [specific missing-route example](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r4a_attachments_2026_09_09_complete/missing_route_example.png)
shows a newly available proposal, explicitly labeled unselected. R3 mask exports
remain frozen; this package exports paths and assignments, not new R3 masks.

## Measured motivation and result

R1 terminates a route at every foreign anchor node. Those nodes can be projected
onto an otherwise continuous skeleton by nearest-head attachment rules. Eight
of the nine approximate missing-route observations have a reachable foreign
anchor; none of their retained K=5 proposals actually ends at one. Reachability
therefore suggested an experiment, not proof that this was their common cause.

The targeted probe improves the longest available route in **one of nine**
observations: 40x Prssly B, tail 13/head 13, from **279.37 to 703.48 pixels of
arclength**, by passing the attachment for head 15. The baseline raster has 721
pixels, an approximate comparison rather than biological truth. That new route
is **not selected** by joint assignment. Two B6 D observations gain other
pass-through alternatives without increasing their longest route. The remaining
six do not change. Two dense Prssly KO A searches still hit their resource cap.
The single-anchor Prssly B tail-2/head-7 control remains unchanged.

| Full existing-image experiment | Result |
| --- | ---: |
| Images | 24 |
| Original R1 non-null route records retained | 977 |
| Original component/head null records retained | 209 |
| Added non-null alternatives | 60 |
| Changed selected heads / images | 12 / 7 |
| Selected nulls, frozen R2 / R4a | 13 / 17 |
| Approximate missing-route observations, before / after | 9 / 8 |
| Limited extension searches | 4 |
| Image-level finite-pool assignment statuses | All optimal |
| End-to-end report runtime / image analysis and export | 51.42 s / 29.98 s |

Seven changed selections use pass-through routes, including one formerly null
head. Five formerly non-null heads become null, for a net increase of four
nulls. The change can allow a path to occupy evidence formerly assigned to
another head. Without trusted labels, this might remove a false attachment or
might wrongly absorb another sperm. **No real-image accuracy gain is claimed.**
The original accepted/rejected/ambiguous decisions remain unchanged.

## Constructed geometry and regression checks

The mechanism fixture is a T graph with a left head, a foreign head attachment
at the junction, and right/down branches. Its geometry was fixed before results:
128×128 canvas, left anchor `(20,64)`, junction `(64,64)`, right endpoint
`(116,64)`, down endpoint `(64,116)`. Head centers are `(14,64)` and `(64,57)`,
radius 7. The declared solution assigns left-to-right to h1 and the downward
branch to h2. Rendered inference receives constructed head/tail masks and uses
the existing nearest-anchor and thinning code; truth paths are used only by the
explicit graph layer and evaluator.

One explicit case and 15 rendered cases cover drawing widths 1/3/5 and identity,
horizontal flip, vertical flip, 180° rotation and padded translation `(11,7)`.
All retain the original routes and make the complete h1 continuation available.
The explicit selected h1 centerline recall rises from 0.4845 to 1.0; both selected
tracks match the declared graph pairing. Rendered new selected/oracle recall is
at least 0.98, with a two-pixel centerline tolerance. The report contains all old
and new per-head measurements, including the poor old h2 selections in several
rendered variants. Sixteen cases report zero declared invariant failures.

A separate regression check runs all eight existing canonical geometries as
explicit graphs and rendered masks: all 16 retain exactly the same selected IDs
and add no routes. Existing crossing, loop and indeterminate-case limitations
are preserved, not counted as solved. Both strict isolated real baseline controls
remain unchanged. These are inspected development cases, not untouched validation.

## Implementation and decisions

`head_anchors/attachment_routes.py` reuses the existing bounded R1 search with a
shallow graph replacement containing only the queried head in its anchor map.
Nodes, segments, radii, evidence, original anchor records and masks are unchanged.
The wrapper preserves every old hypothesis and diagnostic, then appends at most
five distinct non-null routes that pass a foreign anchor in their interior.
A route ending at a foreign anchor is still partial. Added route metadata lists
all passed foreign head IDs. Collocated starting anchors do not count as passage.
Repeated-edge and loop limits remain those of the original search.

The extension asks the existing search for `additional_k + baseline_record_count`
ranked candidates before filtering for foreign-anchor passages. With the usual
five non-null baseline routes plus null, this is 11 candidates. Consequently,
useful pass-through routes can still be pruned. Expansion limits and diagnostic
termination remain visible. This is not an exhaustive route generator.

The unchanged R2 pool builder and joint solver select from the expanded pool.
Null, partial and coverage weights remain 1.5, 0.85 and 1.0. Coverage normalization
uses the longest available route for the same head, so old choices can receive
changed coverage costs after expansion. This is an explicit consequence of the
existing uncalibrated objective, not a new head-confidence model.

The viewer supplies frozen R2, R4a, a complete R4a runner-up, per-route alternatives,
selected null/partial records and raw JSON with all constrained alternatives.
Colors consistently identify heads; unordered baseline controls remain scatter
geometry. Frozen 1400-pixel previews map to the full source-coordinate canvas.

## Files and verification

Added `head_anchors/{__init__,attachment_routes}.py`,
`experiments/{r4_adapter,r4_benchmark,r4_viewer,run_r4a,audit_r4a}.py`,
`experiments/r4a_manifest.json`, `configs/overlap_r4a.json`, this report, and six
R4 test files: attachment routes, benchmark, viewer, viewer validation,
runner/audit, and canonical regressions. Updated root/SCP/experiments READMEs.
No dependency was added. The production predictor hash remains
`cd10122dba110b7cffa55e2e72dadb96f66b6e2d6f9e884d40a5bf6f3a65ac51`.

From the project root:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_r4a.py --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r4a_attachments_2026_09_09_complete
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/audit_r4a.py Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r4a_attachments_2026_09_09_complete
.venv/bin/python -m py_compile Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py Algorithmic_Pixel_Mask_For_Tails/test_tail_assignment.py Algorithmic_Pixel_Mask_For_Tails/head_anchors/*.py Algorithmic_Pixel_Mask_For_Tails/experiments/*r4*.py Algorithmic_Pixel_Mask_For_Tails/tests/test_r4*.py
git diff --check
```

From the SCP directory:

```bash
../.venv/bin/python -m unittest discover -s . -p 'test*.py'
```

Final results: **201 tests pass**, full inventory and source integrity pass, and
the audit passes for 24 images, 60 additions, 1,186 retained route records and
112 local viewer links. It rebuilds finite pools, validates graph-supported
geometry and passed-head metadata, and re-solves every assignment. Determinism
comparison excludes recorded wall-clock timing; selection, costs, solver states,
bounds and expansion counts remain compared. The corrected post-run auditor has
its own source hash in the release manifest.

All 25 page scripts passed `node --check /tmp/scp_r4a_viewer_controls.js` after
extraction. The explicit mechanism panel and real missing-route crop were
visually inspected. Browser interaction remains unverified. The run root also
contains `canonical_regression.json`, `targeted_missing_route_probe.json` and the
annotated review PNG; these supplementary checks are separate from generated
fixture checkpoints.

Three incomplete attempt directories remain preserved with `review_status.json`:
`..._reviewed` stopped at duplicate fixture metadata writing; `..._verified`
stopped at a tuple/list JSON comparison; `..._release` stopped at the first image's
incorrect native-preview-size assertion. The completed root above includes the
fixes. Early development also encountered nonexistent test names, a wrong test
write directory, and a frozen-dataclass mutation in a test; final checks pass.
The first determinism audit compared wall-clock durations and falsely failed;
correcting that verifier did not change route or assignment artifacts.

## Exit criteria and remaining scope

| R4a criterion | Outcome |
| --- | --- |
| Intended continuation becomes available | Passed explicit, rendered, and one measured real proposal case |
| Persists across fixed variations | Passed all 15 rendered variations |
| Existing fixtures and isolated controls retained | Passed 16 canonical checks and both real controls |
| Added alternatives are inspectable | Passed metadata, route controls, source coordinates and report assets |
| Every R4a exit criterion satisfied | Yes at tested algorithm/artifact level; browser interaction unverified |

R4a repairs attachment stopping semantics. It does not implement continuous
pre-threshold color/ridge evidence, new physical neck anchors, supported gap
edges or endpoint evidence recovery. Those remain R4 work to select and bound
using the remaining failures. R5 shared connectors/corridors and R6 consolidation
have not begun. The full annotation-optional implementation goal remains active.
