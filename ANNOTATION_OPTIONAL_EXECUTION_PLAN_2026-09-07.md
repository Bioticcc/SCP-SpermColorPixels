# Annotation-optional SCP improvement plan — September 7, 2026

**Active direction:** improve and demonstrate SCP using the existing repository data, deterministic algorithms, and synthetic fixtures. New lab images and human annotations are optional future validation resources. Their absence must not block this development track.

The user's September 7 instruction supersedes the annotation prerequisites in the July roadmap for the work packages below. The algorithmic direction remains SCP-first. This document changes execution order and development acceptance criteria; it does not claim completion of the old gold-dataset or accuracy criteria. Each implementation request remains one bounded work package.

The immediate product is a repeatable demonstration of individual head-to-tail paths through supported overlaps, clean logical instance masks, and visible alternatives where ownership is unresolved. No model training is required by this plan. New annotations can be integrated later without throwing away these components.

**Resources available now**

- 24 existing Ward RGB source images, including both magnifications, crossings, close parallel tails, and loops.
- SCP masks, component detection, skeleton utilities, independent path-v2 assignment, diagnostic outputs, and candidate scoring.
- Saved canonical predictions and 131 older manually reviewed candidate rows. Those labels describe specific old candidates; they must not automatically transfer to changed predictions.
- 33 passing existing tests, as verified in the preceding investigation.
- An annotation subsystem to retain for possible later use. Finishing the GUI, drawing new annotations, and building a gold evaluator are not prerequisites here.

**Execution sequence**

| Package | Concrete outcome | Demonstration | Main dependency |
| --- | --- | --- | --- |
| R0 | Frozen comparison and a small geometry benchmark | Reproducible current results and first synthetic failure examples | Existing repository only |
| R1 | Direction-aware junctions and multiple routes per head | Current route beside plausible alternatives on synthetic and real images | R0 |
| R2 | Joint selection across competing heads | Two conflicting local choices replaced by a compatible complete assignment | R1 |
| R3 | Local-width, track-aware instance masks | Clean per-instance masks beside current branch-contaminated outputs | R2 |
| R4 | Targeted image-evidence, neck-anchor, and gap improvements | Previously missing supported routes enter the candidate set | Failure evidence from R1–R3 |
| R5 | Parallel-contact and extended-overlap handling | Separate visible ridges; retain identity alternatives through unresolved shared regions | R2–R4 and explicit overlap cases |
| R6 | Stable demonstration release and engineering quality rules | All 24 images browsable, with changes, alternatives, and limitations | R1–R5 results |

Ship a viewable artifact after every package. R1 is the first algorithm milestone; the project does not wait until R6 to show progress. Packages are implementation scopes, not calendar estimates or guaranteed accuracy gains.

**R0 — Small baseline and benchmark foundation**

Record effective CLI settings, dependency versions, source-image hashes, Git revision, and thinning backend. Preserve existing outputs. Run a fresh comparison under a new experiment root and explain any discrepancy with the canonical path-v2 run before attributing later differences to new code. Avoid copying historical output trees or broadly refactoring the main script.

Add a compact procedural fixture generator using NumPy/OpenCV. Store its seed, true centerlines, heads, identities, and expected behavior separately from rendered masks/images. Begin with isolated curves, a same-color X, a curved crossing, two heads competing for paths, a loop/self-crossing, a T-contact, a supported sharp bend, and a deliberately indeterminate case. Include a graph-only layer to test traversal independently of thresholding and skeletonization, plus rendered fixtures that exercise those earlier stages.

For the indeterminate fixture, construct distinct logical identities that produce the same rendered evidence and specify that abstention/alternatives are expected. Do not score an arbitrary hidden identity as the uniquely recoverable answer. Geometry fixtures are engineering tests under declared assumptions.

Choose and save a fixed six-image demo manifest spanning both magnifications, an isolated control, crossing/loop cases, and parallel contacts. Include 40x Prssly C and 100x Prssly KO Exemplary B from the investigation; record the remaining selections before inspecting new algorithm results. Run all 24 images at package completion, so the demonstration cannot hide regressions outside its selected panel.

Expected additions: `configs/overlap_demo_baseline.json`, `Algorithmic_Pixel_Mask_For_Tails/experiments/` for manifests/comparison helpers, and `tests/fixtures/` for synthetic generation. Add only the metadata hooks needed in the existing CLI. Proposed scripts/paths are to be finalized in R0, not assumed to exist now.

Exit: baseline provenance saved; all old tests pass; fixtures regenerate exactly; baseline failure cases are recorded; the six-image comparison and full-24 inventory are reproducible. No requirement for gold labels or a large synthetic corpus.

**R1 — Direction-aware alternatives: first visible algorithm improvement**

Extract only the graph helpers needed for this package. Group adjacent junction pixels into one neighborhood and split the skeleton into branch segments with ports at its boundary. Fit branch tangents outside the crossing center at several local-width scales. Score incoming/outgoing continuations using local direction, curvature consistency, and existing image/color evidence. Keep smoothness soft so image-supported abnormalities survive.

Search with incoming-segment state instead of one predecessor per spatial pixel. Retain distinct paths to the same endpoint and allow a self-crossing to revisit a spatial junction in a different directional state. Prevent immediate backtracking, repeated ordinary-edge exploitation, and unbounded cycling. Include partial/boundary/null possibilities rather than always forcing a full tail. Start with configurable K=5 distinct paths per head, with a diagnostic K=10 run to reveal pruning sensitivity; these are initial resource settings, not claims of sufficient coverage.

Do not initially build a full-frame position-orientation volume. Use skeleton tangents and local oriented responses where needed. Preserve multiple orientation peaks at crossings; a single angle is insufficient. Retain the baseline route and record why each alternative differs, along with decomposed costs and termination/pruning reasons.

Expected modules: `graph_construction/`, `junction_transitions/`, `path_hypotheses/`, focused tests, and a small diagnostic viewer. Keep experimental output separate from current accepted/rejected/ambiguous decisions.

Exit: intended paths survive into top-K on all named determinable canonical fixtures; indeterminate fixtures retain alternatives; self-loop routes are bounded; non-junction baseline behavior remains unchanged. A local HTML viewer/contact sheet shows original image, baseline path, alternative paths, junction pairings, and path scores for the fixed real-image panel. This demonstrates representation and search improvements without asserting real-image correctness.

**R2 — Joint assignment across heads**

Select a compatible set of R1 hypotheses per connected component. Use at most one non-null hypothesis per head, ordinary-branch ownership constraints, endpoint compatibility, and explicitly permitted sharing at modeled crossing centers. Permit missing/false heads, partial tails, headless fragments, and unused evidence; do not force every colored pixel into a sperm.

Begin with deterministic branch-and-bound over the compressed hypothesis conflict graph, requiring no new dependency. Check it against exhaustive enumeration on small fixtures. Use deterministic expansion limits and report termination/bounds; record wall-clock runtime separately. On larger cases retain unresolved alternatives when the search limit is reached. Add an external solver only if measured component complexity justifies it.

Compute a runner-up complete assignment where feasible. Also compute a constrained alternative changing a particular head/continuation, because a component-wide runner-up may differ elsewhere. Name values score margins, not calibrated probabilities. Distinguish insufficient visual evidence from incomplete search.

Expected module: `global_assignment/`, solver tests, joint-solution records, and viewer overlays comparing independent and joint choices.

Exit: solver agrees with exhaustive solutions on small cases; ordinary ownership constraints hold; synthetic competing-head cases improve over independent selection; ambiguity remains in tied cases; isolated controls retain their baseline assignment. Report all changed real-image assignments, including apparently worse results, without calling a higher accepted count an accuracy gain.

**R3 — Reconstruct clean logical instance masks**

Grow around selected centerlines using radius estimates along each track. Estimate width away from crossings and interpolate through them; the union's crossing width is not the width of either individual tail. Restrict candidate pixels by original evidence, local orientation, and competing-track ownership. Share pixels only within explicitly modeled crossing regions.

Separate raw rectangular crop content from selected-mask contamination. An RGB crop can legitimately contain another sperm at a crossing even when logical instance masks are correctly separated. Preserve source pixels and export original crop, per-instance mask, highlighted view, and uncertainty metadata. Do not invent hidden texture or present inpainted crops as measurements.

Expected module: `mask_reconstruction/`, reconstruction fixtures, and updated demonstration exports.

Exit: selected supported centerline pixels remain included; width fixtures retain intended widths; synthetic extra-branch pixels decrease without a completeness regression; no broad duplicate ownership occurs outside modeled overlap regions. Viewer shows masks separately and superimposed, making branch leakage observable on the real panel.

**R4 — Repair specific missing evidence and anchors**

Use the preceding diagnostics to choose one measured failure category per subpackage: wrong neck anchor, missing faint segment, skeleton shortcut, or broken tail. This avoids rebuilding the entire upstream pipeline before testing joint assignment.

Preserve continuous SCP evidence before thresholding alongside the unchanged baseline mask. Estimate local ridge/orientation evidence over a small scale range derived from visible tail widths. Propose neck attachment candidates from head-boundary contact plus nearby tail direction; do not treat the center or ellipse axis of a hooked mouse head as an unquestionable neck.

Introduce bounded gap edges only when surrounding direction and image evidence support continuation. Log each gap and its unsupported distance. Avoid hard typical-length or straight-tail rules. Retain baseline masking and anchor modes for ablation.

Expected module: `tail_evidence/` and/or `head_anchors/` as justified, plus the corresponding generator variations and diagnostics. Do not expand the monolithic CLI with all new algorithms.

Exit for each subpackage: the intended synthetic route/anchor becomes available; the improvement persists across a fixed variation sweep; unaffected fixtures and isolated real controls retain acceptable behavior; all new gaps and anchors are inspectable. On unlabeled real images, call the output a changed proposal, not a confirmed correction.

**R5 — Parallel tails and extended shared regions**

Implement two bounded subpackages. First separate adjacent visible ridges using cross-tail profiles and direction consistency. Then represent a merge-and-separate corridor with its possible entering-to-exiting identity assignments. Width alone must not determine the number of sperm in a bundle. Use visible branches and head/path compatibility as evidence.

Permit conditional shared logical ownership only within the modeled corridor and retain alternative exit pairings when evidence cannot distinguish them. Distinguish a shared region from ordinary branch reuse by two duplicate paths. Preserve unaffected parts of each track even when ownership inside the corridor remains uncertain.

Expected changes: extensions to graph, hypothesis, assignment, and reconstruction modules, with merge/split and parallel-ridge fixtures. Implement subpackages sequentially.

Exit: resolved synthetic parallel ridges remain distinct; identical-evidence corridor cases do not produce falsely certain identity assignments; supported crossing/loop behavior does not regress. Real examples display entry/exit alternatives and shared regions. Resolving an information-limited corridor is not a mandatory release criterion.

**R6 — Demonstration release and quality rules**

Consolidate the already shipping viewers into one local, reproducible report for all 24 images. Support toggles for original pixels, baseline/new centerlines, selected masks, alternate identities, and uncertainty. Include a concise fixed demonstration sequence from easy case through crossing, loop, parallel contact, and unresolved case. Each example states exactly what changed and what is still uncertain.

Replace inappropriate experimental heuristics with separate evidence-support, assignment-margin, completeness-evidence, and solver-status fields. Remove reliance on the fraction discarded from a shared component as a standalone quality judgment. Keep scores explicitly uncalibrated until trustworthy labels exist. Preserve the old baseline CLI/status behavior in its own mode; expose new results through a named experimental resolver mode.

Publish a benchmark table with exact synthetic route/assignment metrics, mask metrics, runtime, solver limits, determinism, transformation sensitivity, and real-image output changes. A fixed corpus of generated cases should be held separate from parameter development; if inspected failures guide subsequent changes, record that reuse instead of continuing to call it untouched validation. Save generator versions, seeds, and configurations.

Exit: one documented command builds the report from available data; tests and the full 24-image smoke pass; expected invariants hold or deviations are explained; all failures remain visible; the report makes no unsupported real precision/recall claim. A successful research demonstration does not certify morphology/clinical readiness.

**Validation that works without new labels**

| Evidence | What it can establish | What it cannot establish |
| --- | --- | --- |
| Graph fixtures | Correct branch pairing enumeration, route availability, cycle handling, exact small-instance optimization | Whether the real image produced the right graph |
| Rendered synthetic fixtures | Centerline recovery, identity assignment, completeness, mask contamination under known rendering conditions | General accuracy on Ward or unseen lab images |
| Real-image regressions | Determinism, preserved isolated behavior, runtime, changed proposals and failure visibility | Whether every changed proposal is biologically correct |
| Transformation checks | Sensitivity to horizontal/vertical flips, 180° rotation, and padded integer translation | Correct identity: consistently wrong or empty output can also be invariant |
| Existing candidate review | Specific documented historical errors and fixed examples to inspect | Labels for altered predictions or overall dataset recall |
| Optional future labels | Real association/completeness accuracy and calibration | Broad generalization without diverse independent images |

For transformations, inverse-map outputs and compare centerline distance, path availability, topology, and status. Set tolerances before inspecting new results. Exact graph fixtures should preserve topology; raster thresholds need documented pixel tolerances. Record baseline sensitivity and demand no unexplained regression, rather than assuming the current pipeline is perfectly invariant. Restrict primary checks to transformations without interpolation and account for added borders. Combine these checks with nonempty synthetic recovery and baseline coverage checks so abstaining on everything cannot pass.

Semi-synthetic overlaps made from existing image fragments are a later stress test, not a prerequisite. They can contain cutout seams, unrealistic stain mixing, focus/blur artifacts, and uncertain source masks. Their labels establish the construction geometry only. Keep derivatives grouped by source image, document provenance, and avoid presenting automated SCP-derived masks as human gold annotations. No scientific source data is overwritten.

**How future images or annotations fit**

If they arrive, retain them as independent evaluation material initially. Map centerlines/heads/crossings into the existing versioned schema and run the deferred gold evaluator against frozen predictions. Use that evidence to measure and calibrate the resolver, then decide whether a narrowly scoped learned component is warranted. If they never arrive, R0–R6 remain executable and yield a useful, auditable demonstration with clearly stated evidence limits.

**Subagent protocol for implementation**

The primary agent owns the active package, architecture, integration, and final verification. Delegate a bounded implementation module to Terra and independent fixture/inventory checks to Luna when useful. Reserve independent review for invariants such as ownership, cycle handling, coordinate preservation, and solver correctness. Do not delegate later packages or allow multiple agents to modify the same files simultaneously. Each package reports added/changed files, exact verification commands, artifacts, known failures, and its own exit criteria before stopping.

**First executable work request**

> Implement R0 of `ANNOTATION_OPTIONAL_EXECUTION_PLAN_2026-09-07.md` only. Use the existing 24 Ward images and existing review artifacts. Freeze a reproducible comparison without overwriting canonical outputs; add a compact procedural graph/rendered fixture benchmark and a fixed six-image demo manifest. Keep SCP predictions and thresholds unchanged. Do not require new annotations, build a gold evaluator, train a model, broadly refactor SCP, or begin R1. Run existing tests and the new fixture checks, reproduce the baseline in a fresh output directory, record differences, and report exact commands and artifacts.

This planning task adds this document and a precedence notice to the July roadmap. No algorithm phase is implemented by these documentation edits; test/smoke execution belongs to the subsequent bounded implementation packages. The preceding investigation's verification results are historical and are not presented as a new test run.
