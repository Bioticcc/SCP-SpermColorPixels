# Tail overlap investigation — September 7, 2026

The original goal remains plausible for evidence-supported crossings with no neural-network training. The repository has not yet implemented or evaluated the strongest deterministic approach in its own roadmap: direction-aware path hypotheses selected jointly for all heads. Current results therefore do not demonstrate that extensive AI training is necessary. They also do not establish an achievable accuracy rate.

This is an investigation and recommendation work package, not implementation of a roadmap phase. Three bounded subagents audited code, artifacts/annotation state, and primary literature. The primary agent checked their findings against code and images, ran verification, and prepared this report. No algorithm, thresholds, dependencies, human annotations, or canonical predictions were changed.

**Verified project state**

| Item | Current evidence | Implication |
| --- | --- | --- |
| Source images | 24 usable, single-frame RGB TIFFs: 15 at 40x and 9 at 100x | No existing TIFF time sequence or focus stack to exploit |
| Current saved path-v2 run | 209 candidates: 22 accepted, 45 rejected, 142 ambiguous; 393 tail components and 325 head candidates | These are proposal/status counts, not accuracy or true sperm counts |
| Older candidate review | 131 labeled rows: 92 partial/cut off, 30 good full sperm, 6 false positives/debris, 2 merged, 1 unclear | Useful failure evidence, but a deliberately selected review queue cannot estimate overall precision or recall |
| Older mask review | 66 partial, 20 wrong tail, 11 extra tail, 10 wrong head, 12 good, 12 unclear | Both path ownership and completeness require measurement |
| Current path-v2 review | 184 queued candidates; human review fields blank | The apparent improvement from stricter rejection is not validated by this queue |
| Gold annotations | One in-progress image JSON, with one low-certainty head ellipse and no neck, centerline, endpoint, or crossing | No completed gold instance dataset or crossing benchmark exists |
| Reproducibility | Git works at `5217d2f`; root `requirements.txt` exists but has unpinned dependencies | July statements that Git and a dependency manifest are absent are stale |
| Phase readiness | Annotation modules and tests exist; no baseline-specific configuration/freeze package or gold evaluator found in the inspected project layout | Phase 0 is partial; Phase 1A tooling is present; pilot, locked gold evaluation, and Phase 2 exit criteria are not demonstrated |

The root README still calls the older 213-candidate/97-accepted run the latest verification. The SCP README correctly identifies path-v2. The analysis and roadmap should be read as historical evidence and intended work, respectively; their unchecked checklists do not substitute for inspecting artifacts.

Evidence: [path-v2 summary](Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs_path_v2/summary.json), [older reviewed queue](Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs/candidate_review/candidate_review_queue.csv), [current review queue](Algorithmic_Pixel_Mask_For_Tails/outputs/updated_outputs_path_v2/candidate_review/candidate_review_queue.csv), [annotation directory](Algorithmic_Pixel_Mask_For_Tails/annotations/ward_gold_v1/json/), [roadmap](PROJECT_NEXT_STEPS_IMPLEMENTATION_PLAN.md).

**What actually limits the current algorithm**

References below are to `Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py` at the inspected revision.

1. **A correct route can disappear before scoring.** `skeleton_graph` (line 758) represents eight-neighbor pixel connectivity. `shortest_path_tree` (870) uses distance-only Dijkstra with one predecessor per pixel. `score_head_tail_paths` (1438) scores just that route to each endpoint, limiting endpoints to 40 and using six farthest nodes when there are none. Different routes to the same endpoint and routes that must revisit a crossing are not fully represented. Later color or curvature scoring cannot recover an absent route. Increasing the endpoint limit would not fix this.
2. **Heads never negotiate competing paths.** The assignment builder (1743) selects each head's local best. `mark_path_conflicts` (1704) flags substantial shared skeleton coverage but never tries runner-up combinations. Later duplicate arbitration (2875) likewise does not solve path assignment. A slightly worse local choice could yield a much better compatible solution for the whole component, but the current pipeline cannot make that trade.
3. **Direction is scored too indirectly.** Curvature is an aggregate property of a preselected path, not a cost for entering and leaving a junction. The anchor is the skeleton pixel closest to any head pixel (774), not an estimated neck with an outward direction. The same-color crossing test (`test_tail_assignment.py:220`) explicitly expects ambiguity; it does not establish that geometry cannot resolve that crossing.
4. **Reconstruction can add neighboring structure.** `reconstruct_path_tail_mask` (1402) already uses a centerline, but reduces width estimates to a single dilation radius for the selected path, clipped to the original binary component. It does not enforce local orientation or competing-track ownership. This creates a mechanism for extra branch pixels near a selected path; its actual error rate needs gold evaluation.
5. **Some risk measurements conflate separation with error.** The trim fraction measures how much of the entire shared component was left out. For three similarly sized sperm in one component, a correct individual mask should omit roughly two-thirds of that component. Such omission is not itself evidence of an incomplete sperm. The scoring logic around 2666–2754 combines this with other risks; inspect that behavior against labels before changing it. Likewise, unrelated pixels inside a rectangular crop are not necessarily unrelated pixels inside the selected instance mask.
6. **The isolated fast path can hide upstream mistakes.** The bypass at 3162 uses detected head count and topology/overlap flags. If an additional head or overlap is missed, a biologically shared component can be treated as isolated. This is confirmed control flow, not a measured frequency.

Reusable foundations include SCP color evidence, skeletonization, head-contact and collar checks, width/color utilities, output records, diagnostics, and the existing baseline and synthetic tests. A replacement of the entire pipeline is unnecessary.

**What visual inspection adds**

[Original crops beside current candidate overlays](Algorithmic_Pixel_Mask_For_Tails/outputs/investigation_2026_09_07/overlap_examples.png) show why a single generic overlap rule is insufficient. Prssly KO Exemplary B includes adjacent tails and close head-tail contacts. Prssly KO Exemplary A includes an adjacent faint structure beside a stronger selected track. A Prssly KO Exemplary C candidate marked as a path conflict has a tiny selected head-colored region near an elongated structure, warranting an anchor/head audit as well as path analysis.

The [40x Prssly C overview](Algorithmic_Pixel_Mask_For_Tails/outputs/investigation_2026_09_07/Prssly_C.tif.png) contains a crossing near a loop. The [100x Prssly B overview](Algorithmic_Pixel_Mask_For_Tails/outputs/investigation_2026_09_07/Prssly_KO_Exemplary_B.TIF.png) shows how far tails can run beside one another. These are qualitative inspections, not new human gold labels or claims that every continuation is determinable.

**Recommended algorithm and refinements beyond the existing roadmap**

Keep SCP as the evidence source. Trace several plausible centerlines from each neck, compare compatible combinations for all heads, and reconstruct each chosen mask using local image evidence. The main architecture is already in Phases 2B–2F. The following refinements make it more precise.

| Recommendation | Concrete formulation | Roadmap fit |
| --- | --- | --- |
| Preserve multiple orientations | Retain multiple peaks of an oriented ridge/filter response near crossings, not one angle per pixel. A single structure tensor direction or a single `cos(2θ), sin(2θ)` pair cannot represent both arms of an X. Start with local junction neighborhoods rather than a costly full-frame position-orientation volume. | Refine 2B/2C |
| Treat a junction as alternative branch pairings | Collapse a junction pixel cluster into a neighborhood with incoming/outgoing branch ports. Fit tangents and short curve segments outside its contaminated center; score continuation across the neighborhood. At a four-port crossing, retain the three complete pairings plus applicable partial/termination options. | 2C/2D |
| Put direction into search | Search states include the incoming segment/direction, so a turn changes the cost while searching. Preserve multiple routes to the same endpoint and physically supported self-crossings. Bound repeated edges and route length to prevent unlimited loops; do not ban every repeated spatial junction. | 2C/2D |
| Solve all heads together | Select one hypothesis per head including an explicit null option. Enforce ordinary-segment ownership, endpoint compatibility, and allowed crossing sharing. For arbitrary candidate conflicts, use a set-packing integer optimization or bounded branch-and-bound; ordinary bipartite head-endpoint matching alone is insufficient. | 2E |
| Represent extended overlap regions | Treat a resolved parallel pair as two ridges. When image evidence supports a merge-and-separate corridor, retain possible entering-to-exiting identity pairings and conditional shared ownership. A small crossing disk cannot model all long overlaps. If the image resolves only one ridge, retain uncertainty about internal separation. | Proposed refinement to 2C/2E/2F; requires explicit scope when implemented |
| Measure uncertainty for each disputed continuation | Compare the best complete solution with a constrained alternative that changes that specific branch pairing/head path. A component-wide runner-up may differ only on another sperm. Distinguish a search limit from visual ambiguity, and mark only affected identities/segments uncertain where possible. | Refine 2E/2G |
| Reconstruct with local width | Estimate radius on uncontaminated neighboring portions and interpolate through junctions. Use orientation and competing-track compatibility to reject branch spill. Distance-transform width at a crossing is the width of the union, not necessarily either tail. | Refine 2F |

Color should be supporting evidence. Compare matched windows before and after a junction and reduce the influence of its central mixed pixels. Missing color contrast between two tails is not itself proof of identity ambiguity. Conversely, a clear geometric preference is evidence under a continuity prior, not guaranteed biological truth.

Smoothness must remain a soft preference: this is a morphology project, and strong preferences for straight tails, typical length, or typical head shape could erase the abnormalities being studied. Estimate tangents over several local-width scales; preserve image-supported bends and loops. Use nearby tail exits together with head geometry to propose neck anchors; a head ellipse alone does not determine the attachment.

A simple global objective is to maximize supported path quality while penalizing unsupported gaps, incompatible branch reuse, and implausible stopping. Include null/partial alternatives so the solver can reject an erroneous head or incomplete tail. Do not force coverage of every stained pixel: debris, headless fragments, and unresolvable structures exist. Record decomposed costs and solver status; an optimum of the chosen objective is not proof of a correct biological assignment.

**Relevant primary research**

There are established non-neural precedents for these components. Their transfer to Ward sperm remains a proposal requiring evaluation.

| Source | Supported contribution | Limit for this project |
| --- | --- | --- |
| [Steger, 1998, An Unbiased Detector of Curvilinear Structures](https://www.howardzzh.com/research/papers/vision/1998.PAMI.Steger.UnbiasedDetector.pdf) | Subpixel line localization and width estimation from an explicit line model | Extracts line evidence; does not determine sperm identity |
| [Bekkers et al., 2014, Multi-Orientation Analysis for Retinal Vessel Tracking](https://arxiv.org/abs/1212.3530) | Position-orientation analysis for crossings, parallel vessels, varying widths, and curved vessels | Vessel assumptions differ from head-anchored sperm tracks |
| [Chen, Mirebeau and Cohen, 2015, Curvature Penalized Minimal Paths](https://www.bmva-archive.org.uk/bmvc/2015/papers/paper086/index.html) | Orientation-lifted curvature-aware paths with an optimization guarantee for the specified path energy | Solves an endpoint-conditioned path, not all competing sperm identities |
| [Xu et al., 2015, SOAX](https://www.nature.com/articles/srep09081) | Open active contours extract filament centerlines and junctions; applicable to 2D as well as 3D | Needs parameter selection and sufficient filament separation; not a ready-made sperm resolver |
| [Zhang, Nishimura and Kanchanawong, 2017, SIFNE](https://pmc.ncbi.nlm.nih.gov/articles/PMC5231901/) | Separates global network extraction from individual filament identification/assignment | Developed for superresolution microtubule data, not these RGB images |

The most useful immediate literature contribution is multi-orientation evidence. SOAX/SIFNE are comparison candidates or sources of specific ideas, not reasons to replace SCP. No reviewed paper demonstrates the complete proposed pipeline on this Ward dataset.

**How to establish whether it works with minimal training**

The needed initial manual work is evaluation annotation, not neural-network training. Follow the existing order: finish the missing Phase 0 freeze/reproducibility work; verify the annotation tool's remaining interactive acceptance criteria; complete the three-image pilot; annotate and lock all 24 images; build the Phase 1D evaluator; then implement bounded Phase 2 subphases. The representative images above are pilot candidates, not a frozen selection.

During Phase 2, measure three distinct ceilings:

1. Does the evidence/graph contain the annotated route at all?
2. Does that route survive into the retained top-K hypotheses?
3. Does joint selection choose it without damaging other sperm assignments?

If the correct route is absent, changing acceptance thresholds or adding a global solver cannot recover it. If it is present but not selected, investigate transition costs and joint assignment before assuming the image features need learning. Then evaluate mask completeness and extra branches separately from centerline identity and raw crop cleanliness.

Report correct crossing continuation, wrong-tail rate, visible-sperm recall, centerline completeness, duplicates, and the fraction deferred for review. Separate X crossings, parallel contacts, long merged regions, self-loops, partials, and both magnifications. Maintain isolated-sperm regression checks. Reserve independent source images from parameter selection where feasible; multiple cells or synthetic crops from one source do not create independent evaluation images.

Synthetic curves with controlled crossings are useful for testing pairing, loop handling, and optimization without training. Later photometrically plausible synthetic overlaps can diagnose robustness, but their success does not establish real-image accuracy. Keep all synthetic derivatives grouped with their source if used in any learned experiment.

Only after the residual-error audit should a small auxiliary predictor be considered, targeting a demonstrated deficit such as neck localization or tail orientation. The available evidence does not support promising either that such a model will be necessary or that 24 images will suffice for it. If full automation is unnecessary, a later review tool could ask a human to choose between two continuation overlays and recompute the affected component; that would trade a few decisions for review labor, not require extensive training.

Where two identity assignments produce indistinguishable observed image evidence, a single still image cannot uniquely establish ownership. More training can supply a preference but cannot guarantee the missing identity information. Any supported partial track and alternate assignment should remain explicit. True physical contact also means a raw RGB crop cannot always exclude the other sperm; per-instance logical masks and provenance are essential outputs.

**Verification and completion record**

Exact test/smoke commands, run from `Algorithmic_Pixel_Mask_For_Tails/`:

```bash
../.venv/bin/python -m py_compile algorithmic_tail_mask.py test_tail_assignment.py
../.venv/bin/python -m unittest discover -p 'test_*.py'
../.venv/bin/python annotation_gui.py --input-root Raw_Ward_Data --annotation-root /tmp/scp_overlap_audit_2026_09_07/annotation_check --scp-output-root outputs/updated_outputs_path_v2 --check-only
../.venv/bin/python algorithmic_tail_mask.py --output-root /tmp/scp_overlap_audit_2026_09_07/scp_smoke --limit 2
```

- Compilation passed; all 33 existing tests passed (21 SCP and 12 annotation tests). No tests were added or changed for this investigation.
- Annotation check found all 24 images readable, loadable-or-blank annotation records for all 24, and 24 overlay aids. This is not a claim that 24 annotations exist or that every interactive GUI criterion was tested.
- SCP smoke completed on the first two 100x images: 8 candidates, 1 accepted, 1 rejected, 6 ambiguous. A Python CSV comparison found zero differences in common non-path summary fields against the same images in the saved path-v2 run. This is not bitwise output equivalence or full 24-image reproducibility.
- Inspection used `git status --short`, `git log -5 --oneline`, file discovery, targeted source reads, and Python JSON/CSV/Pillow inventory. `rg` was unavailable, so discovery used `find`, `grep`, and Python. TIFF metadata inspection confirmed one frame per usable image.
- Added report: `TAIL_OVERLAP_INVESTIGATION_2026-09-07.md`. Existing tracked files changed: none. No new dependency or algorithm implementation.
- Derived investigation artifacts: `Algorithmic_Pixel_Mask_For_Tails/outputs/investigation_2026_09_07/verification.json`, `overlap_examples.png`, `Prssly_C.tif.png`, and `Prssly_KO_Exemplary_B.TIF.png`. Smoke outputs and initial inspection images are under `/tmp/scp_overlap_audit_2026_09_07/`. Raw inputs, reviewed CSVs, annotations, and canonical output directories were preserved.
- Deviations: no roadmap phase was implemented. Extended overlap-region handling and per-decision confidence are proposed refinements, not silent changes to the authoritative plan.
- Investigation exit criteria: repository/artifact/code audit, primary-source research, representative visual inspection, existing tests, smoke verification, and recommendations completed. Roadmap phase exit criteria are not certified by this task. Real crossing accuracy, fully interactive annotation readiness, and generalization beyond Ward remain unresolved.
