# R0 baseline and geometry benchmark — implementation record

R0 is complete as a baseline/benchmark work package. The overall R0–R6 goal
remains active; R1 is next. No new overlap resolver, threshold change, neural
training, or annotation dependency was introduced in R0.

The full unchanged SCP run processed 24 Ward images and generated 209 candidates:
19 accepted, 44 rejected, 146 ambiguous. Its predictions match the saved
`ablation_2026_07_03_hybrid` run structurally across all 24 images after the
comparator's documented output-diagnostic exclusions. Candidate identities,
statuses, assignments, and compared numerical values agree. Byte-for-byte image
equivalence is not asserted.

The older canonical `updated_outputs_path_v2` artifacts are from an earlier
implementation. They have 22 accepted, 45 rejected, 142 ambiguous candidates.
The comparison identifies four changed statuses and 17 changed assignments
across nine images; 152 candidates have changed shared numerical fields.
All 24 images and candidate identities remain accounted for. One concrete
historical difference is 89 recorded endpoint paths on Prssly KO Exemplary A,
where current code and saved hybrid record the current 40-path cap. Current
adaptive crop padding also explains expanded boxes in the fresh/hybrid run.
The exact old source snapshot is unavailable, so remaining differences are
reported rather than assigned speculative causes. These changes predate R0.

**Added files and principal decisions**

- `configs/overlap_demo_baseline.json`: every CLI setting pinned explicitly;
  changes in parser/config keys fail instead of silently adopting defaults.
- `experiments/demo_manifest.json`: fixed six-image panel chosen before new
  resolver development, including both magnifications and baseline controls.
- `experiments/provenance.py`, `configuration.py`, `integrity.py`: input/code/
  config hashes, dependency/backend records, safe metadata writes, effective
  arguments, structural comparisons, and final inventory integrity.
- `experiments/run_baseline.py`: invokes the unchanged CLI on all inputs in a
  fresh directory, recording progress and results. Existing roots are refused.
- `experiments/benchmark.py`, `viewer.py`, `refresh_report.py`: constructed-case
  baseline measurements, local HTML toggles, and separately provenanced report
  regeneration from existing predictions without repeating a full image run.
- `tests/fixtures/synthetic_fixtures.py` and package initializers: eight seeded
  graph/rendered fixtures with separate logical truth and image evidence.
- `tests/test_synthetic_fixtures.py` and four `test_experiment_*.py` modules:
  fixture validity/ambiguity, configuration, metrics, provenance, and integrity.
- `experiments/README.md` and `experiments/baseline_manifest.json`: commands,
  limitations, and the frozen run's reference metadata.

Existing files changed for R0: root `README.md` and the SCP `README.md`.
The previously created investigation/plan files and roadmap precedence edit
remain present. `algorithmic_tail_mask.py` was not modified. No dependency was
added; raw TIFFs, human annotation JSON, reviewed CSVs, and canonical outputs
were preserved.

**Verification commands and results**

From the SCP directory:

```bash
../.venv/bin/python -m unittest discover -p 'test_*.py'
../.venv/bin/python -m py_compile algorithmic_tail_mask.py experiments/*.py tests/fixtures/*.py
```

57 tests pass; compilation passes. The initial existing-only suite passed 33
tests before implementation. Added tests cover graph edge/path/head consistency,
deterministic fixture regeneration, real rendering of alternative identities,
junction revisits without repeated edges, spatial centerline metrics, safe
configuration, missing/changed/reordered candidates, numeric-to-null changes,
schema additions versus regressions, atomic metadata protection, and changed
input/config/code detection.

From the project root:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/run_baseline.py --output-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r0_baseline_2026_09_07
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/experiments/refresh_report.py --run-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r0_baseline_2026_09_07 --report-root Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r0_baseline_2026_09_07/review
git diff --check
```

The full baseline/fixture run completed in about 467 seconds. Its wrapper
returned exit 2, correctly identifying differences against the older canonical
reference; the pipeline subprocess succeeded. Input, source, and config
integrity checks passed. The reporting refresh succeeded with independent
reporting provenance and final integrity checks, preserving full-image
predictions. A separate `compare_runs(saved_hybrid, fresh_predictions)` check
returned no prediction differences, no schema regressions, and no additive
schema differences across 24 images. Its full result is saved below.

An additional fixture-only smoke run completed under
`/tmp/scp_r0_fixture_smoke_jqxgoej2`. The synthetic loop visibly loses its loop
under the existing shortest-path selection. The isolated curved case's selected
endpoint is `(101,34)` versus the constructed `(108,27)`, recording a distal
completeness issue for later investigation. The same-color X retains its
straight paths, illustrating that correct geometry and current ambiguity flags
are separate questions. Partial T-contact tracks are reported separately from
full-track success counts. Indeterminate identities receive no accuracy score
and both logical options are shown.

The final HTML was parsed and its 46 distinct local links/assets verified;
18 toggle controls and 14 image panels were found. Generated real-image and
synthetic panels were visually inspected. Headless Firefox screenshot launch
failed in this environment with an X/graphics initialization error (exit 11),
so browser-executed interaction is not certified. The report uses simple local
image switching, with no server, fetch requests, or external resources.

**Artifacts**

All principal artifacts are under
`Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r0_baseline_2026_09_07/`:

- `predictions/`: frozen current-code full-image outputs.
- `provenance.json`, `input_manifest.json`, config/demo snapshots, and
  `integrity.json`: original prediction-run evidence.
- `comparison.json`: original historical comparison, retained for audit.
- `review/index.html`: current reviewed demonstration.
- `review/comparison.json`: historical comparison with the corrected schema
  distinction; `review/comparison_current_hybrid.json`: independent matching run.
- `review/fixtures/`, `review/fixture_report.json`: eight constructed cases and
  measured baseline failures.
- `review/provenance.json`, `review/integrity.json`: separate reporting integrity.
- `review/panel_contact_sheet.png`: inspected six-image overview.

**Exit audit and deviations**

Baseline provenance, old-test preservation, deterministic fixtures, recorded
baseline failure examples, fixed six-image demonstration, and full 24-image
inventory/smoke are satisfied. The known historical mismatch is explained and
the working-code baseline is independently corroborated by the saved hybrid
run. The numerical exit criteria refer to constructed fixtures and structural
reproduction, not real-image accuracy.

Deviations: baseline provenance is implemented by an external wrapper rather
than hooks in the main script; fixture tests use the discoverable `tests/`
package; reporter corrections were regenerated with separate provenance rather
than rerunning unchanged full-image predictions. Browser interaction remains
unverified because the local headless browser failed. None of these changes
requires additional annotations or begins a later algorithm package.

The next package is R1: direction-aware junction representation and alternative
head-to-tail routes, using this frozen current-code baseline and the constructed
loop/curvature failures. Keep the full R0–R6 user goal active.
