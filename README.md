# Sperm Morphology Recognition

This repository is currently focused on one practical bottleneck: creating
clean, one-sperm-per-crop outputs from microscopy images that contain many mouse
sperm.

## Current Main Method: Sperm Color Pixels

The active crop-generation method is **Sperm Color Pixels (SCP)**:

```text
Algorithmic_Pixel_Mask_For_Tails/
```

SCP is a deterministic pixel/color-mask pipeline, not a trained model. It uses
the high-contrast brown/olive sperm signal in Ward RGB TIFF images to detect
sperm-colored pixels, identify likely heads and tails, associate tails to heads,
highlight detected structures, and create crop candidates.

The current direction is to strengthen SCP first because it is the strongest
available method in this repository for full mouse sperm crops. YOLO, NMA, and
manual labels are supporting resources, not the primary cropper at this stage.

See:

- `ProjectContext.md` for the project direction and current evidence.
- `AGENT.md` for handoff notes.
- `Algorithmic_Pixel_Mask_For_Tails/README.md` for SCP-specific commands and
  output details.

## What SCP Currently Produces

The pipeline now separates crop candidates into:

- `accepted`: cleaner final full-sperm crop candidates.
- `rejected`: tiny fragments or lower-scoring duplicates.
- `ambiguous`: unresolved overlaps, crossings, or crowded crops that should be
  reviewed rather than treated as clean final crops.

Current verified R0 baseline (September 7, 2026):

- 24 real Ward images processed.
- 209 crop candidates: 19 accepted, 44 rejected, 146 ambiguous.
- Matches the saved July hybrid run's structural predictions.
- Preserves the current algorithm; no new overlap resolver is enabled yet.

The historical `updated_outputs_path_v2` folder contains 22 accepted, 45 rejected,
and 142 ambiguous candidates and predates the current code. The older
`updated_outputs` run contains 213 candidates with 97 accepted.
See [the annotation-optional execution plan](ANNOTATION_OPTIONAL_EXECUTION_PLAN_2026-09-07.md)
and [R0 commands and artifacts](Algorithmic_Pixel_Mask_For_Tails/experiments/README.md).

The experimental R1 route generator is now available separately from the
baseline: all 24 images processed, all 209 baseline candidates retained, and
977 alternative routes recorded. It preserves crossing/loop routes that the
single-predecessor search could discard; local ranking remains incomplete.
See [R1 results and limitations](R1_IMPLEMENTATION_REPORT.md) and the
[local comparison viewer](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r1_routes_2026_09_09_verified/index.html).

R2 adds deterministic joint selection across competing heads. Its full 24-image
experiment removes conflicting branch/endpoint claims, but rendered synthetic
tests expose regressions from unmodeled shared connectors. It remains an
experimental proposal mode. See [R2 results](R2_IMPLEMENTATION_REPORT.md) and
the [reviewed joint-assignment viewer](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r2_joint_2026_09_09_reviewed/index.html).

R3 now reconstructs individual logical masks around those frozen selections.
All 24 images have source-coordinate exports and explicit ownership uncertainty.
Read [R3 results and limitations](R3_IMPLEMENTATION_REPORT.md) and open the
[mask viewer](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r3_masks_2026_09_09_release/index.html).
Upstream incomplete or incorrect routes remain visible.

R4a adds optional continuations through other projected head attachments while
retaining all original routes. The 24-image experiment adds 60 alternatives;
changed assignments remain unvalidated, with more null choices than R2. See
[R4a results](R4A_IMPLEMENTATION_REPORT.md) and the
[attachment comparison viewer](Algorithmic_Pixel_Mask_For_Tails/outputs/experiments/r4a_attachments_2026_09_09_complete/index.html).

Current outputs are organized under:

```text
Algorithmic_Pixel_Mask_For_Tails/outputs/
```

Important folders:

- `legacy_outputs/`: older SCP runs preserved for comparison.
- `updated_outputs/highlighted_sperm_crops/`: accepted highlighted crops.
- `updated_outputs/sperm_crops/`: accepted original-image crops and masks.
- `updated_outputs/candidate_review/rejected/`: rejected candidate review images.
- `updated_outputs/candidate_review/ambiguous/`: ambiguous candidate review images.
- `updated_outputs/json/`: per-image metadata, including scoring and status.
- `updated_outputs/summary.csv` and `updated_outputs/summary.json`: run totals.

## Run SCP

From the project root, rerun the current SCP output set with:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py \
  --output-root outputs/updated_outputs_path_v2 \
  --overwrite
```

Smoke test:

```bash
.venv/bin/python Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py \
  --output-root outputs/updated_outputs_path_v2_smoke \
  --limit 2 \
  --overwrite
```

The script resolves relative `--output-root` paths inside
`Algorithmic_Pixel_Mask_For_Tails/`.

## Setup On A New Machine

Create a Python 3.10+ virtual environment and install the repo dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The annotation and review GUIs use Tkinter. If your Python build does not
include Tkinter, install the OS package for it, such as `python3-tk` on
Debian/Ubuntu systems.

## Supporting Methods

### YOLOv8

The existing YOLOv8 segmentation work is trained on human sperm pseudo-masks.
Current review outputs do not support replacing SCP with YOLO. YOLO may later
serve as a supporting region proposal, ambiguity flag, or comparison signal.

### Ben Skinner NMA

NMA is useful mainly as a sperm nucleus/head detector. It does not capture full
tails well enough to be the main crop-generation path. It may later help
validate SCP head candidates.

### Manual Labels

Manual labels for full sperm, debris, and partial sperm are valuable for
evaluation and future rejection experiments. They are intentionally not the main
training dependency because the available label set is limited.

## Data Location Notes

Ward source images live in:

```text
Algorithmic_Pixel_Mask_For_Tails/Raw_Ward_Data/
```

The older training data, model outputs, and baseline workflows remain separate
under `Training_Data/`, `datasets/`, `runs/`, and `scripts/`.

## Source References

Generating low-priority mouse positives:

- NMA Software Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6497523/#abstract1
- NMA Software GitHub: https://github.com/bmskinner/nma/wiki

High-priority human positives:

- https://bridges.monash.edu/articles/dataset/Clinically_labelled_live_unstained_human_sperm_dataset/25621500

Model choice reference:

- https://www.mdpi.com/1424-8220/25/10/3093
