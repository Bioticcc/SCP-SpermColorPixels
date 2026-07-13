# Project Context

## Primary Objective

Build a mouse sperm morphology pipeline that can:

- detect individual full sperm in microscopy images,
- isolate one crop per full sperm,
- preserve head, neck, midpiece, and full tail in each crop,
- avoid duplicate, debris, partial-sperm, and merged multi-sperm crops,
- support later anatomy segmentation and morphology classification.

The current bottleneck is reliable full-sperm cropping from raw images that may
contain many nearby, touching, crossing, or overlapping sperm.

## Current Direction: SCP First

The primary method is now the **Sperm Color Pixels (SCP)** pipeline in:

```text
Algorithmic_Pixel_Mask_For_Tails/
```

SCP is a deterministic pixel/color-mask pipeline for the high-contrast Ward RGB
images. It uses brown/olive sperm pixels against a lighter background to build
masks, detect heads and tails, connect tails to heads, split some shared-tail
components, generate highlights, and produce reviewed crop candidates.

This is the main crop-generation path because current repository evidence shows
it is the strongest available method for full mouse sperm crops, especially for
isolated and moderately separated sperm.

The human-trained YOLO models, Ben Skinner NMA software, and limited manual
labels are supporting resources, not replacements for SCP at this stage.

## Current SCP Behavior

The current SCP pipeline:

1. Loads Ward RGB TIFF images.
2. Builds high-recall sperm-color masks from HSV, local darkness, and cleanup
   operations.
3. Detects compact dark head candidates.
4. Detects line-like or sparse tail components after subtracting head pixels.
5. Skeletonizes tail components and records endpoints, branchpoints, width
   spikes, and overlap-suspicion regions.
6. Associates tail components with touching or nearby heads.
7. Splits some multi-head shared-tail components with skeleton ownership and
   compact width-spike barriers.
8. Converts each tail/head assignment into a crop candidate.
9. Scores candidates and marks them as accepted, rejected, or ambiguous.
10. Saves accepted crops as the final high-confidence crop set, while keeping
    rejected and ambiguous candidates for review.

The candidate scoring layer is deliberately conservative. It does not use YOLO
or manual labels yet. It scores existing SCP candidates using deterministic
features such as assigned tail pixels, skeleton length, head reuse, overlap
flags, branchpoints/endpoints, border contact, and foreign head/tail pixels
inside the crop box.

## Current Output Layout

SCP outputs are organized under:

```text
Algorithmic_Pixel_Mask_For_Tails/outputs/
```

Current folders:

- `legacy_outputs/`: older SCP runs preserved for comparison.
- `updated_outputs/`: latest SCP run with candidate scoring and review buckets.

Important updated output folders:

- `updated_outputs/highlighted_sperm_crops/`: accepted highlighted crops.
- `updated_outputs/sperm_crops/`: accepted original-image crops and masks.
- `updated_outputs/candidate_review/rejected/`: rejected fragments or duplicates.
- `updated_outputs/candidate_review/ambiguous/`: unresolved overlaps or crowded crops.
- `updated_outputs/json/`: per-image metadata with crop candidate status fields.
- `updated_outputs/summary.csv` and `updated_outputs/summary.json`: run totals.

Latest SCP full-run totals:

- 24 real Ward images processed.
- 213 crop candidates.
- 97 accepted crops.
- 35 rejected candidates.
- 81 ambiguous candidates.
- 0 accepted reused-head duplicates in the verification run.

## Supporting Methods

### YOLOv8 Models

The YOLOv8 segmentation model is trained on human sperm pseudo-masks, not full
mouse Ward images. Current review outputs show limited transfer, especially on
40x Ward images. Use YOLO as a possible supporting signal for rough regions,
candidate comparison, or later experiments, not as the primary cropper.

### Ben Skinner NMA

NMA is primarily a nucleus/head morphology tool. It can detect sperm nuclei or
head-like regions, but it does not solve full-tail cropping. Its realistic role
is head localization or head-candidate validation, if later testing shows it
improves SCP.

### Manual Labels

Manual labels include full sperm, debris, and partial sperm examples. They are
valuable for evaluation and future rejection/model experiments, but the dataset
is too small for large-scale training to be the main strategy right now.

## Practical Next Steps

1. Review `updated_outputs/` to compare accepted, rejected, and ambiguous SCP
   candidates.
2. Choose a new test set or stress test from existing Ward images and review
   materials.
3. Measure accepted-crop quality with simple review labels:
   - good full sperm,
   - duplicate,
   - merged multi-sperm crop,
   - partial sperm,
   - debris/false positive,
   - missed visible sperm.
4. Tune the SCP scoring thresholds and ambiguity rules based on review results.
5. Only after the SCP baseline is measurable, test YOLO or NMA as supporting
   signals.

## Repository Notes

- The root `.git/` is incomplete; ordinary `git status` currently fails.
- `Training_Data/`, `datasets/`, `runs/`, model checkpoints, and generated SCP
  outputs are large local artifacts and should not be pushed directly to normal
  GitHub history.
- The current implementation uses standard library, `cv2`, `numpy`, and `PIL`.
  Avoid new dependencies unless they clearly improve SCP reliability.
