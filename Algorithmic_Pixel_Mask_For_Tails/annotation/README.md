# Ward Gold Annotation Tool

This folder contains the Phase 1A manual annotation subsystem for Ward source
images. Canonical annotations are schema-versioned JSON files; previews and
rasters are derived artifacts that can be regenerated.

## Launch

From `Algorithmic_Pixel_Mask_For_Tails/`:

```bash
../.venv/bin/python annotation_gui.py \
  --input-root Raw_Ward_Data \
  --annotation-root annotations/ward_gold_v1 \
  --scp-output-root outputs/updated_outputs_path_v2
```

Check image discovery and annotation load/recovery without opening Tk:

```bash
../.venv/bin/python annotation_gui.py \
  --input-root Raw_Ward_Data \
  --annotation-root /tmp/scp_annotation_smoke \
  --scp-output-root outputs/updated_outputs_path_v2 \
  --check-only
```

Regenerate derived outputs:

```bash
../.venv/bin/python annotation/annotation_export.py \
  --input-root Raw_Ward_Data \
  --annotation-root annotations/ward_gold_v1 \
  --export-root annotations/ward_gold_v1/exports
```

## Canonical Schema

Each image JSON uses schema version `1.0.0` and stores original image-space
coordinates. It records image metadata, source hash, sperm instances, and
crossings. Each sperm instance supports head geometry, neck point, tail
centerline, distal endpoint, completion/boundary status, certainty, crossing
IDs, and notes.

## Shortcuts

- `V`: select/edit
- `H` or hold `Space`: pan
- `N`: new sperm instance
- `E`: head ellipse
- `G`: head polygon
- `K`: neck point
- `T`: tail centerline
- `C`: crossing region
- `Enter`: finish polygon/centerline
- `Esc`: cancel current drawing
- `Delete`: delete selected object
- `Ctrl+Z`: undo
- `Ctrl+Y` or `Ctrl+Shift+Z`: redo
- `Ctrl+S`: save
- `Ctrl+Shift+V`: validate
- `PageUp` / `PageDown`: previous/next image
- `O`: toggle read-only SCP aid overlay
- `I`: show selected instance only
- `F`: fit image
- `1`: zoom to 100%

## Safety

Saves are atomic and keep the latest five backups per image under
`backups/<image_id>/`. If the on-disk annotation changed after it was loaded,
the GUI writes a conflict copy under `conflicts/<image_id>/` instead of
overwriting the newer file.
