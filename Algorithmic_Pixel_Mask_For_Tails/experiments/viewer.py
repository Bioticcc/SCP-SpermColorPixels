"""Static, file://-compatible experiment report with local image toggles."""
from __future__ import annotations

import html
import json
from pathlib import Path

from PIL import Image

import algorithmic_tail_mask as atm


def _thumbnail(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((1000, 750))
        image.save(target)


def write_baseline_viewer(output_root: Path, input_root: Path, reference_root: Path,
                          predictions_root: Path, demo: dict, comparison: dict,
                          fixture_report: dict) -> None:
    assets = output_root / "viewer_assets"
    assets.mkdir()
    cards = []
    for index, row in enumerate(demo["images"]):
        source = input_root / row["relative_image"]
        stem = atm.safe_stem(source, input_root)
        relative_paths = {}
        for label, path in {
            "original": source,
            "saved": reference_root / "head_connected_overlays" / f"{stem}_head_connected_overlay.png",
            "reproduced": predictions_root / "head_connected_overlays" / f"{stem}_head_connected_overlay.png",
        }.items():
            # Use actual metadata: historical naming is not a stable interface.
            if label != "original":
                meta_root = reference_root if label == "saved" else predictions_root
                data = json.loads((meta_root / "json" / f"{stem}.json").read_text())
                recorded = Path(data["outputs"]["head_connected"]["overlay"])
                path = meta_root / "head_connected_overlays" / recorded.name
            target = assets / f"{index}_{label}.png"
            _thumbnail(path, target)
            relative_paths[label] = target.relative_to(output_root).as_posix()
        name = html.escape(source.name)
        attrs = " ".join(f'data-{k}="{html.escape(v, quote=True)}"' for k, v in relative_paths.items())
        cards.append(f'<article><h2>{name}</h2><p>{html.escape(row["reason"])}</p>'
                     f'<img {attrs} src="{relative_paths["original"]}" alt="{name}">'
                     '<div><button data-mode="original">Original</button> '
                     '<button data-mode="saved">Saved path-v2</button> '
                     '<button data-mode="reproduced">Current-code baseline</button></div></article>')
    fixture_cards = []
    for row in fixture_report["cases"]:
        name = html.escape(row["name"])
        fixture_cards.append(f'<article><h2>{name}</h2><img src="fixtures/{name}/comparison.png" alt="Synthetic {name}">'
                             f'<p>{html.escape(row["summary"])}</p>'
                             f'<a href="fixtures/{name}/truth.json">Construction truth</a> · '
                             f'<a href="fixtures/{name}/baseline.json">Baseline measurements</a></article>')
    summary = {"historical_comparison_has_differences": comparison['has_differences'],
               "images_compared": len(comparison['images']), "summary_fields": comparison['summary'],
               "note": "Differences here predate any new overlap resolver. See comparison.json for complete candidate-level evidence."}
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SCP R0 baseline</title>
<style>body{font:16px system-ui;margin:2rem;background:#f5f7fa;color:#172434}main{max-width:1500px;margin:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:1rem}article{background:white;padding:1rem;border-radius:10px}
img{width:100%;height:auto}button{padding:.6rem;margin:.2rem;cursor:pointer}h2{font-size:1.1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:500px){.grid{grid-template-columns:1fr}body{margin:.7rem}}</style><main>
<h1>SCP: frozen baseline and geometry benchmark</h1>
<p>R0 freezes the current algorithm without changing predictions. No new overlap resolver is active here.
The saved historical output may differ from the current-code baseline; differences are reported below.
Real-image examples are qualitative; synthetic measurements describe constructed geometry, not real-image accuracy.</p>
<p><a href="provenance.json">Provenance</a> · <a href="comparison.json">Full 24-image comparison</a> ·
<a href="fixture_report.json">Synthetic baseline report</a> · <a href="input_manifest.json">Input inventory</a></p>
<h2>Fixed real-image panel</h2><div class="grid">''' + "\n".join(cards) + '''</div>
<h2>Constructed cases: image, instance truth, baseline paths</h2><div class="grid">''' + "\n".join(fixture_cards) + '''</div>
<h2>Comparison summary</h2><pre>''' + html.escape(json.dumps(summary, indent=2)) + '''</pre></main>
<script>document.querySelectorAll('button[data-mode]').forEach(b=>b.addEventListener('click',()=>{
const img=b.closest('article').querySelector('img');img.src=img.dataset[b.dataset.mode];}));</script></html>'''
    (output_root / "index.html").write_text(page, encoding="utf-8")
