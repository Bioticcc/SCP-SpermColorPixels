"""Static R4a route-comparison pages; no assignment or image processing."""
from __future__ import annotations
import html, json, os
from pathlib import Path

PALETTE=("#00b9e8","#ee5090","#75cf36","#ad79ff","#edb628","#ee6a25")
def _e(v): return html.escape(str(v),quote=True)
def _doc(title,body,script=""):
 return f'<!doctype html><meta charset="utf-8"><title>{_e(title)}</title><style>body{{font:15px system-ui;margin:22px}}svg{{max-width:100%;border:1px solid #ccc}}button{{margin:3px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:5px}}.note{{padding:10px;background:#fff4d8}}</style><body>{body}<script>{script}</script>'
def _route(item, color, cls):
 m=item.get('metadata',{}); pts=m.get('points_xy_ordered') or m.get('pixels_xy_unordered',[]); roi=m.get('roi_xyxy') or [0,0]
 if not pts:return ''
 if m.get('kind')=='baseline_control': return ''.join(f'<circle class="{cls}" cx="{float(x)+roi[0]}" cy="{float(y)+roi[1]}" r="1.4" fill="{color}" style="display:none"/>' for x,y in pts)
 return f'<polyline class="{cls}" points="{" ".join(f"{float(x)+roi[0]},{float(y)+roi[1]}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="2" style="display:none"/>'
def write_image_viewer(directory:Path, original_r1_directory:Path, payload:dict)->dict:
 directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); w,h=int(payload['width']),int(payload['height']); hyps=payload.get('hypotheses',[]); by={x['id']:x for x in hyps}
 if len(by)!=len(hyps):raise ValueError('duplicate hypothesis IDs')
 frozen=payload.get('frozen_r2',{}); oldby={x['id']:x for x in frozen.get('hypotheses',hyps)}
 for sel,mapping in ((payload.get('assignment',{}).get('selected_by_head',{}),by),(frozen.get('assignment',{}).get('selected_by_head',{}),oldby)):
  if any(x not in mapping or str(mapping[x].get('head_id'))!=str(head) for head,x in sel.items()):raise ValueError('selected ID/head absent from hypotheses')
 source=Path(original_r1_directory)/'original.jpg'
 if not source.is_file():raise ValueError('missing original.jpg')
 from PIL import Image
 with Image.open(source) as source_image:
  pw,ph=source_image.size
  valid_preview=max(w,h)>1400 and max(pw,ph)==1400 and pw<=w and ph<=h and abs(pw*h-ph*w)<=max(w,h)
  if source_image.size != (w,h) and not valid_preview:raise ValueError('source dimensions differ')
 for item in hyps+list(oldby.values()):
  m=item.get('metadata',{}); pts=m.get('points_xy_ordered') or m.get('pixels_xy_unordered',[]); roi=m.get('roi_xyxy') or [0,0,w,h]
  if not isinstance(roi,list) or len(roi)!=4 or any(type(x) is not int for x in roi) or not(0<=roi[0]<roi[2]<=w and 0<=roi[1]<roi[3]<=h):raise ValueError('invalid route ROI')
  for x,y in pts:
   if not (isinstance(x,(int,float)) and isinstance(y,(int,float)) and 0<=float(x)+roi[0]<w and 0<=float(y)+roi[1]<h):raise ValueError('invalid route coordinate')
 heads=sorted({str(x.get('head_id')) for x in hyps}); colors={x:PALETTE[i%len(PALETTE)] for i,x in enumerate(heads)}
 def layers(selected,cls,mapping=by): return ''.join(_route(mapping[i],colors[str(mapping[i]['head_id'])],cls) for i in selected.values())
 old=frozen.get('assignment',{}).get('selected_by_head',{}); new=payload.get('assignment',{}).get('selected_by_head',{})
 added=[x for x in hyps if ':r4_attachment:' in x['id']]
 buttons=''.join(f'<button data-route="{_e(x["id"])}" onclick="one(this.dataset.route)">{_e(x["id"])}</button>' for x in added)
 allroutes=''.join(_route(x,colors[str(x['head_id'])],f"route r-{x['id']}") for x in added)
 rel=os.path.relpath(original_r1_directory,directory).replace(os.sep,'/')
 selected_rows=[by[i] for i in new.values() if by[i].get('termination') in ('null','partial')]
 runner=payload.get('assignment',{}).get('runner_up')
 if runner and (set(runner['selected_by_head'])!=set(new) or any(i not in by or str(by[i]['head_id'])!=str(head) for head,i in runner['selected_by_head'].items())):raise ValueError('Incomplete or invalid R4a runner-up')
 runner_button='<button onclick="show(\'runner\')">Runner-up complete assignment</button>' if runner else ''
 body=f'<p><a href="../../index.html">All images</a> · <a href="assignment.json">R4a record</a></p><h1>{_e(payload["relative_image"])}</h1><p class="note">R4a experimental optional head-attachment continuations. Frozen R2 and R4a selections are shown separately; no accuracy claim is made. The resized preview covers the full source image; routes use original-image coordinates.</p><button onclick="show(\'old\')">Frozen R2 selected</button><button onclick="show(\'new\')">R4a selected</button>{runner_button}<button onclick="clearAll()">Original</button><span id="label">Original</span>{buttons}<svg viewBox="0 0 {w} {h}"><image href="{_e(rel)}/original.jpg" width="{w}" height="{h}"/>{layers(old,"old",oldby)}{layers(new,"new")}{allroutes}{layers(runner.get("selected_by_head",{}),"runner") if runner else ""}</svg><h2>Selected null / partial choices</h2><pre>{_e(json.dumps(selected_rows,indent=2))}</pre><h2>Extension diagnostics</h2><pre>{_e(json.dumps(payload.get("extension_diagnostics",[]),indent=2))}</pre>'
 script="function clearAll(){document.querySelectorAll('.old,.new,.runner,.route').forEach(x=>x.style.display='none');document.getElementById('label').textContent='Original'}function show(c){clearAll();document.querySelectorAll('.'+c).forEach(x=>x.style.display='');document.getElementById('label').textContent=c==='old'?'Frozen R2 selected':c==='new'?'R4a selected':'Runner-up complete assignment'}function one(id){clearAll();document.querySelectorAll('.r-'+CSS.escape(id)).forEach(x=>x.style.display='');document.getElementById('label').textContent='Attachment alternative '+id}"
 (directory/'index.html').write_text(_doc(payload['relative_image'],body,script))
 return {'relative_image':payload['relative_image'],'page':f'images/{directory.name}/index.html','added_routes':len(added),'null_choices':sum(by[i].get('termination')=='null' for i in new.values())}
def write_index(root:Path,summaries:list,fixture_report:dict,demo_manifest:dict)->None:
 fixed={x.get('relative_image'):x.get('reason','') for x in demo_manifest.get('images',[])}
 rows=''.join(f'<tr><td><a href="{_e(x["page"])}">{_e(x["relative_image"])}</a></td><td>{_e(fixed.get(x["relative_image"],"Full inventory"))}</td><td>{x["added_routes"]}</td></tr>' for x in sorted(summaries,key=lambda x:(x['relative_image'] not in fixed,x['relative_image'])))
 fixtures=''.join(f'<li><a href="{_e(x.get("comparison_file",""))}">{_e(x.get("name","fixture"))}</a></li>' for x in fixture_report.get('cases',[]))
 Path(root,'index.html').write_text(_doc('R4a attachment continuations',f'<h1>R4a experimental continuations</h1><p class="note">Optional supported attachment routes; no accuracy claim.</p><table>{rows}</table><ul>{fixtures}</ul>'))
