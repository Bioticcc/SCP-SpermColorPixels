"""R3 reconstruction benchmark conditioned on exact frozen R2 routes."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw
from experiments.r1_benchmark import _rendered_graph
from experiments.r3_adapter import _crossings
from mask_reconstruction import CrossingRegion, Track, reconstruct_tracks
from tests.fixtures import generate_case
from tests.fixtures.r2_competing import generate_competing_case

_DEFAULTS={"fixture_seed":0,"min_radius_px":.5,"crossing_exclusion_px":2,"tie_epsilon":1e-9}
_WIDTHS=(1,3,5,7)
_PALETTE=((0,150,230),(225,70,120),(60,170,60),(160,80,220))
_INVARIANTS={"supported_centerline_missing_pixels":0,"ordinary_duplicate_pixels":0,
 "isolated_width_tail_truth_recall_min":.95,"unequal_width_tail_truth_recall_min":.90,"extra_branch_leak_pixels_max":0,
 "fixed_r2_determinate_tail_truth_recall_min":.90,"fixed_r2_determinate_extra_vs_truth_pixels_max":0,
 "fixed_r2_determinate_missing_selection_count_max":0}

def _metrics(pred,truth):
 inter=int((pred&truth).sum()); union=int((pred|truth).sum())
 return {"tail_truth_recall":inter/int(truth.sum()) if truth.any() else 0.,"iou":inter/union if union else 1.,
  "missing_truth_pixels":int((truth&~pred).sum()),"extra_vs_truth_pixels":int((pred&~truth).sum()),
  "predicted_pixels":int(pred.sum()),"truth_pixels":int(truth.sum())}

def _draw(shape,paths,width):
 mask=np.zeros(shape,np.uint8)
 for points in paths: cv2.polylines(mask,[np.asarray(points,np.int32)],False,1,width,lineType=cv2.LINE_8)
 return mask.astype(bool)

def _source(evidence):
 image=np.full((*evidence.shape,3),242,np.uint8); image[evidence]=(135,82,40); return image

def _direct_case(name,widths):
 shape=(128,128)
 if name=="isolated_width":
  # A straight control isolates raster width recovery from bend/corner effects.
  path=[[10,64],[118,64]]; evidence=_draw(shape,[path],widths[0])
  return _source(evidence),evidence,[Track("isolated","h_isolated",path)],[],{"isolated":evidence.copy()}
 if name=="unequal_crossing":
  h=[[10,64],[118,64]]; v=[[64,10],[64,118]]
  truth={"horizontal":_draw(shape,[h],widths[0]),"vertical":_draw(shape,[v],widths[1])}; evidence=truth["horizontal"]|truth["vertical"]
  crossing=np.zeros(shape,np.uint8); cv2.circle(crossing,(64,64),max(widths)//2+2,1,-1)
  return _source(evidence),evidence,[Track("horizontal","h_left",h),Track("vertical","h_top",v)],[CrossingRegion("junction",crossing.astype(bool),("horizontal","vertical"))],truth
 if name=="extra_branch":
  path=[[10,64],[118,64]]; truth=_draw(shape,[path],widths[0]); evidence=truth|_draw(shape,[[[64,64],[64,17]]],widths[0])
  return _source(evidence),evidence,[Track("main","h_main",path)],[],{"main":truth}
 path=[[12,64],[44,64],[73,29],[111,64],[73,99],[44,64]] if name=="loop" else [[0,64],[45,64],[116,64]]
 if name not in ("loop","boundary"): raise ValueError(name)
 evidence=_draw(shape,[path],widths[0]); return _source(evidence),evidence,[Track(name,"h_"+name,path)],[],{name:evidence.copy()}

def _route_hash(points):
 return hashlib.sha256(json.dumps(points,separators=(",",":"),ensure_ascii=True).encode("ascii")).hexdigest()

def _fixture(payload,seed):
 base=payload["name"].removesuffix("_rendered")
 return generate_competing_case(seed=seed,width_px=int(base.rsplit("_",1)[1])) if base.startswith("r2_competing_endpoint_width_") else generate_case(base,seed=seed)

def _actual_r2_case(path,seed,directory,root):
 payload=json.loads(path.read_text()); case=_fixture(payload,seed); graph,_,_=_rendered_graph(case)
 hypotheses={x["id"]:x for x in payload["hypotheses"]}; instance_by_head={x["head_id"]:x["id"] for x in case.truth["instances"]}
 tracks=[]; provenance={}; missing=[]
 for head,hid in sorted(payload["joint"]["selected_by_head"].items()):
  hyp=hypotheses[hid]; route=hyp.get("metadata",{}).get("r1_route"); iid=instance_by_head.get(head)
  if iid is None: raise ValueError(f"R2 head {head!r} has no declared instance")
  if not route or not route.get("points_xy"):
   missing.append({"id":iid,"head_id":head,"selected_hypothesis_id":hid,"reason":"fixed_r2_null_or_missing_route"}); continue
  tracks.append(Track(iid,head,route["points_xy"],ordered=True)); provenance[iid]={"selected_hypothesis_id":hid,
   "r1_route_points_sha256":_route_hash(route["points_xy"]),"r1_route_node_ids":route.get("node_ids",[]),"termination":hyp.get("termination")}
 node_ids={track.id:tuple(provenance[track.id]["r1_route_node_ids"]) for track in tracks}
 crossings,crossing_metadata=_crossings(graph.to_dict(),case.tail_mask,tracks,node_ids,directory,root)
 return payload,case,tracks,crossings,crossing_metadata,provenance,missing

def _overlay(base,masks):
 image=base.copy()
 for i,(_,mask) in enumerate(sorted(masks.items())): image[mask]=((image[mask].astype(np.uint16)+np.asarray(_PALETTE[i%len(_PALETTE)]))//2).astype(np.uint8)
 return image

def _panel(path,source,evidence,truth,centerlines,masks,centerline_label):
 blank=np.zeros_like(source); ev=np.dstack([evidence.astype(np.uint8)*255]*3)
 panels=[("Source",source),("Evidence",ev),("Declared truth",_overlay(blank,truth) if truth is not None else blank),(centerline_label,_overlay(blank,centerlines)),("R3 instance masks",_overlay(source,masks))]
 h,w=source.shape[:2]; out=Image.new("RGB",(w*len(panels),h+24),"white"); draw=ImageDraw.Draw(out)
 for i,(label,pixels) in enumerate(panels): draw.text((i*w+3,4),label,fill="black"); out.paste(Image.fromarray(pixels.astype(np.uint8)),(i*w,24))
 out.save(path)

def _evaluate(name,source,evidence,tracks,crossings,crossing_metadata,truth,missing,provenance,settings,directory,mode,width,determinate,identity_options=None):
 result=reconstruct_tracks(evidence,tracks,crossings,{k:settings[k] for k in ("min_radius_px","crossing_exclusion_px","tie_epsilon")}); instances=[]
 for track in tracks:
  mask=result["masks"][track.id]; center=result["centerlines"][track.id]; Image.fromarray(mask.astype(np.uint8)*255).save(directory/f"{track.id}_mask.png")
  item={"id":track.id,"head_id":track.head_id,"supported_centerline_pixels":int(center.sum()),"supported_centerline_missing_pixels":int((center&~mask).sum()),
   "radii":result["radii"][track.id],"radius_points_xy":result["radius_points_xy"][track.id],"provenance":provenance.get(track.id)}
  item.update(_metrics(mask,truth[track.id]) if determinate and truth is not None and track.id in truth else {"tail_truth_recall":None,"iou":None,"missing_truth_pixels":None,"extra_vs_truth_pixels":None,"predicted_pixels":int(mask.sum()),"truth_pixels":None})
  item["whole_component_baseline"]=_metrics(evidence,truth[track.id]) if determinate and truth is not None and track.id in truth else None
  instances.append(item)
 for item in missing:
  pixels=int(truth[item["id"]].sum()) if determinate and truth is not None else None
  instances.append({**item,"supported_centerline_pixels":0,"supported_centerline_missing_pixels":0,"tail_truth_recall":0. if determinate else None,"iou":0. if determinate else None,
   "missing_truth_pixels":pixels,"extra_vs_truth_pixels":0 if determinate else None,"predicted_pixels":0,"truth_pixels":pixels,"radii":[],"radius_points_xy":[],"provenance":None,
   "whole_component_baseline":_metrics(evidence,truth[item["id"]]) if determinate and truth is not None and item["id"] in truth else None})
 _panel(directory/"comparison.png",source,evidence,truth if determinate else None,result["centerlines"],result["masks"],"Fixed R2 centerlines" if mode.startswith("fixed_r2") else "Declared centerlines")
 union=np.logical_or.reduce(list(result["masks"].values())) if result["masks"] else np.zeros_like(evidence); truth_union=np.logical_or.reduce(list(truth.values())) if truth and determinate else None
 record={"name":name,"width_px":width,"mode":mode,"determinate":determinate,"instances":sorted(instances,key=lambda x:x["id"]),"diagnostics":result["diagnostics"],
  "whole_component_reference_pixels":int(evidence.sum()),"new_instance_union_pixels":int(union.sum()),"union_extra_vs_truth_pixels":int((union&~truth_union).sum()) if truth_union is not None else None,
  "union_missing_truth_pixels":int((truth_union&~union).sum()) if truth_union is not None else None,"fixed_r2_missing_selection_count":len(missing),
  "crossing_regions":crossing_metadata,"indeterminate_identity_options":identity_options or []}
 (directory/"metadata.json").write_text(json.dumps(record,indent=2)+"\n"); return record

def run_r3_fixtures(root:Path,config:dict[str,Any])->dict[str,Any]:
 root=Path(root)
 if root.exists(): raise ValueError("R3 fixture output root must not already exist")
 reconstruction=config.get("reconstruction",{})
 if not isinstance(reconstruction,dict): raise ValueError("reconstruction must be a mapping")
 settings={**_DEFAULTS,**{k:v for k,v in reconstruction.items() if k in _DEFAULTS},**{k:v for k,v in config.items() if k in _DEFAULTS}}
 r2=config.get("r2_run")
 if not r2: raise ValueError("r2_run is required for fixed-R2 reconstruction mode")
 rendered=sorted((Path(r2)/"fixtures").glob("*/rendered/assignment.json"))
 if not rendered: raise ValueError("r2_run has no rendered fixture assignment records")
 root.mkdir(parents=True); rows=[]
 for path in rendered:
  payload=json.loads(path.read_text()); name=payload["name"]; directory=root/name; directory.mkdir()
  payload,case,tracks,crossings,crossing_metadata,provenance,missing=_actual_r2_case(path,int(settings["fixture_seed"]),directory,root); determinate=bool(payload.get("expected_behavior",{}).get("determinate",True))
  rows.append(_evaluate(name,case.rgb,case.tail_mask,tracks,crossings,crossing_metadata,case.instance_masks,missing,provenance,settings,directory,"fixed_r2_selected_rendered_assignment",case.truth.get("width_px"),determinate,case.truth.get("identity_solutions",[]) if not determinate else []))
 specs=[("isolated_width",(w,)) for w in _WIDTHS]+[("unequal_crossing",(1,7)),("unequal_crossing",(3,5)),("extra_branch",(5,)),("loop",(5,)),("boundary",(5,))]
 for kind,widths in specs:
  name=f"{kind}_width_{'_'.join(map(str,widths))}"; directory=root/name; directory.mkdir(); source,evidence,tracks,crossings,truth=_direct_case(kind,widths)
  rows.append(_evaluate(name,source,evidence,tracks,crossings,[],truth,[],{},settings,directory,"direct_declared_track_reconstruction",list(widths) if len(widths)>1 else widths[0],True))
 failures=[]
 for row in rows:
  if row["diagnostics"]["ordinary_duplicate_pixels"]: failures.append({"case":row["name"],"invariant":"ordinary_duplicate_pixels"})
  if row["determinate"] and row["fixed_r2_missing_selection_count"]>_INVARIANTS["fixed_r2_determinate_missing_selection_count_max"]:
   failures.append({"case":row["name"],"invariant":"fixed_r2_determinate_missing_selection_count_max","actual":row["fixed_r2_missing_selection_count"]})
  for item in row["instances"]:
   if item["supported_centerline_missing_pixels"]: failures.append({"case":row["name"],"instance":item["id"],"invariant":"supported_centerline_missing_pixels"})
   recall=item["tail_truth_recall"]
   if row["name"].startswith("isolated_width") and recall<_INVARIANTS["isolated_width_tail_truth_recall_min"]: failures.append({"case":row["name"],"instance":item["id"],"invariant":"isolated_width_tail_truth_recall_min","actual":recall})
   if row["name"].startswith("unequal_crossing") and recall<_INVARIANTS["unequal_width_tail_truth_recall_min"]: failures.append({"case":row["name"],"instance":item["id"],"invariant":"unequal_width_tail_truth_recall_min","actual":recall})
   if row["name"].startswith("extra_branch") and item["extra_vs_truth_pixels"]>0: failures.append({"case":row["name"],"instance":item["id"],"invariant":"extra_branch_leak_pixels_max","actual":item["extra_vs_truth_pixels"]})
   if row["mode"]=="fixed_r2_selected_rendered_assignment" and row["determinate"] and recall is not None:
    if recall<_INVARIANTS["fixed_r2_determinate_tail_truth_recall_min"]: failures.append({"case":row["name"],"instance":item["id"],"invariant":"fixed_r2_determinate_tail_truth_recall_min","actual":recall})
    if item["extra_vs_truth_pixels"]>_INVARIANTS["fixed_r2_determinate_extra_vs_truth_pixels_max"]: failures.append({"case":row["name"],"instance":item["id"],"invariant":"fixed_r2_determinate_extra_vs_truth_pixels_max","actual":item["extra_vs_truth_pixels"]})
 all_instances=[x for row in rows for x in row["instances"]]
 report={"schema_version":"r3.fixture-benchmark.v2","config":settings,"r2_run_reference":str(r2),"predeclared_invariants":_INVARIANTS,"cases":rows,
  "acceptance":{"case_count":len(rows),"instance_count":len(all_instances),"null_or_missing_fixed_r2_count":sum(r["fixed_r2_missing_selection_count"] for r in rows),"ordinary_duplicate_pixels":sum(r["diagnostics"]["ordinary_duplicate_pixels"] for r in rows),"failure_count":len(failures),"failures":failures},
  "scope":"Conditional mask reconstruction from exact frozen R2 routes plus declared direct geometry; no path-selection or real-image accuracy claim."}
 (root/"fixture_report.json").write_text(json.dumps(report,indent=2)+"\n"); return report
