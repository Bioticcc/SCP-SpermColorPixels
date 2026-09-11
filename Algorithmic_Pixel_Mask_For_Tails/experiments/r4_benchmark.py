"""R4a benchmark for optional routes through incidental foreign head anchors."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw
from experiments.benchmark import centerline_measurements
from experiments.r1_benchmark import _rendered_graph
from experiments.r2_assignment import assign_pool
from experiments.r2_candidates import build_candidate_pool
from experiments.r4_adapter import resolve_image
from graph_construction import from_explicit_graph
from path_hypotheses import search_hypotheses

_DEFAULTS={"additional_k":5,"search_max_expansions":20000,"solver_max_expansions":100000,
 "null_cost":1.5,"partial_cost":.85,"coverage_weight":1.0,"fixture_seed":0}
_WIDTHS=(1,3,5); _TRANSFORMS=("identity","horizontal_flip","vertical_flip","rotate_180","padded_translation")
_COLORS={"h1":(0,160,230),"h2":(225,70,120)}

def _geometry():
 return {"nodes":[{"id":"h1","xy":[20,64],"kind":"head_anchor"},{"id":"b","xy":[64,64],"kind":"junction"},
  {"id":"right","xy":[116,64],"kind":"endpoint"},{"id":"down","xy":[64,116],"kind":"endpoint"}],
  "edges":[{"id":"left_b","start":"h1","end":"b","points":[[20,64],[64,64]]},
   {"id":"b_right","start":"b","end":"right","points":[[64,64],[116,64]]},
   {"id":"b_down","start":"b","end":"down","points":[[64,64],[64,116]]}],
  "anchors":{"h1":"h1","h2":"b"}}

def _transform(name,shape=(128,128)):
 h,w=shape
 if name=="identity": return lambda p:p,shape
 if name=="horizontal_flip": return lambda p:[w-1-p[0],p[1]],shape
 if name=="vertical_flip": return lambda p:[p[0],h-1-p[1]],shape
 if name=="rotate_180": return lambda p:[w-1-p[0],h-1-p[1]],shape
 if name=="padded_translation": return lambda p:[p[0]+11,p[1]+7],(h+14,w+22)
 raise ValueError(name)

def _case(width,transform_name):
 transform,shape=_transform(transform_name); graph=_geometry()
 for node in graph["nodes"]: node["xy"]=transform(node["xy"])
 for edge in graph["edges"]: edge["points"]=[transform(p) for p in edge["points"]]
 heads=[{"id":"h1","center_xy":transform([14,64]),"radius_px":7},{"id":"h2","center_xy":transform([64,57]),"radius_px":7}]
 tail=np.zeros(shape,np.uint8)
 for edge in graph["edges"]: cv2.polylines(tail,[np.asarray(edge["points"],np.int32)],False,1,width,lineType=cv2.LINE_8)
 head=np.zeros(shape,np.uint8)
 for item in heads: cv2.circle(head,tuple(item["center_xy"]),item["radius_px"],1,-1,lineType=cv2.LINE_8)
 rgb=np.full((*shape,3),242,np.uint8); rgb[tail.astype(bool)]=(135,82,40); rgb[head.astype(bool)]=(92,60,32)
 edges={x["id"]:x for x in graph["edges"]}
 paths=[{"instance_id":"s1","node_ids":["h1","b","right"],"edge_ids":["left_b","b_right"]},
        {"instance_id":"s2","node_ids":["h2","down"],"edge_ids":["b_down"]}]
 instances=[]
 for item in paths:
  points=[]
  for eid in item["edge_ids"]: points.extend(edges[eid]["points"] if not points else edges[eid]["points"][1:])
  instances.append({"id":item["instance_id"],"head_id":item["node_ids"][0],"centerline_xy":points})
 truth={"heads":heads,"instances":instances,"true_edge_paths":paths,"graph":graph,
        "expected_behavior":{"determinate":True,"outcome":"foreign_head_anchor_passthrough","expected_instance_count":2}}
 return SimpleNamespace(rgb=rgb,tail_mask=tail.astype(bool),head_mask=head.astype(bool),truth=truth)

def _searches(graph,settings):
 return {head:search_hypotheses(graph,head,k=5,max_expansions=settings["search_max_expansions"]) for head in sorted(graph.anchors)}

def _record(case,graph,searches,name):
 return {"relative_image":name,"width":case.tail_mask.shape[1],"height":case.tail_mask.shape[0],"baseline_candidates":0,
  "components":[{"tail_id":"1","roi_xyxy":[0,0,case.tail_mask.shape[1],case.tail_mask.shape[0]],"graph":graph.to_dict(),"searches":searches,"baseline":[]}]}

def _frozen(record,settings):
 pool=build_candidate_pool(record,settings); assignment=assign_pool(pool["hypotheses"],settings["solver_max_expansions"])
 return {"relative_image":record["relative_image"],"hypotheses":[asdict(x) for x in pool["hypotheses"]],"assignment":assignment}

def _raster(points,shape):
 out=np.zeros(shape,np.uint8)
 if len(points)>1: cv2.polylines(out,[np.asarray(points,np.int32)],False,1,1,lineType=cv2.LINE_8)
 return out

def _metrics(case,selected,hypotheses):
 by_id={x["id"]:x for x in hypotheses}; rows=[]
 for truth in case.truth["instances"]:
  hid=selected.get(truth["head_id"]); item=by_id.get(hid); route=(item or {}).get("metadata",{}).get("r1_route",{})
  points=route.get("points_xy",[]); metric=centerline_measurements(_raster(points,case.tail_mask.shape),truth["centerline_xy"])
  rows.append({"instance_id":truth["id"],"head_id":truth["head_id"],"selected_hypothesis_id":hid,
   "selected_route_available":bool(points),"selected_null_or_missing":not bool(points),"centerline":metric})
 return rows

def _oracle(case,hypotheses):
 rows=[]
 for truth in case.truth["instances"]:
  candidates=[x for x in hypotheses if x["head_id"]==truth["head_id"] and x.get("metadata",{}).get("r1_route",{}).get("points_xy")]
  scored=[(centerline_measurements(_raster(x["metadata"]["r1_route"]["points_xy"],case.tail_mask.shape),truth["centerline_xy"]),x) for x in candidates]
  metric,item=max(scored,key=lambda pair:pair[0]["f1"]) if scored else ({"precision":0.,"recall":0.,"f1":0.},None)
  rows.append({"instance_id":truth["id"],"head_id":truth["head_id"],"hypothesis_id":item["id"] if item else None,"centerline":metric})
 return rows

def _draw(base,selected,hypotheses):
 image=base.copy(); by_id={x["id"]:x for x in hypotheses}
 for head,hid in sorted(selected.items()):
  points=by_id.get(hid,{}).get("metadata",{}).get("r1_route",{}).get("points_xy",[])
  if points: cv2.polylines(image,[np.asarray(points,np.int32)],False,_COLORS[head],2,lineType=cv2.LINE_8)
 return image

def _truth_image(case):
 image=case.rgb.copy()
 for row in case.truth["instances"]: cv2.polylines(image,[np.asarray(row["centerline_xy"],np.int32)],False,_COLORS[row["head_id"]],2,lineType=cv2.LINE_8)
 return image

def _panel(path,case,old,new,old_hypotheses,new_hypotheses):
 panels=[("Original",case.rgb),("Old R2 selected",_draw(case.rgb,old,old_hypotheses)),
         ("R4a selected",_draw(case.rgb,new,new_hypotheses)),("Declared truth",_truth_image(case))]
 h,w=case.rgb.shape[:2]; out=Image.new("RGB",(w*4,h+24),"white"); draw=ImageDraw.Draw(out)
 for i,(label,pixels) in enumerate(panels): draw.text((i*w+3,4),label,fill="black"); out.paste(Image.fromarray(pixels),(i*w,24))
 out.save(path)

def _one(case,name,mode,directory,settings):
 if mode=="explicit": graph=from_explicit_graph(case.truth["graph"],shape=case.tail_mask.shape,anchors=case.truth["graph"]["anchors"])
 else: graph,_,_=_rendered_graph(case)
 searches=_searches(graph,settings); record=_record(case,graph,searches,name); frozen=_frozen(record,settings)
 extended,payload=resolve_image(record,frozen,settings); old=frozen["assignment"]["selected_by_head"]; new=payload["assignment"]["selected_by_head"]
 old_routes={head:{x["id"]:x for x in search["hypotheses"]} for head,search in searches.items()}
 new_searches=extended["components"][0]["searches"]
 retained=all(all({x["id"]:x for x in new_searches[head]["hypotheses"]}.get(route_id)==route for route_id,route in routes.items()) for head,routes in old_routes.items())
 new_hypotheses=payload["hypotheses"]; old_metrics=_metrics(case,old,frozen["hypotheses"]); new_metrics=_metrics(case,new,new_hypotheses); oracle=_oracle(case,new_hypotheses)
 h1_full_available=any(x["head_id"]=="h1" and x.get("metadata",{}).get("r1_route",{}).get("termination")=="endpoint" and
  x.get("metadata",{}).get("r1_route",{}).get("metadata",{}).get("passed_foreign_head_ids")==["h2"] and x["metadata"]["r1_route"].get("node_ids",[])[-1]!=graph.anchors["h2"] for x in new_hypotheses)
 _panel(directory/"comparison.png",case,old,new,frozen["hypotheses"],new_hypotheses)
 record_out={"name":name,"mode":mode,"comparison_file":f"fixtures/{name}/comparison.png","width_px":int(name.split("width_")[-1].split("_")[0]) if "width_" in name else None,
  "baseline_routes_retained_exactly":retained,"h1_full_passthrough_available":h1_full_available,"old_selected_metrics":old_metrics,"new_selected_metrics":new_metrics,
  "expanded_pool_oracle_metrics":oracle,"old_selected_by_head":old,"new_selected_by_head":new,"changed_head_ids":payload["comparison"]["changed_head_ids"],
  "extension_diagnostics":payload["extension_diagnostics"],"old_hypotheses":frozen["hypotheses"],"new_hypotheses":new_hypotheses,
  "truth_paths":case.truth["true_edge_paths"],"scope":"Known synthetic geometry; route availability is separate from joint selection."}
 (directory/"metadata.json").write_text(json.dumps(record_out,indent=2)+"\n"); return record_out

def run_r4a_fixtures(output_root:Path,config:dict[str,Any])->dict[str,Any]:
 settings={**_DEFAULTS,**config}; root=Path(output_root); fixtures=root/"fixtures"; report_path=root/"fixture_report.json"
 root.mkdir(parents=True,exist_ok=True)
 if fixtures.exists() or report_path.exists(): raise FileExistsError("Refusing to overwrite R4a fixture artifacts")
 fixtures.mkdir(); cases=[]
 explicit=_case(3,"identity"); directory=fixtures/"explicit_mechanism"; directory.mkdir(); cases.append(_one(explicit,"explicit_mechanism","explicit",directory,settings))
 for width in _WIDTHS:
  for transform in _TRANSFORMS:
   name=f"rendered_width_{width}_{transform}"; directory=fixtures/name; directory.mkdir(); cases.append(_one(_case(width,transform),name,"rendered",directory,settings))
 failures=[]
 for row in cases:
  if not row["baseline_routes_retained_exactly"]: failures.append({"case":row["name"],"invariant":"baseline_routes_retained_exactly"})
  if not row["h1_full_passthrough_available"]: failures.append({"case":row["name"],"invariant":"h1_full_passthrough_available"})
  for label in ("old_selected_metrics","new_selected_metrics"):
   for item in row[label]:
    if item["selected_null_or_missing"] and item["centerline"]["recall"]!=0: failures.append({"case":row["name"],"invariant":"null_recall_zero","selection":label,"head_id":item["head_id"]})
 report={"schema_version":"r4a.fixture-benchmark.v1","config":settings,"cases":cases,"acceptance":{"case_count":len(cases),"failure_count":len(failures),"failures":failures},
  "scope":"Synthetic attachment-passthrough mechanism only; no real-image accuracy claim and no R2 solver change."}
 report_path.write_text(json.dumps(report,indent=2)+"\n"); return report
