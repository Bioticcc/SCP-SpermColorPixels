"""Compose source graphs, supported gaps, and optional head-node attachments."""
from __future__ import annotations
from dataclasses import replace
import math
from typing import Any
import numpy as np
from graph_construction.gap_overlay import connected_subgraphs,merge_source_graphs,overlay_gap_edges
from graph_construction import SegmentGraph
from head_anchors.candidates import propose_head_anchors
from tail_evidence.gaps import propose_gap_edges

_DEFAULTS={"gap":{},"max_head_distance_px":18.0,"max_head_anchors":3}

def _config(config):
 if config is None: config={}
 if not isinstance(config,dict) or set(config)-set(_DEFAULTS) or not isinstance(config.get("gap",{}),dict): raise ValueError("invalid graph proposal configuration")
 out={**_DEFAULTS,**config}; distance=out["max_head_distance_px"]; count=out["max_head_anchors"]
 if isinstance(distance,bool) or not isinstance(distance,(int,float)) or not math.isfinite(distance) or distance<0 or type(count)is not int or count<0: raise ValueError("invalid head proposal limits")
 return out

def _endpoint_records(graph):
 anchor_nodes=set(graph.anchors.values()); records=[]
 for node_id,node in sorted(graph.nodes.items()):
  outgoing=graph.outgoing(node_id)
  if len(outgoing)!=1 or node_id in anchor_nodes or node.kind in {"boundary","boundary_port"}: continue
  segment_id,forward=outgoing[0]; points=graph.points(segment_id,forward).astype(float); segment=graph.segments[segment_id]
  radii=np.asarray(segment.radii if forward else segment.radii[::-1],float)
  radius=float(radii[0]) if len(radii) else 1.0; target=max(3.,3.*radius)
  lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
  if not len(lengths) or not np.any(lengths>0): continue
  index=min(len(points)-1,int(np.searchsorted(np.cumsum(lengths),target))+1); inward=points[index]-points[0]; norm=float(np.linalg.norm(inward))
  if norm<=0: continue
  records.append({"id":node_id,"point_xy":[float(x) for x in node.xy],"outward_xy":(-inward/norm).tolist(),
   "radius_px":radius,"component_id":node_id.split(":",1)[0],"incident_segment_id":segment_id,"tangent_arclength_px":target})
 return records

def propose_image_graph(record:dict,score:np.ndarray,head_labels:np.ndarray,config:dict|None=None)->dict[str,Any]:
 """Return immutable base/gap graphs plus JSON-ready proposal diagnostics."""
 cfg=_config(config)
 if not isinstance(record,dict) or not isinstance(record.get("components"),list): raise ValueError("base record requires components")
 if type(record.get("height")) is not int or type(record.get("width")) is not int: raise ValueError("base record requires integer dimensions")
 shape=(record["height"],record["width"]); score=np.asarray(score); labels=np.asarray(head_labels)
 if score.dtype!=np.float32 or score.shape!=shape or not np.isfinite(score).all() or np.any((score<0)|(score>1)): raise ValueError("score must be full-image finite float32 in [0,1]")
 if labels.shape!=shape or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0): raise ValueError("head labels must be a nonnegative full-image integer array")
 base,provenance=merge_source_graphs(record["components"],shape); endpoints=_endpoint_records(base)
 gap_result=propose_gap_edges(score,endpoints,cfg["gap"],head_labels=labels); gap_graph=overlay_gap_edges(base,gap_result["candidates"])
 aliases=dict(gap_graph.anchors); alias_provenance=dict(provenance["anchor_aliases"]); anchor_rows=[]
 represented={int(row["actual_head_id"]) for row in alias_provenance.values() if str(row["actual_head_id"]).isdigit()}
 actual_ids=sorted(int(value) for value in np.unique(labels) if value)
 for actual in actual_ids:
  baseline_aliases=sorted(alias for alias,row in alias_provenance.items() if str(row["actual_head_id"])==str(actual))
  baseline_nodes=[alias_provenance[alias]["node_id"] for alias in baseline_aliases]
  proposals=propose_head_anchors(base,labels==actual,score,baseline_nodes,max_distance_px=cfg["max_head_distance_px"],max_additional=cfg["max_head_anchors"])
  additions=[]
  for proposal in proposals:
   if proposal["baseline"]: continue
   node_id=proposal["node_id"]; alias=f"proposal@{actual}@{node_id}"
   if alias in aliases: raise ValueError("head proposal alias collision")
   aliases[alias]=node_id; metadata={"actual_head_id":str(actual),"node_id":node_id,"tail_id":node_id.split(":",1)[0],
    "baseline":False,"proposal":proposal,"node_kind_after_gap_overlay":gap_graph.nodes[node_id].kind,
    "policy":"Candidate was valid on the original degree-one node and remains an optional alias after gap overlay."}
   alias_provenance[alias]=metadata; additions.append(metadata)
  anchor_rows.append({"actual_head_id":str(actual),"already_represented":actual in represented,"baseline_aliases":baseline_aliases,
   "baseline_node_ids":baseline_nodes,"proposals":proposals,"added_aliases":[f"proposal@{actual}@{row['node_id']}" for row in additions]})
 proposed_graph=replace(gap_graph,anchors=aliases,diagnostics={**gap_graph.diagnostics,"head_anchor_proposals":{"rows":anchor_rows}})
 diagnostics={"schema_version":"scp.image-graph-proposals.v1","config":cfg,"endpoint_candidates":endpoints,"gap_proposals":gap_result,
  "head_anchor_proposals":anchor_rows,"alias_provenance":alias_provenance,"base_alias_provenance":provenance,
  "connected_subgraphs":[{"node_ids":sorted(view.nodes),"segment_ids":sorted(view.segments),"anchor_aliases":sorted(view.anchors)} for view in connected_subgraphs(proposed_graph)],
  "limitations":"Straight gaps and existing-node head aliases are optional hypotheses; no head-center chord, route selection, assignment, or ownership relaxation is introduced."}
 return {"base_graph":base,"gap_graph":gap_graph,"proposed_graph":proposed_graph,"diagnostics":diagnostics}
