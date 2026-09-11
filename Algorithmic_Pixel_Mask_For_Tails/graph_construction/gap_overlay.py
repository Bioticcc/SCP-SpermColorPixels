"""Compose source-coordinate component graphs and overlay supported gap edges."""
from __future__ import annotations
from dataclasses import replace
import math
from typing import Any
import cv2
import numpy as np
from .segment_graph import Node, Segment, SegmentGraph, from_explicit_graph

def _shape(value):
 if not isinstance(value,(list,tuple)) or len(value)!=2 or any(type(x) is not int or x<=0 for x in value): raise ValueError("shape must be positive integer HxW")
 return tuple(value)

def merge_source_graphs(components:list,shape)->tuple[SegmentGraph,dict]:
 """Namespace ROI-local graphs while preserving real head identity as provenance."""
 shape=_shape(shape)
 if not isinstance(components,list): raise ValueError("components must be a list")
 nodes={}; segments={}; anchors={}; aliases={}; segment_maps={}; seen_tails=set()
 for component in components:
  tail_id=str(component.get("tail_id",""))
  if not tail_id or tail_id in seen_tails: raise ValueError("tail IDs must be unique nonempty values")
  seen_tails.add(tail_id); roi=component.get("roi_xyxy")
  if not isinstance(roi,(list,tuple)) or len(roi)!=4: raise ValueError("component ROI is required")
  x0,y0,x1,y1=roi
  if any(type(x) is not int for x in roi) or not(0<=x0<x1<=shape[1] and 0<=y0<y1<=shape[0]): raise ValueError("component ROI outside source shape")
  raw=component.get("graph")
  if not isinstance(raw,dict): raise ValueError("component graph is required")
  graph=from_explicit_graph(raw,shape=(y1-y0,x1-x0),anchors=raw.get("anchors"))
  node_map={old:f"{tail_id}:{old}" for old in graph.nodes}; segment_map={old:f"{tail_id}:{old}" for old in graph.segments}
  for old,node in graph.nodes.items():
   xy=(node.xy[0]+x0,node.xy[1]+y0); pixels=tuple((x+x0,y+y0) for x,y in node.pixels_xy)
   if not(0<=xy[0]<shape[1] and 0<=xy[1]<shape[0]) or any(not(0<=x<shape[1] and 0<=y<shape[0]) for x,y in pixels): raise ValueError("shifted node outside source shape")
   nodes[node_map[old]]=Node(node_map[old],xy,node.kind,pixels)
  for old,segment in graph.segments.items():
   points=tuple((x+x0,y+y0) for x,y in segment.points_xy)
   if any(not(0<=x<shape[1] and 0<=y<shape[0]) for x,y in points): raise ValueError("shifted segment outside source shape")
   segments[segment_map[old]]=Segment(segment_map[old],node_map[segment.start],node_map[segment.end],points,segment.radii,segment.evidence)
  for actual,node in graph.anchors.items():
   alias=f"{tail_id}@{actual}"
   if alias in anchors: raise ValueError("anchor alias collision")
   anchors[alias]=node_map[node]; aliases[alias]={"actual_head_id":str(actual),"node_id":node_map[node],"tail_id":tail_id}
  segment_maps[tail_id]={old:new for old,new in sorted(segment_map.items())}
 provenance={"anchor_aliases":aliases,"original_segment_maps":segment_maps,
  "resource_naming":"Namespaced segment IDs equal existing R2 resources: {tail_id}:{original_segment_id}."}
 return SegmentGraph(nodes,segments,anchors,shape,{"source_merge_provenance":provenance}),provenance

def _dense_line(a,b,shape):
 mask=np.zeros(shape,np.uint8); cv2.line(mask,tuple(map(int,a)),tuple(map(int,b)),1,1,lineType=cv2.LINE_8)
 yy,xx=np.nonzero(mask); points=np.column_stack((xx,yy)); delta=np.asarray(b,float)-np.asarray(a,float)
 order=np.argsort((points-np.asarray(a,float))@delta,kind="stable")
 return tuple((float(x),float(y)) for x,y in points[order])

def overlay_gap_edges(graph:SegmentGraph,gap_candidates:list)->SegmentGraph:
 """Add optional gap resources without changing any existing graph object."""
 if not isinstance(graph,SegmentGraph) or not isinstance(gap_candidates,list): raise ValueError("invalid gap overlay input")
 nodes=dict(graph.nodes); segments=dict(graph.segments); anchors=dict(graph.anchors); anchor_nodes=set(anchors.values())
 degree={node:0 for node in nodes}
 for segment in graph.segments.values(): degree[segment.start]+=1; degree[segment.end]+=1
 seen_pairs=set(); metadata=[]
 for gap in gap_candidates:
  gap_id=gap.get("id"); endpoint_ids=gap.get("endpoint_ids"); points=gap.get("points_xy")
  if not isinstance(gap_id,str) or not gap_id or gap_id in segments: raise ValueError("duplicate or invalid gap ID")
  if not isinstance(endpoint_ids,(list,tuple)) or len(endpoint_ids)!=2 or endpoint_ids[0]==endpoint_ids[1]: raise ValueError("gap requires two distinct endpoints")
  a,b=map(str,endpoint_ids); pair=tuple(sorted((a,b)))
  if a not in nodes or b not in nodes: raise ValueError("gap endpoint node is missing")
  if pair in seen_pairs: raise ValueError("duplicate gap endpoint pair")
  if degree[a]!=1 or degree[b]!=1: raise ValueError("gap endpoints must have base degree one")
  if a in anchor_nodes or b in anchor_nodes: raise ValueError("gap endpoints cannot be anchor nodes")
  if nodes[a].kind in {"boundary","boundary_port"} or nodes[b].kind in {"boundary","boundary_port"}: raise ValueError("gap endpoints cannot be boundary nodes")
  arr=np.asarray(points,float)
  if arr.shape!=(2,2) or not np.isfinite(arr).all() or not np.array_equal(arr,np.rint(arr)) or not np.allclose(arr[0],nodes[a].xy) or not np.allclose(arr[1],nodes[b].xy): raise ValueError("gap coordinates must match integer endpoint nodes")
  radius=gap.get("radius_px"); support=gap.get("mean_support")
  if type(radius) not in (int,float) or not math.isfinite(radius) or radius<=0 or type(support) not in (int,float) or not math.isfinite(support) or support<=0: raise ValueError("gap radius/support must be finite positive")
  dense=_dense_line(arr[0],arr[1],graph.shape)
  segments[gap_id]=Segment(gap_id,a,b,dense,tuple(float(radius) for _ in dense),float(support))
  nodes[a]=replace(nodes[a],kind="ordinary"); nodes[b]=replace(nodes[b],kind="ordinary")
  seen_pairs.add(pair); metadata.append({"id":gap_id,"endpoint_ids":[a,b],"points_xy":arr.tolist(),"radius_px":float(radius),"mean_support":float(support),
   "distance_px":gap.get("distance_px"),"minimum_support":gap.get("minimum_support"),"unsupported_distance_px":gap.get("unsupported_distance_px"),
   "max_unsupported_run_px":gap.get("max_unsupported_run_px"),"endpoint_alignment":gap.get("endpoint_alignment"),
   "scope":"Optional resource; ordinary ownership capacity is unchanged."})
 diagnostics=dict(graph.diagnostics); diagnostics["gap_overlay"]={"added_count":len(metadata),"edges":metadata,"ownership_capacity_relaxed":False}
 return SegmentGraph(nodes,segments,anchors,graph.shape,diagnostics)

def connected_subgraphs(graph:SegmentGraph)->list[SegmentGraph]:
 """Return deterministic reachable views with unchanged global coordinates."""
 if not isinstance(graph,SegmentGraph): raise ValueError("graph must be a SegmentGraph")
 neighbors={node:set() for node in graph.nodes}
 for segment in graph.segments.values(): neighbors[segment.start].add(segment.end); neighbors[segment.end].add(segment.start)
 unseen=set(graph.nodes); groups=[]
 while unseen:
  start=min(unseen); pending=[start]; found=set()
  while pending:
   node=pending.pop()
   if node in found: continue
   found.add(node); pending.extend(sorted(neighbors[node]-found,reverse=True))
  unseen-=found; groups.append(found)
 groups.sort(key=lambda group:min(group)); result=[]; used=set()
 for group in groups:
  segments={key:value for key,value in graph.segments.items() if value.start in group and value.end in group}; used.update(segments)
  anchors={key:value for key,value in graph.anchors.items() if value in group}; diagnostics=dict(graph.diagnostics)
  diagnostics["connected_subgraph"]={"minimum_node_id":min(group),"node_count":len(group),"segment_count":len(segments)}
  result.append(SegmentGraph({key:graph.nodes[key] for key in sorted(group)},segments,anchors,graph.shape,diagnostics))
 if used!=set(graph.segments): raise RuntimeError("segment was not assigned to exactly one connected subgraph")
 return result
