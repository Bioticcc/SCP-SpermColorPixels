"""Pure R4b extension of a frozen R4a/R2 hypothesis pool."""
from __future__ import annotations
import copy, math
import numpy as np
from global_assignment import Hypothesis
from experiments.r2_candidates import _controls

def extend_pool(frozen_payload:dict, routes:list[dict], config:dict)->dict:
 """Return old serialized hypotheses plus deterministic source-coordinate R4b routes.
 Routes require head_id, tail_id, segment_ids, points_xy, termination and costs.
 Segment IDs are already globally namespaced and are preserved verbatim.
 """
 controls=_controls(config); old=copy.deepcopy(frozen_payload.get('hypotheses',[])); ids={x['id'] for x in old}
 if len(ids)!=len(old):raise ValueError('duplicate frozen IDs')
 allrows=old+copy.deepcopy(routes); lengths={}
 for r in allrows:
  h=str(r.get('head_id')); p=np.asarray(r.get('metadata',{}).get('points_xy_ordered',r.get('points_xy',[])),float)
  if len(p)>1:lengths[h]=max(lengths.get(h,0.),float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()))
 out=list(old); added=[]
 for n,r in enumerate(routes,1):
  required={'head_id','tail_id','segment_ids','points_xy','termination','costs'}
  if not required<=set(r):raise ValueError('route schema')
  p=np.asarray(r['points_xy'],float)
  if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('route points')
  h=str(r['head_id']); c=r['costs']; d,cu,e=(float(c.get(k,0)) for k in ('direction','curvature','evidence'))
  if not all(math.isfinite(x) for x in (d,cu,e)):raise ValueError('route costs')
  length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) if len(p)>1 else 0.; coverage=0 if not lengths.get(h) else 1-length/lengths[h]; term=str(r['termination']); total=(d+cu)/max(1,len(r['segment_ids'])-1)+e+controls['coverage_weight']*coverage+(controls['partial_cost'] if term=='partial' else 0)
  ident=f"r4b:{r['tail_id']}:{h}:{n}"
  if ident in ids:raise ValueError('ID collision')
  ids.add(ident); meta={'kind':'r4b_route','tail_id':str(r['tail_id']),'head_id':h,'roi_xyxy':[0,0,int(frozen_payload['width']),int(frozen_payload['height'])],'points_xy_ordered':p.tolist(),'attachment':r.get('attachment',{}),'cost_decomposition':{'direction':d,'curvature':cu,'evidence':e,'coverage_deficit':coverage,'total':total}}
  added.append(Hypothesis(ident,h,total,tuple(str(x) for x in r['segment_ids']),r.get('endpoint_id'),term,meta))
 out.extend(copy.deepcopy([x.__dict__ for x in added])); return {'hypotheses':out,'added':added,'old_hypotheses':old}
