"""Bounded, image-supported endpoint gap proposals; no evidence is overwritten."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from .continuous import sample_orientation_peaks

_DEFAULTS = {'max_gap_px':18.0,'max_radius_multiple':6.0,'min_alignment':0.75,
             'min_mean_support':0.25,'min_point_support':0.15,'max_unsupported_run_px':2.0,
             'min_ridge_response':0.05,'min_ridge_alignment':0.85,
             'max_evaluations':2000,'max_candidates':128}


def _settings(config):
    config={} if config is None else config
    if not isinstance(config,dict) or set(config)-set(_DEFAULTS): raise ValueError('Unknown gap settings')
    values={**_DEFAULTS,**config}
    for key,value in values.items():
        if type(value) not in (int,float) or not math.isfinite(value): raise ValueError('Gap settings must be finite numbers')
        if key in ('max_evaluations','max_candidates'):
            if type(value) is not int or value<1: raise ValueError('Gap budgets must be positive integers')
        elif key in ('max_gap_px','max_radius_multiple'):
            if value<=0: raise ValueError('Gap distances must be positive')
        elif key=='max_unsupported_run_px':
            if value<0: raise ValueError('Unsupported run bound must be nonnegative')
        elif not 0<=value<=1: raise ValueError('Gap support/alignment must be in [0,1]')
    return values


def _sample(field,points):
    """Bilinear interpolation for validated source-image points."""
    x,y=points[:,0],points[:,1]; x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,field.shape[1]-1);y1=np.minimum(y0+1,field.shape[0]-1)
    dx=x-x0;dy=y-y0
    return (1-dx)*(1-dy)*field[y0,x0]+dx*(1-dy)*field[y0,x1]+(1-dx)*dy*field[y1,x0]+dx*dy*field[y1,x1]


def _endpoints(items,shape):
    result=[];seen=set()
    for item in items:
        key=item['id'];p=np.asarray(item['point_xy'],float);v=np.asarray(item['outward_xy'],float);r=item['radius_px']
        if not isinstance(key,str) or not key or key in seen: raise ValueError('Endpoint IDs must be unique nonempty strings')
        if p.shape!=(2,) or v.shape!=(2,) or not np.isfinite(p).all() or not np.isfinite(v).all(): raise ValueError('Invalid endpoint geometry')
        if not (0<=p[0]<=shape[1]-1 and 0<=p[1]<=shape[0]-1) or np.linalg.norm(v)<=0: raise ValueError('Invalid endpoint position or tangent')
        if type(r) not in (int,float) or not math.isfinite(r) or r<=0: raise ValueError('Invalid endpoint radius')
        seen.add(key);result.append({**item,'point_xy':p,'outward_xy':v/np.linalg.norm(v),'radius_px':float(r)})
    return sorted(result,key=lambda item:item['id'])


def propose_gap_edges(score,endpoints,config=None,head_labels=None):
    """Return inspected straight gap proposals and explicit search limitations.

    Each endpoint has id, point_xy, outward_xy and radius_px. Optional component_id
    is provenance only: a shared pixel component is not assumed to be one sperm.
    No graph edge, tail mask or head label is mutated by this function.
    """
    score=np.asarray(score)
    if score.dtype!=np.float32 or score.ndim!=2 or not score.size or not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        raise ValueError('score must be a nonempty finite float32 field in [0,1]')
    labels=np.zeros(score.shape,np.int32) if head_labels is None else np.asarray(head_labels)
    if labels.shape!=score.shape or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0): raise ValueError('Invalid head labels')
    cfg=_settings(config);items=_endpoints(endpoints,score.shape);cells=defaultdict(list)
    for index,item in enumerate(items): cells[tuple(np.floor(item['point_xy']/cfg['max_gap_px']).astype(int))].append(index)
    pairs=[]
    for i,a in enumerate(items):
        cx,cy=np.floor(a['point_xy']/cfg['max_gap_px']).astype(int)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for j in cells.get((cx+dx,cy+dy),[]):
                    if j<=i:continue
                    b=items[j];delta=b['point_xy']-a['point_xy'];distance=float(np.linalg.norm(delta))
                    if distance<=0 or distance>min(cfg['max_gap_px'],cfg['max_radius_multiple']*min(a['radius_px'],b['radius_px'])):continue
                    direction=delta/distance;alignment=(float(a['outward_xy']@direction),float(b['outward_xy']@(-direction)))
                    if min(alignment)<cfg['min_alignment']:continue
                    pairs.append((distance,a['id'],b['id'],i,j,alignment))
    pairs.sort(key=lambda item:item[:3]);accepted=[];rejected=defaultdict(int)
    for distance,aid,bid,i,j,alignment in pairs[:cfg['max_evaluations']]:
        a,b=items[i],items[j];n=max(2,int(math.ceil(distance*2)))
        samples=a['point_xy']+(b['point_xy']-a['point_xy'])*((np.arange(n)+.5)/n)[:,None]
        support=_sample(score,samples);step=distance/n;low=support<cfg['min_point_support'];run=longest=0
        for bad in low:
            run=run+1 if bad else 0;longest=max(longest,run)
        unsupported_run=longest*step;mean=float(support.mean())
        rounded=np.rint(samples).astype(int)
        touched=sorted(int(value) for value in np.unique(labels[rounded[:,1],rounded[:,0]]) if value)
        if touched:rejected['detected_head_intersection']+=1;continue
        if mean<cfg['min_mean_support']:rejected['low_mean_support']+=1;continue
        if unsupported_run>cfg['max_unsupported_run_px']+1e-9:rejected['unsupported_run']+=1;continue
        probes=a['point_xy']+(b['point_xy']-a['point_xy'])*np.array([.25,.5,.75])[:,None]
        radii=sorted({max(.5,min(a['radius_px'],b['radius_px'])),max(.5,(a['radius_px']+b['radius_px'])/2)})
        peaks=sample_orientation_peaks(score,probes,radii)
        angle=math.atan2(b['point_xy'][1]-a['point_xy'][1],b['point_xy'][0]-a['point_xy'][0])%math.pi
        aligned=[max((peak['response'] for peak in row if abs(math.cos(peak['angle_radians_mod_pi']-angle))>=cfg['min_ridge_alignment']),default=0.0) for row in peaks]
        if sum(response>=cfg['min_ridge_response'] for response in aligned)<2:rejected['ridge_support']+=1;continue
        accepted.append({'id':f'gap:{aid}->{bid}','endpoint_ids':[aid,bid],
            'component_ids':[a.get('component_id'),b.get('component_id')],
            'points_xy':[a['point_xy'].tolist(),b['point_xy'].tolist()],
            'distance_px':distance,'endpoint_alignment':list(alignment),'radius_px':min(a['radius_px'],b['radius_px']),
            'mean_support':mean,'minimum_support':float(support.min()),
            'unsupported_distance_px':float(low.sum()*step),'max_unsupported_run_px':unsupported_run,
            'ridge_probe_points_xy':probes.tolist(),'ridge_peaks':peaks,'aligned_ridge_responses':aligned,
            'sample_points_xy':samples.tolist(),'sample_support':support.tolist(),
            'scope':'Optional image-supported gap; biological continuity is unvalidated.'})
    accepted.sort(key=lambda row:(-row['mean_support'],-sum(row['aligned_ridge_responses']),row['distance_px'],row['id']))
    return {'schema_version':'scp.gap-proposals.v1','candidates':accepted[:cfg['max_candidates']],
        'config':cfg,'diagnostics':{'endpoint_count':len(items),'geometry_eligible_pairs':len(pairs),
            'evaluated_pairs':min(len(pairs),cfg['max_evaluations']),'rejected':dict(sorted(rejected.items())),
            'supported_candidates_before_limit':len(accepted),
            'truncated':len(pairs)>cfg['max_evaluations'] or len(accepted)>cfg['max_candidates']}}
