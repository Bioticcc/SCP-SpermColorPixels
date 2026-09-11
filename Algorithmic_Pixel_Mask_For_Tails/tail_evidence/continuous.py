"""Uncalibrated continuous SCP evidence and sparse local orientation peaks."""
from __future__ import annotations
import math
import cv2
import numpy as np

def continuous_scp_evidence(rgb, settings, include_terms=True):
 if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or not rgb.size: raise ValueError('rgb must be uint8 HxWx3')
 required=('sat_min','local_dark_min','value_max')
 if not isinstance(settings,dict) or set(required)-set(settings): raise ValueError('missing settings')
 vals={}
 for k in required:
  v=settings[k]
  if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or int(v)!=v: raise ValueError('invalid settings')
  vals[k]=int(v)
 if not(1<=vals['sat_min']<=255 and 1<=vals['local_dark_min']<=255 and 0<=vals['value_max']<=254):raise ValueError('settings bounds')
 h,w=rgb.shape[:2]; hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); hue,sat,value=cv2.split(hsv); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
 sigma=max(25.0,min(w,h)/30.0); bg=cv2.GaussianBlur(gray,(0,0),sigmaX=sigma,sigmaY=sigma); dark=np.clip(bg.astype(np.int16)-gray.astype(np.int16),0,255).astype(np.uint8)
 huef=hue.astype(np.float32); hue_support=np.clip(np.minimum(huef/6,(88-huef)/6),0,1); hue_support[(hue>=6)&(hue<=82)]=1
 sat_support=np.clip(sat.astype(np.float32)/vals['sat_min'],0,1); dark_support=np.clip(dark.astype(np.float32)/vals['local_dark_min'],0,1); value_support=np.clip((255-value.astype(np.float32))/(255-vals['value_max']),0,1)
 score=np.minimum.reduce((hue_support,sat_support,dark_support,value_support)).astype(np.float32)
 out={'hsv':hsv,'hue':hue,'sat':sat,'value':value,'gray':gray,'local_dark':dark,'score':score,'metadata':{'sigma':sigma,'thresholds':vals,'calibration':'uncalibrated soft bottleneck'}}
 if include_terms: out['terms']={'hue':hue_support.astype(np.float32),'sat':sat_support.astype(np.float32),'local_dark':dark_support.astype(np.float32),'value':value_support.astype(np.float32)}
 return out

def sample_orientation_peaks(score,points_xy,radii_px,angles=12,max_peaks=2):
 s=np.asarray(score); p=np.asarray(points_xy,np.float32); r=np.asarray(radii_px,np.float32)
 if p.size == 0: p=np.empty((0,2),np.float32)
 if s.dtype!=np.float32 or s.ndim!=2 or not s.size or not np.isfinite(s).all() or np.any((s<0)|(s>1)) or p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or r.ndim!=1 or not len(r) or not np.isfinite(r).all() or np.any(r<=0) or type(angles) is not int or angles<2 or type(max_peaks)is not int or max_peaks<1: raise ValueError('invalid orientation input')
 if not len(p): return []
 h,w=s.shape; out=[]; ts=np.linspace(-3,3,7)
 for x,y in p:
  candidates=[]
  for rad in sorted(r.tolist()):
   for i in range(angles):
    a=i*math.pi/angles; dx,dy=math.cos(a),math.sin(a); nx,ny=-dy,dx
    coords=np.concatenate([np.column_stack((x+ts*rad*dx,y+ts*rad*dy)),np.column_stack((x+ts*rad*dx+2*rad*nx,y+ts*rad*dy+2*rad*ny)),np.column_stack((x+ts*rad*dx-2*rad*nx,y+ts*rad*dy-2*rad*ny))]).astype(np.float32)
    if np.any(coords[:,0]<0)|np.any(coords[:,0]>w-1)|np.any(coords[:,1]<0)|np.any(coords[:,1]>h-1): continue
    v=cv2.remap(s,coords[:,0],coords[:,1],cv2.INTER_LINEAR); response=max(float(v[:7].mean()-(v[7:14].mean()+v[14:].mean())/2),0)
    if response>0:candidates.append((response,a,float(rad)))
  chosen=[]
  for item in sorted(candidates,key=lambda z:(-z[0],z[2],z[1])):
   if all(min(abs(item[1]-q[1]),math.pi-abs(item[1]-q[1]))>=math.pi/6-1e-9 for q in chosen): chosen.append(item)
   if len(chosen)>=max_peaks:break
  out.append([{'angle_radians_mod_pi':a,'response':v,'radius_px':rr} for v,a,rr in chosen])
 return out
