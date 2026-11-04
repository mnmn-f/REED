"""Causal score-only boundary detectors. RFF NEWMA is a research adaptation.
No labels, condition identity, or sequence length enter update().
"""
from collections import deque
import numpy as np

class OnlineBoundary:
 def __init__(self,normal,config):
  self.cfg=dict(config);normal=np.asarray(normal,float)
  self.center=np.maximum(np.median(np.abs(normal),axis=0),1e-8)
  z=normal.copy() if self.cfg.get("transform")=="identity" else np.log1p(np.maximum(normal,0)/self.center)
  self.loc=np.median(z,axis=0);self.scale=np.maximum(1.4826*np.median(abs(z-self.loc),axis=0),.02)
  self.dim=normal.shape[1];self.rng=np.random.default_rng(29)
  if self.cfg['family']=='newma_rff':
   zz=(z-self.loc)/self.scale;dist=np.linalg.norm(zz[:64,None]-zz[None,:64],axis=-1)
   bandwidth=max(float(np.median(dist[dist>0])) if (dist>0).any() else 1.,.1)*self.cfg.get('bandwidth',1.)
   self.w=self.rng.normal(size=(self.dim,64))/bandwidth;self.b=self.rng.uniform(0,2*np.pi,64)
  self.route=None;self.pending_route=None;self.route_pending=0;self.n=0;self.last_metrics={};self.last_event=-100000;self.reset_statistics()
 def reset_statistics(self):
  self.history=deque(maxlen=self.cfg.get('long',16)+self.cfg.get('short',2))
  self.fast=None;self.slow=None;self.mu=None;self.pos=np.zeros(self.dim);self.neg=np.zeros(self.dim);self.pending=0;self.count=0
 def transform(self,score):
  raw=np.asarray(score,float).reshape(-1)
  value=raw if self.cfg.get("transform")=="identity" else np.log1p(np.maximum(raw,0)/self.center)
  z=(value-self.loc)/self.scale
  if self.cfg['family']=='newma_rff':return np.sqrt(2/64)*np.cos(z@self.w+self.b)
  return z
 def update(self,score,route):
  x=self.transform(score);route=tuple(bool(v) for v in route);self.n+=1
  if self.route is None:self.route=route
  self.route_pending=(self.route_pending+1 if route==self.pending_route else 1) if route!=self.route else 0
  self.pending_route=route
  route_event=self.route_pending>=self.cfg.get('route_patience',2)
  family=self.cfg['family'];stat=0.
  if family=='window':
   self.history.append(x.copy());s=self.cfg['short'];l=self.cfg['long']
   if len(self.history)>=s+l:
    h=np.asarray(self.history);stat=float(np.linalg.norm(h[-s:].mean(0)-h[-s-l:-s].mean(0))/np.sqrt(len(x)))
  elif family=='cusum':
   if self.mu is not None:
    d=x-self.mu;k=self.cfg.get('drift',.5)
    self.pos=np.maximum(0,self.pos+d-k);self.neg=np.maximum(0,self.neg-d-k)
    stat=float(max(np.linalg.norm(self.pos),np.linalg.norm(self.neg))/np.sqrt(len(x)))
   self.mu=x.copy() if self.mu is None else (1-self.cfg.get('rate',.1))*self.mu+self.cfg.get('rate',.1)*x
  elif family.startswith('newma'):
   if self.fast is None:self.fast=x.copy();self.slow=x.copy()
   else:
    f=self.cfg['fast'];s=self.cfg['slow'];self.fast=(1-f)*self.fast+f*x;self.slow=(1-s)*self.slow+s*x
    stat=float(np.linalg.norm(self.fast-self.slow)/np.sqrt(len(x)))
  else:raise ValueError(family)
  self.count+=1
  self.pending=self.pending+1 if stat>self.cfg['threshold'] else 0
  event=self.n-self.last_event>self.cfg.get('cooldown',2) and (route_event or (self.count>=self.cfg.get('warmup',3) and self.pending>=self.cfg.get('patience',1)))
  self.last_metrics={'boundary_statistic':float(stat),'boundary_threshold':float(self.cfg['threshold']),'boundary_route_pending':int(self.route_pending)}
  if event:
   self.last_event=self.n;self.route=route;self.route_pending=0;self.reset_statistics()
  return bool(event)
