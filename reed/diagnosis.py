"""Frozen three-family diagnosis and causal global change clock."""
import json,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist,pdist
from sklearn.covariance import LedoitWolf
from .statistics import rank_transform, fuse

from .statistics import correlation_redundancy
from .boundary import OnlineBoundary

IDS=(0,2,4,11,13,14,15)

def bank(reference):
    features = np.asarray(reference, dtype=np.float64)
    bandwidth = max(float(np.median(pdist(features, "sqeuclidean"))), 1e-10)
    within = cdist(features, features, "sqeuclidean")
    return SimpleNamespace(x=features, mean=features.mean(0),
                           precision=LedoitWolf().fit(features).precision_,
                           bw=bandwidth, kref=np.exp(-within/(2*bandwidth)).mean())


def score_all(x, bank):
    x = np.asarray(x, dtype=np.float64)
    delta = x.mean(0) - bank.mean
    cross = cdist(x, bank.x, "sqeuclidean")
    within = cdist(x, x, "sqeuclidean")
    # Preserve the three column indices in the archived calibration tables.
    scores = np.zeros(37)
    scores[14] = np.sqrt(max(float(delta @ bank.precision @ delta), 0.0))
    scores[2] = correlation_redundancy(x)
    scores[4] = (bank.kref + np.exp(-within/(2*bank.bw)).mean()
                 - 2*np.exp(-cross/(2*bank.bw)).mean())
    return scores


class JointRouter(object):
 def __init__(self,path):
  with np.load(path) as z:
   self.config=json.loads(str(z['config']));self.layer='h';self.seed=int(z['seed']);self.batch=int(z['batch']);self.normal=z['normalizer'].copy();self.calibration=z['threshold'].copy();self.banks=[bank(z['reference_'+m]) for m in ('audio','video')]
  self.mean_only=False;self.normal_u=rank_transform(self.normal,self.normal,False,True);cal=rank_transform(self.calibration,self.normal,False,True);ids=self.config['ids'];self.cutoff=np.quantile(fuse(cal[...,ids],self.config['fusion'],self.normal_u[...,ids]),self.config['q'],axis=0,method='higher');self.reset_segments()
 def reset_segments(self):pass
 def predict(self,audio,video,**kwargs):
  ids=self.config['ids'];raw=np.asarray([[score_all(x,b)[ids] for x,b in zip((audio,video),self.banks)]]);u=rank_transform(raw,self.normal[...,ids],False,True);score=fuse(u,self.config['fusion'],self.normal_u[...,ids])[0]
  return dict(corrupted=score>self.cutoff,fusion_score=score,threshold=self.cutoff.copy(),raw_experts=raw[0],selected_names=self.config['experts'].split('+'),dasp_redundancy=None,segments=[{},{}])

class GlobalChangeClock:
 def __init__(self,router,mode='relative'):
  self.mode=mode;self.impl=None;self.last_route=None;self.mean=None;self.pending=0
  config=router.config['change'];self.config=config
  with np.load(router.root/'change_features.npz') as z:
   self.projection=z['projection'].copy();self.source_mean=z['source_mean'].copy();self.source_scale=z['source_scale'].copy();normal=z['normal'][:512].copy()
  if config['input']=='experts':normal=router.calibration[...,router.config['ids']].reshape(len(router.calibration),-1)
  self.detector=OnlineBoundary(normal,config)
  self.banks=router.banks;self.ids=router.config['ids'];self.window=int(config.get('fixed_window_size',64));self.buffers=[np.empty((0,768)),np.empty((0,768))]
 def update(self,diagnosis,features=None):
  self.buffers=[np.concatenate([old,np.asarray(x,dtype=np.float64)]) for old,x in zip(self.buffers,features)]
  self.last_route=tuple(diagnosis['corrupted'])
  if len(self.buffers[0])<self.window:return False
  values=[x[:self.window] for x in self.buffers];self.buffers=[x[self.window:] for x in self.buffers]
  if self.config['input']=='experts':x=np.concatenate([score_all(v,b)[self.ids] for v,b in zip(values,self.banks)])
  else:
   parts=[]
   for m,value in enumerate(values):
    h=((value-self.source_mean[m])/self.source_scale[m])@self.projection
    parts.append(np.r_[h.mean(0),np.log(np.maximum(h.var(0,ddof=1),1e-4))])
   x=np.concatenate(parts)
  event=self.detector.update(x,diagnosis['corrupted']);self.pending=self.detector.pending
  return event
