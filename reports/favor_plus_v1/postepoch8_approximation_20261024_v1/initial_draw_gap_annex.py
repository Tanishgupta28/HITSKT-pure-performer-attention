"""CPU-only prospective draw-gap follow-up. Frozen training arrays, no optimizer."""
from pathlib import Path
import json,time
import numpy as np
import torch
from ktbench.favor_plus_post50_diagnostics import attention_diagnostics,describe
from ktbench.models.favor_plus import orthogonal_gaussian_projection
from scripts.investigate_favor_plus_epoch8 import RUN,HERE,DES,guard,setup,check_plan,load_array

start=time.time();setup();guard(start);check_plan()
annex=json.loads((HERE/'INITIAL_DRAW_GAP_ANNEX.json').read_text());rows=[]
pools={seed:orthogonal_gaussian_projection(4,512,64,seed).numpy()[:,:128] for seed in annex['seeds']}
for state in annex['states']:
 for case in range(len(DES['training_cohort'])):
  guard(start)
  for stage in DES['stages']:
   a=load_array(state,case,stage);q,k,v,w,qm,km=[a[x] for x in ('q','k','v','w','qm','km')]
   saved=attention_diagnostics(q,k,v,w,qm,km);fresh=[dict(seed=seed,metrics=attention_diagnostics(q,k,v,pool,qm,km)) for seed,pool in pools.items()]
   masses=[];cosines=[];dominant=[]
   for h in range(q.shape[0]):
    for i in np.unique(np.linspace(0,q.shape[1]-1,min(q.shape[1],8)).astype(int)):
     if not qm[i]:continue
     js=np.flatnonzero(km[:i+1])
     for j in js[np.unique(np.linspace(0,len(js)-1,min(len(js),12)).astype(int))] if len(js) else []:
      z=q[h,i].astype(float)/64**.25+k[h,j].astype(float)/64**.25;score=w[h].astype(float)@z;mass=np.exp(score-score.max());mass/=mass.sum()
      masses.append(float(mass.max()));dominant.append((h,int(score.argmax())))
      cosines.append(float(np.max(score/(np.linalg.norm(w[h],axis=-1)*max(np.linalg.norm(z),1e-30)))))
   from collections import Counter
   perhead=[]
   for h in range(q.shape[0]):
    counts=Counter(i for head,i in dominant if head==h);perhead.append(max(counts.values())/sum(counts.values()))
   rows.append(dict(state=state,case=case,stage=stage,saved=saved,fresh=fresh,original_pair_max_feature_mass=describe(masses),original_max_projection_cosine=describe(cosines),original_dominant_feature_max_share_per_head=perhead))
 print('draw gap',state,flush=True)
result=dict(records=rows,host=guard(start),seconds=time.time()-start,optimizer_calls=0,test_evaluations=0,cuda_initialized=torch.cuda.is_initialized(),interpretation='Cross-sectional frozen-array contrasts; supports projection-specific learning alignment if gap emerges after training, not a randomized causal training intervention.')
(RUN/'INITIAL_DRAW_GAP_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
