"""No-grad CPU replay on frozen TRAIN arrays; no new model predictions/AUC."""
from dataclasses import asdict
from pathlib import Path
import json,time,statistics
import torch
from ktbench.models.favor_plus_antithetic_experimental import AntitheticFavorConfig,antithetic_projection,VARIANT,FORMAT
from ktbench.models.favor_plus import causal_favor_attention
from ktbench.favor_plus_post50_diagnostics import attention_diagnostics
from scripts.investigate_favor_plus_epoch8 import RUN,HERE,DES,setup,guard,check_plan,load_array

start=time.time();setup();guard(start);check_plan();records=[];cache={}
for state in ('7','8'):
 for case in range(len(DES['training_cohort'])):
  guard(start)
  for stage in DES['stages']:
   a=load_array(state,case,stage);seed=42+['action_encoder','session_encoder','correct_padding_encoder','decoder'].index(stage.split('.')[0])
   q,k,v=[torch.from_numpy(a[x])[None] for x in ('q','k','v')];qm,km=[torch.from_numpy(a[x])[None] for x in ('qm','km')]
   for m in (128,256,512):
    key=(seed,m)
    if key not in cache:cache[key]=antithetic_projection(4,m,64,seed).float()
    w=cache[key]
    with torch.no_grad():actual=causal_favor_attention(q,k,v,w,query_mask=qm,key_mask=km,chunk_size=32)
    stats=attention_diagnostics(a['q'],a['k'],a['v'],w.numpy(),a['qm'],a['km'],actual.numpy()[0])
    if stats['feature_kernel_agreement']!='PASS':raise FloatingPointError('prototype feature-reference mismatch')
    records.append(dict(state=state,case=case,stage=stage,features_total=m,projection_seed=seed,metrics=stats))
benchmark=[];a=load_array('8',12,'correct_padding_encoder.layers.0.attention');q,k,v=[torch.from_numpy(a[x])[None] for x in ('q','k','v')];qm,km=[torch.from_numpy(a[x])[None] for x in ('qm','km')]
for m in (128,256,512):
 w=cache[(44,m)];measurements=[]
 with torch.no_grad():
  for rep in range(4):
   stamp=time.perf_counter();output=causal_favor_attention(q,k,v,w,query_mask=qm,key_mask=km,chunk_size=32)
   seconds=time.perf_counter()-stamp
   if rep:measurements.append(seconds)
 benchmark.append(dict(features_total=m,cpu_forward_median_seconds=statistics.median(measurements),measurements=measurements,linear_weighted_state_bytes=1*4*m*64*4,projection_seed=44,positions=q.shape[2]))
(RUN/'ANTITHETIC_PROTOTYPE_REPLAY.json').write_text(json.dumps(dict(variant=VARIANT,format_version=FORMAT,config=asdict(AntitheticFavorConfig()),records=records,benchmark=benchmark,host=guard(start),seconds=time.time()-start,cuda_initialized=torch.cuda.is_initialized(),optimizer_calls=0,test_evaluations=0,notes='DEFAULT-stage-seed frozen representations from unchanged older model. Counterfactual raw attention only, no candidate training or validation inference. CPU latency is not a GPU throughput estimate.'),indent=2,sort_keys=True)+'\n')
print('prototype CPU replay PASS',len(records),flush=True)
