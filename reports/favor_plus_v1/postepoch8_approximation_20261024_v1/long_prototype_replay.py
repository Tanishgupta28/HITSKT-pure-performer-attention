"""Extreme frozen TRAIN rows, full lengths, prototype default512, CPU only."""
import json,time
import torch
from ktbench.models.favor_plus_antithetic_experimental import antithetic_projection
from ktbench.models.favor_plus import causal_favor_attention
from ktbench.favor_plus_post50_diagnostics import attention_diagnostics
from scripts.investigate_favor_plus_epoch8 import RUN,DES,setup,guard,check_plan,load_array

start=time.time();setup();guard(start);check_plan();rows=[]
for state in ('7','8'):
 for case in (12,13):
  for stage in DES['stages']:
   guard(start);a=load_array(state,case,stage,10000);seed=42+['action_encoder','session_encoder','correct_padding_encoder','decoder'].index(stage.split('.')[0]);w=antithetic_projection(4,512,64,seed).float()
   q,k,v=[torch.from_numpy(a[x])[None] for x in ('q','k','v')];qm,km=[torch.from_numpy(a[x])[None] for x in ('qm','km')]
   with torch.no_grad():out=causal_favor_attention(q,k,v,w,query_mask=qm,key_mask=km,chunk_size=32)
   stats=attention_diagnostics(a['q'],a['k'],a['v'],w.numpy(),a['qm'],a['km'],out.numpy()[0])
   assert stats['feature_kernel_agreement']=='PASS'
   rows.append(dict(state=state,case=case,stage=stage,length=q.shape[2],metrics=stats))
(RUN/'LONG_PROTOTYPE_REPLAY.json').write_text(json.dumps(dict(records=rows,host=guard(start),seconds=time.time()-start,optimizer_calls=0,test_evaluations=0,cuda_initialized=torch.cuda.is_initialized()),indent=2,sort_keys=True)+'\n')
print('long prototype PASS',len(rows),'maximum length',max(x['length'] for x in rows),flush=True)
