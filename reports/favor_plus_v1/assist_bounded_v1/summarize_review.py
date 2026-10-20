"""Summarize saved frozen CPU review only; no training or CUDA APIs."""
import json
from pathlib import Path
import numpy as np
from scripts.favor_plus_quality_common import describe,independent_attention,output_errors
from ktbench.favor_plus_bounded_assist import query_errors
from ktbench.favor_plus_pipeline import sha256
ROOT=Path('/workspace/capstone')
D=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist_bounded_review_20261020_v2'
rows=json.loads((D/'frozen_feature_comparisons.json').read_text())
summary=[]
for state in ('initial','after_five'):
 for stage in ('action_encoder','session_encoder','correct_padding_encoder','decoder'):
  for m in (128,256,512):
   selected=[r for r in rows if r['prefix'].startswith(state+'_') and ('_'+stage+'_layers') in r['prefix'] and r['features']==m]
   summary.append(dict(state=state,stage=stage,features=m,comparisons=len(selected),
    relative_l2=describe([r['error']['relative_l2'] for r in selected]),
    pair_ess_medians=describe([r['kernel_statistics']['feature_ess']['median'] for r in selected]),
    attention_tv=describe([r['kernel_statistics']['mean_attention_total_variation'] for r in selected])))
old=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/real_assist_five_updates_20261020_v1'
name='correct_padding_encoder_layers_0_attention'
a,b=[dict(np.load(old/f'prefix_{step:02d}_{name}.npz')) for step in (0,5)]
changes={}
for key in ('q','k','v'):
 x,y=a[key].astype(float),b[key].astype(float)
 cos=np.sum(x*y,-1)/np.maximum(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1),1e-30)
 changes[key]=dict(cosine=describe(cos),relative_change=float(np.linalg.norm(y-x)/np.linalg.norm(x)),
  norm_ratio=describe(np.linalg.norm(y,axis=-1)/np.maximum(np.linalg.norm(x,axis=-1),1e-30)))
deployed=[]
for path in sorted(D.glob('*_e*_*.npz')):
 with np.load(path) as z:
  feature,exact,stats=independent_attention(z['q'],z['k'],z['v'],z['projection'],z['query_mask'],z['key_mask'])
  deployed.append(dict(prefix=path.name,implementation_error=output_errors(z['actual'],feature,z['query_mask']),
   feature_error=output_errors(feature,exact,z['query_mask']),per_query_relative_l2=query_errors(feature,exact,z['query_mask']),
   kernel_statistics=stats))
result=dict(frozen_feature_grid=summary,saved_cpe_qkv_change=changes,deployed_projections=deployed,
 prepared_plan_sha256=sha256(D/'prepared_plan.json'), review_decision=json.loads((D/'review_decision.json').read_text()),
 maximum_independent_feature_kernel_error=max(r['implementation_error']['max_abs'] for r in deployed),
 decision='Proceed with authorized bounded diagnostic pilot only; original REVIEW_REQUIRED retained; no numerical correctness defect observed',
 limitations='six examples/first32 valid-position prefixes, not full attention contexts; fixed deployed projection can respond adversely to learned Q/K directions; finite-seed diagnostic variance is not statistical significance or proof of full-training quality')
path=Path(__file__).with_name('PHASE1_REVIEW_SUMMARY.json')
with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_feature_grid','deployed_projections','review_decision')},indent=2))
