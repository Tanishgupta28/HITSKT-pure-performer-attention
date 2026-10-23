"""Independent CPU/NumPy replay of saved25/50 Action and CPE prefixes only.

No model construction, optimizer, data loading, new random features or CUDA.
Direct pairwise positive-feature formula; no production scan/diagnostic helper.
"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1'
FOLDERS={25:RUN/'stage_030',50:RUN/'stage_050'}


def output(q,k,v,projection,qm,km):
    q,k,v,w=[np.asarray(x,dtype=np.float64) for x in (q,k,v,projection)]
    feature=np.zeros_like(v);exact=np.zeros_like(v);d=q.shape[-1];m=w.shape[1]
    for h in range(q.shape[1]):
        for i in range(q.shape[2]):
            if not qm[0,i]:continue
            js=np.flatnonzero(km[0,:i+1])
            if not len(js):continue
            logkernels=[];dots=[]
            x=q[0,h,i]/d**.25
            for j in js:
                y=k[0,h,j]/d**.25
                terms=w[h]@(x+y)-(x@x+y@y)/2
                logkernels.append(np.logaddexp.reduce(terms)-np.log(m))
                dots.append(q[0,h,i]@k[0,h,j]/np.sqrt(d))
            for target,scores in ((feature,logkernels),(exact,dots)):
                a=np.asarray(scores);a=np.exp(a-a.max());a/=a.sum()
                target[0,h,i]=a@v[0,h,js]
    return feature,exact


def review(arrays):
    feature,exact=output(*(arrays[k] for k in ('q','k','v','projection','query_mask','key_mask')))
    eligible=np.broadcast_to(arrays['query_mask'][:,None,:,None],feature.shape)
    reference=exact[eligible];delta=(feature-exact)[eligible]
    impl=float(np.max(np.abs((arrays['actual']-feature)[eligible])))
    assert impl<5e-5
    perquery=np.linalg.norm(feature-exact,axis=-1)/np.maximum(np.linalg.norm(exact,axis=-1),1e-30)
    valid=np.broadcast_to(arrays['query_mask'][:,None,:],perquery.shape)
    return dict(feature_vs_softmax_relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
        production_vs_feature_max_abs=impl,head_query_relative_l2_p95=float(np.percentile(perquery[valid],95)),
        head_query_relative_l2_max=float(np.max(perquery[valid])),valid_head_queries=int(valid.sum()))


result={}
for name in ('action_encoder.layers.0.attention','correct_padding_encoder.layers.0.attention'):
    cases=[]
    for i in range(6):
        arrays={};reviews={}
        for step,folder in FOLDERS.items():
            file=folder/f'm{step:03d}_sample{i}_e0_{name.replace(".","_")}.npz'
            with np.load(file) as z:arrays[step]={k:z[k] for k in z.files}
            reviews[step]=review(arrays[step])
            stored=json.loads((folder/f'm{step:03d}_diagnostics.json').read_text())[i][name]
            assert abs(reviews[step]['feature_vs_softmax_relative_l2']-stored['feature_versus_exact_softmax']['relative_l2'])<1e-12
        for key in ('query_mask','key_mask','projection'):assert np.array_equal(arrays[25][key],arrays[50][key])
        drift={}
        for key,mask in (('q','query_mask'),('k','key_mask'),('v','key_mask')):
            a,b=[arrays[step][key].astype(np.float64) for step in (25,50)]
            valid=np.broadcast_to(arrays[25][mask][:,None,:],a.shape[:3]);a=a[valid];b=b[valid]
            cosine=np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-30)
            drift[key]=dict(relative_l2=float(np.linalg.norm(b-a)/np.linalg.norm(a)),
                minimum_direction_cosine=float(cosine.min()),median_direction_cosine=float(np.median(cosine)))
        cases.append(dict(case=i,update25=reviews[25],update50=reviews[50],representation_drift25_to50=drift))
    result[name]=dict(cases=cases,projection_unchanged=True,masks_unchanged=True,
        interpretation='Descriptive six correlated training prefixes; no significance or generalization test. Fixed random projection, evolving learned Q/K/V; no new projection draw or architecture change.')
summary=dict(status='PASS_INDEPENDENT_DOUBLE_PAIRWISE_REPLAY;APPROXIMATION_REVIEW_RETAINED',
    cpu_only=True,new_optimizer_calls=0,new_cuda_calls=0,prefixes_replayed=24,stages=result,
    root_cause='Persistent finite-feature softmax approximation/alignment error, not float32 implementation error in these prefixes. Learned-state drift changes error without requiring exploding norms. CPE worsening warrants additional review; high-norm/long-history adequacy not established.')
with (HERE/'SAVED_PREFIX_REVIEW.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
for name,info in result.items():
    cases=info['cases']
    print(name,'error25→50',np.median([c['update25']['feature_vs_softmax_relative_l2'] for c in cases]),
        np.median([c['update50']['feature_vs_softmax_relative_l2'] for c in cases]),
        'max_impl',max(c[f'update{s}']['production_vs_feature_max_abs'] for c in cases for s in (25,50)),
        'max_query_tail50',max(c['update50']['head_query_relative_l2_max'] for c in cases),
        'QKV_max_drift',{k:max(c['representation_drift25_to50'][k]['relative_l2'] for c in cases) for k in ('q','k','v')})
