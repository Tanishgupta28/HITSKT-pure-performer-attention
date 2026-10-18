#!/usr/bin/env python3
"""All-target diagnostics of saved CPU scores; no model inference or CUDA access."""
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.audit_hitskt_replay import independent_metrics,METRIC_ABSOLUTE_TOLERANCE
from scripts.diagnose_hitskt_common import DATASETS,OUT,now,old_evaluation,sha,write


def confusion_metrics(c):
    tp,fp,tn,fn=(c[k] for k in ['tp','fp','tn','fn'])
    return dict(accuracy=(tp+tn)/(tp+fp+tn+fn),precision=tp/(tp+fp),
                recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn))


def infer_confusion(metrics):
    """Infer integer counts from rounded aggregate metrics; verify uniqueness.

    Eight ULP here allows representation error in equivalent metric formulas;
    it is NOT the acceptance tolerance for reproduced experimental metrics.
    """
    n=metrics['targets'];p=metrics['precision'];r=metrics['recall'];a=metrics['accuracy']
    estimate=n*(1-a)/(1/p+1/r-2)
    candidates=[]
    for tp in range(round(estimate)-2,round(estimate)+3):
        fp=round(tp*(1/p-1));fn=round(tp*(1/r-1));tn=n-tp-fp-fn
        if min(tp,fp,fn,tn)<0:continue
        c=dict(tp=tp,fp=fp,tn=tn,fn=fn)
        calculated=confusion_metrics(c)
        if all(abs(calculated[k]-metrics[k])<=8*math.ulp(metrics[k]) for k in calculated):candidates.append(c)
    if len(candidates)!=1:raise ValueError(('No unique confusion matrix',candidates))
    return candidates[0]


def disagreement_bounds(first,second):
    assert first['tp']+first['fn']==second['tp']+second['fn']
    assert first['tn']+first['fp']==second['tn']+second['fp']
    positive=first['tp']+first['fn'];negative=first['tn']+first['fp']
    lo=abs(first['tp']-second['tp'])+abs(first['fp']-second['fp'])
    hi=min(first['tp']+second['tp'],2*positive-first['tp']-second['tp'])
    hi+=min(first['fp']+second['fp'],2*negative-first['fp']-second['fp'])
    return dict(minimum=lo,maximum=hi,parity=lo%2,
        net_correct_change=second['tp']+second['tn']-first['tp']-first['tn'],
        exact_total_disagreements='NOT VERIFIED: historical per-target scores/classifications absent',
        scope='Sharp bounds using confusion counts only; net correct change is not total changed classifications')


def diagnose(dataset):
    old=old_evaluation(dataset);cache=Path(old['prediction_artifacts_local'])
    for name,value in old['prediction_sha256'].items():assert sha(cache/name)==value
    p=np.load(cache/'probability.npy',mmap_mode='r');z=np.load(cache/'logit.npy',mmap_mode='r');y=np.load(cache/'label.npy',mmap_mode='r')
    assert len(y)==old['expected_targets']
    zz=z.astype(np.float64);pp=p.astype(np.float64)
    p64=np.empty_like(zz);positive=zz>=0
    p64[positive]=1/(1+np.exp(-zz[positive]));e=np.exp(zz[~positive]);p64[~positive]=e/(1+e)
    cpu=old['recalculated_metrics']['confusion'];historical=infer_confusion(old['stored_metrics'])
    assert historical['tp']+historical['fn']==int(y.sum())
    bounds=disagreement_bounds(historical,cpu)
    values=independent_metrics(p64,y,z)
    loss32=float(F.binary_cross_entropy_with_logits(torch.tensor(np.asarray(z)),torch.tensor(np.asarray(y),dtype=torch.float32)).item())
    metric64=old['recalculated_metrics']['bce_float64'];batch32=old['recalculated_metrics']['historical_batch_weighted_bce']
    result=dict(dataset=dataset,status='COMPLETE',targets=len(y),historical_confusion_inferred=historical,cpu_confusion=cpu,
        original_cuda_vs_cpu_classification_bounds=bounds,
        all_target_probability_rounding=dict(maximum_absolute_probability_difference=float(np.max(np.abs(pp-p64))),
            changed_threshold_classifications=int(np.count_nonzero((pp>=.5)!=(p64>=.5))),
            logit_sign_vs_cached_probability_changes=int(np.count_nonzero((zz>=0)!=(pp>=.5))),
            p32_equal_half=int(np.count_nonzero(pp==.5)),logits_equal_zero=int(np.count_nonzero(zz==0)),
            minimum_absolute_logit=float(np.min(np.abs(zz))),minimum_probability_margin=float(np.min(np.abs(pp-.5))),
            probability_margin_counts={str(b):int(np.count_nonzero(np.abs(pp-.5)<=b)) for b in [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]},
            distinction='Same cached FP32 logits; changes sigmoid/metric precision, not FP64 model inference or historical CUDA replay'),
        metrics_float64_sigmoid_of_cached_float32_logits=values,
        bce_decomposition=dict(historical_cuda=old['stored_metrics']['loss'],cached_cpu_historical_batch_weighted_fp32=batch32,
            cached_cpu_float64_logit_bce=metric64,cached_cpu_single_global_fp32_bce=loss32,
            cpu_batch32_minus_float64=batch32-metric64,original_cuda_minus_cpu_batch32=old['stored_metrics']['loss']-batch32,
            original_cuda_minus_cpu_float64=old['stored_metrics']['loss']-metric64,
            conclusion='Changing metric accumulation precision alone does not remove the historical/CPU discrepancy'),
        maximum_original_cuda_logit_difference='NOT VERIFIED: no historical per-target logits',
        maximum_original_cuda_probability_difference='NOT VERIFIED: no historical per-target probabilities',
        stored_vs_cpu_comparison=old['comparison'],unchanged_absolute_metric_tolerance=METRIC_ABSOLUTE_TOLERANCE,
        checkpoint_sha256=old['historical_artifact_sha256']['best_model.pt'],selected_epoch=old['best_checkpoint_epoch'],
        source_cpu_manifest_sha256=sha(Path('reports/independent_correctness_audit')/(f'{dataset}_evaluation_manifest.json' if dataset=='assist2017' else f'{dataset}_evaluation_guarded_manifest.json')))
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    try:Path('/proc/self/oom_score_adj').write_text('1000')
    except PermissionError:pass
    torch.set_num_threads(1)
    records={}
    for d in DATASETS:
        records[d]=diagnose(d)
        print(d,records[d]['original_cuda_vs_cpu_classification_bounds'],flush=True)
    write(OUT/'cached_metrics_diagnostics.json',dict(utc=now(),status='COMPLETE',records=records,implementation_sha256=sha(__file__)))


if __name__=='__main__':main()
