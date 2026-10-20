"""Read-only accounting and independent cached validation metric check."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from ktbench.favor_plus_pipeline import ROOT,sha256
HERE=Path(__file__).resolve().parent
D=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist_bounded100_20261020_v1_s020'
rows=[json.loads(p.read_text()) for p in sorted(D.glob('update_*.json'))]
base=json.loads((D/'milestone_000.json').read_text())
fail=json.loads((D/'stage_failure.json').read_text())
with np.load(D/'m000_validation_predictions.npz') as f:
 p,y=f['probability'].astype(float),f['label'].astype(int)
pos=y==1;negative=y==0;prediction=p>=.5
tp=int(np.sum(prediction&pos));fp=int(np.sum(prediction&negative));fn=int(np.sum(~prediction&pos));tn=int(np.sum(~prediction&negative))
auc=float((rankdata(p)[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*negative.sum()))
prec=tp/(tp+fp) if tp+fp else 0.
rec=tp/(tp+fn) if tp+fn else 0.
independent=dict(roc_auc=auc,accuracy=float(np.mean(prediction==y)),precision=prec,recall=rec,
 f1=2*prec*rec/(prec+rec) if prec+rec else 0.,mse=float(np.mean((y-p)**2)),
 bce_from_cached_float32_probabilities=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))))
compare={k:abs(v-base['sampled_validation'][k]) for k,v in independent.items() if k in base['sampled_validation']}
assert len(rows)==fail['optimizer_updates']==fail['optimizer_step_calls']==14
assert [r['update'] for r in rows]==list(range(1,15))
assert all(r['gradients_finite'] and r['model_adam_finite'] and r['parameter_delta_l2']>0 for r in rows)
assert not any(D.glob('*.pt')) and not list(D.glob('milestone_020.json')) and not (D/'_BOUNDED_STAGE_COMPLETE').exists()
# Previous70 pilot/evidence files are small enough to rehash without repeating
# the full375-original dataset/research audit.
inventory=json.loads((ROOT/'reports/favor_plus_v1/first_real_cuda_pilot_20261020_v1/artifact_inventory.json').read_text())
prior_failures=[]
for name,entry in inventory['files'].items():
 path=ROOT/name
 if not path.exists() or path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
  prior_failures.append(name)
assert not prior_failures
result=dict(overall_status='RESOURCE_ABORTED;scientific larger-pilot milestones NOT VERIFIED',
 stopped_at=fail,completed_updates=len(rows),supervised_training_targets=sum(r['supervised_targets'] for r in rows),
 training_bce_first=rows[0]['training_bce'],training_bce_last=rows[-1]['training_bce'],
 training_bce_target_weighted=sum(r['training_bce']*r['supervised_targets'] for r in rows)/sum(r['supervised_targets'] for r in rows),
 gradient_l2_range=[min(r['gradient_l2'] for r in rows),max(r['gradient_l2'] for r in rows)],
 all_recorded_gradients_adam_finite=True,all_recorded_parameters_changed=True,
 synchronized_training_interval_seconds=sum(r['training_seconds'] for r in rows),
 per_update_seconds_mean=float(np.mean([r['training_seconds'] for r in rows])),
 per_update_seconds_max=max(r['training_seconds'] for r in rows),
 peak_recorded_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),
 host_peak_at_baseline_bytes=base['host_memory']['peak_process_rss_bytes'],
 final_process_peak_rss=None,final_peak_cuda_reserved_bytes=None,
 memory_limitations='failure manifest omitted final RSS/reserved high-water; baseline values are lower bounds; per-update allocated maxima saved; NVML peak not sampled',
 milestone_0=base,milestones_20_50_100='NOT REACHED / NOT VERIFIED',
 independent_cached_validation=independent,cached_metric_max_abs_differences=compare,
 cached_bce_difference_from_logit_loss=abs(independent['bce_from_cached_float32_probabilities']-base['sampled_validation']['loss']),
 cache_limit='float32 probabilities support metric/count check; full logit-based BCE independently replay requires logits/checkpoint',
 old_pilot_preservation=dict(files_checked=len(inventory['files']),failures=prior_failures,
  prior_inventory_sha256=sha256(ROOT/'reports/favor_plus_v1/first_real_cuda_pilot_20261020_v1/artifact_inventory.json')),
 original_attention_training_core_sha256=json.loads((ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist_bounded_review_20261020_v2/prepared_plan.json').read_text())['source']['source_sha256'],
 checkpoint_14='NOT SAVED; abort preceded first20 checkpoint; no optimizer reset/recovery/replay',
 test_evaluations=0,full_training_ready=False,next_stage='REVIEW; identify GPU-holding conflict and improve failure telemetry before separately authorized new bounded attempt; NO-GO full training')
with (HERE/'ATTEMPT_SUMMARY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('milestone_0','original_attention_training_core_sha256')},indent=2))
