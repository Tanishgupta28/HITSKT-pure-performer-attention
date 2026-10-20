"""Account for saved observations; never trains, evaluates a model or uses CUDA."""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from ktbench.favor_plus_pipeline import ROOT,sha256
from ktbench.favor_plus_pilot_recovery import digest,validate
from ktbench.favor_plus_assist20 import source_identity
import torch

HERE=Path(__file__).resolve().parent
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist20_recovery_20261020_v1'
plan=json.loads((RUN/'prepared_plan.json').read_text())
contract={**plan['contract_base'],'prepared_plan_sha256':sha256(RUN/'prepared_plan.json')}
assert not (RUN/'study_failure.json').exists()
assert json.loads((RUN/'study_result.json').read_text())['optimizer_updates']==20
assert source_identity()['source_sha256']==contract['source_sha256']
rows=[];stages=[];recoveries=[];monitor_rows=[];checkpoints=[];diagnostics=[];cached=[]
for stage in (5,10,20):
 directory=RUN/f'stage_{stage:03d}'
 result=json.loads((directory/'stage_result.json').read_text());assert result['status']=='PASS';stages.append(result)
 for path in sorted(directory.glob('update_*.json')):rows.append(json.loads(path.read_text()))
 verification=json.loads((RUN/f'verify_{stage:03d}'/'recovery_result.json').read_text())
 assert verification['status']=='PASS' and verification['optimizer_calls_during_recovery']==0
 recoveries.append(verification)
 path=RUN/'checkpoints'/f'checkpoint_{stage:03d}.pt'
 payload=torch.load(path,map_location='cpu',weights_only=True)
 check=validate(payload,contract)
 assert sha256(path)==result['checkpoint']['sha256']==verification['checkpoint_sha256']
 adam=payload['optimizer_state_dict']
 second_min=min(float(v['exp_avg_sq'].min()) for v in adam['state'].values())
 assert second_min>=0
 moment_stats=dict(second_moment_minimum=second_min,second_moment_maximum=max(float(v['exp_avg_sq'].max()) for v in adam['state'].values()),
  first_moment_global_l2=math.sqrt(sum(float(v['exp_avg'].double().square().sum()) for v in adam['state'].values())),
  param_groups=[{k:v for k,v in g.items() if k!='params'} for g in adam['param_groups']])
 checkpoints.append(dict(update=stage,sha256=sha256(path),bytes=path.stat().st_size,adam_statistics=moment_stats,
  projection_sha256=check['projection_sha256'],sampler_cursor=check['sampler_cursor'],
  adam_parameter_states=check['adam_parameter_states'],component_sha256=check['component_sha256']))
 del payload
 for path in sorted(directory.glob('diagnostics_result_*.json')):
  d=json.loads(path.read_text());entries=json.loads((directory/f"m{d['update']:03d}_diagnostics.json").read_text())
  norms={}
  for name in entries[0]:
   norms[name]={kind:dict(p95_across_six=[e[name]['norms_all_valid_tokens'][kind]['p95'] for e in entries],
    maximum_token_norm=max(e[name]['norms_all_valid_tokens'][kind]['maximum'] for e in entries)) for kind in ('query','key')}
  flags=sorted({flag for e in entries for r in e.values() for flag in r['review_flags']})
  diagnostics.append(dict(update=d['update'],aggregate=d['containment']['aggregate'],
    causal=d['causal'],containment_status=d['containment']['status'],norms=norms,review_flags=flags))
  if 'sampled_validation' in d:
   with np.load(directory/f"validation_predictions_{d['update']:03d}.npz") as z:
    prob=z['probability'].astype(float);y=z['label'].astype(int)
   pos=y==1;neg=y==0;pred=prob>=.5
   tp=int(np.sum(pred&pos));fp=int(np.sum(pred&neg));fn=int(np.sum(~pred&pos));tn=int(np.sum(~pred&neg))
   precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
   metric=dict(roc_auc=float((rankdata(prob)[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum())),
    accuracy=float(np.mean(pred==y)),precision=precision,recall=recall,
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.,mse=float(np.mean((prob-y)**2)))
   differences={k:abs(value-d['sampled_validation'][k]) for k,value in metric.items()}
   bce=float(-np.mean(y*np.log(prob)+(1-y)*np.log1p(-prob)))
   cached.append(dict(update=d['update'],reported_metrics=d['sampled_validation'],sampled_training_eval=d['sampled_training_eval'],
    independent_cached_metrics=metric,max_abs_metric_differences=differences,
    bce_from_float32_cached_probabilities=bce,bce_difference=abs(bce-d['sampled_validation']['loss']),
    positives=int(pos.sum()),negatives=int(neg.sum()),confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn)))
for path in RUN.glob('*/monitor_event_*.json'):
 event=json.loads(path.read_text())
 for row in event['snapshot']['holders']:
  assert row['verified_read_only'] and row['classification']=='trusted_read_only_monitor'
  monitor_rows.append(dict(evidence=str(path.relative_to(ROOT)),phase=event['phase'],identity=row))
rows.sort(key=lambda r:r['update']);diagnostics.sort(key=lambda d:d['update']);cached.sort(key=lambda d:d['update'])
assert [r['update'] for r in rows]==list(range(1,21))
assert all(r['gradients_finite'] and r['adam_and_model_finite'] and r['parameter_update_l2']>0 for r in rows)
assert all(c['projection_sha256']==checkpoints[0]['projection_sha256'] for c in checkpoints)
assert all(d['causal']['status']=='PASS' and d['containment_status']=='CONTINUE_BOUNDED_DIAGNOSTIC_ONLY' for d in diagnostics)
assert [d['update'] for d in cached]==[0,20]
assert len({r['target_session'] for r in contract['exact_training_and_probe_samples'][:20]})==20
for a,b in ((5,6),(10,11)):
 assert json.loads((RUN/f'stage_{10 if a==5 else 20:03d}'/'actual_next_step_matches_checkpoint.json').read_text())['update']==b
# Strong preservation check: old153-file failed study + old70-file CUDA/five pilot.
preservation=[]
for inventory_name in ('assist_bounded_v1/NEW_EXPERIMENT_ARTIFACT_INVENTORY.json','first_real_cuda_pilot_20261020_v1/artifact_inventory.json'):
 inventory=json.loads((HERE.parent/inventory_name).read_text());fail=[]
 for relative,entry in inventory['files'].items():
  path=ROOT/relative
  if not path.exists() or path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:fail.append(relative)
 assert not fail
 preservation.append(dict(inventory=str((HERE.parent/inventory_name).relative_to(ROOT)),files=len(inventory['files']),failures=fail))
# Repeat the completed14 training observations exactly without retrying that run.
old=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist_bounded100_20261020_v1_s020'
replicated=[]
for row in rows[:14]:
 prev=json.loads((old/f"update_{row['update']:03d}.json").read_text())
 replicated.append(dict(update=row['update'],loss_abs_difference=abs(row['training_bce']-prev['training_bce']),
  gradient_l2_abs_difference=abs(row['gradient_l2']-prev['gradient_l2']),parameter_delta_abs_difference=abs(row['parameter_update_l2']-prev['parameter_delta_l2'])))
resources=[s['resources'] for s in stages]+[v['resources'] for v in recoveries]
result=dict(kind='assist20_verified_observation_summary',status='20_UPDATES_AND_RECOVERY_PASS;APPROXIMATION_REVIEW_REMAINS',
 base_commit='8d4175b88bb73bbf42ef220a41abada1cf6c2bf3',executed_source_commit='1595785',
 optimizer_updates=20,optimizer_calls_in_all_recovery_processes=0,supervised_training_targets=sum(r['supervised_targets'] for r in rows),
 training_loss_first=rows[0]['training_bce'],training_loss_last=rows[-1]['training_bce'],
 training_bce_target_weighted=sum(r['training_bce']*r['supervised_targets'] for r in rows)/sum(r['supervised_targets'] for r in rows),
 all_recorded_gradient_model_adam_finite=True,all_parameter_deltas_positive=True,
 gradient_l2_range=[min(r['gradient_l2'] for r in rows),max(r['gradient_l2'] for r in rows)],
 rows=rows,checkpoints=checkpoints,diagnostics=diagnostics,validation=cached,
 peak_cuda_allocated_bytes=max(s['peak_allocated_bytes'] for s in resources),
 peak_cuda_reserved_bytes=max(s['peak_reserved_bytes'] for s in resources),
 peak_process_rss_bytes=max(s['host']['peak_process_rss_bytes'] for s in resources),
 minimum_observed_cgroup_headroom_bytes=min(s['host']['cgroup_headroom_bytes'] for s in resources),
 synchronized_training_seconds=sum(r['synchronized_training_seconds'] for r in rows),
 update_seconds_mean=float(np.mean([r['synchronized_training_seconds'] for r in rows])),
 update_seconds_max=max(r['synchronized_training_seconds'] for r in rows),
 training_stage_execution_seconds=sum(s['stage_execution_seconds'] for s in stages),
 recovery_execution_seconds=sum(s['recovery_execution_seconds'] for s in recoveries),
 campaign_elapsed_seconds=recoveries[-1]['resources']['campaign_elapsed_seconds'],
 monitoring=dict(events=len(list(RUN.glob('*/monitor_event_*.json'))),unique_pids=sorted({r['identity']['pid'] for r in monitor_rows}),
  observed_holders=len(monitor_rows),all_observed_others_verified_read_only=True,rows=monitor_rows,
  old2751270_identity='UNKNOWN; cannot retrospectively infer from current monitoring'),
 preservation=preservation,first14_comparison=replicated,
 test_evaluations=0,full_training_ready=False,
 next_decision='GO for separately approved50-TOTAL-update diagnostic continuation (30more); REVIEW approximation; NO-GO full training',
 limitations='sampled validation only; single seed42;6 train diagnostic cases/first32-position exact prefixes; original batch/history backward shapes unprofiled; PyTorch peaks exclude driver/continuous NVML peak; source/data/checkpoint access needed to reproduce')
with (HERE/'VERIFICATION_SUMMARY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
small={k:v for k,v in result.items() if k not in ('rows','checkpoints','diagnostics','validation','monitoring','first14_comparison')}
print(json.dumps(small,indent=2))
