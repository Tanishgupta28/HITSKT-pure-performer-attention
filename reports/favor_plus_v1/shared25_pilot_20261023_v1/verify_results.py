"""CPU-only verification of saved shared pilot evidence. No new optimization."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from ktbench import favor_plus_shared25 as p
from ktbench import favor_plus_shared_gpu as shared
from ktbench.favor_plus_continuation_recovery import validate,optimizer_groups_equal

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist25_shared_cuda_20261023_v1'


def load(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def cached_metrics(probability,label):
    prob=probability.astype(np.float64);truth=label.astype(np.int64);pred=prob>=.5
    tp=int(np.sum(pred & (truth==1)));fp=int(np.sum(pred & (truth==0)))
    fn=int(np.sum(~pred & (truth==1)));n=len(truth);positives=int(truth.sum());negatives=n-positives
    order=np.argsort(prob,kind='stable');ranks=np.empty(n);i=0
    while i<n:
        j=i+1
        while j<n and prob[order[j]]==prob[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return dict(roc_auc=float((ranks[truth==1].sum()-positives*(positives+1)/2)/(positives*negatives)),
        accuracy=float(np.mean(pred==truth)),precision=precision,recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        mse=float(np.mean((prob-truth)**2)),
        bce_from_rounded_probability=float(-np.mean(truth*np.log(prob)+(1-truth)*np.log1p(-prob))))


plan_path=RUN/'prepared_plan.json';plan,c,_,_=p.load_plan(plan_path)
parent=torch.load(p.PARENT_CHECKPOINT,map_location='cpu',weights_only=True)
child_path=RUN/'checkpoints/checkpoint_025.pt';child=torch.load(child_path,map_location='cpu',weights_only=True)
p.parent_recovery.validate(parent,c['parent_contract']);integrity=validate(child,c)
assert sha(p.PARENT_CHECKPOINT)==p.PARENT_SHA256
assert optimizer_groups_equal(child['optimizer_state_dict']['param_groups'],parent['optimizer_state_dict']['param_groups'])
assert child['projection_sha256']==parent['projection_sha256']
assert all(float(v['step'])==25 for v in child['optimizer_state_dict']['state'].values())
min_second=min(float(v['exp_avg_sq'].min()) for v in child['optimizer_state_dict']['state'].values())
max_second=max(float(v['exp_avg_sq'].max()) for v in child['optimizer_state_dict']['state'].values())
delta=math.hypot(*(float((child['state_dict'][k].double()-v.double()).norm()) for k,v in parent['state_dict'].items()))
assert delta>0 and min_second>=0
done=load(RUN/'study_result.json');recovery=load(RUN/'verify_025/recovery_result.json')
assert done['optimizer_updates']==25 and done['additional_optimizer_calls']==5
assert recovery['status']=='PASS' and recovery['optimizer_calls_during_recovery']==0
assert recovery['eval_prediction_max_abs_difference']==0 and recovery['next_train_forward_loss_all_gradient_digest']=='EXACT'
assert sha(child_path)==done['checkpoint_sha256']==recovery['checkpoint_sha256']
assert not (RUN/'study_failure.json').exists()
assert sorted(x.name for x in (RUN/'checkpoints').glob('*.pt'))==['checkpoint_025.pt']
rows=[load(path) for path in sorted((RUN/'stage_025').glob('update_*.json'))]
assert [r['update'] for r in rows]==[21,22,23,24,25]
assert all(r['gradients_finite'] and r['adam_and_model_finite'] and r['parameter_update_l2']>0 for r in rows)
assert all(r['sample']['split']=='train' and r['padded_tokens']<=2048 for r in rows)
baseline=load(RUN/'stage_025/parent20_validation_replay.json');assert baseline['status']=='PASS'
assert load(RUN/'stage_025/actual_next_step_matches_checkpoint_021.json')['status']=='PASS'
diag={step:load(RUN/f'stage_025/diagnostics_result_{step:03d}.json') for step in (20,25)}
stages={step:load(RUN/f'stage_025/m{step:03d}_diagnostics.json') for step in (20,25)}
approx={}
for name in stages[20][0]:
    info={}
    for step in (20,25):
        values=[r[name] for r in stages[step]]
        a=diag[step]['containment']['aggregate'][name]
        info[str(step)]=dict(median_relative_l2=a['relative_l2']['median'],max_relative_l2=a['relative_l2']['maximum'],
            prefixes_above_point20=sum(x['feature_versus_exact_softmax']['relative_l2']>.2 for x in values),
            median_pair_ess=a['pair_ess_medians']['median'],minimum_individual_pair_ess=min(x['kernel_statistics']['feature_ess']['minimum'] for x in values),
            max_feature_kernel_abs_error=a['max_implementation_abs'],max_qk_p95_growth_from_initial=a['max_p95_growth_ratio'])
    info['max_qk_p95_growth_20_to25']=max(stages[25][i][name]['norms_all_valid_tokens'][q]['p95']/
        stages[20][i][name]['norms_all_valid_tokens'][q]['p95'] for i in range(6) for q in ('query','key'))
    approx[name]=info
assert all(diag[step]['containment']['status']!='STOP' for step in diag)
metric_checks={}
for step in (20,25):
    with np.load(RUN/f'stage_025/validation_predictions_{step:03d}.npz') as z:
        observed=cached_metrics(z['probability'],z['label'])
    stored=diag[step]['sampled_validation']
    max_diff=max(abs(observed[k]-stored[k]) for k in ('roc_auc','accuracy','precision','recall','f1','mse'))
    assert max_diff<1e-12
    metric_checks[str(step)]=dict(max_score_metric_difference=max_diff,
        rounded_probability_bce_difference=observed['bce_from_rounded_probability']-stored['loss'])
resources={}
for phase in ('stage_025','verify_025'):
    samples=[load(x) for x in (RUN/phase).glob('allocator_sample_*.json')]
    all_obs=[load(x) for x in (RUN/phase).glob('resources_*.json')]
    assert samples and all(x['decision']['status']!='STOP' and not x['pid_mapping_proven'] for x in all_obs)
    resources[phase]=dict(peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in samples),
        peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in samples),
        max_sampled_nvml_global_used_bytes=max(x['observation']['device']['used_bytes'] for x in all_obs),
        minimum_observed_free_bytes=min(x['observation']['device']['free_bytes'] for x in all_obs),
        max_global_used_minus_own_reserved_bytes=max(x['global_used_minus_own_reserved_bytes'] for x in samples),
        max_global_growth_minus_own_reserved_bytes=max(x['global_growth_minus_own_reserved_bytes'] for x in samples),
        peak_process_rss_bytes=max(x['host']['peak_process_rss_bytes'] for x in all_obs),
        minimum_host_headroom_bytes=min(x['host']['cgroup_headroom_bytes'] for x in all_obs),
        resource_observations=len(all_obs),attributed_process_total_vram='NOT MEASURABLE: host PID mapping unresolved')
    assert resources[phase]['peak_allocated_bytes']<=shared.POLICY['allocator_cap_bytes']
    assert resources[phase]['peak_reserved_bytes']<=shared.POLICY['reserved_cap_bytes']
    assert resources[phase]['minimum_observed_free_bytes']>=shared.POLICY['runtime_free_bytes']
    assert resources[phase]['peak_process_rss_bytes']<=shared.POLICY['host_process_rss_bytes']
    assert resources[phase]['minimum_host_headroom_bytes']>=shared.POLICY['host_headroom_bytes']
for name,expected in c['source_sha256'].items():assert sha(ROOT/name)==expected,name
release=load(HERE/'resource_release.json')
assert release['light_hold_sha256']=='65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234'
assert not release['observation']['local']['light_processes']
tests=ET.parse(HERE/'cpu_tests.xml').getroot().findall('.//testcase')
assert len(tests)==75 and not any(any(t.find(x) is not None for x in ('failure','error','skipped')) for t in tests)
stage=load(RUN/'stage_025/stage_result.json')
targets=sum(r['supervised_targets'] for r in rows);intervals=sum(r['synchronized_training_seconds'] for r in rows)
result=dict(status='PASS_SHARED_ENGINEERING;APPROXIMATION_REVIEW_REQUIRED',verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    actual_total_optimizer_updates=25,actual_additional_optimizer_calls=5,recovery_optimizer_calls=0,
    checkpoint25_sha256=sha(child_path),checkpoint25_bytes=child_path.stat().st_size,parent20_sha256=sha(p.PARENT_CHECKPOINT),
    checkpoint_integrity=integrity,adam_second_moment_min=min_second,adam_second_moment_max=max_second,
    parameter_state_delta_l2_20_to25=delta,model_adam_and_gradients_finite=True,cuda_or_memory_errors_observed=0,
    training_rows=rows,weighted_training_bce=sum(r['training_bce']*r['supervised_targets'] for r in rows)/targets,
    supervised_training_targets=targets,synchronized_update_intervals_seconds=intervals,
    synchronized_interval_updates_per_second=5/intervals,synchronized_interval_targets_per_second=targets/intervals,
    stage_execution_seconds=stage['stage_execution_seconds'],recovery_execution_seconds=recovery['recovery_execution_seconds'],
    resource_summary=resources,approximation=approx,sampled_validation={str(k):diag[k]['sampled_validation'] for k in diag},
    cached_score_metric_checks=metric_checks,cpu_tests=dict(passed=75,new_pilot=20,existing_shared=55,failures=0,skips=0),
    all_33_source_files_unchanged=True,pid_mapping_proven=False,resource_budget_increased=False,
    test_evaluations=0,automatic_retries=0,other_process_signals=0,full_training_ready=False,
    next_recommendation='GO with REVIEW retained for separately approved shared25→50 continuation from NEW25 checkpoint; no execution now')
with (HERE/'VERIFICATION_SUMMARY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('training_rows','checkpoint_integrity')},indent=2))
