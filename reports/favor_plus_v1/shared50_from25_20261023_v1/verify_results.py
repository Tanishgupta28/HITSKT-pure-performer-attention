"""Read-only CPU verification of saved evidence; never initializes CUDA or Adam."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from ktbench import favor_plus_shared25_to50 as p
from ktbench import favor_plus_shared_gpu as shared
from ktbench.favor_plus_continuation_recovery import validate,optimizer_groups_equal

ROOT=p.ROOT
HERE=Path(__file__).resolve().parent
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1'
OLD=p.PARENT_PLAN.parent


def load(path):return json.loads(path.read_text())


def sha(path):return p.sha256(path)


def cached_metrics(probability,label):
    prob=probability.astype(np.float64);truth=label.astype(np.int64);pred=prob>=.5
    tp=int(np.sum(pred & (truth==1)));fp=int(np.sum(pred & (truth==0)));fn=int(np.sum(~pred & (truth==1)))
    positives=int(truth.sum());negatives=len(truth)-positives
    order=np.argsort(prob,kind='stable');ranks=np.empty(len(truth));i=0
    while i<len(truth):
        j=i+1
        while j<len(truth) and prob[order[j]]==prob[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return dict(roc_auc=float((ranks[truth==1].sum()-positives*(positives+1)/2)/(positives*negatives)),
        accuracy=float(np.mean(pred==truth)),precision=precision,recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        mse=float(np.mean((prob-truth)**2)),
        bce_from_rounded_probability=float(-np.mean(truth*np.log(prob)+(1-truth)*np.log1p(-prob))))


assert not torch.cuda.is_initialized()
plan,c,_,_=p.load_plan(RUN/'prepared_plan.json')
parent=torch.load(p.PARENT_CHECKPOINT,map_location='cpu',weights_only=True)
validate(parent,c['parent_contract']);assert sha(p.PARENT_CHECKPOINT)==p.PARENT_SHA256
assert c['config']==parent['contract']['config'] and c['dataset_identity']==parent['contract']['dataset_identity']
assert c['exact_training_and_probe_samples'][:26]==parent['sampler']['exact_samples']
assert len(c['exact_training_and_probe_samples'])==51
done=load(RUN/'study_result.json')
assert done['status']=='FIFTY_TOTAL_SHARED_FROM25_COMPLETED'
assert done['optimizer_updates']==50 and done['additional_optimizer_calls']==25
assert not (RUN/'study_failure.json').exists()
assert sorted(x.name for x in (RUN/'checkpoints').glob('*.pt'))==[
    f'checkpoint_{i:03d}.pt' for i in (30,35,40,45,50)]
checkpoint_checks={}
for step in (30,35,40,45,50):
    path=RUN/f'checkpoints/checkpoint_{step:03d}.pt'
    child=torch.load(path,map_location='cpu',weights_only=True);integrity=validate(child,c)
    assert child['optimizer_updates']==step and child['additional_optimizer_calls']==step-20
    assert optimizer_groups_equal(child['optimizer_state_dict']['param_groups'],parent['optimizer_state_dict']['param_groups'])
    assert child['projection_sha256']==parent['projection_sha256']
    assert all(float(v['step'])==step for v in child['optimizer_state_dict']['state'].values())
    delta=math.hypot(*(float((child['state_dict'][k].double()-v.double()).norm()) for k,v in parent['state_dict'].items()))
    assert delta>0
    checkpoint_checks[str(step)]=dict(sha256=sha(path),bytes=path.stat().st_size,integrity=integrity,
        parameter_state_delta_l2_from25=delta,
        adam_second_moment_min=min(float(v['exp_avg_sq'].min()) for v in child['optimizer_state_dict']['state'].values()),
        adam_second_moment_max=max(float(v['exp_avg_sq'].max()) for v in child['optimizer_state_dict']['state'].values()))
    del child
assert done['checkpoint_sha256']==checkpoint_checks['50']['sha256']
recovery={};stageresults={}
for step in (30,40,50):
    recovery[step]=load(RUN/f'verify_{step:03d}/recovery_result.json')
    r=recovery[step]
    assert r['status']=='PASS' and r['optimizer_calls_during_recovery']==0
    assert r['eval_prediction_max_abs_difference']==0 and r['next_train_forward_loss_all_gradient_digest']=='EXACT'
    assert r['checkpoint_sha256']==checkpoint_checks[str(step)]['sha256']
    stageresults[step]=load(RUN/f'stage_{step:03d}/stage_result.json')
    assert stageresults[step]['status']=='PASS' and stageresults[step]['optimizer_updates']==step
    incoming=load(RUN/f'stage_{step:03d}/incoming_recovery_and_migration.json')
    assert incoming['status']=='PASS' and incoming['unchanged_model_adam_rng_projections']
rows=[load(path) for path in sorted(RUN.glob('stage_*/update_*.json'))]
assert [r['update'] for r in rows]==list(range(26,51))
assert all(r['gradients_finite'] and r['adam_and_model_finite'] and r['parameter_update_l2']>0 for r in rows)
assert all(r['sample']['split']=='train' and r['padded_tokens']<=2048 and r['additional_optimizer_calls']==r['update']-25 for r in rows)
assert load(RUN/'stage_030/parent25_validation_replay.json')['status']=='PASS'
for step in (26,31,36,41,46):
    stage=30 if step<=30 else 40 if step<=40 else 50
    assert load(RUN/f'stage_{stage:03d}/actual_next_step_matches_checkpoint_{step:03d}.json')['status']=='PASS'

folders={20:OLD/'stage_025',25:RUN/'stage_030',30:RUN/'stage_030',40:RUN/'stage_040',50:RUN/'stage_050'}
diag={step:load(folder/f'diagnostics_result_{step:03d}.json') for step,folder in folders.items()}
stages={step:load(folder/f'm{step:03d}_diagnostics.json') for step,folder in folders.items()}
approx={}
for name in stages[25][0]:
    info={}
    for step in folders:
        values=[r[name] for r in stages[step]];a=diag[step]['containment']['aggregate'][name]
        info[str(step)]=dict(median_relative_l2=a['relative_l2']['median'],max_relative_l2=a['relative_l2']['maximum'],
            case_errors=[x['feature_versus_exact_softmax']['relative_l2'] for x in values],
            prefixes_above_point20=sum(x['feature_versus_exact_softmax']['relative_l2']>.2 for x in values),
            median_pair_ess=a['pair_ess_medians']['median'],
            minimum_individual_pair_ess=min(x['kernel_statistics']['feature_ess']['minimum'] for x in values),
            max_feature_kernel_abs_error=a['max_implementation_abs'],max_qk_p95_growth_from_initial=a['max_p95_growth_ratio'],
            max_qk_p95_growth_from25=max(stages[step][i][name]['norms_all_valid_tokens'][q]['p95']/
                stages[25][i][name]['norms_all_valid_tokens'][q]['p95'] for i in range(6) for q in ('query','key')))
    info['paired_case_error_change25_to50']=[b-a for a,b in zip(info['25']['case_errors'],info['50']['case_errors'])]
    approx[name]=info
assert all(d['containment']['status']!='STOP' for d in diag.values())
metric_checks={}
for step,folder in folders.items():
    with np.load(folder/f'validation_predictions_{step:03d}.npz') as z:observed=cached_metrics(z['probability'],z['label'])
    stored=diag[step]['sampled_validation']
    maxdiff=max(abs(observed[k]-stored[k]) for k in ('roc_auc','accuracy','precision','recall','f1','mse'))
    assert maxdiff<1e-12
    metric_checks[str(step)]=dict(max_score_metric_difference=maxdiff,
        rounded_probability_bce_difference=observed['bce_from_rounded_probability']-stored['loss'])

resources={}
for step in (30,40,50):
    for phase in (f'stage_{step:03d}',f'verify_{step:03d}'):
        samples=[load(x) for x in (RUN/phase).glob('allocator_sample_*.json')]
        obs=[load(x) for x in (RUN/phase).glob('resources_*.json')]
        assert samples and all(x['decision']['status']!='STOP' and not x['pid_mapping_proven'] for x in obs)
        resources[phase]=dict(peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in samples),
            peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in samples),
            max_sampled_nvml_global_used_bytes=max(x['observation']['device']['used_bytes'] for x in obs),
            minimum_observed_free_bytes=min(x['observation']['device']['free_bytes'] for x in obs),
            max_global_used_minus_own_reserved_bytes=max(x['global_used_minus_own_reserved_bytes'] for x in samples),
            max_global_growth_minus_own_reserved_bytes=max(x['global_growth_minus_own_reserved_bytes'] for x in samples),
            peer_and_self_pids_unattributed=sorted({r['pid'] for x in obs for r in x['decision']['reported_compute_processes']}),
            peak_process_rss_bytes=max(x['host']['peak_process_rss_bytes'] for x in obs),
            minimum_host_headroom_bytes=min(x['host']['cgroup_headroom_bytes'] for x in obs),resource_observations=len(obs),
            attributed_process_total_vram='NOT MEASURABLE: host PID mapping unresolved; peer baseline is not own usage')
        r=resources[phase]
        assert r['peak_allocated_bytes']<=shared.POLICY['allocator_cap_bytes'] and r['peak_reserved_bytes']<=shared.POLICY['reserved_cap_bytes']
        assert r['minimum_observed_free_bytes']>=shared.POLICY['runtime_free_bytes']
        assert r['peak_process_rss_bytes']<=shared.POLICY['host_process_rss_bytes']
        assert r['minimum_host_headroom_bytes']>=shared.POLICY['host_headroom_bytes']
for name,expected in c['source_sha256'].items():assert sha(ROOT/name)==expected,name
preservation=load(HERE/'PRESERVATION_BEFORE.json')
for row in preservation['manifests']:
    assert sha(ROOT/row['manifest'])==row['manifest_sha256']
    for filename,identity in load(ROOT/row['manifest'])[row['key']].items():
        path=Path(filename);path=path if path.is_absolute() else ROOT/path
        assert sha(path)==identity['sha256'],str(path)
for filename,expected in preservation['protected_files'].items():assert sha(Path(filename))==expected
release=load(HERE/'resource_release.json')
assert release['light_hold_sha256']==preservation['protected_files'][str(shared.LIGHT_HOLD)]
assert not release['observation']['local']['light_processes']
assert release['container_memory_oom_control'].split('oom_kill ')[-1].strip()=='2'
tests={}
for file in ('cpu_tests.xml','postrun_regression_tests.xml'):
    cases=ET.parse(HERE/file).getroot().findall('.//testcase')
    assert cases and not any(any(t.find(x) is not None for x in ('failure','error','skipped')) for t in cases)
    tests[file]=dict(passed=len(cases),failures=0,skips=0)
targets=sum(r['supervised_targets'] for r in rows);intervals=sum(r['synchronized_training_seconds'] for r in rows)
assert not torch.cuda.is_initialized()
result=dict(status='PASS_SHARED_ENGINEERING;ACTION_APPROXIMATION_REVIEW_REQUIRED',verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    actual_total_optimizer_updates=50,actual_new_optimizer_calls=25,recovery_optimizer_calls=0,checkpoint25_sha256=sha(p.PARENT_CHECKPOINT),
    checkpoints=checkpoint_checks,model_adam_and_gradients_finite=True,cuda_or_memory_errors_observed=0,
    training_rows=rows,weighted_new_training_bce=sum(r['training_bce']*r['supervised_targets'] for r in rows)/targets,
    supervised_new_training_targets=targets,synchronized_update_intervals_seconds=intervals,
    synchronized_interval_updates_per_second=25/intervals,synchronized_interval_targets_per_second=targets/intervals,
    stage_results={str(k):dict(stage_execution_seconds=v['stage_execution_seconds'],
        training_interval_seconds=v['training_interval_seconds'],new_calls=v['additional_optimizer_calls']) for k,v in stageresults.items()},
    recovery_execution_seconds={str(k):v['recovery_execution_seconds'] for k,v in recovery.items()},
    resource_summary=resources,approximation=approx,sampled_validation={str(k):d['sampled_validation'] for k,d in diag.items()},
    sampled_training_eval={str(k):d['sampled_training_eval'] for k,d in diag.items()},cached_score_metric_checks=metric_checks,
    cpu_tests=tests,all_35_sources_unchanged=True,preserved_prior_artifacts=preservation['manifests'],
    pid_mapping_proven=False,resource_budget_increased=False,test_evaluations=0,automatic_retries=0,other_process_signals=0,
    full_training_ready=False,next_recommendation='GO only separately approved longer bounded diagnostic; Action REVIEW persists; NO-GO full training now')
with (HERE/'VERIFICATION_SUMMARY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('training_rows','checkpoints')},indent=2))
