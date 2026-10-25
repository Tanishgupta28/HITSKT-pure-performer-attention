"""Explicit common-batch proposal after full-shape failure; CPU preparation only.

This starts fresh models in a new namespace, never resumes the failed profile.
It preserves all sample/label order and full lengths. It changes Adam group sizes
for BOTH arms; it is not claimed equivalent to the historical1454-step trajectory.
"""
from __future__ import annotations
import copy,json,os
from pathlib import Path
from ktbench import favor_plus_paired_epochs as pair


def rebatch(schedules,splits,max_batch_size):
    if type(max_batch_size) is not int or not 1<=max_batch_size<=8:raise ValueError('common1..8 candidate only')
    output={}
    for split,epochs in schedules.items():
        output[split]={}
        for epoch,rows in epochs.items():
            chunks=[row['indices'][i:i+max_batch_size] for row in rows for i in range(0,len(row['indices']),max_batch_size)]
            new=pair.shape_manifest(splits[split],chunks)
            if [t for r in rows for t in r['target_sessions']]!=[t for r in new for t in r['target_sessions']]:raise ValueError('rebatch changed target order')
            if sum(r['supervised_targets'] for r in rows)!=sum(r['supervised_targets'] for r in new):raise ValueError('target count changed')
            if max(r['padded_tokens'] for r in new)>max(r['padded_tokens'] for r in rows):raise ValueError('rebatching increased padded cost')
            output[split][epoch]=new
    return output


def prepare(parent_path,run_id,max_batch_size=8):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU preparation only')
    parent_path=Path(parent_path).resolve();old,splits=pair.load_plan(parent_path)
    failure=parent_path.parent/'study_failure.json'
    if not failure.exists():raise ValueError('preserved source failure evidence required for this follow-up')
    plan=copy.deepcopy(old);plan.update(run_id=run_id,training_approved=False,
        common_batch_policy=dict(max_batch_size=max_batch_size,whole_examples=True,truncate=False,skip=False,gradient_accumulation=False,
            note='Both arms fresh; smaller Adam groups change trajectory versus historical run. Same new groups/order/update count for primary pair.'),
        source_failed_plan_sha256=pair.sha256(parent_path),source_failure_sha256=pair.sha256(failure),
        notice='CPU-prepared follow-up only. NO automatic retry/profile/training approval. New prospective profiling scope required.')
    plan['schedules']=rebatch(old['schedules'],splits,max_batch_size)
    plan['source_sha256'].update({str(parent_path.relative_to(pair.ROOT)):pair.sha256(parent_path),
        str(failure.relative_to(pair.ROOT)):pair.sha256(failure),
        **{name:pair.sha256(pair.ROOT/name) for name in ('ktbench/favor_plus_paired_rebatch.py','scripts/prepare_favor_plus_paired_rebatch.py')}})
    for c in plan['contracts'].values():
        c.update(sampler_sha256=pair.digest(plan['schedules']),
            sampler_schedule={s:{e:[r['target_sessions'] for r in rr] for e,rr in ee.items()} for s,ee in plan['schedules'].items()},
            source_sha256=plan['source_sha256'],common_batch_policy=plan['common_batch_policy'])
    root=pair.fresh_run('assist2017',run_id)
    for arm in pair.ARMS:
        for sub in ('checkpoints','metrics'):(root/arm/sub).mkdir(parents=True,exist_ok=False)
    pair.write_json(root/'prepared_plan.json',plan)
    for name in ('PROFILE_RECORD_TEMPLATE.json','TRAINING_RECORD_TEMPLATE.json'):
        value=json.loads((parent_path.parent/name).read_text());value.update(prepared_plan_sha256=pair.sha256(root/'prepared_plan.json'),approval_is_proposal=True,
            actual_approval_evidence='',profile_approved=False,training_approved=False,starts_utc='SET_AFTER_NEW_APPROVAL',expires_utc='SET_AFTER_NEW_APPROVAL',campaign_started_unix=None)
        pair.write_json(root/name,value)
    pair.load_plan(root/'prepared_plan.json')
    return dict(status='CPU_PREPARED_NEW_PROFILE_REQUIRED',plan=str(root/'prepared_plan.json'),plan_sha256=pair.sha256(root/'prepared_plan.json'),
        maximum_batch_size=max_batch_size,train_updates_per_arm=sum(map(len,plan['schedules']['train'].values())),
        train_batches_per_epoch={e:len(r) for e,r in plan['schedules']['train'].items()},
        max_padded_tokens=max(r['padded_tokens'] for ee in plan['schedules'].values() for rr in ee.values() for r in rr),
        real_optimizer_updates=0,CUDA_initialized=False,training_runtime_estimate='NOT ESTIMABLE before new profile; old2h proposal is not validated')
