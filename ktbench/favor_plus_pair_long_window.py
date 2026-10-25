"""Approval-gated outer budget for unchanged, profiled paired epoch workers.

The7h parent clock and4h arm clocks never reset. Legacy workers receive bounded
<=1h leases that expire before both parent deadlines. This is explicitly a NEW
longer campaign, never a claim that its execution fits the old2h campaign.
No tensor/attention/optimizer implementation or GPU resource guard is modified.
"""
from __future__ import annotations
import datetime as dt
import json,os,subprocess,sys,time
from pathlib import Path
from ktbench import favor_plus_paired_epochs as pair

ACTION='favor_plus_B8_matched8epochs_7h_parent_v1'
PAIR_SECONDS=25200;ARM_SECONDS=14400;MAX_WORKER_LEASE=3600
SOURCES=('ktbench/favor_plus_pair_long_window.py','scripts/run_favor_plus_pair_long_window.py')
START_HEADROOM={'train-epoch':{'orf128':1500,'orf512':1800},'verify-epoch':120,'validate-best':300}


def template(path,output):
    path,output=Path(path),Path(output);plan,_=pair.load_plan(path);profiles={}
    for arm in pair.ARMS:
        file=path.parent/arm/'full_shape_profile/profile_result.json';v=json.loads(file.read_text())
        if v['status']!='PASS_ZERO_UPDATE_FULL_SHAPES' or v['plan_sha256']!=pair.sha256(path):raise ValueError('both exact-plan profiles required')
        profiles[arm]=pair.sha256(file)
    value=dict(action=ACTION,prepared_plan_sha256=pair.sha256(path),resource_uuid=pair.shared.TRANSFERRED_MIG,limits=pair.LIMITS,
        source_sha256={n:pair.sha256(pair.ROOT/n) for n in SOURCES},full_shape_profile_sha256=profiles,
        shared_permission_confirmed=True,known_explicit_restriction=False,known_incompatible_reservation=False,
        training_approved=False,new_parent_runtime_budget_approved=False,approval_is_proposal=True,actual_approval_evidence='',
        pair_seconds=PAIR_SECONDS,per_arm_seconds=ARM_SECONDS,max_epochs_per_arm=8,arms=list(pair.ARMS),
        minimum_start_headroom_seconds=START_HEADROOM,
        starts_utc='SET_ONLY_AFTER_ACTUAL_APPROVAL',expires_utc='SET_ONLY_AFTER_ACTUAL_APPROVAL',campaign_started_unix=None,
        notice='PROPOSAL. Actual new7hpair/4harm approval required. Worker leases are subordinate to unchanged absolute parent/arm deadlines; no automatic clock reset or furthertraining.')
    if output.exists():raise FileExistsError(output)
    pair.write_json(output,value);return dict(status='UNAPPROVED_TEMPLATE_ONLY',path=str(output),training_approved=False)


def authorize(scope,path,*,now=None):
    if (scope.get('action')!=ACTION or scope.get('training_approved') is not True or scope.get('new_parent_runtime_budget_approved') is not True
        or scope.get('approval_is_proposal') is not False or not isinstance(scope.get('actual_approval_evidence'),str) or not scope['actual_approval_evidence'].strip()
        or scope.get('prepared_plan_sha256')!=pair.sha256(path) or scope.get('limits')!=pair.LIMITS
        or scope.get('resource_uuid')!=pair.shared.TRANSFERRED_MIG or os.environ.get('CUDA_VISIBLE_DEVICES')!=pair.shared.TRANSFERRED_MIG
        or scope.get('shared_permission_confirmed') is not True or scope.get('known_explicit_restriction') is not False or scope.get('known_incompatible_reservation') is not False
        or scope.get('pair_seconds')!=PAIR_SECONDS or scope.get('per_arm_seconds')!=ARM_SECONDS or scope.get('max_epochs_per_arm')!=8 or scope.get('arms')!=list(pair.ARMS)):
        raise PermissionError('actual NEW7hpair/4harm training scope required; profile/old2h approval is insufficient')
    if set(scope.get('source_sha256',{}))!=set(SOURCES) or set(scope.get('full_shape_profile_sha256',{}))!=set(pair.ARMS):raise ValueError('complete coordinator/profile identities required')
    if scope.get('minimum_start_headroom_seconds')!=START_HEADROOM:raise ValueError('measured stage headroom policy changed')
    current=now or dt.datetime.now(dt.timezone.utc);start=dt.datetime.fromisoformat(scope['starts_utc']);end=dt.datetime.fromisoformat(scope['expires_utc']);origin=scope.get('campaign_started_unix')
    if start.tzinfo is None or end.tzinfo is None or not start<=current<end or not 0<(end-start).total_seconds()<=PAIR_SECONDS:raise PermissionError('bounded7h parent validity required')
    if not isinstance(origin,(float,int)) or not start.timestamp()<=origin<=current.timestamp() or current.timestamp()-origin>=PAIR_SECONDS:raise TimeoutError('immutable parent clock expired')
    for n,want in scope['source_sha256'].items():
        if pair.sha256(pair.ROOT/n)!=want:raise ValueError('outer coordinator source changed')
    for arm,want in scope['full_shape_profile_sha256'].items():
        if pair.sha256(Path(path).parent/arm/'full_shape_profile/profile_result.json')!=want:raise ValueError('profile receipt changed')
    return min(origin+PAIR_SECONDS,end.timestamp())


def worker_lease(scope,parent_sha,arm,arm_start,parent_deadline,*,now=None):
    clock=now or dt.datetime.now(dt.timezone.utc);origin=clock.timestamp();deadline=min(origin+MAX_WORKER_LEASE,parent_deadline,arm_start+ARM_SECONDS)
    if arm not in pair.ARMS or not scope['campaign_started_unix']<=arm_start<=origin or parent_deadline>scope['campaign_started_unix']+PAIR_SECONDS:raise ValueError('invalid parent/arm clock')
    if deadline-origin<=30:raise TimeoutError('no worker lease within remaining parent/arm headroom')
    return dict(action=pair.TRAIN_ACTION,prepared_plan_sha256=scope['prepared_plan_sha256'],resource_uuid=scope['resource_uuid'],limits=scope['limits'],
        shared_permission_confirmed=True,known_explicit_restriction=False,known_incompatible_reservation=False,
        approval_is_proposal=False,actual_approval_evidence=scope['actual_approval_evidence'],training_approved=True,eight_epoch_pair_runtime_approved=True,
        actual_new_time_budget_evidence='Explicit7hparent; bounded worker lease, not an old2h overall campaign',
        total_seconds=7200,per_arm_seconds=3600,max_epochs_per_arm=8,
        starts_utc=clock.isoformat(),expires_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat(),campaign_started_unix=origin,arm_started_unix={arm:origin},
        parent_scope_sha256=parent_sha,parent_campaign_started_unix=scope['campaign_started_unix'],parent_deadline_unix=parent_deadline,
        parent_arm_started_unix=arm_start,parent_arm_deadline_unix=arm_start+ARM_SECONDS,
        lease_notice='LOCAL per-worker lease only.7h global and4h arm clocks are immutable in the parent; no overallbudget reset.')


def run(path,scope_path):
    path,scope_path=Path(path),Path(scope_path);scope=json.loads(scope_path.read_text());parent_deadline=authorize(scope,path);pair.load_plan(path)
    if (path.parent/'study_failure.json').exists():raise ValueError('cannot retry failed study')
    out=path.parent/'approved_7h_parent_campaign';out.mkdir(exist_ok=False);arm_starts={};stages=[];parent_sha=pair.sha256(scope_path)
    try:
        for arm in pair.ARMS:
            arm_starts[arm]=time.time();pair.write_json(out/f'CLOCKS_{arm}.json',dict(parent_campaign_started_unix=scope['campaign_started_unix'],parent_deadline_unix=parent_deadline,arm_started_unix=arm_starts))
            actions=[(a,e) for e in range(1,9) for a in ('train-epoch','verify-epoch')]+[('validate-best',None)]
            for action,epoch in actions:
                if pair.sha256(scope_path)!=parent_sha:raise ValueError('parent authorization record changed')
                authorize(scope,path);remaining=min(parent_deadline,arm_starts[arm]+ARM_SECONDS)-time.time()
                need=scope['minimum_start_headroom_seconds'][action];need=need[arm] if isinstance(need,dict) else need
                if remaining<need:raise TimeoutError('stop before launching stage without measured headroom')
                lease=worker_lease(scope,parent_sha,arm,arm_starts[arm],parent_deadline);file=out/f'{arm}_{action}_{epoch or 0:03d}_lease.json';pair.write_json(file,lease)
                command=[sys.executable,'scripts/run_favor_plus_paired_epochs.py',action,'--plan',str(path),'--record',str(file),'--arm',arm,'--execute-cuda']
                if epoch is not None:command+=['--epoch',str(epoch)]
                stamp=time.perf_counter()
                with (out/f'{arm}_{action}_{epoch or 0:03d}.log').open('xb') as log:subprocess.run(command,cwd=pair.ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,check=True)
                stages.append(dict(arm=arm,action=action,epoch=epoch,seconds=time.perf_counter()-stamp));pair.write_json(out/f'COMPLETED_STAGES_{len(stages):02d}.json',stages)
                authorize(scope,path);print(arm,action,epoch,'PASS',flush=True)
        return pair.compare_results(path)
    except BaseException as exc:
        pair.write_json(out/'STOP_NO_RETRY.json',dict(status='STOP_NO_RETRY',type=type(exc).__name__,message=str(exc),parent_campaign_started_unix=scope['campaign_started_unix'],arm_started_unix=arm_starts,
            completed_stages=stages,no_other_job_signals=True,checkpoint_writes_left_to_worker_atomic_protocol=True));raise
