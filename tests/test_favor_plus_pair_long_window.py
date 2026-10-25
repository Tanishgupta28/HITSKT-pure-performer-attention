"""Pure CPU outer-budget scope/clock checks; no GPU or training subprocess."""
import datetime as dt,json,subprocess
import pytest
from ktbench import favor_plus_pair_long_window as p


def scope(path):
    now=dt.datetime.now(dt.timezone.utc)
    profiles={}
    for arm in p.pair.ARMS:
        file=path.parent/arm/'full_shape_profile/profile_result.json';file.parent.mkdir(parents=True,exist_ok=True);file.write_text('{}');profiles[arm]=p.pair.sha256(file)
    return dict(action=p.ACTION,training_approved=True,new_parent_runtime_budget_approved=True,approval_is_proposal=False,actual_approval_evidence='UNIT_NOT_REAL_APPROVAL',prepared_plan_sha256=p.pair.sha256(path),limits=p.pair.LIMITS,
        resource_uuid=p.pair.shared.TRANSFERRED_MIG,shared_permission_confirmed=True,known_explicit_restriction=False,known_incompatible_reservation=False,
        pair_seconds=25200,per_arm_seconds=14400,max_epochs_per_arm=8,arms=list(p.pair.ARMS),starts_utc=now.isoformat(),expires_utc=(now+dt.timedelta(hours=7)).isoformat(),campaign_started_unix=now.timestamp(),source_sha256={n:p.pair.sha256(p.pair.ROOT/n) for n in p.SOURCES},full_shape_profile_sha256=profiles,
        minimum_start_headroom_seconds={'train-epoch':{'orf128':1500,'orf512':1800},'verify-epoch':120,'validate-best':300})


@pytest.mark.parametrize('key,value',[('training_approved',False),('approval_is_proposal',True),('new_parent_runtime_budget_approved',False),('pair_seconds',30000),('actual_approval_evidence','')])
def test_profile_or_unapproved_bigger_budget_never_grants_training(tmp_path,monkeypatch,key,value):
    path=tmp_path/'plan.json';path.write_text('{}');s=scope(path);s[key]=value;monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.pair.shared.TRANSFERRED_MIG)
    with pytest.raises(PermissionError):p.authorize(s,path)


def test_child_lease_is_bounded_by_unreset_parent_and_arm_deadlines(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');s=scope(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.pair.shared.TRANSFERRED_MIG)
    now=dt.datetime.now(dt.timezone.utc);origin=now.timestamp();arm_start=origin-14000;s['campaign_started_unix']=origin-18000;s['starts_utc']=dt.datetime.fromtimestamp(origin-18000,dt.timezone.utc).isoformat();s['expires_utc']=dt.datetime.fromtimestamp(origin-18000+25200,dt.timezone.utc).isoformat();deadline=p.authorize(s,path)
    lease=p.worker_lease(s,'parent','orf128',arm_start,deadline,now=now)
    end=dt.datetime.fromisoformat(lease['expires_utc']).timestamp()
    assert end<=deadline and end<=arm_start+14400 and end<=origin+3600
    assert lease['parent_campaign_started_unix']==s['campaign_started_unix'] and lease['parent_arm_started_unix']==arm_start
    p.pair.authorize(lease,path,p.pair.TRAIN_ACTION,'orf128',now=now)


def test_future_arm_start_cannot_extend_parent_budget(tmp_path):
    path=tmp_path/'plan.json';path.write_text('{}');s=scope(path);now=dt.datetime.now(dt.timezone.utc)
    with pytest.raises(ValueError):p.worker_lease(s,'parent','orf128',now.timestamp()+1,s['campaign_started_unix']+25200,now=now)


def test_expired_parent_cannot_be_extended_by_a_fresh_child_clock(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');s=scope(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.pair.shared.TRANSFERRED_MIG)
    now=dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=7,seconds=1)
    with pytest.raises((PermissionError,TimeoutError)):p.authorize(s,path,now=now)


def test_unapproved_parent_refused_before_any_worker(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');s=scope(path);s['training_approved']=False;file=tmp_path/'scope.json';file.write_text(json.dumps(s));monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.pair.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:pytest.fail('no training subprocess without approval'))
    with pytest.raises(PermissionError):p.run(path,file)
    assert not (tmp_path/'approved_7h_parent_campaign').exists()


@pytest.mark.parametrize('fail_first',[False,True])
def test_parent_workflow_keeps_one_global_clock_and_stops_without_retry(tmp_path,monkeypatch,fail_first):
    path=tmp_path/'plan.json';path.write_text('{}');s=scope(path);file=tmp_path/'scope.json';file.write_text(json.dumps(s));monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.pair.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(p.pair,'load_plan',lambda *a:({},{}));monkeypatch.setattr(p.pair,'compare_results',lambda *a:{'UNIT':True});calls=[];leases=[]
    def fake(command,**kwargs):
        calls.append(command);lease=json.loads(open(command[command.index('--record')+1]).read());leases.append(lease)
        if fail_first:raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(p.subprocess,'run',fake)
    if fail_first:
        with pytest.raises(subprocess.CalledProcessError):p.run(path,file)
        assert len(calls)==1 and (tmp_path/'approved_7h_parent_campaign/STOP_NO_RETRY.json').exists()
    else:
        assert p.run(path,file)=={'UNIT':True} and len(calls)==34
        assert len({x['parent_campaign_started_unix'] for x in leases})==1
        for arm in p.pair.ARMS:assert len({x['parent_arm_started_unix'] for x in leases if arm in x['arm_started_unix']})==1
