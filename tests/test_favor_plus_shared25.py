"""CPU-only scope/atomic recovery tests for the five-call shared CUDA pilot."""
import datetime as dt
import json
from pathlib import Path

import pytest
import torch
from ktbench import favor_plus_shared25 as p
from ktbench import favor_plus_shared_assist50 as old
from ktbench import favor_plus_shared_gpu as shared
from test_favor_plus_shared_continuation import simulation


def record():
    now=dt.datetime.now(dt.timezone.utc)
    return dict(mode=shared.MODE,exclusive=False,shared_permission_confirmed=True,execution_approved=True,
        budget_and_window_acknowledged=True,limited_pid_visibility_acknowledged=True,
        permission_evidence='UNIT_ACTUAL_PERMISSION',execution_evidence='UNIT_ONLY_FIVE_UPDATES',
        coordination_evidence='UNIT_APPROVED_FIXED_BUDGET',operator='UNIT_OPERATOR',resource_uuid=shared.TRANSFERRED_MIG,
        action=p.PILOT_ACTION,prepared_plan_sha256='a'*64,policy_sha256='b'*64,max_total_optimizer_updates=25,
        max_additional_optimizer_calls=5,known_explicit_restriction=False,known_incompatible_reservation=False,
        starts_utc=(now-dt.timedelta(minutes=1)).isoformat(),expires_utc=(now+dt.timedelta(hours=1)).isoformat())


def test_five_step_permission_is_independent_of_older_fifty_step_scope():
    r=record()
    assert p.authorize_record(r,shared.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)==shared.TRANSFERRED_MIG
    with pytest.raises(ValueError):shared.authorize(r,shared.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)


@pytest.mark.parametrize('key,value',[('max_total_optimizer_updates',50),('max_additional_optimizer_calls',30),
    ('action',shared.ACTION),('execution_approved',False),('budget_and_window_acknowledged',False),
    ('policy_sha256','c'*64),('known_explicit_restriction',True)])
def test_expanded_or_unapproved_scope_refused(key,value):
    r=record();r[key]=value
    with pytest.raises(ValueError):p.authorize_record(r,shared.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)


@pytest.mark.parametrize('endpoint',[20,24,26,30,40,50,100,True,25.0])
def test_no_endpoint_other_than25_before_admission(monkeypatch,endpoint):
    monkeypatch.setattr(p,'context',lambda *a:(_ for _ in ()).throw(AssertionError('must reject before CUDA')))
    with pytest.raises(ValueError):p.execute(Path('UNUSED'),{},endpoint)
    with pytest.raises(ValueError):p.verify(Path('UNUSED'),{},endpoint)


def test_exact_five_sample_interval_and_frozen_resource_limits():
    assert list(p.stage_interval(20,25))==[21,22,23,24,25]
    assert shared.POLICY['allocator_cap_bytes']==1024**3
    assert shared.POLICY['advisory_total_vram_budget_bytes']==2*1024**3
    assert shared.POLICY['admission_free_bytes']==8*1024**3 and shared.POLICY['runtime_free_bytes']==6*1024**3


def test_all_existing_shared_sources_remain_bound_unchanged():
    previous=old.source_identity()['source_sha256'];current=p.source_identity()['source_sha256']
    assert len(current)==len(previous)+2 and all(current[k]==v for k,v in previous.items())


@pytest.fixture
def pilot(simulation,monkeypatch):
    root,path,parent,observe=simulation
    for name in ('context','initialize','runtime','capture_checked','PARENT_PLAN','PARENT_CHECKPOINT'):
        monkeypatch.setattr(p,name,getattr(old,name))
    monkeypatch.setattr(p,'authorize',lambda *a:shared.TRANSFERRED_MIG)
    original_context=p.context
    def context(*args):
        output,plan,c,train,val=original_context(*args)
        c['pilot_scope']=p.SCOPE;c['exact_training_and_probe_samples']=c['exact_training_and_probe_samples'][:26]
        c['execution_limits'].update(max_total_optimizer_updates=25,max_additional_optimizer_calls=5)
        return output,plan,c,train,val
    monkeypatch.setattr(p,'context',context)
    return root,path,parent


def test_cpu_pilot_takes_five_calls_saves25_and_recovers_with_zero_calls(pilot,monkeypatch):
    root,path,parent=pilot;before=p.sha256(parent);calls=[];step=torch.optim.Adam.step
    def counted(self,*a,**kw):calls.append(1);return step(self,*a,**kw)
    monkeypatch.setattr(torch.optim.Adam,'step',counted)
    result=p.execute(path,{'prepared_plan_sha256':'UNIT'},25)
    assert result['status']=='PASS' and result['optimizer_updates']==25 and result['additional_optimizer_calls']==5
    assert len(calls)==5 and len(list(root.glob('stage_025/update_*.json')))==5
    assert [x.name for x in (root/'checkpoints').glob('*.pt')]==['checkpoint_025.pt']
    proof=p.verify(path,{'prepared_plan_sha256':'UNIT'},25)
    assert proof['status']=='PASS' and proof['optimizer_calls_during_recovery']==0 and len(calls)==5
    done=json.loads((root/'study_result.json').read_text())
    assert done['status']=='TWENTY_FIVE_TOTAL_SHARED_PILOT_COMPLETED' and done['optimizer_updates']==25
    assert p.sha256(parent)==before and not torch.cuda.is_initialized()
    samples=[json.loads(x.read_text()) for x in (root/'stage_025').glob('allocator_sample_*.json')]
    assert samples and all(x['overhead_attribution'].startswith('UNATTRIBUTED') for x in samples)
    with pytest.raises(RuntimeError,match='completed'):p.execute(path,{'prepared_plan_sha256':'UNIT'},25)
    assert len(calls)==5
