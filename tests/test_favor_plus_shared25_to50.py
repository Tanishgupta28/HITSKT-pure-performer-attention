"""CPU simulations of exact25-parent continuation, never real CUDA training."""
import copy
import datetime as dt
import json
from pathlib import Path

import pytest
import torch
from ktbench import favor_plus_shared25_to50 as p
from ktbench import favor_plus_shared25 as old25
from ktbench import favor_plus_shared_assist50 as old50
from ktbench import favor_plus_shared_gpu as shared
from test_favor_plus_shared_continuation import simulation


def record():
    now=dt.datetime.now(dt.timezone.utc)
    return dict(mode=shared.MODE,exclusive=False,shared_permission_confirmed=True,execution_approved=True,
        budget_and_window_acknowledged=True,limited_pid_visibility_acknowledged=True,
        permission_evidence='UNIT_ACTUAL_PERMISSION',execution_evidence='UNIT_25_MORE',
        coordination_evidence='UNIT_FIXED_BUDGET',operator='UNIT_OPERATOR',resource_uuid=shared.TRANSFERRED_MIG,
        action=p.PILOT_ACTION,prepared_plan_sha256='a'*64,policy_sha256='b'*64,
        initial_optimizer_updates=25,parent_checkpoint_sha256=p.PARENT_SHA256,
        max_total_optimizer_updates=50,max_additional_optimizer_calls=25,
        known_explicit_restriction=False,known_incompatible_reservation=False,
        starts_utc=(now-dt.timedelta(minutes=1)).isoformat(),expires_utc=(now+dt.timedelta(hours=1)).isoformat())


def test_exact25_permission_is_separate_from_legacy_scopes():
    r=record()
    assert p.authorize_record(r,shared.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)==shared.TRANSFERRED_MIG
    for authorizer in (old25.authorize_record,shared.authorize):
        with pytest.raises(ValueError):authorizer(r,shared.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)


@pytest.mark.parametrize('key,value',[
    ('max_total_optimizer_updates',51),('max_additional_optimizer_calls',30),('initial_optimizer_updates',20),
    ('parent_checkpoint_sha256','c'*64),('action',shared.ACTION),('execution_approved',False),
    ('budget_and_window_acknowledged',False),('policy_sha256','c'*64),('known_explicit_restriction',True)])
def test_unapproved_or_wrong_parent_scope_refused(key,value):
    r=record();r[key]=value
    with pytest.raises(ValueError):p.authorize_record(r,shared.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)


@pytest.mark.parametrize('endpoint',[20,25,26,35,45,51,100,True,30.0])
def test_invalid_endpoint_rejected_before_admission(monkeypatch,endpoint):
    monkeypatch.setattr(p,'context',lambda *a:(_ for _ in ()).throw(AssertionError('must reject before CUDA')))
    with pytest.raises(ValueError):p.execute(Path('UNUSED'),{},endpoint)
    with pytest.raises(ValueError):p.verify(Path('UNUSED'),{},endpoint)


def test_stage_intervals_and_all_prior_sources_remain_bound():
    assert [list(p.stage_interval(a,b)) for a,b in ((25,30),(30,40),(40,50))]==[
        list(range(26,31)),list(range(31,41)),list(range(41,51))]
    with pytest.raises(ValueError):p.stage_interval(20,30)
    previous=old25.source_identity()['source_sha256'];current=p.source_identity()['source_sha256']
    assert len(current)==len(previous)+2 and all(current[k]==v for k,v in previous.items())
    assert shared.POLICY['allocator_cap_bytes']==1024**3
    assert shared.POLICY['advisory_total_vram_budget_bytes']==2*1024**3


@pytest.fixture
def continuation(simulation,monkeypatch,tmp_path):
    oldroot,oldpath,_,_=simulation
    # Generate a genuine format2 CPU fixture25 using the already verified controller.
    for name in ('context','initialize','runtime','capture_checked','PARENT_PLAN','PARENT_CHECKPOINT'):
        monkeypatch.setattr(old25,name,getattr(old50,name))
    monkeypatch.setattr(old25,'authorize',lambda *a:shared.TRANSFERRED_MIG)
    old25.execute(oldpath,{'prepared_plan_sha256':'UNIT'},25)
    parent=oldroot/'checkpoints/checkpoint_025.pt'
    payload=torch.load(parent,map_location='cpu',weights_only=True)
    pc=payload['contract'];c=copy.deepcopy(pc)
    c.update(run_id='UNIT_FROM25',parent_contract=pc,parent_component_sha256=payload['component_sha256'],
        parent_projection_sha256=payload['projection_sha256'],parent_checkpoint_sha256=p.PARENT_SHA256,
        parent_optimizer_param_groups=payload['optimizer_state_dict']['param_groups'])
    c['execution_limits'].update(max_total_optimizer_updates=50,max_additional_optimizer_calls=25)
    root=tmp_path/'new25parent';(root/'checkpoints').mkdir(parents=True)
    path=root/'prepared_plan.json';path.write_text('{}');plan=dict(contract_base=c)
    def context(path,record,phase,update):
        if (root/'study_failure.json').exists() or (root/'study_result.json').exists():
            raise RuntimeError('failed/completed campaign cannot restart')
        output=root/f'{phase}_{update:03d}';output.mkdir()
        return output,plan,c,None,None
    monkeypatch.setattr(p,'context',context)
    for name in ('initialize','runtime','capture_checked'):
        monkeypatch.setattr(p,name,getattr(old50,name))
    monkeypatch.setattr(p,'authorize',lambda *a:shared.TRANSFERRED_MIG)
    monkeypatch.setattr(p,'PARENT_CHECKPOINT',parent)
    monkeypatch.setattr(p,'PARENT_PLAN',oldpath)
    return root,path,parent,c


def test_exact25_to50_has_only25_new_calls_and_zero_call_fresh_recovery(continuation,monkeypatch):
    root,path,parent,c=continuation;before=p.sha256(parent);calls=[];step=torch.optim.Adam.step
    def counted(self,*a,**kw):calls.append(1);return step(self,*a,**kw)
    monkeypatch.setattr(torch.optim.Adam,'step',counted)
    for endpoint in (30,40,50):
        stage=p.execute(path,{'prepared_plan_sha256':'UNIT'},endpoint)
        assert stage['optimizer_updates']==endpoint and stage['additional_optimizer_calls']==endpoint-25
        result=p.verify(path,{'prepared_plan_sha256':'UNIT'},endpoint)
        assert result['status']=='PASS' and result['optimizer_calls_during_recovery']==0
    assert len(calls)==25 and len(list(root.glob('stage_*/update_*.json')))==25
    assert sorted(x.name for x in (root/'checkpoints').glob('*.pt'))==[
        f'checkpoint_{i:03d}.pt' for i in (30,35,40,45,50)]
    final=torch.load(root/'checkpoints/checkpoint_050.pt',map_location='cpu',weights_only=True)
    assert all(float(s['step'])==50 for s in final['optimizer_state_dict']['state'].values())
    assert final['additional_optimizer_calls']==30  # unchanged historical format, not25 new-call journal
    assert json.loads((root/'study_result.json').read_text())['additional_optimizer_calls']==25
    assert json.loads((root/'stage_030/actual_next_step_matches_checkpoint_026.json').read_text())['status']=='PASS'
    assert p.sha256(parent)==before and not torch.cuda.is_initialized()
    with pytest.raises(RuntimeError,match='completed'):p.execute(path,{'prepared_plan_sha256':'UNIT'},50)
    assert len(calls)==25


def test_no_later_stage_without_fresh_previous_recovery(continuation):
    root,path,parent,c=continuation
    p.execute(path,{'prepared_plan_sha256':'UNIT'},30)
    with pytest.raises(FileNotFoundError):p.execute(path,{'prepared_plan_sha256':'UNIT'},40)
    assert len(list(root.glob('stage_*/update_*.json')))==5


def test_cuda_style_failure_does_not_retry_or_modify_parent(continuation,monkeypatch):
    root,path,parent,c=continuation;before=p.sha256(parent);calls=[];step=torch.optim.Adam.step
    def failed(self,*a,**kw):
        calls.append(1);step(self,*a,**kw)
        raise RuntimeError('UNIT CUDA-style failure after partial Adam')
    monkeypatch.setattr(torch.optim.Adam,'step',failed)
    with pytest.raises(RuntimeError,match='CUDA-style'):p.execute(path,{'prepared_plan_sha256':'UNIT'},30)
    failure=json.loads((root/'study_failure.json').read_text())
    assert failure['optimizer_step_calls']==26 and failure['optimizer_updates']==25
    assert failure['no_automatic_restart'] and failure['no_cuda_calls_in_failure_handler']
    assert p.sha256(parent)==before and len(calls)==1
    with pytest.raises(RuntimeError,match='cannot restart'):p.execute(path,{'prepared_plan_sha256':'UNIT'},30)
    assert len(calls)==1


def test_pressure_stops_before_allocator_query_or_adam(continuation,monkeypatch):
    root,path,parent,c=continuation
    def pressure(uuid):
        return dict(observed_utc=p.original.utc().isoformat(),resource_uuid=uuid,
            device=dict(total_bytes=40*shared.GIB,used_bytes=39*shared.GIB,free_bytes=shared.GIB,compute_processes=[]),
            utilization=dict(status='unsupported',nvml_status=3),local=dict(holders=[],inaccessible_fd_pids=[],light_processes=[]))
    monkeypatch.setattr(shared,'observe',pressure)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *a:(_ for _ in ()).throw(AssertionError('unsafe CUDA query')))
    with pytest.raises(shared.SharedResourceStop):p.execute(path,{'prepared_plan_sha256':'UNIT'},30)
    assert json.loads((root/'study_failure.json').read_text())['optimizer_step_calls']==25
    assert not list(root.glob('stage_*/update_*.json'))
