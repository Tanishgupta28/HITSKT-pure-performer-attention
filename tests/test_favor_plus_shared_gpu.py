"""Shared policy CPU tests. No real NVML, CUDA or real-data optimizer calls."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from ktbench import favor_plus_shared_gpu as s
from ktbench import favor_plus_shared_assist50 as p


@pytest.fixture
def observation():
    return dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),resource_uuid=s.TRANSFERRED_MIG,
        device=dict(total_bytes=40*s.GIB,free_bytes=20*s.GIB,used_bytes=20*s.GIB,
                    compute_processes=[dict(pid=111,used_gpu_memory_bytes=2*s.GIB),dict(pid=222)]),
        utilization=dict(status='ok',gpu_percent=25,memory_percent=30),
        local=dict(holders=[dict(pid=333,classification='unknown_or_compute_conflict')],
                   inaccessible_fd_pids=[444],light_processes=[]))


@pytest.fixture
def host():
    return dict(peak_process_rss_bytes=s.GIB,cgroup_headroom_bytes=20*s.GIB)


@pytest.fixture
def record():
    now=dt.datetime.now(dt.timezone.utc)
    return dict(mode=s.MODE,exclusive=False,shared_permission_confirmed=True,execution_approved=True,
        budget_and_window_acknowledged=True,limited_pid_visibility_acknowledged=True,
        permission_evidence='UNIT_USER_PERMISSION',execution_evidence='UNIT_APPROVED_BOUNDED_SCOPE',
        coordination_evidence='UNIT_SHARED_BUDGET_AND_WINDOW',operator='UNIT_OPERATOR',
        resource_uuid=s.TRANSFERRED_MIG,action=s.ACTION,prepared_plan_sha256='a'*64,policy_sha256='b'*64,
        max_total_optimizer_updates=50,max_additional_optimizer_calls=30,
        known_explicit_restriction=False,known_incompatible_reservation=False,
        starts_utc=(now-dt.timedelta(minutes=1)).isoformat(),expires_utc=(now+dt.timedelta(hours=1)).isoformat())


def test_explicit_shared_permission_accepts_nonexclusive_record(record):
    assert s.authorize(record,s.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)==s.TRANSFERRED_MIG


@pytest.mark.parametrize('field,value',[
    ('mode','exclusive'),('exclusive',True),('shared_permission_confirmed',False),('execution_approved',False),
    ('budget_and_window_acknowledged',False),('limited_pid_visibility_acknowledged',False),
    ('permission_evidence',''),('execution_evidence',''),('coordination_evidence',''),
    ('resource_uuid','GPU-unknown'),('action','full_training'),('max_total_optimizer_updates',51),
    ('max_additional_optimizer_calls',31),('prepared_plan_sha256','c'*64),('policy_sha256','c'*64),
    ('known_explicit_restriction',True),('known_incompatible_reservation',True),
    ('expires_utc','2000-01-01T00:00:00+00:00'),('starts_utc','2000-01-01T00:00:00'),
])
def test_permission_scope_policy_and_window_fail_closed(record,field,value):
    record[field]=value
    with pytest.raises(ValueError):s.authorize(record,s.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)


def test_legacy_record_and_wrong_visible_device_cannot_select_shared(record):
    with pytest.raises(ValueError):s.authorize({'exclusive':True},s.TRANSFERRED_MIG,plan_sha256='a'*64,policy_sha256='b'*64)
    with pytest.raises(ValueError):s.authorize(record,'0',plan_sha256='a'*64,policy_sha256='b'*64)


def test_multiple_unattributed_peers_logged_without_claiming_alias_or_exclusivity(observation,host):
    r=s.assess(observation,host,admission=True)
    assert r['status']=='ADMIT_SHARED_WITH_LIMITED_ATTRIBUTION'
    assert [p['pid'] for p in r['reported_compute_processes']]==[111,222]
    assert r['local_holders']==observation['local']['holders'] and r['local_visibility_gaps']==[444]
    assert not r['pid_mapping_proven'] and not r['exclusive_ownership_claimed']


@pytest.mark.parametrize('kind',['memory_floor','missing_memory','negative_memory','boolean_memory','missing_processes',
    'bad_pid','missing_local','protected_light','util_query_error','util_invalid','host_rss','host_headroom','stale'])
def test_unsafe_pressure_visibility_and_corrupt_telemetry_stop(observation,host,kind):
    if kind=='memory_floor':observation['device']['free_bytes']=s.POLICY['runtime_free_bytes']-1
    elif kind=='missing_memory':del observation['device']['free_bytes']
    elif kind=='negative_memory':observation['device']['free_bytes']=-1
    elif kind=='boolean_memory':observation['device']['free_bytes']=True
    elif kind=='missing_processes':del observation['device']['compute_processes']
    elif kind=='bad_pid':observation['device']['compute_processes']=[{'pid':True}]
    elif kind=='missing_local':del observation['local']
    elif kind=='protected_light':observation['local']['light_processes']=[{'pid':999}]
    elif kind=='util_query_error':observation['utilization']={'status':'error','nvml_status':4}
    elif kind=='util_invalid':observation['utilization']['gpu_percent']=float('nan')
    elif kind=='host_rss':host['peak_process_rss_bytes']=s.POLICY['host_process_rss_bytes']+1
    elif kind=='host_headroom':host['cgroup_headroom_bytes']=s.POLICY['host_headroom_bytes']-1
    else:observation['observed_utc']=(dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=10)).isoformat()
    assert s.assess(observation,host)['status']=='STOP'


def test_admission_and_runtime_have_distinct_memory_floors(observation,host):
    observation['device'].update(free_bytes=7*s.GIB,used_bytes=33*s.GIB)
    assert s.assess(observation,host,admission=True)['status']=='STOP'
    assert s.assess(observation,host)['status']!='STOP'


def test_unsupported_mig_utilization_is_unknown_not_zero(observation,host):
    observation['utilization']=dict(status='unsupported',nvml_status=3,gpu_percent=None,memory_percent=None)
    r=s.assess(observation,host,admission=True)
    assert r['status']!='STOP' and any('load UNKNOWN' in x for x in r['warnings'])
    assert observation['utilization']['gpu_percent'] is None


def test_busy_admission_defers_but_runtime_load_cannot_be_attributed_to_a_peer(observation,host):
    observation['utilization']['gpu_percent']=99
    assert s.assess(observation,host,admission=True)['status']=='STOP'
    assert s.assess(observation,host)['status']!='STOP'


@pytest.mark.parametrize('key',['allocated_bytes','reserved_bytes'])
def test_allocated_and_reserved_caps_enforced(observation,host,key):
    own=dict(allocated_bytes=1,reserved_bytes=1);own[key]=s.GIB+1
    assert s.assess(observation,host,own_cuda=own)['status']=='STOP'


def test_policy_cannot_be_silently_relaxed(observation,host):
    policy=copy.deepcopy(s.POLICY);policy['allocator_cap_bytes']=8*s.GIB
    with pytest.raises(ValueError):s.assess(observation,host,policy=policy)


def test_boundary_records_pressure_and_does_not_query_cuda_after_stop(tmp_path,monkeypatch,observation,host,record):
    monkeypatch.setattr(p,'authorize',lambda *a:s.TRANSFERRED_MIG)
    monkeypatch.setattr(p.original,'check_host_resources',lambda:host)
    observation['device']['free_bytes']=1
    monkeypatch.setattr(s,'observe',lambda *a:observation)
    def forbidden(*args):raise AssertionError('CUDA query after unsafe resource observation')
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    b=p.Boundaries(record,p.original.utc().isoformat(),{'max_execution_seconds':2700},tmp_path)
    with pytest.raises(s.SharedResourceStop):b.check('pre_adam')
    assert b.last['decision']['status']=='STOP' and list(tmp_path.glob('resources_*.json'))


def test_peer_churn_is_tracked_without_static_whitelist(tmp_path,monkeypatch,observation,host,record):
    monkeypatch.setattr(p,'authorize',lambda *a:s.TRANSFERRED_MIG)
    monkeypatch.setattr(p.original,'check_host_resources',lambda:host)
    monkeypatch.setattr(s,'observe',lambda *a:observation)
    b=p.Boundaries(record,p.original.utc().isoformat(),{'max_execution_seconds':2700},tmp_path)
    assert b.check('first',cuda=False)['reported_pids_added']==[111,222]
    observation['device']['compute_processes']=[{'pid':555}]
    r=b.check('second',cuda=False)
    assert r['reported_pids_added']==[555] and r['reported_pids_departed']==[111,222]
    assert not r['pid_mapping_proven']


def test_execution_rejected_before_plan_model_or_cuda(monkeypatch,tmp_path):
    path=tmp_path/'prepared_plan.json';path.write_text('{}')
    monkeypatch.setattr(p,'authorize',lambda *a:(_ for _ in ()).throw(ValueError('execution approval required')))
    monkeypatch.setattr(p,'initialize',lambda *a:(_ for _ in ()).throw(AssertionError('CUDA/model must not run')))
    with pytest.raises(ValueError,match='approval'):p.execute(path,{},30)


@pytest.mark.parametrize('to',[20,25,51,60,100,True,30.0])
def test_total_scope_cannot_expand_or_restart(monkeypatch,to):
    monkeypatch.setattr(p,'context',lambda *a:(_ for _ in ()).throw(AssertionError('no GPU admission')))
    with pytest.raises(ValueError):p.execute(Path('unused'),{},to)


def test_shared_source_map_preserves_every_legacy_source():
    old=p.exclusive.source_identity()['source_sha256'];new=p.source_identity()['source_sha256']
    assert all(new[k]==v for k,v in old.items()) and len(new)==len(old)+3


def test_importing_policy_cannot_load_torch_or_cuda():
    result=subprocess.run([sys.executable,'-c',
        'import sys;import ktbench.favor_plus_shared_gpu;assert "torch" not in sys.modules;assert not any("libcuda.so" in x for x in open("/proc/self/maps"))'],
        text=True,capture_output=True,timeout=15,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert result.returncode==0,result.stderr


def test_utilization_permission_error_cannot_masquerade_as_unsupported(observation,host):
    observation['utilization']=dict(status='unsupported',nvml_status=4)
    assert s.assess(observation,host)['status']=='STOP'


def test_allocator_cap_is_applied_before_first_model_allocation(monkeypatch):
    import ktbench.config
    events=[]
    monkeypatch.setattr(ktbench.config,'seed_everything',lambda seed:events.append(('seed',seed)))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *args:SimpleNamespace(total_memory=40*s.GIB))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda value,device:events.append(('cap',value)))
    class FakeNet:
        def __init__(self,*args):self.parameter=torch.nn.Parameter(torch.ones(1))
        def to(self,device):events.append(('model_to',device));return self
        def parameters(self):return [self.parameter]
    monkeypatch.setattr(p.original,'HiTSKTFavorPlus',FakeNet)
    plan={'contract_base':{'parameter_tensor_count':1,'config':{
        'model_config':{'num_questions':8,'num_skills':3},'profile':{'learning_rate':5e-5,'weight_decay':0}}}}
    model,opt=p.initialize(plan)
    assert events==[('seed',42),('cap',1/40),('model_to','cuda:0')]
    assert opt.param_groups[0]['lr']==5e-5 and not torch.cuda.is_initialized()
