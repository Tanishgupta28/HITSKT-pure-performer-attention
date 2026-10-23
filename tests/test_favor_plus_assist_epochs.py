"""New full-epoch controller tests on tiny CPU fixtures, never production training."""
import copy
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from ktbench import favor_plus_assist_epochs as p
from ktbench.data.session_store import RollingSessionDataset
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree
from ktbench.training import COMMON_EARLY_STOPPING,capture_rng_state
from test_favor_plus_pipeline import synthetic_store


def test_patience3_is_local_strict_auc_and_f1_selection_is_separate():
    state={};decisions=[]
    for epoch,(auc,f1) in enumerate([(0.5,.3),(.6,.32),(.6,.35),(.55,.4),(.58,.38)],1):
        state,improved=p.selection(state,epoch,dict(roc_auc=auc,f1=f1));decisions.append(improved)
    assert state['stop'] and state['bad_epochs']==3 and state['best_epoch']==2
    assert state['best_f1_epoch']==4 and state['best_auc']==.6 and decisions==[True,True,False,False,False]
    assert COMMON_EARLY_STOPPING.patience==5


@pytest.mark.parametrize('epoch',[0,11,True,1.0])
def test_no_epoch_outside_approved_integer_ceiling(epoch):
    with pytest.raises(ValueError):p.selection({},epoch,dict(roc_auc=.6,f1=.5))
    with pytest.raises(ValueError):p.execute_epoch(Path('UNUSED'),{},epoch)


def test_10th_epoch_stops_even_with_continuous_improvement():
    state={}
    for e in range(1,11):state,_=p.selection(state,e,dict(roc_auc=.5+e*.01,f1=.4))
    assert state['stop'] and state['best_epoch']==10


@pytest.mark.parametrize('field,value',[('budget_increase_approved',False),('actual_budget_approval_evidence',''),('approval_is_proposal',True)])
def test_proposed_budget_never_becomes_approval(field,value):
    record=dict(budget_profile='requested_8gib',budget_increase_approved=True,actual_budget_approval_evidence='UNIT_ACTUAL_RESPONSE',approval_is_proposal=False)
    record[field]=value
    with pytest.raises(p.BudgetBlocked):p.budget(record)


def test_existing_budget_cannot_admit_complete_shapes_or_silently_skip():
    plan=dict(schedules={'train':{'1':[dict(batch_size=4,padded_tokens=7606)]},'validation':{'0':[dict(batch_size=5,padded_tokens=9000)]}})
    result=p.shape_admission(plan,p.budget({'budget_profile':'existing_1gib'}))
    assert result['status']=='BLOCKED_RESOURCE_BUDGET' and result['incompatible_batches']==2 and result['data_skipped']==0
    assert p.budget({'budget_profile':'existing_1gib'})['allocator_bytes']==1024**3


def test_explicit_budget_is_bounded_and_does_not_change_shared_pilot_policy():
    old=copy.deepcopy(p.shared.POLICY)
    limits=p.budget(dict(budget_profile='requested_8gib',budget_increase_approved=True,actual_budget_approval_evidence='UNIT_REAL_RESPONSE',approval_is_proposal=False))
    assert limits['allocator_bytes']==8*1024**3 and limits['admission_free_bytes']==16*1024**3
    assert limits['runtime_free_bytes']==old['runtime_free_bytes']==6*1024**3
    assert old==p.shared.POLICY


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1);torch.random.default_generator.manual_seed(42)
    store=synthetic_store(tmp_path/'store');train=RollingSessionDataset(store,'train',history_sessions=15)
    net=HiTSKTFavorPlus(HiTSKTConfig(9,4,width=8,heads=2,feedforward_width=16,dropout=.1),FavorPlusConfig(features=8,chunk_size=3))
    opt=torch.optim.Adam(net.parameters(),lr=5e-5);rows=p.shape_manifest(train,[[i] for i in range(len(train))])
    contract=dict(model_config=net.checkpoint()['config'],favor_config=net.checkpoint()['favor_config'],parameter_tensor_count=len(list(net.parameters())),
        expected_optimizer_groups=cpu_tree(opt.state_dict()['param_groups']),
        sampler_schedule={'train':{'1':[row['target_sessions'] for row in rows]}},
        expected_projection_sha256={k:p.digest(v) for k,v in net.state_dict().items() if k.endswith('.projection')},seed=42,patience=3)
    progress=dict(epoch=1,next_batch=0,optimizer_step_calls=0,optimizer_updates=0,best_auc=None,best_f1=None,bad_epochs=0)
    boundary=type('CPUFixtureBoundary',(),{'check':lambda self,*a,**k:{'UNIT_CPU':True}})()
    proof=p.make_batch(train,rows[0]);return train,net,opt,rows,contract,progress,boundary,proof,tmp_path


def test_complete_epoch_and_validation_count_every_target_without_test_access(fixture):
    train,net,opt,rows,c,progress,b,proof,root=fixture
    metrics,prob,y,updates=p.epoch_data(net,opt,train,rows,torch.device('cpu'),b,progress,training=True)
    assert metrics['targets']==14 and len(updates)==len(train)==4 and progress['optimizer_updates']==4
    assert len(prob)==len(y)==14 and all(x['gradient_l2']>=0 for x in updates)
    val=RollingSessionDataset(train.store,'validation',history_sessions=15)
    vr=p.shape_manifest(val,p.schedule(val,0,False));result=p.epoch_data(net,None,val,vr,torch.device('cpu'),b,{},training=False)
    assert result[0]['targets']==4 and not result[3]
    assert progress['optimizer_updates']==4 and not torch.cuda.is_initialized()


def test_actual_partial_checkpoint_resume_preserves_next_updates_and_metrics(fixture):
    train,net,opt,rows,c,progress,b,proof,root=fixture
    def interrupted(cursor):
        if cursor==2:
            p.save_checkpoint(net,opt,c,progress,proof,root/'partial.pt')
            raise InterruptedError('UNIT bounded interruption after valid checkpoint')
    with pytest.raises(InterruptedError):p.epoch_data(net,opt,train,rows,torch.device('cpu'),b,progress,training=True,checkpoint_callback=interrupted,checkpoint_interval=2)
    checkpoint=torch.load(root/'partial.pt',map_location='cpu',weights_only=True)
    expected=p.epoch_data(net,opt,train,rows,torch.device('cpu'),b,progress,training=True)
    weights=cpu_tree(net.state_dict());adam=cpu_tree(opt.state_dict());rng=cpu_tree(capture_rng_state())
    other=HiTSKTFavorPlus(net.config,net.favor_config);otheropt=torch.optim.Adam(other.parameters(),lr=5e-5)
    recovery=p.restore_checkpoint(checkpoint,other,otheropt,c,proof)
    resumed=p.epoch_data(other,otheropt,train,rows,torch.device('cpu'),b,copy.deepcopy(checkpoint['progress']),training=True)
    assert recovery['status']=='PASS' and recovery['optimizer_calls']==0
    assert equal_tree(weights,cpu_tree(other.state_dict())) and equal_tree(adam,cpu_tree(otheropt.state_dict())) and equal_tree(rng,cpu_tree(capture_rng_state()))
    assert resumed[0]==expected[0] and np.array_equal(resumed[1],expected[1]) and np.array_equal(resumed[2],expected[2])
    assert [x['batch'] for x in resumed[3]]==[2,3]


def test_atomic_checkpoint_and_fresh_process_recovery(fixture):
    train,net,opt,rows,c,progress,b,proof,root=fixture
    p.epoch_data(net,opt,train,rows,torch.device('cpu'),b,progress,training=True)
    path=root/'checkpoint.pt';receipt=p.save_checkpoint(net,opt,c,progress,proof,path)
    assert receipt['atomic_no_overwrite'] and receipt['optimizer_updates']==4
    with pytest.raises(FileExistsError):p.save_checkpoint(net,opt,c,progress,proof,path)
    script=root/'fresh.py'
    script.write_text('''import json,torch
from pathlib import Path
from ktbench import favor_plus_assist_epochs as p
from ktbench.data.session_store import SessionStore,RollingSessionDataset
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
torch.set_num_threads(1)
root=Path(__file__).parent
v=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
net=HiTSKTFavorPlus(HiTSKTConfig(**v['config']),FavorPlusConfig(**v['favor_config']))
opt=torch.optim.Adam(net.parameters(),lr=5e-5)
ds=RollingSessionDataset(SessionStore(root/'store'),'train',history_sessions=15)
batch=p.make_batch(ds,p.shape_manifest(ds,[[0]])[0])
result=p.restore_checkpoint(v,net,opt,v['contract'],batch)
assert not torch.cuda.is_initialized()
print(json.dumps(result))
''')
    result=subprocess.run([sys.executable,str(script)],check=True,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert json.loads(result.stdout)['optimizer_calls']==0


@pytest.mark.parametrize('key',['state_dict','optimizer_state_dict','rng_state','progress','contract'])
def test_checkpoint_component_corruption_refused(fixture,key):
    train,net,opt,rows,c,progress,b,proof,root=fixture
    p.epoch_data(net,opt,train,rows,torch.device('cpu'),b,progress,training=True)
    path=root/'checkpoint.pt';p.save_checkpoint(net,opt,c,progress,proof,path);v=torch.load(path,map_location='cpu',weights_only=True)
    v[key]={'corrupted':True}
    with pytest.raises((ValueError,KeyError)):p.checkpoint_validate(v,c)


def test_resealed_optimizer_config_change_is_not_allowed(fixture):
    train,net,opt,rows,c,progress,b,proof,root=fixture
    p.epoch_data(net,opt,train,rows,torch.device('cpu'),b,progress,training=True)
    path=root/'checkpoint.pt';p.save_checkpoint(net,opt,c,progress,proof,path);v=torch.load(path,map_location='cpu',weights_only=True)
    v['optimizer_state_dict']['param_groups'][0]['lr']=1e-3
    v['component_sha256']['optimizer_state_dict']=p.digest(v['optimizer_state_dict'])
    with pytest.raises(ValueError,match='Adam settings'):p.checkpoint_validate(v,c)


def test_future_epoch_cli_never_accepts_test_split():
    from scripts import train_assist_favor_plus_epochs
    assert 'test' not in train_assist_favor_plus_epochs.main.__code__.co_consts


def test_budget_blocked_before_any_gpu_observation_or_initialization(tmp_path,monkeypatch):
    root=tmp_path/'experiments/hitskt_favor_plus_v1/assist2017/UNIT';root.mkdir(parents=True)
    path=root/'prepared_plan.json';path.write_text('{}')
    plan=dict(contract={'run_id':'UNIT'},schedules={'train':{'1':[dict(batch_size=4,padded_tokens=9000)]}})
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'authorize',lambda *a:p.budget({'budget_profile':'existing_1gib'}))
    monkeypatch.setattr(p,'load_plan',lambda *a:(plan,{}))
    monkeypatch.setattr(p.shared,'observe',lambda *a:(_ for _ in ()).throw(AssertionError('budget must refuse before GPU observation')))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:(_ for _ in ()).throw(AssertionError('no CUDA initialization')))
    with pytest.raises(p.BudgetBlocked):p.context(path,{},'must_not_exist')
    assert not (root/'must_not_exist').exists() and not torch.cuda.is_initialized()


@pytest.mark.parametrize('gib,expected',[(1,'reject'),(8,'accept')])
def test_resource_guard_obeys_only_the_explicit_cap(tmp_path,monkeypatch,gib,expected):
    limits=dict(allocator_bytes=gib*p.GIB,admission_free_bytes=8*p.GIB,runtime_free_bytes=6*p.GIB)
    monkeypatch.setattr(p,'authorize',lambda *a:limits)
    monkeypatch.setattr(p.shared,'check_exposure_and_hold',lambda *a:None)
    monkeypatch.setattr(p,'host_memory',lambda:dict(peak_process_rss_bytes=p.GIB,cgroup_headroom_bytes=10*p.GIB))
    obs=dict(observed_utc=p.dt.datetime.now(p.dt.timezone.utc).isoformat(),resource_uuid=p.shared.TRANSFERRED_MIG,
        device=dict(total_bytes=40*p.GIB,free_bytes=20*p.GIB,used_bytes=20*p.GIB,compute_processes=[{'pid':123}]),
        utilization=dict(status='unsupported',nvml_status=3),local=dict(holders=[],inaccessible_fd_pids=[],light_processes=[]))
    monkeypatch.setattr(p.shared,'observe',lambda *a:obs)
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda *a:2*p.GIB)
    boundary=p.Boundaries(Path('UNIT'),{},tmp_path,p.time.time())
    if expected=='reject':
        with pytest.raises(MemoryError):boundary.check('UNIT')
    else:assert boundary.check('UNIT')['allocated_bytes']==2*p.GIB
    assert p.shared.POLICY['allocator_cap_bytes']==p.GIB and not boundary.last['pid_mapping_proven']


def test_pressure_stop_retains_cache_without_querying_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'authorize',lambda *a:dict(allocator_bytes=p.GIB,admission_free_bytes=8*p.GIB,runtime_free_bytes=6*p.GIB))
    monkeypatch.setattr(p.shared,'check_exposure_and_hold',lambda *a:None)
    monkeypatch.setattr(p,'host_memory',lambda:dict(peak_process_rss_bytes=p.GIB,cgroup_headroom_bytes=10*p.GIB))
    obs=dict(observed_utc=p.dt.datetime.now(p.dt.timezone.utc).isoformat(),resource_uuid=p.shared.TRANSFERRED_MIG,
        device=dict(total_bytes=40*p.GIB,free_bytes=5*p.GIB,used_bytes=35*p.GIB,compute_processes=[]),
        utilization=dict(status='unsupported',nvml_status=3),local=dict(holders=[],inaccessible_fd_pids=[],light_processes=[]))
    monkeypatch.setattr(p.shared,'observe',lambda *a:obs)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *a:(_ for _ in ()).throw(AssertionError('no CUDA after pressure')))
    boundary=p.Boundaries(Path('UNIT'),{},tmp_path,p.time.time());boundary.allocator_cached={'peak_allocated_bytes':17}
    with pytest.raises(p.shared.SharedResourceStop):boundary.check('pressure')
    assert boundary.last['cached_allocator']['peak_allocated_bytes']==17


def test_curves_only_use_completed_epoch_metrics(tmp_path):
    (tmp_path/'metrics').mkdir();p.curves(tmp_path)
    assert not list(tmp_path.glob('*.png'))
    (tmp_path/'metrics/epoch_001.json').write_text(json.dumps(dict(epoch=1,train={'loss':.7},validation={'loss':.69,'roc_auc':.51,'f1':.3})))
    p.curves(tmp_path)
    assert (tmp_path/'curves_through_epoch_001.png').stat().st_size>1000


@pytest.mark.parametrize('duration,accept',[(7200.,True),(7200.000001,False)])
def test_operator_window_is_exactly_bounded_without_tolerance_relaxation(tmp_path,monkeypatch,duration,accept):
    path=tmp_path/'plan.json';path.write_text('{}');now=dt.datetime.now(dt.timezone.utc);start=now-dt.timedelta(seconds=1)
    record=dict(action=p.ACTION,shared_permission_confirmed=True,full_training_approved=True,max_epochs=10,patience=3,
        resource_uuid=p.shared.TRANSFERRED_MIG,prepared_plan_sha256=p.sha256(path),known_explicit_restriction=False,
        known_incompatible_reservation=False,budget_profile='existing_1gib',starts_utc=start.isoformat(),expires_utc=(start+dt.timedelta(seconds=duration)).isoformat())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.shared.TRANSFERRED_MIG)
    if accept:assert p.authorize(record,path,now=now)['allocator_bytes']==p.GIB
    else:
        with pytest.raises(ValueError,match='operator record'):p.authorize(record,path,now=now)
