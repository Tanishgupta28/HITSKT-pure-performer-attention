"""CPU-only independent accumulation, recovery, masks and authorization checks."""
import copy
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ktbench import favor_plus_paired_epochs as pair
from ktbench import favor_plus_microbatch as micro
from ktbench import favor_plus_micro_profile as policy
from ktbench import favor_plus_micro_epochs as epochs
from ktbench.data.session_store import RollingSessionDataset
from ktbench.models.hitskt import HiTSKTConfig
from test_favor_plus_pipeline import synthetic_store


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(micro.time,'sleep',lambda *a:None)
    torch.set_num_threads(1)


@pytest.fixture(params=pair.ARMS)
def fixture(request,tmp_path):
    ds=RollingSessionDataset(synthetic_store(tmp_path/'store'),'train',history_sessions=15)
    net=pair.fresh_model(HiTSKTConfig(9,4,width=8,heads=2,feedforward_width=16,dropout=0.),request.param)
    optimizer=torch.optim.Adam(net.parameters(),lr=5e-5)
    rows=pair.shape_manifest(ds,[[0,1,2,3]])
    boundary=type('CPUOnly',(),{'check':lambda self,*a,**k:{'unit_cpu':True}})()
    c=dict(arm=request.param,model_config=asdict(net.config),comparison_config=asdict(net.comparison),
        expected_projection_sha256={n:pair.digest(v) for n,v in net.named_buffers() if n.endswith('.projection')},
        parameter_tensor_count=len(list(net.parameters())),expected_optimizer_groups=pair.cpu_tree(optimizer.state_dict()['param_groups']),
        sampler_schedule={'train':{'1':[r['target_sessions'] for r in rows]}},microbatch_policy=dict(physical_batch=2))
    progress=dict(epoch=1,next_batch=0,optimizer_updates=0,optimizer_step_calls=0,last_selected_epoch=0)
    return ds,net,optimizer,rows,boundary,c,progress,tmp_path


@pytest.mark.parametrize('physical',[1,2,4])
def test_partition_preserves_variable_lengths_targets_order_masks(fixture,physical):
    ds,net,opt,rows,b,*_=fixture
    row=rows[0];parts=micro.partition(ds,row,physical)
    full=pair.make_batch(ds,row);truth=full.target_labels[full.target_metric_mask].numpy()
    assert max(p['batch_size'] for p in parts)<=physical
    observed=np.concatenate([pair.make_batch(ds,p).target_labels[pair.make_batch(ds,p).target_metric_mask].numpy() for p in parts])
    assert np.array_equal(truth,observed) and len(truth)==14
    assert [i for p in parts for i in p['indices']]==row['indices']


@pytest.mark.parametrize('physical',[0,3,8,True,2.0])
def test_unfrozen_physical_policy_rejected(fixture,physical):
    ds,_,_,rows,*_=fixture
    with pytest.raises(ValueError):micro.partition(ds,rows[0],physical)


@pytest.mark.parametrize('physical',[1,2])
def test_unequal_target_weighting_matches_independent_full_batch_gradient_and_Adam(fixture,physical):
    ds,net,opt,rows,b,c,progress,root=fixture
    full=pair.fresh_model(net.config,c['arm']);fullopt=torch.optim.Adam(full.parameters(),lr=5e-5)
    batch=pair.make_batch(ds,rows[0]);fullopt.zero_grad();logits=full(batch)
    # Independent scalar objective, no use of helper or a microbatch mean.
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[batch.target_metric_mask],batch.target_labels[batch.target_metric_mask].float())
    loss.backward();gradient={n:p.grad.clone() for n,p in full.named_parameters()};fullopt.step()
    result,*_=micro.logical_batch(net,opt,ds,rows[0],physical,'cpu',b,training=True,progress=progress)
    assert result['loss']==pytest.approx(float(loss),rel=2e-6,abs=1e-7)
    for name,p in net.named_parameters():
        torch.testing.assert_close(p.grad,gradient[name],rtol=3e-4,atol=2e-6)
        torch.testing.assert_close(p,dict(full.named_parameters())[name],rtol=3e-4,atol=3e-6)
    assert progress['optimizer_updates']==1 and progress['optimizer_step_calls']==1
    assert all(float(s['step'])==1 for s in opt.state.values())


def test_averaging_micro_means_would_be_wrong_for_unequal_target_counts():
    sums=torch.tensor([1.,18.]);counts=torch.tensor([1.,9.])
    correct=(sums/counts*counts/counts.sum()).sum()
    assert correct==pytest.approx(1.9) and not torch.isclose(correct,(sums/counts).mean())


def test_failure_before_last_microbatch_never_calls_Adam_or_advances_cursor(fixture):
    ds,net,opt,rows,b,c,progress,root=fixture;before=pair.learned_digest(net)
    def check(phase,**kwargs):
        if phase=='micro_train' and check.calls==1:raise RuntimeError('synthetic shared resource stop')
        if phase=='micro_train':check.calls+=1
    check.calls=0;b.check=check
    opt.step=lambda *a,**k:pytest.fail('Adam forbidden before all microbatches pass')
    with pytest.raises(RuntimeError):micro.logical_batch(net,opt,ds,rows[0],2,'cpu',b,training=True,progress=progress)
    assert progress['optimizer_updates']==0 and progress['next_batch']==0 and pair.learned_digest(net)==before


def test_microbatch_eval_preserves_outputs_and_never_changes_state_or_RNG(fixture):
    ds,net,opt,rows,b,c,progress,root=fixture
    full=pair.fixed_logits(net,pair.make_batch(ds,rows[0]));mask=pair.make_batch(ds,rows[0]).target_metric_mask
    before=pair.digest(pair.cpu_tree(dict(model=net.state_dict(),adam=opt.state_dict(),rng=pair.capture_rng_state())))
    result,prob,y,logits=micro.logical_batch(net,None,ds,rows[0],2,'cpu',b,training=False)
    np.testing.assert_allclose(logits,full[mask].numpy(),rtol=2e-5,atol=1e-6)
    assert before==pair.digest(pair.cpu_tree(dict(model=net.state_dict(),adam=opt.state_dict(),rng=pair.capture_rng_state())))


def test_partial_logical_epoch_checkpoint_resume_is_exact_and_namespace_separate(fixture):
    ds,net,opt,_,b,c,progress,root=fixture;rows=pair.shape_manifest(ds,[[0,1],[2,3]])
    net=pair.fresh_model(replace(net.config,dropout=.1),c['arm']);opt=torch.optim.Adam(net.parameters(),lr=5e-5)
    c['model_config']=asdict(net.config)
    c['sampler_schedule']={'train':{'1':[r['target_sessions'] for r in rows]}}
    proof=pair.make_batch(ds,pair.shape_manifest(ds,[[0]])[0])
    def interrupted(cursor):
        if cursor==1:
            epochs.save_checkpoint(net,opt,c,progress,proof,root/'partial.pt')
            raise InterruptedError('synthetic')
    with pytest.raises(InterruptedError):
        micro.epoch_data(net,opt,ds,rows,'cpu',b,progress,training=True,physical_batch=1,checkpoint_callback=interrupted,checkpoint_interval=1)
    value=torch.load(root/'partial.pt',weights_only=True)
    expected=micro.epoch_data(net,opt,ds,rows,'cpu',b,progress,training=True,physical_batch=1)
    final=pair.cpu_tree(dict(model=net.state_dict(),adam=opt.state_dict(),rng=pair.capture_rng_state()))
    other=pair.fresh_model(net.config,c['arm']);otheropt=torch.optim.Adam(other.parameters(),lr=5e-5)
    assert epochs.restore(value,other,otheropt,c,proof)['optimizer_calls']==0
    actual=micro.epoch_data(other,otheropt,ds,rows,'cpu',b,copy.deepcopy(value['progress']),training=True,physical_batch=1)
    assert actual[0]==expected[0] and np.array_equal(actual[1],expected[1])
    assert pair.equal_tree(final,pair.cpu_tree(dict(model=other.state_dict(),adam=otheropt.state_dict(),rng=pair.capture_rng_state())))
    with pytest.raises(ValueError):pair.PairedEpochModel.from_checkpoint(root/'partial.pt')
    with pytest.raises(FileExistsError):epochs.save_checkpoint(other,otheropt,c,value['progress'],proof,root/'partial.pt')


def test_fresh_process_recovers_model_Adam_projection_RNG_and_next_computation(fixture):
    ds,net,opt,rows,b,c,progress,root=fixture
    net=pair.fresh_model(replace(net.config,dropout=.1),c['arm']);opt=torch.optim.Adam(net.parameters(),lr=5e-5)
    c['model_config']=asdict(net.config)
    micro.epoch_data(net,opt,ds,rows,'cpu',b,progress,training=True,physical_batch=2)
    proof=pair.make_batch(ds,pair.shape_manifest(ds,[[0]])[0])
    epochs.save_checkpoint(net,opt,c,progress,proof,root/'saved.pt')
    script=root/'recover.py'
    script.write_text('''import json,torch
from pathlib import Path
from ktbench import favor_plus_paired_epochs as p
from ktbench import favor_plus_micro_epochs as e
from ktbench.data.session_store import SessionStore,RollingSessionDataset
assert torch.version.cuda is None and not torch.cuda.is_initialized()
torch.set_num_threads(1)
root=Path(__file__).parent
v=torch.load(root/'saved.pt',map_location='cpu',weights_only=True);c=v['contract']
net=p.fresh_model(p.HiTSKTConfig(**c['model_config']),c['arm']);opt=torch.optim.Adam(net.parameters(),lr=5e-5)
ds=RollingSessionDataset(SessionStore(root/'store'),'train',history_sessions=15)
proof=p.make_batch(ds,p.shape_manifest(ds,[[0]])[0])
result=e.restore(v,net,opt,c,proof)
assert result['optimizer_calls']==0 and result['status']=='PASS_EXACT'
assert not torch.cuda.is_initialized()
print(json.dumps(result))
''')
    result=subprocess.run([sys.executable,str(script)],env=os.environ,cwd=pair.ROOT,text=True,capture_output=True,check=True)
    assert json.loads(result.stdout)['next_dropout_loss_gradients']=='EXACT'


def test_profile_sandbox_checkpoint_cannot_be_loaded_as_training(fixture):
    ds,net,opt,rows,b,c,progress,root=fixture
    micro.logical_batch(net,opt,ds,rows[0],2,'cpu',b,training=True,progress=progress)
    proof=pair.make_batch(ds,pair.shape_manifest(ds,[[0]])[0]);file=root/'profile.pt'
    receipt=policy.save_profile_checkpoint(net,opt,proof,progress,file,'TEST')
    assert receipt['actual_Adam_state'] and not receipt['training_checkpoint']
    with pytest.raises(ValueError):epochs.validate_checkpoint(torch.load(file,weights_only=True),c)


def record(path,action=policy.PROFILE):
    now=dt.datetime.now(dt.timezone.utc)
    return dict(action=action,prepared_plan_sha256=pair.sha256(path),limits=pair.LIMITS,
        resource_uuid=pair.shared.TRANSFERRED_MIG,shared_permission_confirmed=True,
        known_explicit_restriction=False,known_incompatible_reservation=False,
        profile_approved=True,training_approved=False,approval_is_proposal=False,actual_approval_evidence='UNIT_NOT_REAL_AUTHORIZATION',
        max_optimizer_updates=160,total_seconds=900,starts_utc=now.isoformat(),expires_utc=(now+dt.timedelta(seconds=900)).isoformat(),campaign_started_unix=now.timestamp())


@pytest.mark.parametrize('key,value',[('profile_approved',False),('training_approved',True),('approval_is_proposal',True),
    ('max_optimizer_updates',161),('total_seconds',1800),('limits',{}),('known_explicit_restriction',True),('actual_approval_evidence','')])
def test_missing_new_scope_or_changed_budget_rejected_before_CUDA(tmp_path,monkeypatch,key,value):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);r[key]=value
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',pair.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(pair,'initialize',lambda *a:pytest.fail('CUDA must not start'))
    with pytest.raises(PermissionError):policy.profile(path,r,'orf128',2)


def test_previous_zero_step_scope_never_grants_real_Adam_or_training(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',pair.shared.TRANSFERRED_MIG)
    r['action']=pair.PROFILE_ACTION
    with pytest.raises(PermissionError):policy.authorize(r,path,policy.PROFILE)
    r=record(path)
    with pytest.raises(PermissionError):policy.authorize(r,path,policy.TRAIN)


def test_expired_clock_cannot_be_reset_by_extending_expiry(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',pair.shared.TRANSFERRED_MIG)
    with pytest.raises(PermissionError):policy.authorize(r,path,policy.PROFILE,now=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=901))


@pytest.mark.parametrize('free_gib',[15,20])
def test_same_shared_admission_and_16GiB_floor(tmp_path,monkeypatch,free_gib):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',pair.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(pair.shared,'check_exposure_and_hold',lambda *a:None)
    obs=dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),resource_uuid=pair.shared.TRANSFERRED_MIG,
        device=dict(total_bytes=40*pair.GIB,free_bytes=free_gib*pair.GIB,used_bytes=(40-free_gib)*pair.GIB,compute_processes=[dict(pid=123)]),
        utilization=dict(status='unsupported',nvml_status=3),local=dict(holders=[],inaccessible_fd_pids=[],light_processes=[]))
    monkeypatch.setattr(pair.shared,'observe',lambda *a:obs);monkeypatch.setattr(pair,'host_memory',lambda:dict(peak_process_rss_bytes=pair.GIB,cgroup_headroom_bytes=20*pair.GIB))
    b=policy.Boundaries(path,r,tmp_path,policy.PROFILE)
    if free_gib==15:
        with pytest.raises(pair.shared.SharedResourceStop):b.check('admit',cuda=False,admission=True)
    else:
        v=b.check('admit',cuda=False,admission=True)
        assert not v['decision']['pid_mapping_proven'] and v['decision']['reported_compute_processes'][0]['pid']==123


def test_candidate_without_actual_profiles_cannot_generate_training_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(policy,'estimate',lambda *a:dict(configurations={'2':dict(arms={'orf128':dict(memory_status='REVIEW_INSUFFICIENT_HEADROOM')})}))
    with pytest.raises(PermissionError):policy.training_template(tmp_path/'plan.json',tmp_path/'scope.json')
    assert not (tmp_path/'scope.json').exists()


def test_eight_GiB_cap_and_runtime_memory_floor_retained(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',pair.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(pair.shared,'check_exposure_and_hold',lambda *a:None)
    obs=dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),resource_uuid=pair.shared.TRANSFERRED_MIG,device=dict(total_bytes=40*pair.GIB,free_bytes=20*pair.GIB,used_bytes=20*pair.GIB,compute_processes=[]),
        utilization=dict(status='unsupported',nvml_status=3),local=dict(holders=[],inaccessible_fd_pids=[],light_processes=[]))
    monkeypatch.setattr(pair.shared,'observe',lambda *a:obs);monkeypatch.setattr(pair,'host_memory',lambda:dict(peak_process_rss_bytes=pair.GIB,cgroup_headroom_bytes=20*pair.GIB))
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda:9*pair.GIB)
    boundary=policy.Boundaries(path,r,tmp_path,policy.PROFILE)
    with pytest.raises(MemoryError):boundary.check('cap')
    obs['device']['free_bytes']=5*pair.GIB
    with pytest.raises(pair.shared.SharedResourceStop):boundary.check('pressure',cuda=False)


def test_new_campaign_dispatches_exactly8epochs_twoarms_fresh_recoveries_no_real_work(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path,policy.TRAIN)
    r.update(profile_approved=False,training_approved=True,new_runtime_budget_approved=True,
        physical_batch=2,max_epochs_per_arm=8,total_seconds=3600,per_arm_seconds=1800)
    r['expires_utc']=(dt.datetime.fromisoformat(r['starts_utc'])+dt.timedelta(seconds=3600)).isoformat()
    file=tmp_path/'scope.json';file.write_text(json.dumps(r));monkeypatch.setenv('CUDA_VISIBLE_DEVICES',pair.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(policy,'load_plan',lambda *a:({'accounting':{'2':{'logical_updates':8}}},
        {'schedules':{'train':{str(e):[{}] for e in range(1,9)}}},{}))
    monkeypatch.setattr(epochs,'require_profiles',lambda *a:None)
    monkeypatch.setattr(policy,'estimate',lambda *a:dict(configurations={'2':{'arms':{arm:dict(actual_Adam_training_body_seconds=8,validation_seconds=9,prior_IO_diagnostics_and_startup_proxy_seconds=8) for arm in pair.ARMS}}}))
    calls=[];clocks=[]
    def fake(command,**kwargs):
        calls.append(command);arm=command[command.index('--arm')+1];epoch=int(command[command.index('--epoch')+1])
        clocks.append(json.loads(Path(command[command.index('--clock')+1]).read_text()))
        if command[2]=='train-epoch':
            out=tmp_path/'training_b2'/arm/'metrics';out.mkdir(parents=True,exist_ok=True)
            (out/f'epoch_{epoch:03d}.json').write_text(json.dumps(dict(epoch=epoch,validation=dict(roc_auc=.6+epoch/1000),checkpoint=dict(sha256='UNIT'))))
    monkeypatch.setattr(epochs.subprocess,'run',fake)
    result=epochs.campaign(path,file)
    assert len(calls)==34 and result['test_evaluations']==0
    assert len({c['campaign_started_unix'] for c in clocks})==1
    for arm in pair.ARMS:
        assert len({c['arm_started_unix'] for c in clocks if c['arm']==arm})==1
        assert result['matched_full_validation'][arm]['selected_epoch']==8


def test_persisted_parent_plan_string_is_restored_to_Path_on_load(tmp_path,monkeypatch):
    monkeypatch.setattr(pair,'ROOT',tmp_path)
    parent=tmp_path/'parent.json';parent.write_text('{}')
    out=tmp_path/'experiments/hitskt_favor_plus_v1/assist2017/unit';out.mkdir(parents=True)
    plan=dict(kind=policy.KIND,run_id='unit',limits=pair.LIMITS,physical_choices=[2,4],
        parent_plan=str(parent),parent_sha256=pair.sha256(parent),source_sha256={},
        logical_schedules_sha256=pair.digest({}),profile_design=dict(max_optimizer_steps_total=160))
    file=out/'prepared_plan.json';file.write_text(json.dumps(plan))
    monkeypatch.setattr(pair,'load_plan',lambda path:({'schedules':{}},{}))
    assert policy.load_plan(file)[0]==plan


def test_persisted_parent_scope_string_and_arm_deadline_checked_before_resource_probes(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',pair.shared.TRANSFERRED_MIG)
    file=tmp_path/'scope.json';file.write_text(json.dumps(r))
    clock=dict(parent_path=str(file),parent_sha256=pair.sha256(file),campaign_started_unix=r['campaign_started_unix'],arm_started_unix=r['campaign_started_unix'])
    # Profile clock is used here purely as a CPU path/expiry fixture.
    r['per_arm_seconds']=1
    clock['arm_started_unix']-=2
    boundary=policy.Boundaries(path,r,tmp_path,policy.PROFILE,clock)
    monkeypatch.setattr(pair.shared,'observe',lambda *a:pytest.fail('expired/mismatched clock before telemetry'))
    with pytest.raises(TimeoutError):boundary.check('expired',cuda=False)


def test_post_split_worst_group_added_even_when_not_old_B8_extremum():
    first=dict(indices=[1,2]);different=dict(indices=[3,4]);summary={str(b):dict(max_components={str(m):dict(row=different) for m in (128,512)}) for b in (2,4)}
    result=policy.augment_extremes([first],summary)
    assert result==[first,different]
