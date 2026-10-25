"""Tiny synthetic CPU execution only. No genuine dataset/GPU training."""
import copy,datetime as dt,json,os,subprocess,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest,torch
from ktbench import favor_plus_paired_epochs as p
from ktbench.favor_plus_matched_comparison import MatchedComparisonModel
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.data.session_store import RollingSessionDataset
from ktbench.training import capture_rng_state
from test_favor_plus_pipeline import synthetic_store


@pytest.fixture(autouse=True)
def cpu_only():
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    torch.set_num_threads(1)


@pytest.fixture(params=p.ARMS)
def fixture(request,tmp_path):
    store=synthetic_store(tmp_path/'store');ds=RollingSessionDataset(store,'train',history_sessions=15)
    net=p.fresh_model(HiTSKTConfig(9,4,width=8,heads=2,feedforward_width=16,dropout=.1),request.param);opt=torch.optim.Adam(net.parameters(),lr=5e-5)
    rows=p.shape_manifest(ds,[[i] for i in range(len(ds))]);proof=p.make_batch(ds,rows[0])
    c=dict(arm=request.param,model_config=asdict(net.config),comparison_config=asdict(net.comparison),
        expected_projection_sha256={n:p.digest(v) for n,v in net.named_buffers() if n.endswith('.projection')},
        parameter_tensor_count=len(list(net.parameters())),expected_optimizer_groups=cpu_tree(opt.state_dict()['param_groups']),
        sampler_schedule={'train':{'1':[r['target_sessions'] for r in rows]}})
    progress=dict(epoch=1,next_batch=0,optimizer_updates=0,optimizer_step_calls=0,last_selected_epoch=0)
    boundary=type('CPUOnly',(),{'check':lambda self,*a,**k:{'unit_cpu':True}})()
    return ds,net,opt,rows,c,progress,boundary,proof,tmp_path


def test_fresh_pair_parameters_rng_precision_and_checkpoint_namespace():
    cfg=HiTSKTConfig(9,4,width=8,heads=2,feedforward_width=16,dropout=.1);states=[]
    for arm in p.ARMS:
        n=p.fresh_model(cfg,arm);states.append((p.learned_digest(n),p.digest(torch.random.get_rng_state())))
        assert all(x.dtype==torch.float32 for x in n.parameters())
        assert n.comparison.accumulation_dtype=='float64' and n.checkpoint()['format_version']==5
    assert states[0]==states[1]


def test_equal8epoch_selection_never_stops_for_patience_or_selects_f1():
    state={}
    for e,(auc,f1) in enumerate([(.7,.3),(.69,.4),(.68,.5),(.67,.6),(.66,.7),(.65,.8),(.64,.9),(.63,.95)],1):
        state,_=p.selection(state,e,dict(roc_auc=auc,f1=f1));assert state['stop']==(e==8)
    assert state['best_epoch']==1 and state['best_f1_epoch']==8
    with pytest.raises(ValueError):p.selection({},2,dict(roc_auc=.7,f1=.5))


@pytest.mark.parametrize('epoch',[0,9,True,1.0])
def test_epoch_ceiling_and_integer_type(epoch):
    with pytest.raises(ValueError):p.execute_epoch(Path('unused'),{},'orf128',epoch)


def record(path,action=p.PROFILE_ACTION):
    now=dt.datetime.now(dt.timezone.utc)
    return dict(action=action,prepared_plan_sha256=p.sha256(path),resource_uuid=p.shared.TRANSFERRED_MIG,limits=p.LIMITS,
        shared_permission_confirmed=True,known_explicit_restriction=False,known_incompatible_reservation=False,
        approval_is_proposal=False,actual_approval_evidence='UNIT_FAKE_SCOPE_NOT_REAL_APPROVAL',profile_approved=True,training_approved=False,
        max_optimizer_updates=0,total_seconds=600,per_arm_seconds=600,max_epochs_per_arm=8,
        starts_utc=now.isoformat(),expires_utc=(now+dt.timedelta(hours=2)).isoformat(),campaign_started_unix=now.timestamp(),arm_started_unix={})


def test_profile_permission_never_grants_real_training_before_cuda(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(p,'initialize',lambda *a:pytest.fail('no CUDA under profile permission'))
    p.authorize(r,path,p.PROFILE_ACTION)
    with pytest.raises(PermissionError):p.execute_epoch(path,r,'orf128',1)
    r['action']=p.TRAIN_ACTION
    with pytest.raises(PermissionError):p.authorize(r,path,p.TRAIN_ACTION,'orf128')


@pytest.mark.parametrize('key,value',[('approval_is_proposal',True),('limits',{}),('max_optimizer_updates',1),('total_seconds',1200),('known_explicit_restriction',True)])
def test_scope_or_budget_drift_rejected_without_cuda(tmp_path,monkeypatch,key,value):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);r[key]=value;monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.shared.TRANSFERRED_MIG)
    with pytest.raises((PermissionError,TimeoutError)):p.authorize(r,path,p.PROFILE_ACTION)


def test_profile_campaign_clock_cannot_reset_by_extending_record(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.shared.TRANSFERRED_MIG)
    now=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=601)
    with pytest.raises(TimeoutError):p.authorize(r,path,p.PROFILE_ACTION,now=now)


@pytest.mark.parametrize('free_gib',[15,20])
def test_shared_admission_retains_unknown_peers_and16GiBfloor(tmp_path,monkeypatch,free_gib):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(p.shared,'check_exposure_and_hold',lambda *a:None)
    obs=dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),resource_uuid=p.shared.TRANSFERRED_MIG,
        device=dict(total_bytes=40*p.GIB,free_bytes=free_gib*p.GIB,used_bytes=(40-free_gib)*p.GIB,compute_processes=[dict(pid=12345)]),
        utilization=dict(status='unsupported',nvml_status=3),local=dict(holders=[],inaccessible_fd_pids=[],light_processes=[]))
    monkeypatch.setattr(p.shared,'observe',lambda *a:obs);monkeypatch.setattr(p,'host_memory',lambda:dict(peak_process_rss_bytes=p.GIB,cgroup_headroom_bytes=20*p.GIB))
    b=p.Boundaries(path,r,tmp_path,p.PROFILE_ACTION)
    if free_gib==15:
        with pytest.raises(p.shared.SharedResourceStop):b.check('admit',cuda=False,admission=True)
    else:
        x=b.check('admit',cuda=False,admission=True);assert not x['decision']['pid_mapping_proven'] and x['decision']['reported_compute_processes'][0]['pid']==12345


def test_real_epoch_helper_counts_complete_fixture_targets_and_leaves_val_unchanged(fixture):
    ds,net,opt,rows,c,progress,b,proof,root=fixture
    tm,prob,y,updates=p.epoch_data(net,opt,ds,rows,torch.device('cpu'),b,progress,training=True)
    assert tm['targets']==14 and progress['optimizer_updates']==4 and len(updates)==4
    state=cpu_tree(net.state_dict());adam=cpu_tree(opt.state_dict())
    val=RollingSessionDataset(ds.store,'validation',history_sessions=15);vr=p.shape_manifest(val,[[i] for i in range(len(val))])
    vm,*_=p.epoch_data(net,None,val,vr,torch.device('cpu'),b,{},training=False)
    assert vm['targets']==4 and equal_tree(state,cpu_tree(net.state_dict())) and equal_tree(adam,cpu_tree(opt.state_dict()))


def test_atomic_partial_recovery_resumes_exact_optimizer_trajectory_and_metrics(fixture):
    ds,net,opt,rows,c,progress,b,proof,root=fixture
    def interrupted(cursor):
        if cursor==2:
            p.save_checkpoint(net,opt,c,progress,proof,root/'partial.pt');raise InterruptedError('synthetic fixture interruption')
    with pytest.raises(InterruptedError):p.epoch_data(net,opt,ds,rows,torch.device('cpu'),b,progress,training=True,checkpoint_callback=interrupted,checkpoint_interval=2)
    v=torch.load(root/'partial.pt',weights_only=True);expected=p.epoch_data(net,opt,ds,rows,torch.device('cpu'),b,progress,training=True)
    state=cpu_tree(net.state_dict());adam=cpu_tree(opt.state_dict());rng=cpu_tree(capture_rng_state())
    other=p.fresh_model(net.config,c['arm']);otheropt=torch.optim.Adam(other.parameters(),lr=5e-5)
    assert p.restore_checkpoint(v,other,otheropt,c,proof)['optimizer_calls']==0
    actual=p.epoch_data(other,otheropt,ds,rows,torch.device('cpu'),b,copy.deepcopy(v['progress']),training=True)
    assert actual[0]==expected[0] and np.array_equal(actual[1],expected[1])
    assert equal_tree(state,cpu_tree(other.state_dict())) and equal_tree(adam,cpu_tree(otheropt.state_dict())) and equal_tree(rng,cpu_tree(capture_rng_state()))


def test_fresh_process_checkpoint_prediction_gradient_rng_adam_recovery(fixture):
    ds,net,opt,rows,c,progress,b,proof,root=fixture
    p.epoch_data(net,opt,ds,rows,torch.device('cpu'),b,progress,training=True)
    p.save_checkpoint(net,opt,c,progress,proof,root/'saved.pt')
    with pytest.raises(FileExistsError):p.save_checkpoint(net,opt,c,progress,proof,root/'saved.pt')
    with pytest.raises(ValueError):MatchedComparisonModel.from_checkpoint(root/'saved.pt')
    script=root/'recover.py';script.write_text('''import json,torch
from pathlib import Path
from ktbench import favor_plus_paired_epochs as p
from ktbench.data.session_store import SessionStore,RollingSessionDataset
torch.set_num_threads(1)
root=Path(__file__).parent
v=torch.load(root/'saved.pt',map_location='cpu',weights_only=True)
n=p.PairedEpochModel.from_checkpoint(root/'saved.pt')
o=torch.optim.Adam(n.parameters(),lr=5e-5)
d=RollingSessionDataset(SessionStore(root/'store'),'train',history_sessions=15)
b=p.make_batch(d,p.shape_manifest(d,[[0]])[0])
print(json.dumps(p.restore_checkpoint(v,n,o,v['contract'],b)))
assert torch.version.cuda is None and not torch.cuda.is_initialized()
''')
    r=subprocess.run([sys.executable,str(script)],check=True,text=True,capture_output=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert json.loads(r.stdout)['status']=='PASS_EXACT'


@pytest.mark.parametrize('component',p.COMPONENTS)
def test_checkpoint_component_corruption_refused(fixture,component):
    ds,net,opt,rows,c,progress,b,proof,root=fixture
    p.save_checkpoint(net,opt,c,progress,proof,root/'zero.pt');v=torch.load(root/'zero.pt',weights_only=True);v[component]={'corrupt':True}
    with pytest.raises((ValueError,KeyError)):p.checkpoint_validate(v,c)


def test_resealed_projection_corruption_is_not_accepted(fixture):
    ds,net,opt,rows,c,progress,b,proof,root=fixture
    p.save_checkpoint(net,opt,c,progress,proof,root/'zero.pt');v=torch.load(root/'zero.pt',weights_only=True)
    key=next(k for k in v['state_dict'] if k.endswith('.projection'));v['state_dict'][key]+=1;v['component_sha256']['state_dict']=p.digest(v['state_dict'])
    with pytest.raises(ValueError,match='projections'):p.checkpoint_validate(v,c)


def test_full_validation_event_ids_follow_the_exact_bucket_order(fixture):
    ds,net,opt,rows,c,progress,b,proof,root=fixture
    val=RollingSessionDataset(ds.store,'validation',history_sessions=15);vr=p.shape_manifest(val,[[1,0]])
    ids=p.validation_event_ids({'validation':{'0':vr}},{'validation':val})
    assert ids.tolist()==[23,24,9,10]
    vm,prob,y,_=p.epoch_data(net,None,val,vr,torch.device('cpu'),b,{},training=False)
    assert np.array_equal(y,ds.store.correct[ids])


def test_extended_training_diagnostics_have_fixed_audits_no_updates_and_no_rng_changes(fixture):
    ds,net,opt,rows,c,progress,b,proof,root=fixture
    c.update(diagnostic_samples=[{'dataset_index':0}],audit_projection_seeds=[17,101],review_and_severe_stops={'severe_stops':dict(feature_kernel_max_abs=5e-5,stage_median_relative_l2=.75,any_prefix_relative_l2=1.5,stage_median_pair_ess_min=1.25,qk_p95_growth_ratio=2.)})
    before=(p.learned_digest(net),p.digest(cpu_tree(capture_rng_state())))
    p.diagnostics(net,ds,c,b,root)
    value=json.loads((root/'attention_diagnostics.json').read_text())
    assert len(value['cases'][0])==4 and len(value['extended_prefixes'])==4
    assert all([a['seed'] for a in r['audit_draws']]==[17,101] for r in value['extended_prefixes'])
    assert before==(p.learned_digest(net),p.digest(cpu_tree(capture_rng_state()))) and net.training


def test_unapproved_campaign_stops_before_any_subprocess(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);file=tmp_path/'scope.json';file.write_text(json.dumps(r));monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:pytest.fail('no worker without real training scope'))
    with pytest.raises(PermissionError):p.campaign(path,file)
    assert not (tmp_path/'approved_campaign').exists()


@pytest.mark.parametrize('fail_first',[False,True])
def test_campaign_is_finite_sequential_and_stops_on_first_failure(tmp_path,monkeypatch,fail_first):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path,p.TRAIN_ACTION);r.update(training_approved=True,eight_epoch_pair_runtime_approved=True,actual_new_time_budget_evidence='UNIT_NOT_REAL',total_seconds=7200,per_arm_seconds=3600)
    file=tmp_path/'scope.json';file.write_text(json.dumps(r));monkeypatch.setenv('CUDA_VISIBLE_DEVICES',p.shared.TRANSFERRED_MIG);monkeypatch.setattr(p,'load_plan',lambda *a:({},{}));monkeypatch.setattr(p,'compare_results',lambda *a:{'UNIT':True});calls=[]
    def fake(command,**kwargs):
        calls.append(command)
        if fail_first:raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(p.subprocess,'run',fake)
    if fail_first:
        with pytest.raises(subprocess.CalledProcessError):p.campaign(path,file)
        assert len(calls)==1 and (tmp_path/'approved_campaign/CAMPAIGN_STOP.json').exists()
    else:
        assert p.campaign(path,file)=={'UNIT':True} and len(calls)==34
        assert all(c[c.index('--arm')+1]=='orf128' for c in calls[:17])
        assert all(c[c.index('--arm')+1]=='orf512' for c in calls[17:])
