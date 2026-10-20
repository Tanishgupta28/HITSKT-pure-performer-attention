"""CPU fixtures: immediate identity, atomic checkpoint, fresh no-step recovery."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from ktbench import favor_plus_resource_telemetry as t
from ktbench import favor_plus_pilot_recovery as r
from ktbench import favor_plus_assist20 as p
from ktbench.batching import Session,SessionExample,collate_hitskt
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.favor_plus_pilot_diagnostics import cpu_tree
from ktbench.favor_plus_bounded_assist import continuation_probe


def fixture_process(root,pid=20,*,argv=None,executable=None,ticks=10):
    proc=root/str(pid);(proc/'fd').mkdir(parents=True)
    if executable is None:
        executable=root/'trusted_nvidia-smi';executable.write_bytes(b'UNIT-FIXTURE-BINARY');executable.chmod(0o755)
    (proc/'exe').symlink_to(executable);(proc/'cwd').symlink_to('/workspace/other')
    (proc/'status').write_text('Name:\tnvidia-smi\nState:\tR (running)\nUid:\t0 0 0 0\nPPid:\t1\n')
    (proc/'comm').write_text('nvidia-smi\n')
    (proc/'cmdline').write_bytes(b'\0'.join([str(executable).encode(),*(argv or [b'-L']),b'']))
    (proc/'stat').write_text(f'{pid} (nvidia-smi) R '+' '.join(['0']*18+[str(ticks)]+['0']*5))
    (proc/'fd'/'3').symlink_to('/dev/nvidia0')
    return proc,executable


@pytest.mark.parametrize('argv,allowed',[
    (['nvidia-smi'],True),(['smi','-L'],True),(['smi','-q','-x'],True),
    (['smi','--query-gpu=memory.used,uuid','--format=csv,noheader,nounits'],True),
    (['smi','--query-compute-apps','pid,used_memory','--id','0'],True),
    (['smi','-i','0','-q'],True),(['smi','--gpu-reset'],False),(['smi','-pm','1'],False),
    (['smi','--query-gpu=name','--gpu-reset'],False),(['smi','--id'],False),
    (['smi','--format=--gpu-reset'],False),(['smi','--query-gpu=name\nSECRET'],False),
    (['smi','--loop=1'],False),(['smi','-q','unknown'],False)])
def test_complete_read_only_flag_parser(argv,allowed):
    assert (t.read_only_options(argv) is not None)==allowed


def test_only_actual_immutable_trusted_executable_can_be_monitor(tmp_path):
    proc,exe=fixture_process(tmp_path);trust=t.trusted_binaries([exe])
    assert str(exe) in trust
    result=t.device_snapshot(tmp_path,own_pid=1,trusted=trust)
    assert not result['conflicts'] and result['holders'][0]['verified_read_only']
    exe.chmod(0o777)
    assert t.device_snapshot(tmp_path,own_pid=1,trusted=t.trusted_binaries([exe]))['conflicts']


def test_spoofed_name_secret_arguments_never_trusted_or_logged(tmp_path):
    secret=b'UNIT-SECRET-TOKEN-NOT-TO-LOG'
    fixture_process(tmp_path,argv=[b'train.py',b'--token',secret])
    result=t.device_snapshot(tmp_path,own_pid=1,trusted={})
    assert result['conflicts'] and secret.decode() not in json.dumps(result)


def test_read_only_monitor_exit_race_retains_captured_identity(tmp_path,monkeypatch):
    proc,exe=fixture_process(tmp_path);trust=t.trusted_binaries([exe]);real=os.readlink
    def race(path,*args,**kw):
        target=real(path,*args,**kw)
        if Path(path)==proc/'fd'/'3':
            for name in ('exe','cmdline','stat'):(proc/name).unlink(missing_ok=True)
        return target
    monkeypatch.setattr(os,'readlink',race)
    snap=t.device_snapshot(tmp_path,own_pid=1,trusted=trust)
    assert not snap['conflicts'] and snap['holders'][0]['exited_after_identity']
    assert snap['holders'][0]['trusted_binary_sha256']==trust[str(exe)]['sha256']


def test_unknown_exiting_process_remains_conflict(tmp_path,monkeypatch):
    proc,exe=fixture_process(tmp_path);real=os.readlink
    def race(path,*args,**kw):
        target=real(path,*args,**kw)
        if Path(path)==proc/'fd'/'3':(proc/'stat').unlink(missing_ok=True)
        return target
    monkeypatch.setattr(os,'readlink',race)
    assert t.device_snapshot(tmp_path,own_pid=1,trusted={})['conflicts']


def test_pid_reuse_cannot_inherit_monitor_permission(tmp_path):
    proc,exe=fixture_process(tmp_path);trust=t.trusted_binaries([exe])
    def race(proc):
        if proc.name=='20':
            s=(proc/'stat').read_text();(proc/'stat').write_text(s.replace(' 10 ',' 11 '))
    snap=t.device_snapshot(tmp_path,own_pid=1,trusted=trust,after_identity=race)
    assert snap['conflicts'] and not snap['holders'][0]['verified_read_only']


def test_mutating_smi_blocks_and_other_gpu_is_not_this_allocation(tmp_path):
    proc,exe=fixture_process(tmp_path,argv=[b'--gpu-reset'])
    assert t.device_snapshot(tmp_path,own_pid=1,trusted=t.trusted_binaries([exe]))['conflicts']
    (proc/'fd'/'3').unlink();(proc/'fd'/'3').symlink_to('/dev/nvidia1')
    assert not t.device_snapshot(tmp_path,own_pid=1,trusted={})['conflicts']


def test_no_new_action_no_io_or_cuda(monkeypatch,tmp_path):
    def forbidden(*a,**kw):raise AssertionError('must not reach CUDA or plan I/O')
    monkeypatch.setattr(p,'load_plan',forbidden);monkeypatch.setattr(torch.cuda,'get_device_properties',forbidden)
    with pytest.raises(ValueError,match='specific twenty'):
        p.execute(tmp_path/'missing.json',{'user_approval':{'actions':['favor_plus_assist2017_bounded_100_update_validation_pilot']}},5)


@pytest.mark.parametrize('previous,to',[(0,10),(0,20),(5,5),(5,20),(10,50),(20,20),(20,50),(0,True)])
def test_controller_cannot_skip_restart_or_scale(previous,to):
    with pytest.raises(ValueError):p.stage_interval(previous,to)


def test_exact_twenty_total():
    assert list(p.stage_interval(0,5))+list(p.stage_interval(5,10))+list(p.stage_interval(10,20))==list(range(1,21))


def tiny_payload():
    from ktbench.training import capture_rng_state
    torch.manual_seed(42)
    config=HiTSKTConfig(8,3,width=8,heads=2,feedforward_width=12,dropout=.1)
    favor=FavorPlusConfig(features=8,chunk_size=2)
    model=HiTSKTFavorPlus(config,favor);optimizer=torch.optim.Adam(model.parameters(),lr=5e-5)
    ex=SessionExample([Session([1,2],[1,2],[1,0])],Session([2,3,4],[1,2,1],[0,1,0]),0)
    batch=collate_hitskt([ex],num_questions=8,num_skills=3);item=dict(dataset_index=0)
    from ktbench.metrics import masked_bce_loss
    # Synthetic unit fixture only, five tiny CPU steps create real Adam state.
    for _ in range(5):
        model.train();optimizer.zero_grad(set_to_none=True)
        masked_bce_loss(model(batch),batch.target_labels,batch.target_metric_mask).backward();optimizer.step()
    import ktbench.favor_plus_bounded_assist as bounded
    old=bounded.collate_selected;bounded.collate_selected=lambda *args:batch
    try:probe=continuation_probe(model,optimizer,None,item)
    finally:bounded.collate_selected=old
    prediction=p.fixed_logits(model,batch).cpu()
    contract=dict(run_id='SYNTHETIC_UNIT_FIXTURE',exact_training_and_probe_samples=[item]*21,
        parameter_tensor_count=len(list(model.parameters())),config=dict(model_config=asdict(config),favor_config=asdict(favor)),
        source_sha256={'unit':'fixture'},dataset_identity={'synthetic':True},configuration_sha256='fixture')
    payload=cpu_tree(model.checkpoint())
    payload.update(kind=r.KIND,contract=contract,optimizer_updates=5,optimizer_step_calls=5,
        sampler=dict(cursor=5,epoch=1,seed=42,exact_samples=contract['exact_training_and_probe_samples']),
        optimizer_state_dict=cpu_tree(optimizer.state_dict()),rng_state=cpu_tree(capture_rng_state()),
        reference_eval_logits=prediction,next_update_probe=probe)
    r.seal(payload)
    return payload,contract,batch


@pytest.fixture
def checkpoint_fixture(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    return tiny_payload()


def test_atomic_roundtrip_and_no_overwrite(tmp_path,checkpoint_fixture):
    payload,contract,_=checkpoint_fixture;path=tmp_path/'checkpoint_005.pt'
    result=r.atomic_checkpoint(path,payload,contract)
    assert result['status']=='PASS' and result['atomic_no_overwrite']
    original=path.read_bytes()
    with pytest.raises(FileExistsError):r.atomic_checkpoint(path,payload,contract)
    assert path.read_bytes()==original


def test_interrupted_publication_never_eligible(tmp_path,checkpoint_fixture,monkeypatch):
    payload,contract,_=checkpoint_fixture
    def fail(*args):raise OSError('synthetic filesystem interruption')
    monkeypatch.setattr(os,'link',fail)
    with pytest.raises(OSError):r.atomic_checkpoint(tmp_path/'checkpoint_005.pt',payload,contract)
    assert not (tmp_path/'checkpoint_005.pt').exists() and list(tmp_path.glob('*.partial.pt'))


@pytest.mark.parametrize('corruption',['weights','adam','rng','projection','cursor','samples','source','config','calls'])
def test_corruption_rejected_before_publication(tmp_path,checkpoint_fixture,corruption):
    payload,contract,_=checkpoint_fixture;payload=copy.deepcopy(payload)
    if corruption=='weights':next(iter(payload['state_dict'].values())).view(-1)[0]+=1
    elif corruption=='adam':next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg'].view(-1)[0]+=1
    elif corruption=='rng':payload['rng_state']['torch_cpu'][0]^=1
    elif corruption=='projection':next(v for k,v in payload['state_dict'].items() if k.endswith('.projection')).view(-1)[0]+=1
    elif corruption=='cursor':payload['sampler']['cursor']=4
    elif corruption=='samples':payload['sampler']['exact_samples']=[{'dataset_index':999}]*21
    elif corruption=='source':payload['contract']['source_sha256']['unit']='changed'
    elif corruption=='config':payload['config']['dropout']=0.
    else:payload['optimizer_step_calls']=6
    with pytest.raises((ValueError,FloatingPointError)):
        r.atomic_checkpoint(tmp_path/'checkpoint_005.pt',payload,contract)
    assert not list(tmp_path.iterdir())


def test_nonfinite_checkpoint_not_published(tmp_path,checkpoint_fixture):
    payload,contract,_=checkpoint_fixture
    next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg_sq'].view(-1)[0]=float('inf');r.seal(payload)
    with pytest.raises(FloatingPointError):r.atomic_checkpoint(tmp_path/'checkpoint_005.pt',payload,contract)


def test_fresh_process_recovery_predictions_adam_rng_next_gradients(tmp_path,checkpoint_fixture):
    payload,contract,batch=checkpoint_fixture;path=tmp_path/'checkpoint_005.pt'
    r.atomic_checkpoint(path,payload,contract);torch.save(dict(batch=vars(batch),contract=contract),tmp_path/'inputs.pt')
    code='''
import json,sys,torch
from pathlib import Path
from ktbench.favor_plus_pilot_recovery import no_step_recovery
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.batching import HiTSKTBatch
from ktbench.favor_plus_assist20 import fixed_logits
from ktbench.favor_plus_bounded_assist import continuation_probe
import ktbench.favor_plus_bounded_assist as bounded
torch.set_num_threads(1)
p=Path(sys.argv[1]);payload=torch.load(p/'checkpoint_005.pt',map_location='cpu',weights_only=True)
i=torch.load(p/'inputs.pt',map_location='cpu',weights_only=True);batch=HiTSKTBatch(**i['batch'])
model=HiTSKTFavorPlus.from_checkpoint(p/'checkpoint_005.pt');optimizer=torch.optim.Adam(model.parameters(),lr=5e-5)
bounded.collate_selected=lambda *args:batch
result=no_step_recovery(payload,model,optimizer,i['contract'],evaluate_logits=lambda model:fixed_logits(model,batch),
 probe=lambda model,opt:continuation_probe(model,opt,None,{'dataset_index':0}))
assert all(float(s['step'])==5 for s in optimizer.state.values())
print(json.dumps(result))
'''
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    run=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=env,capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr
    result=json.loads(run.stdout);assert result['status']=='PASS' and result['optimizer_calls_during_recovery']==0


def test_failure_uses_cached_memory_never_cuda(tmp_path,monkeypatch):
    (tmp_path/'checkpoints').mkdir();out=tmp_path/'stage';out.mkdir()
    def forbidden(*a,**kw):raise AssertionError('failure handler may not touch CUDA')
    for name in ('memory_allocated','max_memory_allocated','max_memory_reserved','get_rng_state_all','is_available'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    snap=dict(conflicts=[{'pid':42,'executable':'/fixture/python','identity_complete':True}])
    b=type('B',(),{'last':dict(peak_allocated_bytes=123,peak_reserved_bytes=456,host={'peak_process_rss_bytes':789})})()
    p.record_failure(tmp_path,out,t.ResourceConflict('fixture',snap),'post_update',6,6,b,0.)
    value=json.loads((out/'failure.json').read_text())
    assert value['last_safe_resources']['peak_reserved_bytes']==456 and value['conflict_snapshot']==snap
    assert value['no_automatic_restart'] and value['no_cuda_calls_in_failure_handler']


def test_same_pid_exec_change_cannot_inherit_monitor_permission(tmp_path):
    proc,exe=fixture_process(tmp_path);trust=t.trusted_binaries([exe])
    other=tmp_path/'python';other.write_bytes(b'UNKNOWN-EXECUTABLE')
    def race(path):
        if path.name=='20':(path/'exe').unlink();(path/'exe').symlink_to(other)
    result=t.device_snapshot(tmp_path,own_pid=1,trusted=trust,after_identity=race)
    assert result['conflicts'] and 'executable changed' in result['holders'][0]['identity_error']


def test_partial_identity_still_retains_uid_parent_and_name(tmp_path):
    proc,exe=fixture_process(tmp_path);(proc/'exe').unlink()
    result=t.device_snapshot(tmp_path,own_pid=1,trusted={})
    assert result['conflicts']
    row=result['holders'][0]
    assert row['uid']==[0,0,0,0] and row['parent_pid']==1 and row['comm']=='nvidia-smi'
    assert not row['verified_read_only']


def test_boundary_conflict_does_not_query_cuda_and_keeps_previous_memory(monkeypatch,tmp_path):
    cached={'peak_allocated_bytes':123,'peak_reserved_bytes':456}
    b=p.Boundaries({},t.PILOT_ACTION,'2026-10-20T00:00:00+00:00',{},tmp_path);b.last=cached
    def conflict(*args,**kwargs):raise t.ResourceConflict('UNIT FIXTURE',{'conflicts':[{'pid':42}]})
    def forbidden(*args,**kwargs):raise AssertionError('no CUDA stats after conflict')
    monkeypatch.setattr(p,'admit',conflict)
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    with pytest.raises(t.ResourceConflict):b.check('UNIT FIXTURE')
    assert b.last==cached


@pytest.mark.parametrize('missing',['hold','exposure','conflict','inaccessible'])
def test_actual_hold_exposure_and_unknown_conflicts_remain_required(tmp_path,monkeypatch,missing):
    hold=tmp_path/'hold.json';hold.write_text('{"status":"USER_TRAINING_HOLD_ACTIVE"}')
    exposed=tmp_path/'exposed';exposed.mkdir();(exposed/'fixture').touch()
    monkeypatch.setattr(t,'LIGHT_HOLD',hold);monkeypatch.setattr(t,'DEVICE_EXPOSURE',exposed)
    monkeypatch.setattr(t,'require_gpu_authorization',lambda *args:'fixture')
    snapshot=dict(conflicts=[],inaccessible_fd_pids=[],light_processes=[])
    if missing=='hold':hold.write_text('{"status":"INACTIVE"}')
    elif missing=='exposure':(exposed/'fixture').unlink()
    elif missing=='conflict':snapshot['conflicts']=[{'pid':999}]
    else:snapshot['inaccessible_fd_pids']=[999]
    with pytest.raises(RuntimeError):
        t.admit({'user_approval':{'actions':[t.PILOT_ACTION]}},t.PILOT_ACTION,snapshot_fn=lambda:snapshot)


def test_literal_legacy_race_blocks_verified_monitor_but_identity_first_retains_it(tmp_path,monkeypatch):
    """Reproduce old descriptor-before-identity false conflict, no GPU process."""
    from ktbench.favor_plus_gpu_access import conflicting_device_processes
    binary=Path('/usr/bin/nvidia-smi')
    if not binary.is_file():pytest.skip('fixture requires local management binary, never executes it')
    old_root=tmp_path/'legacy';new_root=tmp_path/'identity_first';old_root.mkdir();new_root.mkdir()
    fixture_process(old_root,executable=binary);fixture_process(new_root,executable=binary)
    trust=t.trusted_binaries([binary]);real=os.readlink
    def exit_after_fd(path,*args,**kwargs):
        target=real(path,*args,**kwargs);path=Path(path)
        if path.name=='3' and path.parent.name=='fd':
            for name in ('exe','cmdline','stat'):(path.parent.parent/name).unlink(missing_ok=True)
        return target
    monkeypatch.setattr(os,'readlink',exit_after_fd)
    assert conflicting_device_processes(old_root,own_pid=1)==[20]
    new=t.device_snapshot(new_root,own_pid=1,trusted=trust)
    assert not new['conflicts'] and new['holders'][0]['verified_read_only']
    assert new['holders'][0]['exited_after_identity']
