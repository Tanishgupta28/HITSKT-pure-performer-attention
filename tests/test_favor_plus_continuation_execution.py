"""Synthetic CPU tests for the new continuation, not real ASSIST training."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from ktbench import favor_plus_assist50 as p
from ktbench import favor_plus_continuation_recovery as r
from ktbench import favor_plus_bounded_assist as bounded
from ktbench.favor_plus_continuation_contract import ACTION,PARENT_SHA256
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree
from ktbench.batching import Session,SessionExample,collate_hitskt
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.metrics import masked_bce_loss


@pytest.fixture(scope='module')
def tiny_state():
    from ktbench.training import capture_rng_state
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1')
    torch.set_num_threads(1);torch.manual_seed(42)
    config=HiTSKTConfig(8,3,width=8,heads=2,feedforward_width=12,dropout=.1)
    favor=FavorPlusConfig(features=8,chunk_size=2)
    net=HiTSKTFavorPlus(config,favor);opt=torch.optim.Adam(net.parameters(),lr=5e-5)
    ex=SessionExample([Session([1,2],[1,2],[1,0])],Session([2,3,4],[1,2,1],[0,1,0]),0)
    batch=collate_hitskt([ex],num_questions=8,num_skills=3)
    for _ in range(25):
        net.train();opt.zero_grad(set_to_none=True)
        masked_bce_loss(net(batch),batch.target_labels,batch.target_metric_mask).backward();opt.step()
    item={'dataset_index':0};old=bounded.collate_selected;bounded.collate_selected=lambda *args:batch
    try:probe=bounded.continuation_probe(net,opt,None,item)
    finally:bounded.collate_selected=old
    weights=cpu_tree(net.state_dict())
    contract=dict(config=dict(model_config=asdict(config),favor_config=asdict(favor)),
        parameter_tensor_count=len(list(net.parameters())),exact_training_and_probe_samples=[item]*51,
        parent_checkpoint_sha256=PARENT_SHA256,parent_optimizer_param_groups=cpu_tree(opt.state_dict()['param_groups']),
        parent_projection_sha256={k:r.digest(v) for k,v in weights.items() if k.endswith('.projection')},
        source_sha256={'synthetic':'CPU_ONLY_UNIT_FIXTURE'})
    contract=json.loads(json.dumps(contract))  # same on-disk JSON representation as real plan
    payload=cpu_tree(net.checkpoint())
    payload.update(kind=r.KIND,contract=contract,optimizer_updates=25,optimizer_step_calls=25,additional_optimizer_calls=5,
        optimizer_state_dict=cpu_tree(opt.state_dict()),rng_state=cpu_tree(capture_rng_state()),
        sampler=dict(cursor=25,epoch=1,seed=42,exact_samples=contract['exact_training_and_probe_samples']),
        reference_eval_logits=p.original.fixed_logits(net,batch).cpu(),next_update_probe=probe)
    r.seal(payload);return payload,contract,batch


def test_new_atomic_roundtrip_and_no_overwrite(tmp_path,tiny_state):
    value,contract,_=tiny_state;path=tmp_path/'checkpoint_025.pt'
    receipt=r.atomic_checkpoint(path,value,contract)
    assert receipt['updates']==25 and receipt['additional_optimizer_calls']==5
    assert equal_tree(value,torch.load(path,map_location='cpu',weights_only=True))
    with pytest.raises(FileExistsError):r.atomic_checkpoint(path,value,contract)


def test_json_plan_preserves_all_adam_group_values(tiny_state):
    value,contract,_=tiny_state
    saved=value['optimizer_state_dict']['param_groups'];planned=contract['parent_optimizer_param_groups']
    assert isinstance(saved[0]['betas'],tuple) and isinstance(planned[0]['betas'],list)
    assert r.optimizer_groups_equal(saved,planned)
    changed=copy.deepcopy(planned);changed[0]['betas'][0]=.8
    assert not r.optimizer_groups_equal(saved,changed)
    assert r.validate(value,contract)['status']=='PASS'


@pytest.mark.parametrize('corruption',['weights','adam','rng','projection','cursor','samples','source','config','calls','additional','groups'])
def test_new_contract_corruption_rejected(tmp_path,tiny_state,corruption):
    value,contract,_=tiny_state;v=copy.deepcopy(value)
    if corruption=='weights':next(iter(v['state_dict'].values())).view(-1)[0]+=1
    elif corruption=='adam':next(iter(v['optimizer_state_dict']['state'].values()))['exp_avg'].view(-1)[0]+=1
    elif corruption=='rng':v['rng_state']['torch_cpu'][0]^=1
    elif corruption=='projection':next(x for k,x in v['state_dict'].items() if k.endswith('.projection')).view(-1)[0]+=1
    elif corruption=='cursor':v['sampler']['cursor']=20
    elif corruption=='samples':v['sampler']['exact_samples']=[{'dataset_index':999}]*51
    elif corruption=='source':v['contract']['source_sha256']['synthetic']='CHANGED'
    elif corruption=='config':v['config']['dropout']=0
    elif corruption=='calls':v['optimizer_step_calls']=26
    elif corruption=='additional':v['additional_optimizer_calls']=25
    else:v['optimizer_state_dict']['param_groups'][0]['lr']=.001;r.seal(v)
    with pytest.raises((ValueError,FloatingPointError)):r.atomic_checkpoint(tmp_path/'checkpoint.pt',v,contract)
    assert not list(tmp_path.iterdir())


def test_resealed_redraw_and_negative_moments_refused(tiny_state):
    value,contract,_=tiny_state
    for corruption in ('projection','second_moment'):
        v=copy.deepcopy(value)
        if corruption=='projection':next(x for k,x in v['state_dict'].items() if k.endswith('.projection')).view(-1)[0]+=1
        else:next(iter(v['optimizer_state_dict']['state'].values()))['exp_avg_sq'].view(-1)[0]=-1
        r.seal(v)
        with pytest.raises(ValueError):r.validate(v,contract)


@pytest.mark.parametrize('step',[0,20,21,24,26,49,51,100,True,25.0])
def test_only_valid_checkpoint_totals(tiny_state,step):
    value,contract,_=tiny_state;v=copy.deepcopy(value)
    v.update(optimizer_updates=step,optimizer_step_calls=step,additional_optimizer_calls=step-20)
    with pytest.raises(ValueError):r.validate(v,contract)


def test_incomplete_write_retains_partial_not_published(tmp_path,tiny_state,monkeypatch):
    value,contract,_=tiny_state
    def fail(*args):raise OSError('UNIT simulated publication interruption')
    monkeypatch.setattr(os,'link',fail)
    with pytest.raises(OSError):r.atomic_checkpoint(tmp_path/'checkpoint_025.pt',value,contract)
    assert not (tmp_path/'checkpoint_025.pt').exists()
    assert len(list(tmp_path.glob('*.partial.pt')))==1


def test_fresh_process_exact_continuation_recovery_zero_steps(tmp_path,tiny_state):
    value,contract,batch=tiny_state
    r.atomic_checkpoint(tmp_path/'checkpoint_025.pt',value,contract)
    torch.save({'batch':vars(batch),'contract':contract},tmp_path/'inputs.pt')
    code='''
import json,sys,torch
from pathlib import Path
from ktbench.favor_plus_continuation_recovery import no_step_recovery
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.batching import HiTSKTBatch
from ktbench.favor_plus_assist20 import fixed_logits
import ktbench.favor_plus_bounded_assist as b
torch.set_num_threads(1);p=Path(sys.argv[1])
v=torch.load(p/'checkpoint_025.pt',map_location='cpu',weights_only=True)
i=torch.load(p/'inputs.pt',map_location='cpu',weights_only=True);batch=HiTSKTBatch(**i['batch'])
model=HiTSKTFavorPlus.from_checkpoint(p/'checkpoint_025.pt');opt=torch.optim.Adam(model.parameters(),lr=5e-5)
b.collate_selected=lambda *args:batch
result=no_step_recovery(v,model,opt,i['contract'],evaluate_logits=lambda m:fixed_logits(m,batch),
 probe=lambda m,o:b.continuation_probe(m,o,None,{'dataset_index':0}))
assert all(float(s['step'])==25 for s in opt.state.values())
assert not torch.cuda.is_initialized()
print(json.dumps(result))
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],text=True,capture_output=True,timeout=60,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    assert result.returncode==0,result.stderr
    check=json.loads(result.stdout)
    assert check['status']=='PASS' and check['optimizer_calls_during_recovery']==0 and check['optimizer_updates']==25


def test_checkpoint_recovery_prohibits_hidden_step(tiny_state,monkeypatch):
    value,contract,batch=tiny_state
    net=HiTSKTFavorPlus(HiTSKTConfig(**value['config']),FavorPlusConfig(**value['favor_config']))
    opt=torch.optim.Adam(net.parameters(),lr=5e-5)
    def hidden_step(m,o):o.step()
    with pytest.raises(RuntimeError,match='must not take'):
        r.no_step_recovery(value,net,opt,contract,evaluate_logits=lambda m:p.original.fixed_logits(m,batch),probe=hidden_step)


def test_new_scope_gate_before_model_plan_or_cuda(monkeypatch,tmp_path):
    def forbidden(*a,**k):raise AssertionError('must reject before model/data/CUDA')
    monkeypatch.setattr(p,'load_plan',forbidden);monkeypatch.setattr(p.original,'initialize',forbidden)
    with pytest.raises(ValueError,match='explicit20→50'):
        p.execute(tmp_path/'missing.json',{'user_approval':{'actions':[p.telemetry.PILOT_ACTION]}},30)


def test_conflict_boundary_retains_cache_without_cuda(monkeypatch,tmp_path):
    def fail(*a,**k):raise p.telemetry.ResourceConflict('UNIT UNKNOWN OWNER',{'conflicts':[{'pid':123}]})
    def forbidden(*a,**k):raise AssertionError('no CUDA after failed ownership check')
    monkeypatch.setattr(p,'admit',fail)
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    b=p.Boundaries({},'2026-10-20T00:00:00+00:00',{},tmp_path);b.last={'peak_allocated_bytes':1}
    with pytest.raises(p.telemetry.ResourceConflict):b.check('UNIT')
    assert b.last=={'peak_allocated_bytes':1}


def test_old_checkpoint_validator_still_refuses_new_kind(tiny_state):
    from ktbench.favor_plus_pilot_recovery import validate
    value,contract,_=tiny_state
    with pytest.raises(ValueError):validate(value,contract)


@pytest.mark.parametrize('same_pid',[False,True])
def test_nvml_identity_requires_exact_own_pid_without_alias_assumptions(tmp_path,monkeypatch,same_pid):
    snap=dict(conflicts=[],holders=[],inaccessible_fd_pids=[],light_processes=[])
    monkeypatch.setattr(p,'admit',lambda *a:('UNIT_UUID',snap))
    monkeypatch.setattr(p.original,'check_host_resources',lambda:{'unit':'CPU fixture'})
    memory=dict(total_bytes=40*1024**3,free_bytes=39*1024**3,used_bytes=1024**3,
        compute_processes=[{'pid':os.getpid() if same_pid else os.getpid()+1}])
    monkeypatch.setattr(p.telemetry,'nvml_snapshot',lambda *a:memory)
    def forbidden(*a,**k):raise AssertionError('must not query CUDA for metadata-only boundary')
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    b=p.Boundaries({},p.original.utc().isoformat(),{'max_execution_seconds':2700},tmp_path)
    b.last={'cached_before_conflict':True}
    if same_pid:assert b.check('UNIT',cuda=False)['nvml']==memory
    else:
        with pytest.raises(p.telemetry.ResourceConflict):b.check('UNIT',cuda=False)
        assert b.last=={'cached_before_conflict':True}
