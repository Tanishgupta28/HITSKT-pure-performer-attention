"""Small CPU-only controls for the separately bounded100-update study."""
import copy
import datetime as dt
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ktbench import favor_plus_bounded_assist as b
from ktbench.batching import Session, SessionExample, collate_hitskt
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus


@pytest.fixture
def design():
    return json.loads(b.DESIGN.read_text())


@pytest.mark.parametrize('previous,milestone,count',[(0,20,20),(20,50,30),(50,100,50)])
def test_exact_campaign_continuation_ranges(previous,milestone,count):
    r=list(b.stage_range(previous,milestone))
    assert len(r)==count and r[0]==previous+1 and r[-1]==milestone


@pytest.mark.parametrize('previous,milestone',[(0,50),(0,100),(5,20),(20,20),(50,50),(100,100),(0,True),(0,101)])
def test_no_skip_restart_extra_or_unapproved_milestone(previous,milestone):
    with pytest.raises(ValueError): b.stage_range(previous,milestone)


@pytest.mark.parametrize('field,value,exception',[
    ('elapsed',2701,TimeoutError),('elapsed',-1,TimeoutError),
    ('calls',101,RuntimeError),('allocated',8*1024**3+1,MemoryError)])
def test_cumulative_deadline_and_resources_across_processes(design,field,value,exception):
    data=dict(elapsed=1,calls=20,allocated=100)
    data[field]=value
    start=dt.datetime(2026,10,20,tzinfo=dt.timezone.utc)
    with pytest.raises(exception):
        b.budget_check(design,start.isoformat(),data['calls'],data['allocated'],clock=start+dt.timedelta(seconds=data['elapsed']))


def test_maximum_authorized_boundaries_are_inclusive(design):
    start=dt.datetime(2026,10,20,tzinfo=dt.timezone.utc)
    b.budget_check(design,start.isoformat(),100,8*1024**3,clock=start+dt.timedelta(seconds=2700))


def test_five_step_approval_cannot_authorize_bounded_training(monkeypatch,tmp_path):
    def forbidden(*a,**k): raise AssertionError('not reached')
    monkeypatch.setattr(b,'require_execution_access',forbidden)
    monkeypatch.setattr(b,'validate_plan',forbidden)
    for name in ('is_available','init','get_device_properties'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    with pytest.raises(ValueError,match='bounded100'):
        b.execute_stage(tmp_path/'missing',{'user_approval':{'actions':['favor_plus_assist2017_five_step_pilot']}},'never',20)
    assert not list(tmp_path.iterdir())


def test_bounded_approval_still_calls_existing_allocation_and_conflict_gate(monkeypatch,tmp_path):
    hold=tmp_path/'hold.json';hold.write_text('{"status":"USER_TRAINING_HOLD_ACTIVE"}')
    monkeypatch.setattr(b,'LIGHT_HOLD',hold)
    seen=[]
    monkeypatch.setattr(b,'require_execution_access',lambda *args:seen.append(args) or 'ok')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','fixture')
    record={'user_approval':{'actions':[b.ACTION]}}
    assert b.authorize_bounded(record)=='ok'
    assert seen==[(record,'fixture',b.CUDA_VALIDATION)]


def diagnostic(error=.25,ess=8.,growth=1.,kernel=1e-7):
    return {'stage':dict(feature_versus_exact_softmax={'relative_l2':error},
        kernel_statistics={'feature_ess':{'median':ess}},implementation_error={'max_abs':kernel},
        p95_ratio_to_initial={'query':growth,'key':1.})}


def test_old_review_trigger_does_not_become_a_pass_claim(design):
    result=b.severe_check([diagnostic()],design)
    assert result['status']=='CONTINUE_BOUNDED_DIAGNOSTIC_ONLY'
    assert result['original_softmax_review_trigger']==.20 and result['full_training_ready'] is False


@pytest.mark.parametrize('kwargs',[{'error':.8},{'error':1.6},{'ess':1.1},{'growth':2.1},{'kernel':5.1e-5},{'error':float('inf')}])
def test_severe_instability_stops_without_tolerance_relaxation(design,kwargs):
    assert b.severe_check([diagnostic(**kwargs)],design)['status']=='STOP'


def test_empty_diagnostics_are_not_verification(design):
    assert b.severe_check([],design)['status']=='STOP'


def tiny_setup():
    torch.manual_seed(9)
    model=HiTSKTFavorPlus(HiTSKTConfig(8,3,width=8,heads=2,feedforward_width=12,dropout=.1),FavorPlusConfig(features=8,chunk_size=2))
    ex=SessionExample([Session([1,2],[1,2],[1,0])],Session([2,3,4],[1,2,1],[0,1,0]),0)
    batch=collate_hitskt([ex],num_questions=8,num_skills=3)
    item={'dataset_index':0}
    dataset=SimpleNamespace(__getitem__=lambda i:ex)
    return model,ex,batch,item,dataset


def test_raw_response_causal_guard_runs_and_detects_a_leaking_model():
    model,ex,batch,item,_=tiny_setup()
    class Dataset:
        def __getitem__(self,i):return ex
    assert b.causal_check(model,Dataset(),item)['status']=='PASS'
    class Leaking(HiTSKTFavorPlus):
        def forward(self,batch):
            # next shifted input contains current raw response
            return batch.target_correct_input.roll(-1,1).float()
    bad=Leaking(model.config,model.favor_config)
    with pytest.raises(RuntimeError,match='raw own/future'):
        b.causal_check(bad,Dataset(),item)


def test_continuation_probe_preserves_rng_and_never_updates_parameters(monkeypatch):
    model,ex,batch,item,dataset=tiny_setup()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    # CPU-only fixture; trap CUDA calls in shared RNG helpers.
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(b,'collate_selected',lambda *args:batch)
    before=b.cpu_tree(model.state_dict());rng=torch.get_rng_state().clone()
    first=b.continuation_probe(model,optimizer,dataset,item)
    second=b.continuation_probe(model,optimizer,dataset,item)
    assert first==second and first['optimizer_step_calls']==0
    assert torch.equal(rng,torch.get_rng_state())
    assert b.equal_tree(before,b.cpu_tree(model.state_dict())) and optimizer.state_dict()['state']=={}


def test_metric_contract_and_target_mask_exclude_eos(monkeypatch):
    model,ex,batch,item,dataset=tiny_setup()
    monkeypatch.setattr(b,'collate_selected',lambda *args:batch)
    result,prob,label=b.evaluate(model,dataset,[item])
    assert result['targets']==3 and len(prob)==len(label)==3
    assert 0<=result['roc_auc']<=1 and set(label.tolist())=={0.,1.}
    assert all(key in result for key in ('loss','accuracy','precision','recall','f1','mse'))


def test_released_restart_hold_refuses_cuda(monkeypatch,tmp_path):
    hold=tmp_path/'hold.json';hold.write_text('{"status":"INACTIVE"}')
    monkeypatch.setattr(b,'LIGHT_HOLD',hold)
    with pytest.raises(RuntimeError,match='not active'):
        b.authorize_bounded({'user_approval':{'actions':[b.ACTION]}})


def test_validation_disables_dropout_and_restores_training_mode(monkeypatch):
    model,ex,batch,item,dataset=tiny_setup()
    monkeypatch.setattr(b,'collate_selected',lambda *args:batch)
    observed=[]
    handle=model.register_forward_pre_hook(lambda module,args:observed.append([
        m.training for m in module.modules() if isinstance(m,torch.nn.Dropout)]))
    before=torch.get_rng_state().clone()
    try:b.evaluate(model,dataset,[item])
    finally:handle.remove()
    assert observed and not any(observed[0])
    assert model.training and all(m.training for m in model.modules() if isinstance(m,torch.nn.Dropout))
    assert torch.equal(before,torch.get_rng_state())


def test_legacy_helper_restores_train_mode_so_no_grad_is_not_eval():
    model,ex,batch,item,dataset=tiny_setup()
    before=torch.get_rng_state().clone()
    b.fixed_example_diagnostics(model,batch)
    assert model.training and torch.equal(before,torch.get_rng_state())
    seen=[]
    handle=model.register_forward_pre_hook(lambda module,args:seen.append([
        m.training for m in module.modules() if isinstance(m,torch.nn.Dropout)]))
    try:
        with torch.no_grad():model(batch)
    finally:handle.remove()
    assert all(seen[0]) and not torch.equal(before,torch.get_rng_state())
