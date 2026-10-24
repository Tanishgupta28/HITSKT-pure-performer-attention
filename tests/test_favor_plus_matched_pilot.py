"""Approval/recovery preparation, CPU only; deliberately never call Adam.step."""
import json
import pytest
import torch
from ktbench import favor_plus_matched_pilot as p
from ktbench import favor_plus_matched_comparison as m
from ktbench.favor_plus_pilot_recovery import digest
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.training import capture_rng_state,restore_rng_state
from test_favor_plus import example,collate


@pytest.mark.parametrize('action',('execute','recover'))
def test_training_proposal_refused_before_any_cuda_or_optimizer(action,tmp_path,monkeypatch):
    record=tmp_path/'record.json';record.write_text(json.dumps({'action':m.VERIFY_ACTION}))
    monkeypatch.setattr(p,'initialize_cuda',lambda *a:pytest.fail('no CUDA before new approval'))
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('no optimizer before new approval'))
    with pytest.raises(ValueError):getattr(p,action)('orf128',record)


@pytest.mark.parametrize('arm',m.ARMS)
def test_zero_step_training_checkpoint_preserves_model_rng_projection_and_probe(arm,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:pytest.fail('no optimizer updates in preparation'))
    cfg=HiTSKTConfig(20,5,width=16,heads=2,feedforward_width=32,dropout=.1)
    net=m.fresh_model(cfg,arm,accumulation_dtype='float64');b=collate([example(4,2),example(2,1)])
    opt=torch.optim.Adam(net.parameters(),lr=5e-5)
    identity={'arm':arm,'projection_sha256':{n:digest(v) for n,v in net.named_buffers() if n.endswith('.projection')}}
    plan={'fresh_identities':[identity]};path=tmp_path/'step0.pt';before=(m.learned_digest(net),digest(cpu_tree(capture_rng_state())))
    receipt=p.checkpoint(net,opt,plan,0,b,path);v=torch.load(path,weights_only=True)
    assert receipt['update']==0 and v['optimizer_state_dict']['state']=={}
    assert (m.learned_digest(net),digest(cpu_tree(capture_rng_state())))==before
    loaded=m.MatchedComparisonModel.from_checkpoint(path);restored=torch.optim.Adam(loaded.parameters(),lr=5e-5)
    restored.load_state_dict(v['optimizer_state_dict']);restore_rng_state(v['rng_state'])
    assert torch.equal(p.fixed_logits(loaded,b),v['reference_logits'])
    assert p.probe(loaded,restored,b)==v['reference_probe']
    assert equal_tree(cpu_tree(restored.state_dict()),v['optimizer_state_dict'])
    assert digest(cpu_tree(capture_rng_state()))==before[1]
    with pytest.raises(FileExistsError):p.checkpoint(net,opt,plan,0,b,path)


def test_missing_adam_state_is_not_valid_for_completed_update(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k:pytest.fail('not allowed'))
    net=m.fresh_model(HiTSKTConfig(20,5,width=16,heads=2,feedforward_width=32,dropout=0.),'orf128',accumulation_dtype='float64')
    with pytest.raises(ValueError,match='Adam state'):
        p.checkpoint(net,torch.optim.Adam(net.parameters()),{},1,collate([example(4,2)]),tmp_path/'invalid.pt')
    assert not (tmp_path/'invalid.pt').exists()
