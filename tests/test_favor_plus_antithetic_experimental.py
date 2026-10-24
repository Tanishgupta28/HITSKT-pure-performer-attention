"""Separate variance-reduction prototype; no optimizer or genuine dataset runs."""
from dataclasses import replace
import pytest
import torch
from ktbench.models.favor_plus import causal_favor_attention,softmax_log_features,orthogonal_gaussian_projection
from ktbench.models.favor_plus_antithetic_experimental import (
    AntitheticFavorConfig,HiTSKTFavorPlusAntithetic,antithetic_projection,VARIANT,
)
from ktbench.models.hitskt import HiTSKT
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.metrics import masked_bce_loss
from ktbench.batching import Session,SessionExample
from test_favor_plus import config,example,collate,explicit_reference


@pytest.fixture(autouse=True)
def cpu_only():
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    torch.set_num_threads(1)


@pytest.mark.parametrize('total',[2,128,512])
def test_antithetic_projection_total_budget_rng_seed_blocks_and_pairs(total):
    torch.random.default_generator.manual_seed(19);rng=torch.random.get_rng_state().clone()
    w=antithetic_projection(4,total,64,101)
    assert w.shape==(4,total,64) and torch.equal(w[:,:total//2],-w[:,total//2:])
    assert torch.equal(rng,torch.random.get_rng_state())
    assert torch.equal(w,antithetic_projection(4,total,64,101))
    assert not torch.equal(w,antithetic_projection(4,total,64,102))
    for block in w[:,:total//2].split(64,dim=1):
        unit=block/block.norm(dim=-1,keepdim=True)
        torch.testing.assert_close(unit@unit.transpose(-1,-2),torch.eye(block.shape[1],dtype=w.dtype).expand(4,-1,-1),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('kwargs',[{'features_total':1},{'features_total':127},{'features_total':True},{'chunk_size':0},{'projection_seed':-1},{'sampling':'legacy'},{'redraw':'every_step'},{'accumulation_dtype':'float16'}])
def test_experimental_policy_must_be_explicit_and_valid(kwargs):
    with pytest.raises(ValueError):AntitheticFavorConfig(**kwargs)


def test_gaussian_marginal_kernel_expectation_and_positive_hyperbolic_scaling():
    w=antithetic_projection(128,128,4,12)
    q=torch.tensor([.2,-.3,.1,.4],dtype=torch.float64);k=torch.tensor([.3,.1,-.2,.2],dtype=torch.float64)
    # Each head is a separate independent estimator; exact target from Gaussian MGF.
    x,y=q/4**.25,k/4**.25
    estimates=torch.exp(torch.einsum('hmd,d->hm',w,x+y)-(x.square().sum()+y.square().sum())/2).mean(-1)
    assert abs(float(estimates.mean())-float(torch.exp(x@y)))<.015
    assert abs(float(w.square().mean())-1)<.04
    features=softmax_log_features(q[None,None,None].expand(1,128,1,4),w).exp()
    assert bool((features>0).all())
    torch.testing.assert_close(features.square().sum(-1),
        torch.exp(torch.einsum('hmd,d->hm',w,2*x)-x.square().sum()).mean(-1)[None,:,None])


def test_projection_matches_preregistered_frozen_nested_pool_arm():
    pool=orthogonal_gaussian_projection(4,512,64,17)
    actual=antithetic_projection(4,128,64,17)
    assert torch.equal(actual,torch.cat((pool[:,:64],-pool[:,:64]),dim=1))
    with pytest.raises(ValueError):AntitheticFavorConfig(features_total=128,projection_pool_features=63)


@pytest.mark.parametrize('length,chunk',[(1,1),(7,3),(17,4)])
@pytest.mark.parametrize('mask_kind',['valid','holes','empty'])
def test_experimental_feature_kernel_outputs_and_gradients_match_independent_reference(length,chunk,mask_kind):
    g=torch.Generator().manual_seed(length);inputs=[torch.randn(2,2,length,d,generator=g,dtype=torch.float64,requires_grad=True) for d in (4,4,3)]
    w=antithetic_projection(2,14,4,7);km=torch.ones(2,length,dtype=torch.bool);qm=km.clone()
    if mask_kind=='holes':km[:,::3]=False;qm[0,-1]=False
    if mask_kind=='empty':km[:]=False
    actual=causal_favor_attention(*inputs,w,query_mask=qm,key_mask=km,chunk_size=chunk)
    expected=explicit_reference(*inputs,w,qm,km)
    torch.testing.assert_close(actual,expected,atol=3e-12,rtol=3e-11)
    a=torch.autograd.grad(actual.square().sum(),inputs,retain_graph=True);b=torch.autograd.grad(expected.square().sum(),inputs)
    for left,right in zip(a,b):torch.testing.assert_close(left,right,atol=4e-11,rtol=4e-10)


def test_finite_difference_gradient_check_crosses_scan_boundaries():
    torch.random.default_generator.manual_seed(3)
    inputs=tuple(torch.randn(1,1,4,2,dtype=torch.float64,requires_grad=True) for _ in range(3));w=antithetic_projection(1,6,2,10);mask=torch.tensor([[False,True,True,True]])
    assert torch.autograd.gradcheck(lambda q,k,v:causal_favor_attention(q,k,v,w,query_mask=mask,key_mask=mask,chunk_size=2),inputs,eps=1e-6,atol=2e-6,rtol=2e-4)


def net():
    torch.random.default_generator.manual_seed(11)
    return HiTSKTFavorPlusAntithetic(config(),AntitheticFavorConfig(features_total=12,chunk_size=3))


def test_original_nonattention_weights_and_initialization_rng_preserved():
    torch.random.default_generator.manual_seed(11);old=HiTSKT(config());rng=torch.random.get_rng_state().clone();new=net()
    assert torch.equal(rng,torch.random.get_rng_state())
    for key,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[key]),key
    assert sum(x.numel() for x in old.parameters())==sum(x.numel() for x in new.parameters())


@pytest.mark.parametrize('length,history',[(1,1),(4,3),(7,2)])
def test_full_model_no_own_or_future_response_label_leakage(length,history):
    model=net().eval();e=example(length,history);batch=collate([e,example(8,2)])
    with torch.no_grad():original=model(batch)
    for i in range(length):
        labels=list(e.target.correct);labels[i]=1-labels[i]
        changed=SessionExample(e.history,Session(e.target.question_ids,e.target.skill_ids,labels),0)
        with torch.no_grad():actual=model(collate([changed,example(8,2)]))
        assert torch.equal(original[0,:i+1],actual[0,:i+1]) and torch.equal(original[1],actual[1])
    batch.target_labels.fill_(99)
    with torch.no_grad():assert torch.equal(original,model(batch))


def test_full_model_padding_eos_gradients_and_checkpoint_identity(tmp_path):
    model=net();batch=collate([example(5,3),example(2,1)])
    loss=masked_bce_loss(model(batch),batch.target_labels,batch.target_metric_mask);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():original=model(batch)
    assert batch.target_attention_mask.sum()==9 and batch.target_metric_mask.sum()==7
    batch.target_correct_input[~batch.target_attention_mask]=1;batch.history_correct[~batch.history_mask]=1
    with torch.no_grad():assert torch.equal(original,model(batch))
    path=tmp_path/'experimental.pt';torch.save(model.checkpoint(),path)
    saved=torch.load(path,weights_only=True);assert saved['variant']==VARIANT and saved['antithetic_config']['features_total']==12
    loaded=HiTSKTFavorPlusAntithetic.from_checkpoint(path).eval()
    with torch.no_grad():assert torch.equal(original,loaded(batch))
    with pytest.raises(ValueError):HiTSKTFavorPlus.from_checkpoint(path)
    with pytest.raises(ValueError):HiTSKT.from_checkpoint(path)
    old=tmp_path/'original.pt';torch.save(HiTSKTFavorPlus(config()).checkpoint(),old)
    with pytest.raises(ValueError):HiTSKTFavorPlusAntithetic.from_checkpoint(old)


@pytest.mark.parametrize('length',[65,257])
def test_long_masked_prefix_gradients_and_future_invariance(length):
    g=torch.Generator().manual_seed(44);q,k,v=[torch.randn(1,2,length,4,generator=g,requires_grad=True) for _ in range(3)]
    w=antithetic_projection(2,16,4,9).float();mask=torch.arange(length)[None]<length-3
    out=causal_favor_attention(q,k,v,w,query_mask=mask,key_mask=mask,chunk_size=16);out.square().mean().backward()
    assert torch.isfinite(out).all() and all(torch.isfinite(x.grad).all() and x.grad[:,:,-3:].eq(0).all() for x in (q,k,v))
    cut=length//2;k2=k.detach().clone();v2=v.detach().clone();k2[:,:,cut:]+=5;v2[:,:,cut:]-=3
    with torch.no_grad():changed=causal_favor_attention(q.detach(),k2,v2,w,query_mask=mask,key_mask=mask,chunk_size=16)
    torch.testing.assert_close(out[:,:,:cut],changed[:,:,:cut],atol=0,rtol=0)
