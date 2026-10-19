"""Independent FAVOR+ oracles. No ELU+1 pass counts as FAVOR+ evidence."""
from pathlib import Path

import pytest
import torch

from ktbench.batching import Session, SessionExample, collate_hitskt
from ktbench.metrics import masked_bce_loss
from ktbench.models.favor_plus import (FavorPlusConfig, causal_favor_attention,
    orthogonal_gaussian_projection, softmax_log_features)
from ktbench.models.hitskt import HiTSKT, HiTSKTConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus


def explicit_reference(q, k, v, omega, qm, km):
    """L² oracle: directly construct log of each positive-feature inner product.

    Does not call the production feature transform, prefix routine or rescaling.
    Double loop intentionally differs from production implementation.
    """
    d, m = q.shape[-1], omega.shape[1]
    rows = []
    for i in range(q.shape[2]):
        scores = []
        for j in range(i+1):
            x, y = q[:, :, i]/d**0.25, k[:, :, j]/d**0.25
            exponent = ((x+y).unsqueeze(-2)*omega.unsqueeze(0)).sum(-1)
            exponent = exponent - (x.square().sum(-1)+y.square().sum(-1)).unsqueeze(-1)/2
            scores.append(torch.logsumexp(exponent, -1)-torch.log(q.new_tensor(m)))
        score = torch.stack(scores, -1)
        valid = km[:, None, :i+1]
        has = valid.any(-1, keepdim=True)
        score = score.masked_fill(~valid, -torch.inf)
        score = torch.where(has, score, 0.0)
        weights = score.softmax(-1) * valid
        out = torch.einsum("bhj,bhjd->bhd", weights, v[:, :, :i+1])
        rows.append(torch.where(qm[:, None, i, None], out, 0.0))
    return torch.stack(rows, 2)


@pytest.mark.parametrize("features,dimension", [(1,1), (3,5), (5,5), (13,5), (64,8)])
def test_projection_blocks_seed_and_rng(features, dimension):
    torch.manual_seed(91)
    state = torch.random.get_rng_state().clone()
    a = orthogonal_gaussian_projection(2, features, dimension, 123)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(a, orthogonal_gaussian_projection(2, features, dimension, 123))
    assert not torch.equal(a, orthogonal_gaussian_projection(2, features, dimension, 124))
    for start in range(0, features, dimension):
        block = a[:, start:start+dimension]
        unit = block/block.norm(dim=-1, keepdim=True)
        identity = torch.eye(block.shape[1], dtype=a.dtype).expand(2,-1,-1)
        torch.testing.assert_close(unit@unit.transpose(-1,-2), identity, atol=1e-12, rtol=1e-12)


def test_projection_gaussian_marginals_and_chi_radii():
    a = orthogonal_gaussian_projection(128, 32, 8, 942)
    assert abs(a.mean().item()) < .025
    assert abs(a.square().mean().item()-1) < .035
    squared_radii = a.square().sum(-1)
    assert abs(squared_radii.mean().item()-8) < .2
    assert abs(squared_radii.var().item()-16) < 1.2
    assert squared_radii.std() > 1  # rejects fixed sqrt(d) radii


def test_features_positive_and_correct_scaling():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2,2,3,4,generator=g,dtype=torch.float64)
    omega = orthogonal_gaussian_projection(2, 7, 4, 7)
    actual = softmax_log_features(x,omega).exp()
    scaled = x/4**.25
    expected = torch.exp((scaled.unsqueeze(-2)*omega[None,:,None]).sum(-1)
                         -scaled.square().sum(-1,keepdim=True)/2)/7**.5
    assert (actual > 0).all()
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-13)


@pytest.mark.parametrize("length,chunk", [(1,1),(2,1),(7,2),(9,4),(11,32)])
@pytest.mark.parametrize("mask_kind", ["all", "holes", "empty"])
def test_independent_output_and_gradient_reference(length,chunk,mask_kind):
    g = torch.Generator().manual_seed(length+chunk)
    inputs = [torch.randn(2,2,length,d,generator=g,dtype=torch.float64,requires_grad=True)
              for d in (4,4,3)]
    q,k,v = inputs
    omega = orthogonal_gaussian_projection(2,9,4,32)
    km = torch.ones(2,length,dtype=torch.bool)
    qm = km.clone()
    if mask_kind == "holes":
        km[:,::3] = False
        qm[0,1::2] = False
    if mask_kind == "empty":
        km[:] = False
    actual = causal_favor_attention(q,k,v,omega,query_mask=qm,key_mask=km,chunk_size=chunk)
    expected = explicit_reference(q,k,v,omega,qm,km)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=2e-11)
    multiplier = torch.randn(actual.shape,generator=g,dtype=actual.dtype)
    ga = torch.autograd.grad((actual*multiplier).sum(),inputs,retain_graph=True)
    ge = torch.autograd.grad((expected*multiplier).sum(),inputs)
    for a,e in zip(ga,ge):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,e,atol=3e-11,rtol=3e-10)


def test_finite_difference_gradcheck_chunk_boundaries():
    torch.manual_seed(9)
    args = tuple(torch.randn(1,1,4,2,dtype=torch.float64,requires_grad=True) for _ in range(3))
    omega = orthogonal_gaussian_projection(1,3,2,65)
    mask = torch.tensor([[False,True,True,True]])
    def call(q,k,v):
        return causal_favor_attention(q,k,v,omega,query_mask=mask,key_mask=mask,chunk_size=2)
    assert torch.autograd.gradcheck(call,args,eps=1e-6,atol=2e-6,rtol=2e-4)


def test_saved_outer_states_are_chunk_boundaries_only():
    q,k,v = [torch.randn(1,2,257,8,requires_grad=True) for _ in range(3)]
    omega = orthogonal_gaussian_projection(2,16,8,4).float()
    mask = torch.ones(1,257,dtype=torch.bool)
    saved_shapes = []
    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape)); return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        out = causal_favor_attention(q,k,v,omega,query_mask=mask,key_mask=mask,chunk_size=16)
    assert not any(len(s) == 5 for s in saved_shapes)
    assert saved_shapes.count((1,2,16,8)) == (257+15)//16
    out.sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in (q,k,v))


@pytest.mark.parametrize("chunk", [1,3,32])
def test_strict_future_invariance_and_zero_future_gradients(chunk):
    torch.manual_seed(13)
    q,k,v = [torch.randn(2,2,9,4,requires_grad=True) for _ in range(3)]
    mask = torch.ones(2,9,dtype=torch.bool)
    omega = orthogonal_gaussian_projection(2,11,4,4).float()
    def run(q,k,v):
        return causal_favor_attention(q,k,v,omega,query_mask=mask,key_mask=mask,chunk_size=chunk)
    out = run(q,k,v)
    for boundary in (1,3,7):
        changed = [t.detach().clone() for t in (q,k,v)]
        for t in changed:
            t[:,:,boundary:] += 1000
        assert torch.equal(out[:,:,:boundary],run(*changed)[:,:,:boundary])
        grads = torch.autograd.grad(out[:,:,:boundary].sum(),(q,k,v),retain_graph=True)
        assert all(g[:,:,boundary:].eq(0).all() for g in grads)


@pytest.mark.parametrize("scale", [1.,30.,1000.])
def test_large_norm_numerics_and_gradients(scale):
    torch.manual_seed(11)
    args = [((torch.randn(1,2,7,4)*scale).double()).requires_grad_() for _ in range(3)]
    omega = orthogonal_gaussian_projection(2,8,4,123)
    mask = torch.ones(1,7,dtype=torch.bool)
    out = causal_favor_attention(*args,omega,query_mask=mask,key_mask=mask,chunk_size=3)
    expected = explicit_reference(*args,omega,mask,mask)
    torch.testing.assert_close(out,expected,atol=1e-7,rtol=1e-8)
    out.square().mean().backward()
    assert all(torch.isfinite(t.grad).all() for t in args)


def test_masked_nonfinite_padding_and_empty_prefix():
    q = torch.randn(2,2,6,4,requires_grad=True)
    k,v = q.detach().clone(),q.detach().clone()
    mask = torch.tensor([[False,False,True,True,False,False],[False]*6])
    for x in (k,v): x[~mask[:,None,:,None].expand_as(x)] = torch.nan
    omega = orthogonal_gaussian_projection(2,7,4,11).float()
    out = causal_favor_attention(q,k,v,omega,query_mask=mask,key_mask=mask,chunk_size=2)
    assert torch.isfinite(out).all()
    assert out[~mask[:,None,:,None].expand_as(out)].eq(0).all()
    out.sum().backward()
    assert torch.isfinite(q.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32,torch.float64,torch.bfloat16])
def test_precision_island_dtype_and_chunk_equivalence(dtype):
    torch.manual_seed(832)
    args = [torch.randn(2,2,17,4,dtype=dtype,requires_grad=True) for _ in range(3)]
    omega = orthogonal_gaussian_projection(2,13,4,42).to(dtype)
    mask = torch.tensor([[True]*13+[False]*4,[False,True]*8+[True]])
    with torch.autocast("cpu",dtype=torch.bfloat16):
        a = causal_favor_attention(*args,omega,query_mask=mask,key_mask=mask,chunk_size=3)
        b = causal_favor_attention(*args,omega,query_mask=mask,key_mask=mask,chunk_size=32)
    assert a.dtype == dtype and torch.equal(a,b)
    a.float().square().mean().backward()
    assert all(torch.isfinite(t.grad).all() for t in args)


@pytest.mark.parametrize("chunk", [2,5,32])
def test_masked_key_has_no_gradient_including_after_holes(chunk):
    torch.manual_seed(4)
    q,k,v = [torch.randn(1,2,11,4,requires_grad=True) for _ in range(3)]
    km = torch.tensor([[False,True,False,True,True,False,True,False,True,False,False]])
    qm = torch.ones_like(km)
    omega = orthogonal_gaussian_projection(2,11,4,2).float()
    out = causal_favor_attention(q,k,v,omega,query_mask=qm,key_mask=km,chunk_size=chunk)
    out.sum().backward()
    for t in (k,v):
        assert t.grad[~km[:,None,:,None].expand_as(t)].eq(0).all()


def test_full_model_gradients_cannot_reach_future_correctness_inputs():
    net = model().eval()
    batch = collate([example(7,2),example(4,1)])
    retained = []
    def hook(module,inputs,output):
        output.retain_grad(); retained.append(output)
    handle = net.correct_padding_encoder.correct.register_forward_hook(hook)
    try:
        logits = net(batch)
        logits[0,:3].sum().backward()
        assert retained[0].grad[0,3:].eq(0).all()
        assert retained[0].grad[1].eq(0).all()
    finally:
        handle.remove()


def test_training_dropout_uses_same_rng_without_response_leakage():
    torch.manual_seed(43)
    cfg = config()
    from dataclasses import replace
    net = HiTSKTFavorPlus(replace(cfg,dropout=.2),FavorPlusConfig(features=12,chunk_size=3)).train()
    a = collate([example(7,2),example(4,1)])
    b = collate([example(7,2),example(4,1)])
    b.target_correct_input[0,3] = 1-b.target_correct_input[0,3]
    rng = torch.random.get_rng_state().clone()
    baseline = net(a)
    torch.random.set_rng_state(rng)
    changed = net(b)
    assert torch.equal(baseline[0,:3],changed[0,:3])
    assert torch.equal(baseline[1],changed[1])


def test_softmax_approximation_prespecified_small_norm_regime():
    # Frozen regime and seeds; diagnostic of approximation, not equality to
    # exact softmax and not a guarantee for learned high-norm Q/K.
    torch.manual_seed(931)
    q,k = [torch.randn(1,1,12,8,dtype=torch.float64)*.35 for _ in range(2)]
    v = torch.randn(1,1,12,4,dtype=torch.float64)
    mask = torch.ones(1,12,dtype=torch.bool)
    scores = (q@k.transpose(-1,-2))/8**.5
    scores.masked_fill_(~torch.ones(12,12,dtype=torch.bool).tril(),-torch.inf)
    exact = scores.softmax(-1)@v
    errors = {}
    for m in (8,256):
        errors[m] = []
        for seed in range(8):
            omega = orthogonal_gaussian_projection(1,m,8,seed)
            out = causal_favor_attention(q,k,v,omega,query_mask=mask,key_mask=mask)
            errors[m].append((out-exact).square().mean().item())
    assert sum(errors[256])/8 < .001
    assert sum(errors[256]) < sum(errors[8])


def config():
    return HiTSKTConfig(20,5,width=16,heads=2,feedforward_width=32,dropout=0.)


def example(length=4,history=2):
    return SessionExample([Session([1,2],[1,2],[0,1]) for _ in range(history)],
                          Session(list(range(1,length+1)),[1]*length,[i%2 for i in range(length)]),0)


def model():
    torch.manual_seed(11)
    return HiTSKTFavorPlus(config(),FavorPlusConfig(features=12,chunk_size=3))


def collate(examples):
    return collate_hitskt(examples,num_questions=20,num_skills=5)


def test_non_attention_weights_and_rng_exactly_preserved():
    torch.manual_seed(11)
    old = HiTSKT(config())
    rng = torch.random.get_rng_state().clone()
    new = model()
    assert torch.equal(rng,torch.random.get_rng_state())
    assert old.state_dict().keys() < new.state_dict().keys()
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    for key,value in old.state_dict().items():
        assert torch.equal(value,new.state_dict()[key]),key


@pytest.mark.parametrize("length", [1,2,4,7])
@pytest.mark.parametrize("history", [1,3])
def test_full_model_own_and_future_label_invariance(length,history):
    net = model().eval()
    e = example(length,history)
    batch = collate([e,example(8,2)])
    with torch.no_grad(): original = net(batch)
    for i in range(length):
        labels = list(e.target.correct)
        labels[i] = 1-labels[i]
        changed = SessionExample(e.history,Session(e.target.question_ids,e.target.skill_ids,labels),0)
        other = collate([changed,example(8,2)])
        assert other.target_correct_input[0,0] == 4
        assert other.target_correct_input[0,i+1] == labels[i]
        with torch.no_grad(): actual = net(other)
        assert torch.equal(original[0,:i+1],actual[0,:i+1])
        assert torch.equal(original[1],actual[1])
    # Labels used for the objective are never read by forward.
    batch.target_labels.fill_(99)
    with torch.no_grad(): assert torch.equal(original,net(batch))


def test_full_model_padding_eos_and_batch_independence():
    net = model().eval()
    a,b = collate([example(3,1)]),collate([example(3,1),example(9,4)])
    with torch.no_grad(): x,y = net(a),net(b)
    torch.testing.assert_close(x[0,:3],y[0,:3],atol=2e-6,rtol=2e-6)
    assert a.target_attention_mask.sum() == 4 and a.target_metric_mask.sum() == 3
    b.target_correct_input[0,4:] = 1
    b.history_correct[~b.history_mask] = 1
    with torch.no_grad(): z = net(b)
    assert torch.equal(y[0,:3],z[0,:3])
    dirty = y.clone(); dirty[~b.target_metric_mask] = 1e5
    assert torch.equal(masked_bce_loss(y,b.target_labels,b.target_metric_mask),
                       masked_bce_loss(dirty,b.target_labels,b.target_metric_mask))


def test_full_model_backward_checkpoint_and_variant_rejection(tmp_path: Path):
    net = model()
    batch = collate([example(5,3),example(2,1)])
    loss = masked_bce_loss(net(batch),batch.target_labels,batch.target_metric_mask)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(not b.requires_grad for n,b in net.named_buffers() if n.endswith('projection'))
    net.eval()
    with torch.no_grad(): before = net(batch)
    path = tmp_path/'favor.pt'; torch.save(net.checkpoint(),path)
    loaded = HiTSKTFavorPlus.from_checkpoint(path).eval()
    with torch.no_grad(): assert torch.equal(before,loaded(batch))
    with pytest.raises(ValueError): HiTSKT.from_checkpoint(path)
    path2 = tmp_path/'elu.pt'; torch.save(HiTSKT(config()).checkpoint(),path2)
    with pytest.raises(ValueError): HiTSKTFavorPlus.from_checkpoint(path2)


@pytest.mark.parametrize("length", [65,257,1025])
def test_long_variable_length_forward_backward(length):
    torch.manual_seed(9)
    q,k,v = [torch.randn(1,1,length,4,requires_grad=True) for _ in range(3)]
    omega = orthogonal_gaussian_projection(1,8,4,4).float()
    mask = torch.arange(length)[None,:] < length-3
    out = causal_favor_attention(q,k,v,omega,query_mask=mask,key_mask=mask,chunk_size=32)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(torch.isfinite(x.grad).all() for x in (q,k,v))
    assert all(x.grad[:,:,length-3:].eq(0).all() for x in (q,k,v))


@pytest.mark.parametrize("kwargs", [{"features":0},{"chunk_size":0},{"redraw":"step"},
                                    {"accumulation_dtype":"float16"}])
def test_configuration_rejects_unsupported_policies(kwargs):
    with pytest.raises(ValueError): FavorPlusConfig(**kwargs)
