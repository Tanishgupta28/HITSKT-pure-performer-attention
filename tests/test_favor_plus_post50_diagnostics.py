"""CPU-only diagnostic-oracle/storage tests; no optimizer or real-data training."""
import numpy as np
import pytest
import torch
from ktbench.favor_plus_post50_diagnostics import (attention_weights,attention_diagnostics,
    direct_log_feature_weights,binary_summary,shape_memory,SavedStorageTracker,require_cpu)
from ktbench.models.favor_plus import orthogonal_gaussian_projection,causal_favor_attention
from ktbench.batching import ExampleShape,padded_token_cost
from ktbench.metrics import binary_metrics


@pytest.mark.parametrize('n',[1,8,17])
@pytest.mark.parametrize('holes',[False,True])
def test_compensated_feature_gauge_matches_direct_pairwise_and_production(n,holes):
    rng=np.random.default_rng(13+n);q,k,v=[rng.normal(size=(2,n,4)) for _ in range(3)]
    w=orthogonal_gaussian_projection(2,13,4,7).numpy();qm=np.ones(n,bool);km=qm.copy()
    if holes and n>1:km[::2]=False;qm[-1]=False
    a,e,_=attention_weights(q,k,w,qm,km);direct=direct_log_feature_weights(q,k,w,qm,km)
    np.testing.assert_allclose(a,direct,atol=1e-12,rtol=1e-11)
    actual=causal_favor_attention(*(torch.tensor(x)[None] for x in (q,k,v)),torch.tensor(w),
        query_mask=torch.tensor(qm)[None],key_mask=torch.tensor(km)[None],chunk_size=3).numpy()[0]
    np.testing.assert_allclose(a@np.where(km[None,:,None],v,0),actual,atol=1e-11,rtol=1e-10)
    assert np.all(a[:,np.triu_indices(n,1)[0],np.triu_indices(n,1)[1]]==0)


def test_masks_empty_keys_and_inert_nonfinite_padding():
    q=np.ones((1,3,2));k=q.copy();q[:,2]=np.nan;k[:,2]=np.inf;w=np.ones((1,4,2))
    a,e,_=attention_weights(q,k,w,np.array([True,True,False]),np.array([False,False,False]))
    assert np.array_equal(a,np.zeros_like(a)) and np.array_equal(e,np.zeros_like(e))


def test_future_changes_do_not_affect_explicit_reference_prefix():
    rng=np.random.default_rng(21);q,k,v=[rng.normal(size=(2,12,4)) for _ in range(3)]
    w=orthogonal_gaussian_projection(2,16,4,3).numpy();mask=np.ones(12,bool)
    before=attention_weights(q,k,w,mask,mask)
    q[:,8:]*=2;k[:,8:]*=2
    after=attention_weights(q,k,w,mask,mask)
    for x,y in zip(before[:2],after[:2]):np.testing.assert_allclose(x[:,:8,:8],y[:,:8,:8],atol=1e-13,rtol=1e-12)


def test_output_error_does_not_uniquely_determine_attention_weight_error():
    rng=np.random.default_rng(34);q,k=[rng.normal(size=(2,12,4))*2 for _ in range(2)]
    w=orthogonal_gaussian_projection(2,8,4,1).numpy();mask=np.ones(12,bool)
    result=attention_diagnostics(q,k,np.ones_like(q),w,mask,mask)
    assert result['attention_total_variation']['mean']>.01
    assert result['relative_output_l2']<1e-14  # constant values hide weight error


def test_relative_output_error_invariant_to_common_value_scale():
    rng=np.random.default_rng(1);q,k,v=[rng.normal(size=(1,9,4)) for _ in range(3)]
    w=orthogonal_gaussian_projection(1,8,4,6).numpy();mask=np.ones(9,bool)
    a=attention_diagnostics(q,k,v,w,mask,mask);b=attention_diagnostics(q,k,v*19,w,mask,mask)
    assert a['relative_output_l2']==pytest.approx(b['relative_output_l2'])
    assert a['output_rmse']*19==pytest.approx(b['output_rmse'])


@pytest.mark.parametrize('threshold',[.4,.5,.6])
def test_independent_metrics_and_ties_match_existing_contract(threshold):
    prob=np.array([.2,.5,.5,.6,.75,.8]);y=np.array([0,1,0,1,0,1])
    a=binary_summary(prob,y,threshold);b=binary_metrics(torch.tensor(prob),torch.tensor(y),loss=0.,threshold=threshold)
    for key in ('accuracy','precision','recall','f1','roc_auc'):assert a[key]==pytest.approx(b[key])
    assert a['tp']+a['fp']+a['tn']+a['fn']==len(y)


def test_false_negative_growth_can_coexist_with_accuracy_and_bce_improvement():
    y=np.array([0,0,0,1,1]);a=binary_summary([.6,.6,.6,.6,.6],y);b=binary_summary([.2,.2,.2,.49,.49],y)
    assert b['recall']<a['recall'] and b['f1']<a['f1']
    assert b['accuracy']>a['accuracy'] and b['rounded_probability_bce']<a['rounded_probability_bce']


def test_shape_accounting_matches_exact_collator_cost_and_has_separate_workspace():
    shapes=[ExampleShape((12,8),10),ExampleShape((7,),5)]
    result=shape_memory(shapes,100,20)
    assert result['token_cost']==padded_token_cost(shapes)
    assert result['persistent_weights_grad_adam_buffers_bytes']==420
    assert result['stages']['action']['rows']==3 and result['stages']['correct']['length']==11
    assert result['max_one_weighted_state_per_recomputed_token_bytes']>0


@pytest.mark.parametrize('n',[3,35])
def test_live_saved_storage_tracking_preserves_forward_and_gradients(n):
    torch.random.default_generator.manual_seed(5)
    inputs=[torch.randn(1,1,n,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    w=orthogonal_gaussian_projection(1,8,4,3);mask=torch.ones(1,n,dtype=torch.bool)
    direct=causal_favor_attention(*inputs,w,query_mask=mask,key_mask=mask,chunk_size=4)
    direct.sum().backward();grad=[x.grad.clone() for x in inputs]
    for x in inputs:x.grad=None
    tracker=SavedStorageTracker()
    with torch.autograd.graph.saved_tensors_hooks(tracker.pack,tracker.unpack):
        output=causal_favor_attention(*inputs,w,query_mask=mask,key_mask=mask,chunk_size=4)
        tracker.phase='backward';output.sum().backward()
    assert torch.equal(output,direct)
    for a,b in zip(inputs,grad):torch.testing.assert_close(a.grad,b,atol=0,rtol=0)
    assert tracker.peak>0 and tracker.phase_peak['backward']>0 and tracker.live==0


def test_cpu_guard_rejects_visible_gpu_before_any_initialization_query(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:(_ for _ in ()).throw(AssertionError('must short circuit')))
    with pytest.raises(RuntimeError):require_cpu()


def test_cpu_autograd_build_guard_refuses_cuda_linked_engine_before_graph(monkeypatch):
    from scripts.favor_plus_post50_cpu import require_cpu_autograd_build
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.version,'cuda','UNIT_CUDA_BUILD')
    with pytest.raises(RuntimeError,match='CPU-only PyTorch'):require_cpu_autograd_build()


def test_cpu_autograd_build_guard_accepts_cpu_only_build_without_gpu_query(monkeypatch):
    from scripts.favor_plus_post50_cpu import require_cpu_autograd_build
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.version,'cuda',None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:(_ for _ in ()).throw(AssertionError('no GPU probe')))
    require_cpu_autograd_build()
