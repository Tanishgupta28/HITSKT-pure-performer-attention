"""Frozen controls only: no dataset optimizer, GPU allocation or outcome tuning."""
import numpy as np
import pytest
from ktbench.favor_plus_frozen_investigation import (
    first_key_center, projection_arm, evaluate_arm, pair_geometry, value_offset_control,
)
from ktbench.favor_plus_post50_diagnostics import attention_weights


@pytest.mark.parametrize('length',[1,8,35])
@pytest.mark.parametrize('holes',[False,True])
def test_first_key_translation_preserves_exact_target_and_causal_feature_prefix(length,holes):
    rng=np.random.default_rng(13);q,k,v=[rng.normal(size=(2,length,4)) for _ in range(3)]
    w=rng.normal(size=(2,12,4));qm=np.ones(length,bool);km=qm.copy()
    if holes and length>1:km[::3]=False;qm[-1]=False
    original=attention_weights(q,k,w,qm,km)[1]
    translated=attention_weights(q,first_key_center(k,km),w,qm,km)[1]
    np.testing.assert_allclose(original,translated,rtol=1e-12,atol=1e-12)
    result=evaluate_arm(q,k,v,w,qm,km,'first_valid_key_centering_ORF')
    assert result['exact_softmax_translation_max_abs']<=1e-12
    if length>3:
        before=attention_weights(q,first_key_center(k,km),w,qm,km)[0];cut=length//2
        k[:,cut:]+=20
        after=attention_weights(q,first_key_center(k,km),w,qm,km)[0]
        np.testing.assert_allclose(before[:,:cut,:cut],after[:,:cut,:cut],rtol=1e-12,atol=1e-12)


def test_first_key_center_masks_nonfinite_pad_and_empty_history():
    k=np.ones((2,4,3));k[:,0]=np.nan;k[:,-1]=np.inf;km=np.array([False,True,True,False])
    output=first_key_center(k,km);assert np.isfinite(output).all() and np.array_equal(output,np.zeros_like(output))
    assert np.array_equal(first_key_center(k,np.zeros(4,bool)),np.zeros_like(output))


def test_projection_counts_are_nested_and_antithetic_budget_is_total_features():
    pool=np.random.default_rng(2).normal(size=(4,512,64));before=pool.copy()
    a=projection_arm(pool,128,'original_ORF');b=projection_arm(pool,256,'original_ORF')
    assert np.array_equal(a,b[:,:128])
    pair=projection_arm(pool,128,'antithetic_ORF')
    assert pair.shape==(4,128,64) and np.array_equal(pair[:,:64],-pair[:,64:])
    assert np.array_equal(pool,before)


@pytest.mark.parametrize('count,arm',[(0,'original_ORF'),(1024,'original_ORF'),(127,'antithetic_ORF'),(128,'unknown')])
def test_unregistered_or_invalid_arm_is_rejected(count,arm):
    with pytest.raises(ValueError):projection_arm(np.ones((1,512,4)),count,arm)


def test_controls_never_modify_frozen_arrays():
    rng=np.random.default_rng(5);arrays=[rng.normal(size=(2,9,4)) for _ in range(3)];w=rng.normal(size=(2,16,4));mask=np.ones(9,bool)
    before=[x.copy() for x in arrays]+[w.copy(),mask.copy()]
    evaluate_arm(*arrays,w,mask,mask,'first_valid_key_centering_ORF')
    value_offset_control(*arrays,w,mask,mask)
    for a,b in zip([*arrays,w,mask],before):assert np.array_equal(a,b)


def test_output_denominator_control_keeps_absolute_error_and_weight_metrics():
    rng=np.random.default_rng(9);q,k,v=[rng.normal(size=(2,11,4)) for _ in range(3)];v+=10
    w=rng.normal(size=(2,12,4));mask=np.ones(11,bool)
    r=value_offset_control(q,k,v,w,mask,mask)
    assert r['absolute_error_invariance_max_abs']<1e-12
    assert r['original']['attention_total_variation']==r['centered_values']['attention_total_variation']
    assert r['centered_values']['relative_output_l2']>r['original']['relative_output_l2']


def test_pair_geometry_uses_scaled_sum_and_alignment_not_only_separate_norms():
    q=np.ones((1,3,4));mask=np.ones(3,bool)
    aligned=pair_geometry(q,q,mask,mask);opposed=pair_geometry(q,-q,mask,mask)
    assert aligned['cosine']['mean']==pytest.approx(1.) and opposed['cosine']['mean']==pytest.approx(-1.)
    assert aligned['scaled_pair_sum_norm_squared']['mean']==pytest.approx(8.)
    assert opposed['scaled_pair_sum_norm_squared']['maximum']==0.


def test_exact_sensitivity_reference_uniform_prefix_and_empty_mask():
    import torch
    from ktbench.frozen_exact_attention_reference import exact_causal_reference
    q=k=torch.zeros(1,1,4,2);v=torch.arange(8.).reshape(1,1,4,2);mask=torch.ones(1,4,dtype=torch.bool)
    result=exact_causal_reference(q,k,v,mask,mask)
    for i in range(4):torch.testing.assert_close(result[:,:,i],v[:,:,:i+1].double().mean(2))
    assert exact_causal_reference(q,k,v,mask,~mask).eq(0).all()


def test_exact_sensitivity_reference_future_and_nonfinite_pad_are_inert():
    import torch
    from ktbench.frozen_exact_attention_reference import exact_causal_reference
    g=torch.Generator().manual_seed(33);q,k,v=[torch.randn(1,2,9,4,generator=g) for _ in range(3)];mask=torch.ones(1,9,dtype=torch.bool);mask[:,-1]=False
    before=exact_causal_reference(q,k,v,mask,mask);k[:,:,6:8]+=50;v[:,:,6:8]-=50;k[:,:,-1]=torch.nan;v[:,:,-1]=torch.inf
    after=exact_causal_reference(q,k,v,mask,mask)
    torch.testing.assert_close(before[:,:,:6],after[:,:,:6],atol=0,rtol=0)
    assert torch.isfinite(after).all() and after[:,:,-1].eq(0).all()
