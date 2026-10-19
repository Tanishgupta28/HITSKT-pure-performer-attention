"""Independent hand-calculated oracle and frozen study-design controls."""
import json
import datetime as dt

import pytest

import numpy as np
import torch

from scripts.favor_plus_quality_common import independent_attention,torch_feature_reference
from scripts.validate_favor_plus_gpu import admit_allocation,execute,plan


def test_numpy_oracle_uniform_kernel_prefix_averages_and_masks():
    q=k=np.zeros((1,1,3,1))
    v=np.array([1.,3.,9.]).reshape(1,1,3,1)
    qm=np.array([[True,True,False]]);km=np.array([[True,False,True]])
    feature,softmax,stats=independent_attention(q,k,v,np.array([[[1.],[-1.]]]),qm,km)
    np.testing.assert_array_equal(feature,np.array([1.,1.,0.]).reshape(1,1,3,1))
    np.testing.assert_array_equal(softmax,feature)
    assert stats['mean_attention_total_variation']==0


def test_numpy_feature_kernel_against_hand_calculated_cosh_identity():
    q=np.array([.2,.4]).reshape(1,1,2,1)
    k=np.array([-.5,.7]).reshape(1,1,2,1)
    v=np.array([2.,-1.]).reshape(1,1,2,1)
    omega=np.array([[[1.],[-1.]]]);mask=np.ones((1,2),bool)
    actual,exact,_=independent_attention(q,k,v,omega,mask,mask)
    a=np.exp(-(.4**2+(-.5)**2)/2)*np.cosh(.4-.5)
    b=np.exp(-(.4**2+.7**2)/2)*np.cosh(.4+.7)
    np.testing.assert_allclose(actual[0,0,:,0],[2.,(2*a-b)/(a+b)],rtol=1e-13)
    e1,e2=np.exp(.4*-.5),np.exp(.4*.7)
    np.testing.assert_allclose(exact[0,0,:,0],[2.,(2*e1-e2)/(e1+e2)],rtol=1e-13)


def test_torch_oracle_gradient_matches_numpy_finite_difference():
    q=torch.tensor([.2,.4],dtype=torch.float64).reshape(1,1,2,1).requires_grad_()
    k=torch.tensor([-.5,.7],dtype=torch.float64).reshape(1,1,2,1).requires_grad_()
    v=torch.tensor([2.,-1.],dtype=torch.float64).reshape(1,1,2,1).requires_grad_()
    omega=torch.tensor([[[1.],[-1.]]],dtype=torch.float64);mask=torch.ones(1,2,dtype=torch.bool)
    out=torch_feature_reference(q,k,v,omega,mask,mask)
    grads=torch.autograd.grad(out.sum(),(q,k,v))
    original=[a.detach().numpy() for a in (q,k,v)]
    for arg,g in enumerate(grads):
        for pos in range(2):
            plus=[a.copy() for a in original];minus=[a.copy() for a in original]
            plus[arg][0,0,pos,0]+=1e-6;minus[arg][0,0,pos,0]-=1e-6
            a=independent_attention(*plus,omega.numpy(),mask.numpy(),mask.numpy())[0].sum()
            b=independent_attention(*minus,omega.numpy(),mask.numpy(),mask.numpy())[0].sum()
            assert abs(g[0,0,pos,0].item()-(a-b)/2e-6)<1e-8


def test_predeclared_design_does_not_select_features_using_test_auc():
    d=json.load(open('reports/favor_plus_v1/numerical_quality_v1/PREDECLARED_DESIGN.json'))
    assert d['features']==[128,256,512]
    assert d['training_sample']['split']=='train only'
    assert d['training_sample']['optimizer_steps']==0
    assert not d['recommendation_rule']['test_auc_used']
    assert not d['recommendation_rule']['change_defaults']


@pytest.mark.parametrize('record,visible',[
    ({},None),
    ({'exclusive':'false','owner':'fixture','external_allocation_evidence':'fixture',
      'resource_uuid':'MIG-unit-test','expires_utc':'2099-01-01T00:00:00Z'},'MIG-unit-test'),
    ({'exclusive':True,'owner':'unit test','resource_uuid':'MIG-unit-test'},'MIG-unit-test'),
    ({'exclusive':True,'owner':'unit test','external_allocation_evidence':'fixture',
      'resource_uuid':'MIG-unit-test','expires_utc':'2000-01-01T00:00:00Z'},'MIG-unit-test'),
    ({'exclusive':True,'owner':'unit test','external_allocation_evidence':'fixture',
      'resource_uuid':'MIG-unit-test','expires_utc':'2099-01-01T00:00:00Z'},'another-device'),
])
def test_gpu_rejected_before_any_cuda_initialization(record,visible,monkeypatch):
    if visible is None:monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    else:monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible)
    def forbidden(*args,**kwargs):raise AssertionError('CUDA API must not be called')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    monkeypatch.setattr(torch.cuda,'get_device_properties',forbidden)
    with pytest.raises(ValueError):execute(record,'must-not-run',128,'junyi')


def test_allocation_metadata_validation_and_plan_are_cpu_only():
    record={'exclusive':True,'owner':'unit-test fixture','external_allocation_evidence':'not a real lease',
            'resource_uuid':'MIG-unit-test','expires_utc':'2030-01-01T00:00:00Z'}
    assert admit_allocation(record,'MIG-unit-test',now=dt.datetime(2026,1,1,tzinfo=dt.timezone.utc))=='MIG-unit-test'
    assert plan()['status']=='PENDING' and not plan()['cuda_initialized']


def test_sample_selection_uses_whole_train_sessions_and_counts_hierarchy_eos():
    from types import SimpleNamespace
    from scripts.inspect_favor_plus_training_norms import select_samples
    from ktbench.batching import ExampleShape,padded_token_cost
    lengths=np.array([n for target in (8,40,80,160) for n in (2,target,target+1,512,512)])
    store=SimpleNamespace(session_offsets=np.r_[0,np.cumsum(lengths)],
        session_student=np.repeat(np.arange(4),5),session_number=np.tile(np.arange(1,6),4),
        session_split=np.tile([0,0,0,1,2],4),student_offsets=np.arange(0,21,5))
    design=json.load(open('reports/favor_plus_v1/numerical_quality_v1/PREDECLARED_DESIGN.json'))
    _,selected=select_samples(store,design)
    assert len(selected)==4 and len({r['student'] for r in selected})==4
    for r in selected:
        assert r['split']==0 and store.session_split[r['target_session']]==0
        assert r['padded_tokens']==padded_token_cost([ExampleShape(tuple(r['lengths'][:-1]),r['lengths'][-1])])
        assert max(r['lengths'])<=256 and r['padded_tokens']<=2048


def test_pipeline_selects_validation_peak_and_reloads_it_for_test(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from ktbench import favor_plus_pipeline as p
    from ktbench.models.favor_plus import FavorPlusConfig
    original=p.profile_for
    monkeypatch.setattr(p,'profile_for',lambda dataset:{**original(dataset),'epoch_ceiling':3})
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'EXPERIMENT_ROOT',tmp_path/'experiments'/p.VARIANT)
    monkeypatch.setattr(p,'source_identity',lambda:{'synthetic_selection_control':True})
    monkeypatch.setattr(p,'store_identity',lambda store:{'synthetic':True})
    monkeypatch.setattr(p,'RollingSessionDataset',lambda store,split,**kw:SimpleNamespace(split=split))
    metrics=dict(targets=2,loss=.7,bce=.7,roc_auc=.6,accuracy=.5,precision=.5,recall=.5,f1=.5,mse=.25,threshold=.5)
    observed=[]
    def fake_epoch(model,dataset,*,device,epoch_index,optimizer=None,prediction_output=None):
        if optimizer is not None:
            with torch.no_grad():model.prediction.bias.fill_(float(epoch_index))
            return {**metrics,'roc_auc':.99}
        if dataset.split=='validation':return {**metrics,'roc_auc':{1:.7,2:.9,3:.8}[epoch_index]}
        observed.append(float(model.prediction.bias.detach()[0]))
        return metrics.copy()
    monkeypatch.setattr(p,'epoch',fake_epoch)
    result=p.train('junyi',SimpleNamespace(metadata={'num_questions':20,'num_skills':5}),
                   'synthetic_selection',device='cpu',favor=FavorPlusConfig(features=8),allow_full_training=True)
    assert result['best_epoch']==2 and result['best_validation_auc']==.9
    assert result['epochs_completed']==3 and observed==[2.]
    root=p.EXPERIMENT_ROOT/'junyi'/'synthetic_selection'
    assert torch.load(root/'last_model.pt',weights_only=True)['epoch']==3
    assert torch.load(root/'best_model.pt',weights_only=True)['epoch']==2
