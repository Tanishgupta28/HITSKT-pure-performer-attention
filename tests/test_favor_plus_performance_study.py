"""No optimizer calls or CUDA: test preparation's scientific comparison contract."""
import copy
import numpy as np
import pytest
from ktbench.favor_plus_performance_study import recipes,validate_plan,validation_record,storage_components,canonical_sha


def plan():
    return dict(execution_approved=False,optimizer_updates_executed=0,test_evaluations_authorized=0,
        schedule_sha256='fixed_order',recipes=recipes(dict(width=256,heads=4,dropout=.1),'fixed_order'),
        budget=dict(allocator_gib=8,advisory_total_gib=10,admission_free_gib=16,runtime_free_gib=6,host_rss_gib=4,host_headroom_gib=2,pacing_seconds=.1))


def test_primary_feature_pair_is_matched_and_optional_modifications_are_unverified():
    p=plan();assert validate_plan(p)=='PASS'
    assert p['recipes'][0]['features']==128 and p['recipes'][1]['features']==512
    assert len({x['namespace'] for x in p['recipes']})==5
    assert all('DESIGN ONLY' in x['implementation_status'] for x in p['recipes'][3:])


@pytest.mark.parametrize('key,value',[('seed',43),('accumulation_dtype','float32'),('projection_policy','native128'),('model_config',dict(width=128)),('pilot_updates',100)])
def test_primary_pair_refuses_confounded_comparisons(key,value):
    p=plan();p['recipes'][1][key]=value
    with pytest.raises(ValueError):validate_plan(p)


@pytest.mark.parametrize('key,value',[('execution_approved',True),('optimizer_updates_executed',1),('test_evaluations_authorized',1)])
def test_preparation_does_not_grant_execution_or_report_unrun_updates(key,value):
    p=plan();p[key]=value
    with pytest.raises(ValueError):validate_plan(p)


@pytest.mark.parametrize('key,value',[('review_threshold',.25),('truncate_actions',True),('drop_targets',True),('schedule_sha256','different'),('seed',43)])
def test_architecture_recipe_cannot_change_protected_protocol(key,value):
    p=plan();p['recipes'][3][key]=value
    with pytest.raises(ValueError):validate_plan(p)


def test_shared_budget_cannot_be_increased_or_pressure_floor_lowered():
    for key,value in [('allocator_gib',16),('runtime_free_gib',1),('admission_free_gib',8)]:
        p=plan();p['budget'][key]=value
        with pytest.raises(ValueError):validate_plan(p)


def test_validation_context_ignores_test_metrics_even_if_malformed():
    row=dict(split='validation',epoch='1',targets='125800',roc_auc='.7',loss='.6',f1='.5',accuracy='.7',precision='.6',recall='.4',threshold='.5')
    result=validation_record([dict(split='test',roc_auc='DO_NOT_PARSE'),row],1)
    assert result['roc_auc']==.7 and result['targets']==125800
    with pytest.raises(ValueError):validation_record([row,{**row,'epoch':'2','roc_auc':'.8'}],1)


def test_storage_accounting_scales_features_without_claiming_a_gpu_peak():
    offsets=np.array([0,10,30]);numbers=np.array([1,2])
    a=storage_components([1],offsets,numbers,128);b=storage_components([1],offsets,numbers,512)
    assert b['saved_boundary_bytes']==4*a['saved_boundary_bytes']
    assert b['two_ff_activation_bytes']==a['two_ff_activation_bytes']
    assert b['max_action_length']==11 and b['max_target_length']==21
    assert b['short_session_retrieval_score_bytes']==1*4*21*15*4


def test_canonical_schedule_hash_preserves_order_but_not_dictionary_key_order():
    assert canonical_sha({'a':[1,2],'b':3})==canonical_sha({'b':3,'a':[1,2]})
    assert canonical_sha({'a':[1,2]})!=canonical_sha({'a':[2,1]})
