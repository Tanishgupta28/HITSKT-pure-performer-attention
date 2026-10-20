"""CPU-only proposal controls; no real experiment or optimizer calls."""
import copy
import pytest
from ktbench import favor_plus_continuation_contract as c


@pytest.fixture
def inputs():
    ordered=[{'target_session':i,'dataset_index':i,'student':i//2} for i in range(100)]
    samples=dict(training=ordered,validation=[{'target_session':101}],diagnostics=[{'target_session':102}])
    contract=dict(exact_training_and_probe_samples=ordered[:21],
        frozen_validation_samples=samples['validation'],diagnostic_samples=samples['diagnostics'],
        config={'unchanged_scientific_configuration':'fixture'})
    parent=dict(optimizer_updates=20,optimizer_step_calls=20,
        sampler=dict(cursor=20,exact_samples=ordered[:21]),contract=contract,
        component_sha256={'all':'preserved'},projection_sha256={'projection':'fixed'})
    return parent,samples


def test_handoff_keeps_cursor_parent_and_probe_without_mutation(inputs):
    parent,samples=inputs;before=copy.deepcopy(inputs)
    plan=c.proposed_handoff(parent,c.PARENT_SHA256,samples)
    assert inputs==before
    assert plan['next_training_sample']==samples['training'][20]
    assert plan['next_probe_only_after50']==samples['training'][50]
    assert plan['original_contract']==parent['contract']
    assert plan['scientific_config']==parent['contract']['config']
    assert plan['requested_additional_optimizer_calls']==30
    assert plan['max_total_optimizer_updates']==50
    assert not plan['explicit_continuation_approval_received']
    plan['scientific_config']['new']='cannot mutate parent'
    assert inputs==before


@pytest.mark.parametrize('corruption',['sha','step','calls','cursor','training','validation','diagnostics','duplicate'])
def test_changed_or_incomplete_handoff_refused(inputs,corruption):
    parent,samples=copy.deepcopy(inputs);sha=c.PARENT_SHA256
    if corruption=='sha':sha='wrong'
    elif corruption=='step':parent['optimizer_updates']=19
    elif corruption=='calls':parent['optimizer_step_calls']=21
    elif corruption=='cursor':parent['sampler']['cursor']=0
    elif corruption=='training':samples['training']=samples['training'][1:]
    elif corruption=='validation':samples['validation']=[]
    elif corruption=='diagnostics':samples['diagnostics']=[]
    elif corruption=='duplicate':samples['training'][25]=samples['training'][24]
    with pytest.raises(ValueError):c.proposed_handoff(parent,sha,samples)


def test_exactly_thirty_additional_updates():
    updates=list(c.stage_interval(20,30))+list(c.stage_interval(30,40))+list(c.stage_interval(40,50))
    assert updates==list(range(21,51)) and len(updates)==30


@pytest.mark.parametrize('previous,to',[(0,30),(20,20),(20,50),(30,50),(50,60),(True,30),(20,True)])
def test_no_restart_skip_reset_or_scope_expansion(previous,to):
    with pytest.raises(ValueError):c.stage_interval(previous,to)


def test_approval_is_a_separate_gate():
    with pytest.raises(ValueError):c.require_explicit_approval({},'plan-hash')
    approval=dict(approved=True,action=c.ACTION,prepared_plan_sha256='plan-hash',
        parent_checkpoint_sha256=c.PARENT_SHA256,additional_optimizer_calls=30,
        max_total_optimizer_updates=50,actual_user_approval_evidence='SYNTHETIC_TEST_ONLY')
    c.require_explicit_approval(approval,'plan-hash')
    for key in approval:
        bad={k:v for k,v in approval.items() if k!=key}
        with pytest.raises(ValueError):c.require_explicit_approval(bad,'plan-hash')
    with pytest.raises(ValueError):c.require_explicit_approval(approval,'different-plan')


def test_old_controller_still_refuses_continuation():
    from ktbench.favor_plus_assist20 import stage_interval
    with pytest.raises(ValueError):stage_interval(20,50)
