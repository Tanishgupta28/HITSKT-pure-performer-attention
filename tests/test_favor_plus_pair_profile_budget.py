"""Pure CPU timing-estimator/design guards. No optimization or CUDA."""
import copy
import pytest
from ktbench import favor_plus_pair_profile_budget as p


def test_strata_weight_full_schedule_and_select_real_midpoints_deterministically():
    rows=[dict(padded_tokens=x,indices=[i]) for i,x in enumerate([3,1,9,2,7,7,8])]
    result=p.stratify(rows,3)
    assert sum(x['weight'] for x in result)==7 and len(result)==3
    assert result==p.stratify(rows,3)
    assert all(x['shape']==rows[x['index']] for x in result)
    assert len(p.stratify(rows,50))==7


@pytest.mark.parametrize('rows,bins',[([],3),([{}],0),([{}],True)])
def test_invalid_design_cannot_produce_empty_or_unweighted_estimate(rows,bins):
    with pytest.raises(ValueError):p.stratify(rows,bins)


def test_time_budget_proxy_counts_all_epochs_replays_checkpoints_and_margin():
    result=dict(strata={'training':[dict(weight=100,body_median_seconds=2.)],'validation':[dict(weight=10,body_median_seconds=1.)]},io_proxies=[dict(total_seconds=1.),dict(total_seconds=3.)],initial_diagnostics_seconds=4.)
    d=dict(validation_passes_per_arm=9,checkpoint_count_per_arm=312,diagnostic_events_per_arm=9,budget_screen_reserve_fraction=.25)
    x=p.estimate_arm(result,d,startup_proxy_seconds=1.)
    assert x['subtotal_seconds']==200+90+624+16+36+17
    assert x['with_planning_reserve_seconds']==x['subtotal_seconds']*1.25
    assert not x['real_Adam_step_cost_measured']


def test_no_step_optimizer_adapter_forbids_any_optimizer_update():
    class Net:
        def zero_grad(self,**kwargs):self.kwargs=kwargs
    n=Net();o=p.NoStepOptimizer(n);o.zero_grad();assert n.kwargs=={'set_to_none':True}
    with pytest.raises(PermissionError):o.step()


def test_untrained_io_proxy_roundtrip_cannot_be_loaded_as_training_checkpoint(tmp_path):
    import torch
    from ktbench.models.hitskt import HiTSKTConfig
    from ktbench.data.session_store import RollingSessionDataset
    from test_favor_plus_pipeline import synthetic_store
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    ds=RollingSessionDataset(synthetic_store(tmp_path/'store'),'train',history_sessions=15)
    net=p.pair.fresh_model(HiTSKTConfig(9,4,width=8,heads=2,feedforward_width=16,dropout=.1),'orf128')
    proof=p.pair.make_batch(ds,p.pair.shape_manifest(ds,[[0]])[0]);buffers=[torch.zeros_like(x) for x in net.parameters() for _ in range(2)]
    boundary=type('CPUOnly',(),{'check':lambda self,*a,**k:{'unit_cpu':True}})()
    before=p.pair.learned_digest(net)
    result=p.io_proxy(net,buffers,{},dict(target_counts={'train':{'labels':14}}),proof,tmp_path,boundary,0)
    assert result['optimizer_updates']==0 and result['bytes']>0 and before==p.pair.learned_digest(net)
    with pytest.raises(ValueError):p.pair.PairedEpochModel.from_checkpoint(result['path'])
