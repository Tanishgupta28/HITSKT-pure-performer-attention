"""New paired-study factory/admission checks; all execute on CPU only."""
import copy,datetime as dt,time
from pathlib import Path
import pytest,torch
from ktbench import favor_plus_matched_comparison as m
from ktbench.favor_plus_pilot_recovery import digest
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.models.favor_plus_antithetic_experimental import HiTSKTFavorPlusAntithetic
from test_favor_plus import example,collate


def cfg():return HiTSKTConfig(20,5,width=16,heads=2,feedforward_width=32,dropout=0.)


def test_four_arms_have_identical_learned_parameters_and_rng_after_initialization():
    nets=[m.fresh_model(cfg(),arm) for arm in m.ARMS]
    assert len({m.learned_digest(n) for n in nets})==1
    rng=[]
    for arm in m.ARMS:
        m.fresh_model(cfg(),arm);rng.append(digest(torch.random.get_rng_state()))
    assert len(set(rng))==1
    assert isinstance(nets[0].core,HiTSKTFavorPlus) and isinstance(nets[2].core,HiTSKTFavorPlusAntithetic)


def test_declared_projection_policy_pairs_all_feature_counts_and_mappings():
    w={arm:m.arm_projection(2,8,m.MatchedArmConfig(arm),2) for arm in m.ARMS}
    assert torch.equal(w['orf128'],w['orf512'][:,:128])
    assert torch.equal(w['anti128'][:,:64],w['orf512'][:,:64])
    assert torch.equal(w['anti128'][:,:64],-w['anti128'][:,64:])
    assert torch.equal(w['anti512'][:,:256],w['orf512'][:,:256])


def test_fp64_attention_option_preserves_matched_fp32_parameters_and_rng():
    baseline=m.fresh_model(cfg(),'orf128');expected=m.learned_digest(baseline);rng=digest(torch.random.get_rng_state())
    for arm in m.ARMS:
        model=m.fresh_model(cfg(),arm,accumulation_dtype='float64')
        assert m.learned_digest(model)==expected and digest(torch.random.get_rng_state())==rng
        assert all(p.dtype==torch.float32 for p in model.parameters())
        assert all(x.config.accumulation_dtype=='float64' for x in model.core.modules() if isinstance(x,m.FavorPlusAttention))


def test_high_norm_independent_reference_regression_uses_supported_fp64_island():
    from ktbench.models.favor_plus import causal_favor_attention
    from scripts.favor_plus_quality_common import torch_feature_reference
    g=torch.Generator().manual_seed(107);xs=[(torch.randn(1,2,65,64,generator=g)*3).requires_grad_() for _ in range(3)]
    w=m.arm_projection(2,64,m.MatchedArmConfig('orf512'),0);mask=torch.ones(1,65,dtype=torch.bool);mask[:,1::7]=False
    expected_inputs=[x.detach().double().requires_grad_() for x in xs]
    ref=torch_feature_reference(*expected_inputs,w.double(),mask,mask)
    actual=causal_favor_attention(*xs,w,query_mask=mask,key_mask=mask,chunk_size=16,accumulation_dtype='float64')
    assert actual.dtype==torch.float32 and (actual.double()-ref).abs().max()<=5e-5
    a=torch.autograd.grad(actual.square().mean(),xs);b=torch.autograd.grad(ref.square().mean(),expected_inputs)
    assert all(torch.allclose(left.double(),right,atol=5e-5,rtol=1e-3) for left,right in zip(a,b))


@pytest.mark.parametrize('arm',m.ARMS)
def test_matched_checkpoint_is_separate_and_exact_cpu_roundtrip(arm,tmp_path):
    net=m.fresh_model(cfg(),arm).eval();b=collate([example(4,2),example(2,1)])
    with torch.no_grad():before=net(b)
    path=tmp_path/'matched.pt';torch.save(net.checkpoint(),path)
    loaded=m.MatchedComparisonModel.from_checkpoint(path).eval()
    with torch.no_grad():assert torch.equal(before,loaded(b))
    with pytest.raises(ValueError):HiTSKTFavorPlus.from_checkpoint(path)
    with pytest.raises(ValueError):HiTSKTFavorPlusAntithetic.from_checkpoint(path)


def record(path):
    now=dt.datetime.now(dt.timezone.utc)
    return dict(action=m.VERIFY_ACTION,resource_uuid=m.shared.TRANSFERRED_MIG,prepared_plan_sha256=m.sha256(path),shared_permission_confirmed=True,known_explicit_restriction=False,known_incompatible_reservation=False,limits=m.LIMITS.copy(),starts_utc=now.isoformat(),expires_utc=(now+dt.timedelta(hours=2)).isoformat(),campaign_started_unix=now.timestamp(),cuda_verification_approved=True,max_optimizer_updates=0,training_approved=False)


def test_verification_does_not_grant_training_approval(tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',m.shared.TRANSFERRED_MIG)
    m.require_scope(r,path,m.VERIFY_ACTION)
    with pytest.raises(ValueError):m.require_scope(r,path,m.TRAIN_ACTION)
    r['action']=m.TRAIN_ACTION
    with pytest.raises(PermissionError,match='NEW explicit'):m.require_scope(r,path,m.TRAIN_ACTION)


@pytest.mark.parametrize('field,value',[('max_optimizer_updates',1),('cuda_verification_approved',False),('known_incompatible_reservation',True),('resource_uuid','OTHER')])
def test_unapproved_scope_refused_without_gpu_calls(tmp_path,monkeypatch,field,value):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);r[field]=value;monkeypatch.setenv('CUDA_VISIBLE_DEVICES',m.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:(_ for _ in ()).throw(AssertionError('no CUDA before authorization')))
    with pytest.raises(ValueError):m.SharedStudyGuard(r,path,tmp_path)


@pytest.mark.parametrize('free_gib,expected',[(15,'STOP'),(20,'ADMIT')])
def test_shared_pressure_guard_retains_peers_and_does_not_require_pid_alias(tmp_path,monkeypatch,free_gib,expected):
    path=tmp_path/'plan.json';path.write_text('{}');r=record(path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',m.shared.TRANSFERRED_MIG)
    monkeypatch.setattr(m.shared,'check_exposure_and_hold',lambda *a:None)
    now=dt.datetime.now(dt.timezone.utc).isoformat();obs=dict(observed_utc=now,resource_uuid=m.shared.TRANSFERRED_MIG,device=dict(total_bytes=40*m.GIB,free_bytes=free_gib*m.GIB,used_bytes=(40-free_gib)*m.GIB,compute_processes=[dict(pid=98765)]),utilization=dict(status='unsupported',nvml_status=3),local=dict(holders=[],inaccessible_fd_pids=[],light_processes=[]))
    monkeypatch.setattr(m.shared,'observe',lambda *a:obs);monkeypatch.setattr(m,'host_memory',lambda:dict(peak_process_rss_bytes=m.GIB,cgroup_headroom_bytes=20*m.GIB))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:(_ for _ in ()).throw(AssertionError('no CUDA on refused admission')))
    guard=m.SharedStudyGuard(r,path,tmp_path)
    if expected=='STOP':
        with pytest.raises(m.shared.SharedResourceStop):guard.check('admission',cuda=False,admission=True)
    else:
        x=guard.check('admission',cuda=False,admission=True)
        assert x['decision']['status']!='STOP' and x['decision']['reported_compute_processes'][0]['pid']==98765
        assert not x['decision']['pid_mapping_proven'] and not x['decision']['exclusive_ownership_claimed']
