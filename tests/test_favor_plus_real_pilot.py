"""Lightweight CPU validation only: no real-data forward/optimization or GPU APIs."""
import copy
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ktbench import favor_plus_real_pilot as pilot
from ktbench.favor_plus_gpu_access import (CUDA_VALIDATION, REAL_PILOT, PROTECTED_RESOURCES,
    require_gpu_authorization)
from ktbench.favor_plus_pilot_diagnostics import (prefix_diagnostics, fixed_example_diagnostics,
    finite_tree, cpu_tree, equal_tree)
from ktbench.models.favor_plus import FavorPlusAttention, FavorPlusConfig, causal_favor_attention
from ktbench.models.performer import CausalLinearAttention

NOW = dt.datetime(2026, 10, 19, tzinfo=dt.timezone.utc)
UUID = "MIG-11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def prohibit_cuda(monkeypatch):
    """Fail tests on accidental probes/RNG/context/memory APIs, even with no GPU."""
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preparation/tests must not call a CUDA API")
    for name in ("is_available", "is_initialized", "init", "_lazy_init", "device_count",
                 "get_device_properties", "get_device_name", "manual_seed", "manual_seed_all",
                 "get_rng_state", "get_rng_state_all", "synchronize", "memory_allocated",
                 "max_memory_allocated", "max_memory_reserved", "reset_peak_memory_stats",
                 "set_per_process_memory_fraction"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def allocation():
    # Test-only assertion; never an actual lease or user approval.
    return dict(resource_uuid=UUID, owner="unit fixture", job_id="unit-fixture",
        exclusive=True, independent=True, external_allocation_evidence="UNIT FIXTURE ONLY",
        starts_utc="2026-01-01T00:00:00Z", expires_utc="2099-01-01T00:00:00Z",
        user_approval=dict(approved=True, evidence="UNIT FIXTURE ONLY",
                           actions=[CUDA_VALIDATION, REAL_PILOT]))


def test_metadata_guard_accepts_only_documented_action_without_cuda():
    record = allocation()
    assert require_gpu_authorization(record, UUID, REAL_PILOT, now=NOW) == UUID
    record["user_approval"]["actions"] = [CUDA_VALIDATION]
    with pytest.raises(ValueError, match="user approval"):
        require_gpu_authorization(record, UUID, REAL_PILOT, now=NOW)


@pytest.mark.parametrize("field,value", [
    ("exclusive", False), ("exclusive", "true"), ("independent", False),
    ("owner", ""), ("job_id", ""), ("external_allocation_evidence", None),
    ("resource_uuid", "MIG-placeholder"), ("starts_utc", "2090-01-01T00:00:00Z"),
    ("expires_utc", "2000-01-01T00:00:00Z"), ("starts_utc", "2026-01-01"),
    ("user_approval", {}), ("user_approval", {"approved": True, "evidence": "flag only"}),
])
def test_metadata_guard_rejects_missing_invalid_expired_or_unapproved(field, value):
    record = allocation(); record[field] = value
    with pytest.raises(ValueError):
        require_gpu_authorization(record, UUID, REAL_PILOT, now=NOW)


@pytest.mark.parametrize("visible", [None, "0", UUID+",another", "GPU-11111111-2222-3333-4444-555555555555"])
def test_metadata_guard_requires_exact_device_isolation(visible):
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        require_gpu_authorization(allocation(), visible, REAL_PILOT, now=NOW)


@pytest.mark.parametrize("resource", sorted(PROTECTED_RESOURCES))
def test_protected_mig_and_whole_parent_refused_even_with_asserted_approval(resource):
    record = allocation(); record["resource_uuid"] = resource
    with pytest.raises(ValueError, match="protected LightSAKT2"):
        require_gpu_authorization(record, resource, REAL_PILOT, now=NOW)


def test_real_execute_refuses_before_io_rng_model_or_cuda(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    def forbidden(*args, **kwargs): raise AssertionError("must fail before plan/data/output access")
    monkeypatch.setattr(pilot, "load_prepared_plan", forbidden)
    monkeypatch.setattr(pilot, "fresh_run", forbidden)
    with pytest.raises(ValueError, match="independent exclusive"):
        pilot.execute(tmp_path/"missing_plan.json", {}, tmp_path/"missing_cuda.json", "never")
    assert not list(tmp_path.iterdir())


def test_existing_cuda_harness_now_requires_actual_user_approval(monkeypatch):
    from scripts import validate_favor_plus_gpu as harness
    record = allocation(); record.pop("user_approval")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    with pytest.raises(ValueError, match="user approval"):
        harness.execute(record, "never", 128, "assist2017")


def store_fixture(students=6):
    lengths = np.tile([2, 3, 4, 20, 20], students)
    return SimpleNamespace(metadata={"num_questions": 8, "num_skills": 3},
        session_offsets=np.r_[0, np.cumsum(lengths)],
        session_student=np.repeat(np.arange(students), 5),
        session_number=np.tile(np.arange(1, 6), students),
        session_split=np.tile([0, 0, 0, 1, 2], students), session_length=lambda i:int(lengths[i]))


def test_train_selection_is_repeatable_whole_sessions_in_original_sampler_order():
    store = store_fixture()
    dataset, first, skipped = pilot.select_training_examples(store, 5)
    _, second, again = pilot.select_training_examples(store, 5)
    assert first == second and skipped == again == 0 and len(first) == 5
    assert len({r["target_session"] for r in first}) == 5
    from ktbench.data.session_store import make_session_dataloader
    _, sampler = make_session_dataloader(dataset, token_budget=2048, maximum_batch_size=1,
        bucket_size=512, shuffle=True, seed=42, workers=0, pin_memory=False)
    sampler.set_epoch(1)
    expected = [indices[0] for indices in list(sampler)[:5]]
    assert [r["dataset_index"] for r in first] == expected
    for row in first:
        a, b = row["history_start"], row["target_session"]
        assert np.all(store.session_split[a:b+1] == 0)
        assert np.all(store.session_student[a:b+1] == row["student"])
        assert row["history_lengths"] == [int(store.session_offsets[i+1]-store.session_offsets[i]) for i in range(a,b)]


def test_selector_skips_oversize_singletons_without_truncating_or_reading_labels(monkeypatch):
    store = store_fixture()
    # No question/skill/correct arrays exist: length-only selection cannot read labels.
    monkeypatch.setitem(pilot.LIMITS, "max_padded_tokens", 10)
    _, rows, skipped = pilot.select_training_examples(store, 5)
    assert skipped > 0
    assert all(r["target_length"] == 3 and r["history_lengths"] == [2] for r in rows)
    assert all(r["padded_tokens"] == 9 for r in rows)  # action EOS and session EOS included


@pytest.mark.parametrize("corruption", ["student", "split"])
def test_corrupt_history_boundaries_fail_instead_of_silent_filtering(corruption):
    store = store_fixture()
    if corruption == "student": store.session_student = np.arange(30)
    else: store.session_split[::5] = 1
    with pytest.raises(ValueError, match="crosses"):
        pilot.select_training_examples(store, 5)


def test_verified_dataset_identity_rejects_modified_processed_bytes(tmp_path):
    path = tmp_path/"question.npy"; path.write_bytes(b"unit processed content")
    store = SimpleNamespace(root=tmp_path, metadata={"unit fixture": True})
    reference = dict(metadata=store.metadata, expected_processed_checksums={"data/question.npy":pilot.sha256(path)})
    expected = pilot.verified_dataset_identity(store, reference)
    assert expected["files"]["question.npy"] == pilot.sha256(path)
    path.write_bytes(b"changed content")
    with pytest.raises(ValueError, match="checksum mismatch"):
        pilot.verified_dataset_identity(store, reference)


def test_prepare_and_reload_use_real_store_and_collator_without_forward_or_optimizer(tmp_path,monkeypatch):
    from ktbench import favor_plus_pipeline as pipeline
    store=store_fixture(3);root=tmp_path/'store';root.mkdir()
    lengths=np.diff(store.session_offsets);n=int(store.session_offsets[-1])
    arrays=dict(question=np.arange(n)%8+1,skill=np.arange(n)%3+1,correct=np.arange(n)%2,
        timestamp_ms=np.arange(n),session_offsets=store.session_offsets,session_student=store.session_student,
        session_number=store.session_number,session_split=store.session_split,student_offsets=np.arange(0,16,5))
    for name,array in arrays.items():np.save(root/f'{name}.npy',array)
    (root/'metadata.json').write_text(json.dumps(store.metadata));(root/'_SUCCESS').write_text('unit complete')
    reference=dict(metadata=store.metadata,expected_processed_checksums={str(p):pilot.sha256(p) for p in root.iterdir()})
    refpath=tmp_path/'reports/favor_plus_v1/numerical_quality_v1/dataset_pipeline_plan.json'
    refpath.parent.mkdir(parents=True);refpath.write_text(json.dumps({'datasets':{'assist2017':reference}}))
    monkeypatch.setattr(pilot,'ROOT',tmp_path);monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'EXPERIMENT_ROOT',tmp_path/'experiments'/pilot.VARIANT)
    monkeypatch.setattr(pilot,'pilot_source_identity',lambda:{'source_sha256':{'unit':'fixture'}})
    def forbidden(*args,**kwargs):raise AssertionError('preparation cannot construct model/optimizer')
    monkeypatch.setattr(pilot,'HiTSKTFavorPlus',forbidden);monkeypatch.setattr(torch.optim,'Adam',forbidden)
    path=pilot.prepare('preflight',store_path=root,steps=5)
    plan,dataset=pilot.load_prepared_plan(path)
    assert len(plan['selected'])==5 and len(dataset)>5
    assert plan['optimizer_updates']==plan['model_forward_passes']==0
    assert plan['cuda_measurements'] is None and not plan['full_training_ready']
    with pytest.raises(FileExistsError):pilot.prepare('preflight',store_path=root,steps=5)


@pytest.mark.parametrize("steps", [0, 6, -1, True, 5.0])
def test_step_bounds_fail_before_dataset_access(steps):
    with pytest.raises(ValueError, match="one through five"):
        pilot.prepare("never", store_path=Path("missing"), steps=steps)


def test_update_counter_never_allows_sixth_call_and_counts_partial_failure():
    class Optimizer:
        calls = 0
        def step(self): self.calls += 1
    optimizer = Optimizer(); counter = pilot.UpdateCounter(5)
    for _ in range(5): counter.apply(optimizer)
    with pytest.raises(RuntimeError, match="budget exhausted"): counter.apply(optimizer)
    assert optimizer.calls == counter.calls == counter.returned == 5
    class Failing:
        def step(self): raise RuntimeError("unit partial mutation")
    counter = pilot.UpdateCounter(1)
    with pytest.raises(RuntimeError): counter.apply(Failing())
    assert counter.calls == 1 and counter.returned == 0


def test_bounded_controller_stops_on_failure_without_retry_or_sixth_update():
    seen, saved, checks = [], [], []
    def step(number, item):
        seen.append(number)
        if number == 3: raise FloatingPointError("unit nonfinite")
        return {"unit_fixture_only":item}
    with pytest.raises(FloatingPointError):
        pilot.bounded_updates(list(range(5)),5,step,lambda:checks.append(True),lambda n,row:saved.append(n))
    assert seen == [1,2,3] and saved == [1,2] and len(checks) == 5
    seen.clear()
    pilot.bounded_updates(list(range(5)),5,lambda n,item:seen.append(n),lambda:None,lambda n,row:None)
    assert seen == [1,2,3,4,5]


def test_resource_failure_stops_before_any_fake_update(monkeypatch):
    monkeypatch.setattr(pilot,"host_memory",lambda:{"peak_process_rss_bytes":1,"cgroup_headroom_bytes":0})
    seen=[]
    with pytest.raises(MemoryError,match="headroom"):
        pilot.bounded_updates([1],1,lambda n,x:seen.append(n),pilot.check_host_resources,lambda n,r:None)
    assert not seen


def test_high_norm_approximation_error_is_not_mislabeled_as_kernel_defect():
    q=torch.tensor([3.,3.],dtype=torch.float64).reshape(1,1,2,1)
    k=torch.tensor([1.,3.],dtype=torch.float64).reshape(1,1,2,1)
    v=torch.tensor([1.,-1.],dtype=torch.float64).reshape(1,1,2,1)
    omega=torch.tensor([[[1.],[-1.]]],dtype=torch.float64);mask=torch.ones(1,2,dtype=torch.bool)
    actual=causal_favor_attention(q,k,v,omega,query_mask=mask,key_mask=mask)
    result=prefix_diagnostics(q.numpy(),k.numpy(),v.numpy(),omega.numpy(),mask.numpy(),mask.numpy(),actual.numpy())
    assert result["feature_kernel_agreement"] == "PASS"
    assert result["implementation_error"]["max_abs"] < 1e-12
    assert result["feature_versus_exact_softmax"]["relative_l2"] > .2
    assert "softmax_relative_l2_above_review_trigger" in result["review_flags"]
    with pytest.raises(FloatingPointError,match="tolerance"):
        prefix_diagnostics(q,k,v,omega,mask,mask,actual.numpy()+1e-3)


def test_prefix_diagnostic_rejects_nonfinite_active_values_and_unbounded_prefix():
    q=np.zeros((1,1,2,1)); mask=np.ones((1,2),bool);omega=np.ones((1,2,1))
    bad=q.copy();bad[0,0,0,0]=np.nan
    with pytest.raises(FloatingPointError):prefix_diagnostics(bad,q,q,omega,mask,mask,q)
    long=np.zeros((1,1,33,1));lm=np.ones((1,33),bool)
    with pytest.raises(ValueError,match="bounded"):prefix_diagnostics(long,long,long,omega,lm,lm,long)


def test_fixed_example_hooks_preserve_rng_mode_and_remove_themselves_on_failure():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention=FavorPlusAttention(CausalLinearAttention(4,1,.1),FavorPlusConfig(features=128),seed=42)
        def forward(self, values):
            mask=torch.ones(values.shape[:2],dtype=torch.bool)
            return self.attention(values,values,values,query_mask=mask,key_mask=mask)
    model=Tiny().train(); values=torch.arange(12,dtype=torch.float32).reshape(1,3,4)/20
    rng=torch.get_rng_state().clone()
    initial=fixed_example_diagnostics(model,values)
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    again=fixed_example_diagnostics(model,values,baseline=initial)
    assert again["attention"]["p95_ratio_to_initial"] == {"query":1.,"key":1.}
    assert not model.attention._forward_pre_hooks
    def failed_save(*args):raise OSError("unit save failure")
    with pytest.raises(OSError):fixed_example_diagnostics(model,values,save_prefix=failed_save)
    assert model.training and not model.attention._forward_pre_hooks


def test_checkpoint_tree_covers_nonfinite_adam_and_exact_serialization(tmp_path):
    state={"optimizer_state_dict":{0:{"exp_avg_sq":torch.tensor([1.,2.])}},"rng":(torch.tensor([3]),4)}
    cpu=cpu_tree(state);path=tmp_path/"unit_checkpoint.pt"
    torch.save(cpu,path);loaded=torch.load(path,weights_only=True)
    assert finite_tree(loaded) and equal_tree(cpu,loaded)
    loaded["optimizer_state_dict"][0]["exp_avg_sq"][1]=torch.inf
    assert not finite_tree(loaded) and not equal_tree(cpu,loaded)


def test_plan_rejects_widened_bounds_before_source_or_dataset_io(tmp_path):
    plan=dict(kind=pilot.PLAN_KIND,format_version=pilot.PLAN_VERSION,status="PREPARED_NOT_RUN",
        dataset="assist2017",model=pilot.VARIANT,protocol=pilot.PROTOCOL,seed=42,
        favor_config=pilot.FAVOR,limits={**pilot.LIMITS,"max_optimizer_updates":6},profile=pilot.profile_for("assist2017"))
    path=tmp_path/"plan.json";path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match="fixed ASSIST2017"):
        pilot.load_prepared_plan(path)


@pytest.mark.parametrize("change", ["missing", "profile", "device", "stage_failure"])
def test_cuda_pass_evidence_cannot_be_missing_different_profile_or_device(tmp_path,change):
    from scripts.validate_favor_plus_gpu import LENGTHS
    report=dict(status="PASS",scope="bounded synthetic GPU verification",profile="assist2017",features=128,
        allocation_resource_uuid=UUID,full_model_shift_gradients_checkpoint="PASS",
        kernels=[dict(length=n,chunk=c,status="PASS",features=128) for n in LENGTHS for c in (16,32,64)])
    if change=="missing":report={}
    elif change=="profile":report["profile"]="junyi"
    elif change=="device":report["allocation_resource_uuid"]="other"
    else:report["kernels"].pop()
    path=tmp_path/"gpu_validation.json";path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="matching complete"):
        pilot.require_cuda_verification(path,allocation())


def test_cuda_pass_parser_checks_policy_sources_stage_bounds_and_checkpoint(tmp_path):
    from scripts.validate_favor_plus_gpu import LENGTHS
    checkpoint=tmp_path/'gpu_model.pt';checkpoint.write_bytes(b'UNIT FIXTURE ONLY: no real CUDA checkpoint')
    source=pilot.source_identity()
    report=dict(status='PASS',scope='bounded synthetic GPU verification',profile='assist2017',features=128,
        allocation_resource_uuid=UUID,full_model_shift_gradients_checkpoint='PASS',source=source,
        runtime=dict(device='cuda:0',float32_matmul_precision='highest',matmul_allow_tf32=False,
            cudnn_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=True,deterministic_algorithms=True),
        script_sha256=pilot.sha256(pilot.ROOT/'scripts/validate_favor_plus_gpu.py'),
        access_guard_sha256=pilot.sha256(pilot.ROOT/'ktbench/favor_plus_gpu_access.py'),
        oracle_sha256=pilot.sha256(pilot.ROOT/'scripts/favor_plus_quality_common.py'),
        checkpoint_sha256=pilot.sha256(checkpoint),kernels=[dict(length=n,chunk=c,status='PASS',features=128,
            max_abs_output=0.,peak_allocated_bytes=1024,seconds=.1) for n in LENGTHS for c in (16,32,64)])
    path=tmp_path/'gpu_validation.json'
    def write(value):path.write_text(json.dumps(value))
    write(report);assert pilot.require_cuda_verification(path,allocation())['status']=='PASS'
    modified=copy.deepcopy(report);modified['runtime']['matmul_allow_tf32']=True;write(modified)
    with pytest.raises(ValueError,match='FP32 policy'):pilot.require_cuda_verification(path,allocation())
    modified=copy.deepcopy(report);modified['source']['source_sha256']['ktbench/models/favor_plus.py']='changed';write(modified)
    with pytest.raises(ValueError,match='stale'):pilot.require_cuda_verification(path,allocation())
    modified=copy.deepcopy(report);modified['oracle_sha256']='changed';write(modified)
    with pytest.raises(ValueError,match='oracle changed'):pilot.require_cuda_verification(path,allocation())
    modified=copy.deepcopy(report);modified['kernels'][0]['seconds']=121;write(modified)
    with pytest.raises(ValueError,match='unchanged gate'):pilot.require_cuda_verification(path,allocation())
    write(report);checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):pilot.require_cuda_verification(path,allocation())
