"""ASSIST2017 real-data preparation and explicitly gated five-update CUDA pilot.

prepare() only hashes/selects/collates CPU data. It does not seed CUDA, build a
model, run a forward pass or optimize. execute() is future work requiring actual
user approval, external independent allocation and a matching staged CUDA PASS.
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import math
import os
import platform
import resource
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ktbench.batching import collate_hitskt, example_shape, padded_token_cost
from ktbench.data.session_store import RollingSessionDataset, SessionStore, make_session_dataloader
from ktbench.favor_plus_gpu_access import REAL_PILOT, require_gpu_authorization
from ktbench.favor_plus_pipeline import (ROOT, PROTOCOL, fresh_run, model_config, profile_for,
    runtime, sha256, source_identity, write_json)
from ktbench.favor_plus_pilot_diagnostics import (cpu_tree, equal_tree, finite_tree,
    fixed_example_diagnostics)
from ktbench.metrics import masked_bce_loss
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus, VARIANT

DATASET = "assist2017"
PLAN_KIND = "assist2017_favor_plus_real_pilot_preparation"
PLAN_VERSION = 1
FAVOR = dict(features=128, projection_seed=42, chunk_size=32, accumulation_dtype="float32")
LIMITS = dict(max_optimizer_updates=5, batch_size=1, max_padded_tokens=2048,
    max_cuda_allocated_bytes=8*1024**3, max_process_rss_bytes=4*1024**3,
    minimum_cgroup_headroom_bytes=2*1024**3, max_step_seconds=120,
    max_execution_seconds=900, diagnostic_prefix_positions=32)


def validate_steps(steps: int) -> None:
    if type(steps) is not int or not 1 <= steps <= LIMITS["max_optimizer_updates"]:
        raise ValueError("pilot requires one through five optimizer updates")


def pilot_source_identity() -> dict:
    identity = source_identity()
    for name in ("ktbench/favor_plus_real_pilot.py", "ktbench/favor_plus_gpu_access.py",
                 "ktbench/favor_plus_pilot_diagnostics.py", "ktbench/config.py", "ktbench/training.py",
                 "scripts/pilot_hitskt_favor_plus_real.py", "scripts/validate_favor_plus_gpu.py",
                 "scripts/favor_plus_quality_common.py"):
        identity["source_sha256"][name] = sha256(ROOT/name)
    return identity


def host_memory() -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024  # Linux bytes
    result = {"peak_process_rss_bytes": peak}
    for base, limit_name, usage_name in (
            (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes"),
            (Path("/sys/fs/cgroup"), "memory.max", "memory.current")):
        try:
            limit, used = int((base/limit_name).read_text()), int((base/usage_name).read_text())
        except (OSError, ValueError):
            continue
        result.update(cgroup_limit_bytes=limit, cgroup_usage_bytes=used, cgroup_headroom_bytes=limit-used)
        break
    return result


def check_host_resources() -> dict:
    values = host_memory()
    if values["peak_process_rss_bytes"] > LIMITS["max_process_rss_bytes"]:
        raise MemoryError("pilot process RSS budget exceeded")
    if values.get("cgroup_headroom_bytes", math.inf) < LIMITS["minimum_cgroup_headroom_bytes"]:
        raise MemoryError("insufficient shared-container RAM headroom")
    return values


def verified_dataset_identity(store: SessionStore, reference: dict) -> dict:
    """Compare to previously audited processed hashes, not a newly invented data ID."""
    if store.metadata != reference["metadata"]:
        raise ValueError("ASSIST2017 metadata differs from verified dataset")
    files = {}
    for previous_path, expected in reference["expected_processed_checksums"].items():
        path = store.root/Path(previous_path).name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"verified ASSIST2017 checksum mismatch: {path.name}")
        files[path.name] = actual
    return {"metadata": store.metadata, "files": files}


def select_training_examples(store: SessionStore, steps: int) -> tuple[RollingSessionDataset, list, int]:
    """First eligible singleton batches from the original seed42/epoch1 sampler.

    Original singleton oversize exceptions are explicitly skipped, never trimmed.
    Selection uses lengths/indices/splits, not correctness or model predictions.
    """
    validate_steps(steps)
    dataset = RollingSessionDataset(store, "train", history_sessions=15)
    _, sampler = make_session_dataloader(dataset, token_budget=LIMITS["max_padded_tokens"],
        maximum_batch_size=1, bucket_size=512, shuffle=True, seed=42, workers=0, pin_memory=False)
    sampler.set_epoch(1)
    selected, skipped = [], 0
    for batch_indices in sampler:
        if len(batch_indices) != 1:
            raise ValueError("pilot sampler must yield singleton batches")
        index = batch_indices[0]
        target = int(dataset.targets[index])
        start, end = dataset._history_bounds(target)
        if (end != target or start >= target
                or not np.all(store.session_student[start:target+1] == store.session_student[target])
                or not np.all(store.session_split[start:target+1] == 0)):
            raise ValueError("pilot history crosses a student or training-split boundary")
        shape = dataset.shape(index)
        cost = padded_token_cost([shape])
        if cost > LIMITS["max_padded_tokens"]:
            skipped += 1
            continue
        selected.append(dict(dataset_index=int(index), target_session=target, history_start=start,
            history_end_exclusive=end, student=int(store.session_student[target]), split="train",
            history_lengths=list(shape.history_lengths), target_length=shape.target_length, padded_tokens=cost))
        if len(selected) == steps:
            return dataset, selected, skipped
    raise ValueError("not enough eligible whole training sessions within the pilot bound")


def collate_selected(dataset, item):
    example = dataset[item["dataset_index"]]
    if padded_token_cost([example_shape(example)]) != item["padded_tokens"]:
        raise ValueError("prepared example shape changed")
    batch = collate_hitskt([example], num_questions=int(dataset.store.metadata["num_questions"]),
        num_skills=int(dataset.store.metadata["num_skills"]))
    if int(batch.target_metric_mask.sum()) != item["target_length"]:
        raise ValueError("pilot supervised-target alignment changed")
    return batch


def prepare(run_id: str, *, store_path: Path | None = None, steps: int = 5) -> Path:
    """CPU data preflight only; no model or optimizer is constructed."""
    validate_steps(steps)
    memory = check_host_resources()
    reference_path = ROOT/"reports/favor_plus_v1/numerical_quality_v1/dataset_pipeline_plan.json"
    reference = json.loads(reference_path.read_text())["datasets"][DATASET]
    store = SessionStore(store_path or ROOT/"data/processed/assist2017/full/session_store")
    identity = verified_dataset_identity(store, reference)
    dataset, selected, skipped = select_training_examples(store, steps)
    for item in selected:
        collate_selected(dataset, item)  # PAD/EOS/shift construction, no forward
    output = fresh_run(DATASET, run_id)
    payload = dict(kind=PLAN_KIND, format_version=PLAN_VERSION, status="PREPARED_NOT_RUN",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), model=VARIANT, dataset=DATASET,
        protocol=PROTOCOL, seed=42, profile=profile_for(DATASET), favor_config=FAVOR,
        model_config=asdict(model_config(DATASET, store.metadata["num_questions"], store.metadata["num_skills"])),
        limits=LIMITS, requested_updates=steps, selected=selected, skipped_oversize_singletons=skipped,
        selection="original seed42 epoch1 sampler; first eligible whole train sessions; no label selection",
        dataset_identity=identity, store_path=str(store.root.resolve()), source=pilot_source_identity(),
        verified_dataset_plan_sha256=sha256(reference_path), host_memory=memory,
        cpu_environment=dict(python=platform.python_version(), torch=importlib.metadata.version("torch"),
                             numpy=np.__version__),
        optimizer_updates=0, model_forward_passes=0, cuda_status="NOT RUN", cuda_measurements=None,
        full_training_ready=False, note="bounded preparation only; no validation/test evaluation")
    path = output/"prepared_plan.json"
    write_json(path, payload)
    return path


def load_prepared_plan(path: Path) -> tuple[dict, RollingSessionDataset]:
    plan = json.loads(path.read_text())
    if (plan.get("kind") != PLAN_KIND or plan.get("format_version") != PLAN_VERSION
            or plan.get("status") != "PREPARED_NOT_RUN" or plan.get("dataset") != DATASET
            or plan.get("model") != VARIANT or plan.get("protocol") != PROTOCOL
            or plan.get("seed") != 42 or plan.get("favor_config") != FAVOR
            or plan.get("limits") != LIMITS or plan.get("profile") != profile_for(DATASET)):
        raise ValueError("prepared pilot configuration is not the fixed ASSIST2017 pilot")
    validate_steps(plan["requested_updates"])
    if plan["source"]["source_sha256"] != pilot_source_identity()["source_sha256"]:
        raise ValueError("pilot source changed; prepare a fresh plan")
    refpath = ROOT/"reports/favor_plus_v1/numerical_quality_v1/dataset_pipeline_plan.json"
    if sha256(refpath) != plan["verified_dataset_plan_sha256"]:
        raise ValueError("verified dataset reference changed")
    reference = json.loads(refpath.read_text())["datasets"][DATASET]
    store = SessionStore(plan["store_path"])
    if verified_dataset_identity(store, reference) != plan["dataset_identity"]:
        raise ValueError("pilot dataset identity changed")
    config = asdict(model_config(DATASET, store.metadata["num_questions"], store.metadata["num_skills"]))
    if plan.get("model_config") != config:
        raise ValueError("pilot model architecture changed")
    dataset, selected, skipped = select_training_examples(store, plan["requested_updates"])
    if selected != plan["selected"] or skipped != plan["skipped_oversize_singletons"]:
        raise ValueError("prepared sample order changed")
    return plan, dataset


def require_cuda_verification(path: Path, allocation: dict) -> dict:
    """Reject absent/failed/stale/different-device or different-profile evidence."""
    report = json.loads(path.read_text())
    from scripts.validate_favor_plus_gpu import LENGTHS
    combinations = {(length, chunk) for length in LENGTHS for chunk in (16, 32, 64)}
    rows = report.get("kernels", [])
    if (report.get("status") != "PASS" or report.get("scope") != "bounded synthetic GPU verification"
            or report.get("profile") != DATASET or report.get("features") != 128
            or report.get("allocation_resource_uuid") != allocation["resource_uuid"]
            or report.get("full_model_shift_gradients_checkpoint") != "PASS"
            or len(rows) != len(combinations)
            or {(r.get("length"), r.get("chunk")) for r in rows} != combinations):
        raise ValueError("matching complete ASSIST2017 128-feature CUDA PASS is required")
    policy = {"device": "cuda:0", "float32_matmul_precision": "highest", "matmul_allow_tf32": False,
              "cudnn_allow_tf32": False, "cudnn_benchmark": False,
              "cudnn_deterministic": True, "deterministic_algorithms": True}
    if any(report.get("runtime", {}).get(k) != value for k, value in policy.items()):
        raise ValueError("CUDA evidence does not use the fixed deterministic FP32 policy")
    current = pilot_source_identity()["source_sha256"]
    recorded = report.get("source", {}).get("source_sha256", {})
    for name in source_identity()["source_sha256"]:
        if recorded.get(name) != current[name]:
            raise ValueError("CUDA evidence has stale model/pipeline source hashes")
    if (report.get("script_sha256") != sha256(ROOT/"scripts/validate_favor_plus_gpu.py")
            or report.get("access_guard_sha256") != sha256(ROOT/"ktbench/favor_plus_gpu_access.py")
            or report.get("oracle_sha256") != sha256(ROOT/"scripts/favor_plus_quality_common.py")):
        raise ValueError("CUDA verification harness, access guard or oracle changed")
    for row in rows:
        if (row.get("status") != "PASS" or row.get("features") != 128
                or not 0 <= row.get("max_abs_output", math.inf) <= 5e-5
                or not 0 <= row.get("peak_allocated_bytes", math.inf) <= LIMITS["max_cuda_allocated_bytes"]
                or not 0 <= row.get("seconds", math.inf) <= 120):
            raise ValueError("CUDA stage evidence fails an unchanged gate")
    checkpoint = path.parent/"gpu_model.pt"
    if not checkpoint.exists() or sha256(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("CUDA verification checkpoint missing or changed")
    return report


@dataclass
class UpdateCounter:
    """Count calls before invoking step: failures can partially mutate optimizers."""
    limit: int
    calls: int = 0
    returned: int = 0

    def __post_init__(self):
        validate_steps(self.limit)

    def apply(self, optimizer):
        if self.calls >= self.limit:
            raise RuntimeError("five-update pilot budget exhausted; no additional step allowed")
        self.calls += 1
        optimizer.step()
        self.returned += 1


def bounded_updates(selected, steps, perform, check, record):
    """Small injectable controller; CPU tests use fake steps, never train a model."""
    validate_steps(steps)
    if len(selected) != steps:
        raise ValueError("prepared sample count must exactly match requested update count")
    for index in range(steps):
        check()
        row = perform(index+1, selected[index])
        check()
        record(index+1, row)


def execute(prepared_path: Path, allocation: dict, cuda_validation: Path, run_id: str):
    """Future CUDA-only execution. Never call without new user approval/allocation."""
    def authorize():
        return require_gpu_authorization(allocation, os.environ.get("CUDA_VISIBLE_DEVICES"), REAL_PILOT)
    authorize()  # before plan IO, seeding, model construction, output creation or ANY CUDA API
    plan, dataset = load_prepared_plan(prepared_path)
    validation = require_cuda_verification(cuda_validation, allocation)
    check_host_resources()
    output = fresh_run(DATASET, run_id)
    counter = UpdateCounter(plan["requested_updates"])
    started = time.perf_counter()
    phase = "initialization"
    last_training_observation = None
    signal_previous = {}
    manifest = dict(kind="bounded_real_assist2017_favor_plus_cuda_pilot", model=VARIANT,
        prepared_plan_sha256=sha256(prepared_path), prepared_plan=plan,
        allocation_record=allocation, cuda_validation_sha256=sha256(cuda_validation),
        cuda_validation_checkpoint_sha256=validation["checkpoint_sha256"], source=pilot_source_identity(),
        full_training_ready=False, validation_test_evaluations=0)
    write_json(output/"pilot_config.json", manifest)

    def boundary():
        authorize()
        memory = check_host_resources()
        if time.perf_counter()-started > LIMITS["max_execution_seconds"]:
            raise TimeoutError("pilot execution deadline exceeded")
        if torch.cuda.memory_allocated(0) > LIMITS["max_cuda_allocated_bytes"]:
            raise MemoryError("pilot GPU allocation budget exceeded")
        return memory

    def interrupted(signum, frame):
        raise InterruptedError("pilot process received termination; no automatic retry")

    try:
        for sig in (signal.SIGTERM,):
            signal_previous[sig] = signal.signal(sig, interrupted)
        authorize()
        # All CUDA/RNG APIs below are reachable only after the external/user gates.
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        from ktbench.config import seed_everything
        seed_everything(42)
        torch.use_deterministic_algorithms(True, warn_only=False)
        capacity = torch.cuda.get_device_properties(0).total_memory
        if capacity < LIMITS["max_cuda_allocated_bytes"]+2*1024**3:
            raise MemoryError("allocated GPU has insufficient budget plus runtime headroom")
        torch.cuda.set_per_process_memory_fraction(LIMITS["max_cuda_allocated_bytes"]/capacity, 0)
        device = torch.device("cuda:0")
        net = HiTSKTFavorPlus(model_config(DATASET, dataset.store.metadata["num_questions"],
            dataset.store.metadata["num_skills"]), FavorPlusConfig(**FAVOR)).to(device)
        profile = profile_for(DATASET)
        optimizer = torch.optim.Adam(net.parameters(), lr=profile["learning_rate"], weight_decay=profile["weight_decay"])
        fixed_batch = collate_selected(dataset, plan["selected"][0]).to(device)
        write_json(output/"runtime.json", runtime(device))
        torch.cuda.reset_peak_memory_stats(0)
        snapshots = []

        def diagnose(step, baseline=None):
            boundary()
            def save(name, arrays):
                filename = f"prefix_{step:02d}_{name.replace('.', '_')}.npz"
                with (output/filename).open("xb") as handle:
                    np.savez(handle, **arrays)
            torch.cuda.synchronize(0)
            stamp = time.perf_counter()
            stages = fixed_example_diagnostics(net, fixed_batch, baseline=baseline, save_prefix=save)
            torch.cuda.synchronize(0)
            row = {"after_optimizer_updates": step, "fixed_training_target": plan["selected"][0]["target_session"],
                   "mode": "eval; fixed input; QKV evolves with weights", "stages": stages,
                   "diagnostic_wall_seconds": time.perf_counter()-stamp}
            write_json(output/f"diagnostics_{step:02d}.json", row)
            snapshots.append(row)
            boundary()
            return stages

        phase = "initial_fixed_example_diagnostics"
        initial = diagnose(0)

        def perform(step, item):
            nonlocal phase, last_training_observation
            phase = f"optimizer_update_{step}"
            batch = collate_selected(dataset, item).to(device)
            net.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(0)
            stamp = time.perf_counter()
            logits = net(batch)
            loss = masked_bce_loss(logits, batch.target_labels, batch.target_metric_mask)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("nonfinite pilot logits/loss; optimizer not stepped")
            loss.backward()
            gradients = [p.grad for p in net.parameters()]
            if any(g is None or not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("missing/nonfinite gradient; optimizer not stepped")
            grad_norm = math.hypot(*(float(g.double().norm().item()) for g in gradients))
            if not math.isfinite(grad_norm):
                raise FloatingPointError("nonfinite global gradient norm")
            boundary()  # memory/time/lease failures do not get an optimizer step
            last_training_observation = dict(update=step, loss=float(loss.detach().cpu()),
                gradient_l2_norm=grad_norm, supervised_targets=item["target_length"],
                optimizer_step_calls_before_update=counter.calls)
            if time.perf_counter()-stamp > LIMITS["max_step_seconds"]:
                raise TimeoutError("pilot forward/backward exceeds120 seconds; optimizer not stepped")
            counter.apply(optimizer)
            torch.cuda.synchronize(0)
            seconds = time.perf_counter()-stamp
            if not finite_tree(net.state_dict()) or not finite_tree(optimizer.state_dict()):
                raise FloatingPointError("nonfinite model/Adam state after update; stop without reset")
            if seconds > LIMITS["max_step_seconds"]:
                raise TimeoutError("pilot update exceeds120-second acceptance budget")
            row = dict(update=step, optimizer_step_calls=counter.calls, optimizer_updates_returned=counter.returned,
                loss=float(loss.detach().cpu()), gradient_l2_norm=grad_norm,
                supervised_targets=int(batch.target_metric_mask.sum()), padded_tokens=item["padded_tokens"],
                synchronized_training_wall_seconds=seconds,
                targets_per_second=item["target_length"]/seconds,
                padded_tokens_per_second=item["padded_tokens"]/seconds,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
            # Preserve accepted loss/gradient/time measurements even if the next
            # diagnostic or lease boundary fails. Do not retry an applied step.
            write_json(output/f"update_{step:02d}.json", row)
            return row

        def after_step(step, row):
            nonlocal phase
            phase = f"post_update_{step}_diagnostics"
            diagnose(step, initial)

        bounded_updates(plan["selected"], plan["requested_updates"], perform, boundary,
            after_step)
        phase = "checkpoint_and_reload"
        boundary()
        net.eval()
        with torch.no_grad():
            before = net(fixed_batch).detach()
        from ktbench.training import capture_rng_state
        payload = cpu_tree(net.checkpoint())
        payload.update(kind="bounded_real_assist2017_favor_plus_cuda_pilot", optimizer_updates=counter.returned,
            optimizer_state_dict=cpu_tree(optimizer.state_dict()), rng_state=cpu_tree(capture_rng_state()),
            pilot_config=manifest)
        checkpoint = output/"pilot_model.pt"
        with checkpoint.open("xb") as handle:
            torch.save(payload, handle)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not equal_tree(payload, loaded):
            raise RuntimeError("checkpoint state/optimizer/RNG serialization changed")
        restored = HiTSKTFavorPlus.from_checkpoint(checkpoint).to(device).eval()
        replay_optimizer = torch.optim.Adam(restored.parameters(), lr=profile["learning_rate"], weight_decay=profile["weight_decay"])
        replay_optimizer.load_state_dict(loaded["optimizer_state_dict"])
        if not equal_tree(payload["optimizer_state_dict"], cpu_tree(replay_optimizer.state_dict())):
            raise RuntimeError("optimizer checkpoint reload changed state")
        with torch.no_grad():
            after = restored(fixed_batch)
        if not torch.equal(before, after):
            raise RuntimeError("checkpoint reload changed fixed-example CUDA logits")
        boundary()
        flags = sorted({flag for snap in snapshots for entry in snap["stages"].values()
                        for flag in entry["review_flags"]})
        result = dict(execution_status="PASS", kind="bounded_real_assist2017_favor_plus_cuda_pilot",
            optimizer_step_calls=counter.calls, optimizer_updates_returned=counter.returned,
            checkpoint_sha256=sha256(checkpoint), checkpoint_reload_max_abs_logit_difference=0.,
            checkpoint_optimizer_rng_integrity="PASS", approximation_review_flags=flags,
            approximation_review_status="REVIEW_REQUIRED" if flags else "NO_TRIGGER_IN_TINY_PILOT",
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0), host_memory=host_memory(),
            execution_wall_seconds=time.perf_counter()-started, runtime=runtime(device),
            full_training_ready=False, validation_test_evaluations=0,
            limitation="five updates do not establish convergence, validation/test AUC or general approximation quality")
        write_json(output/"pilot_manifest.json", result)
        (output/"_PILOT_COMPLETE").write_text("bounded execution only; inspect approximation review; no benchmark AUC\n")
        return result
    except BaseException as exc:
        write_json(output/"pilot_failure.json", dict(execution_status="FAIL", phase=phase,
            optimizer_step_calls=counter.calls, optimizer_updates_returned=counter.returned,
            note="a raising optimizer call may have partially mutated state; no automatic retry or reset",
            failure_type=type(exc).__name__, message=str(exc), execution_wall_seconds=time.perf_counter()-started,
            last_training_observation=last_training_observation,
            full_training_ready=False))
        raise
    finally:
        for sig, previous in signal_previous.items():
            signal.signal(sig, previous)
