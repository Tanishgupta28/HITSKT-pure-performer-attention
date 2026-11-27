#!/usr/bin/env python3
"""Train one full rolling DKT, DKVMN, or SAKT experiment."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ktbench.baseline_registry import BASELINE_TRAINING_CONFIGS, make_baseline_model
from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.fast_rolling import PackedBatchSampler, PackedRollingDataset
from ktbench.data.rolling_targets import (
    RollingLengthBatchSampler,
    RollingTargetDataset,
    collate_rolling_targets,
)
from ktbench.data.session_store import SessionStore
from ktbench.metrics import binary_metrics
from ktbench.models import load_baseline_checkpoint
from ktbench.training import (
    COMMON_EARLY_STOPPING,
    DKTEdNetEarlyStoppingConfig,
    EarlyStoppingConfig,
    ValidationAUCEarlyStopping,
    capture_rng_state,
    restore_rng_state,
)


def _early_stopping_config(
    model_name: str, dataset_name: str, patience: int
) -> EarlyStoppingConfig:
    if patience == COMMON_EARLY_STOPPING.patience:
        return COMMON_EARLY_STOPPING
    if model_name == "dkt" and dataset_name == "ednet_kt1" and patience == 10:
        return DKTEdNetEarlyStoppingConfig()
    raise ValueError("patience 10 is authorized only for DKT/full EdNet")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _loader(
    dataset: RollingTargetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    epoch: int,
    workers: int,
    transport: str = "reference",
) -> tuple[DataLoader, RollingLengthBatchSampler]:
    if transport not in ("reference", "packed64"):
        raise ValueError(f"unknown transport: {transport}")
    sampler = RollingLengthBatchSampler(
        dataset,
        batch_size=batch_size,
        bucket_size=4096,
        shuffle=shuffle,
        seed=PROJECT_SEED,
    )
    sampler.set_epoch(epoch if shuffle else 0)
    if transport == "packed64":
        return DataLoader(
            PackedRollingDataset(dataset),
            sampler=PackedBatchSampler(sampler, block_size=64),
            batch_size=None,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        ), sampler
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_rolling_targets,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return loader, sampler


def _run_epoch(
    model: torch.nn.Module,
    dataset: RollingTargetDataset,
    *,
    batch_size: int,
    device: torch.device,
    workers: int,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float | None,
    transport: str = "reference",
    progress_path: Path | None = None,
    dkt_dense: bool = False,
    gpu_sample_path: Path | None = None,
    verify_target_coverage: bool = False,
    check_finite_train_state: bool = False,
) -> dict[str, float | int]:
    training = optimizer is not None
    started = time.perf_counter()
    model.train(training)
    forward = model.forward_dense if dkt_dense else model
    loader, _ = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        epoch=epoch,
        workers=workers,
        transport=transport,
    )
    probabilities = np.empty(len(dataset), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.uint8)
    target_ids = np.empty(len(dataset), dtype=np.int64) if verify_target_coverage else None
    cursor = 0
    loss_sum = 0.0
    records = []
    batches_processed = 0
    targets_processed = 0

    def progress(status: str) -> None:
        if progress_path is None:
            return
        value = {
            "event": "epoch_progress",
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "epoch": epoch,
            "split": {0: "train", 1: "validation", 2: "test"}[dataset.split],
            "status": status,
            "minibatches_processed": batches_processed,
            "targets_processed": targets_processed,
            "targets_total": len(dataset),
            "elapsed_seconds": time.perf_counter() - started,
        }
        temporary = progress_path.with_suffix(".json.tmp")
        _write_json(temporary, value)
        temporary.replace(progress_path)
        print(json.dumps(value, sort_keys=True), flush=True)

    progress("running")

    def flush():
        nonlocal cursor, loss_sum
        if not records:
            return
        losses = torch.stack([r[0] for r in records]).cpu().numpy()
        probs = torch.cat([r[1] for r in records]).cpu().numpy()
        targets = torch.cat([r[2] for r in records]).cpu().numpy()
        for value, record in zip(losses, records, strict=True):
            loss_sum += float(value) * len(record[2])
        count = len(targets)
        probabilities[cursor:cursor + count] = probs
        labels[cursor:cursor + count] = targets
        if target_ids is not None:
            target_ids[cursor:cursor + count] = torch.cat([r[3] for r in records]).cpu().numpy()
        cursor += count
        records.clear()

    def batches():
        for item in loader:
            if transport == "packed64":
                yield from item.to_batches(device)
            else:
                yield item.to(device)

    trace = None
    for batch in batches():
        if gpu_sample_path is not None and batches_processed == 64:
            from torch.profiler import ProfilerActivity, profile

            torch.cuda.synchronize(device)
            trace = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            trace.start()
            sample_start = time.perf_counter()
        if training:
            optimizer.zero_grad(set_to_none=True)
            logits = forward(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch.target_labels.float(), reduction="mean"
            )
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = forward(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch.target_labels.float(), reduction="mean"
                )
        count = len(batch.target_labels)
        if transport == "packed64":
            records.append((loss.detach(), torch.sigmoid(logits).detach(),
                            batch.target_labels, batch.target_event_id))
            if len(records) == 64:
                flush()
        else:
            loss_sum += float(loss.detach().cpu()) * count
            probabilities[cursor : cursor + count] = torch.sigmoid(logits).detach().cpu().numpy()
            labels[cursor : cursor + count] = batch.target_labels.detach().cpu().numpy()
            if target_ids is not None:
                target_ids[cursor:cursor + count] = batch.target_event_id.cpu().numpy()
            cursor += count
        batches_processed += 1
        targets_processed += count
        if trace is not None and batches_processed == 72:
            torch.cuda.synchronize(device)
            sample_seconds = time.perf_counter() - sample_start
            trace.stop()
            intervals = sorted(
                (event.time_range.start, event.time_range.end) for event in trace.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
                and not event.name.startswith(("Memcpy", "Memset", "[CUDA memcpy", "[CUDA memset"))
            )
            active_us, previous = 0.0, 0.0
            for start, end in intervals:
                active_us += max(0.0, end - max(start, previous))
                previous = max(previous, end)
            if not intervals:
                raise RuntimeError("CUDA profiler observed no training kernels")
            evidence = {
                "event": "training_cuda_activity_sample", "epoch": epoch,
                "minibatches": 8, "cuda_kernels": len(intervals),
                "kernel_active_seconds": active_us / 1e6, "sample_seconds": sample_seconds,
                "process_kernel_active_percent": 100 * active_us / (1e6 * sample_seconds),
                "measurement": "CUPTI kernel intervals over a profiled wall-time window",
            }
            _write_json(gpu_sample_path, evidence)
            print(json.dumps(evidence), flush=True)
            trace = None
        if batches_processed == 1 or batches_processed % 4096 == 0:
            if training and check_finite_train_state:
                _verify_finite_resume_state({"state_dict": model.state_dict(),
                                             "optimizer_state_dict": optimizer.state_dict()})
            progress("running")
    flush()
    if cursor != len(dataset):
        raise RuntimeError(f"evaluated {cursor} targets but expected {len(dataset)}")
    if target_ids is not None:
        observed = np.sort(target_ids)
        if isinstance(dataset, RollingTargetDataset):
            for start in range(0, len(dataset), 1_000_000):
                stop = min(start + 1_000_000, len(dataset))
                _, expected = dataset._events_for_indices(np.arange(start, stop, dtype=np.int64))
                if not np.array_equal(observed[start:stop], expected):
                    raise ValueError("actual evaluation targets differ from the session split")
        elif not np.array_equal(observed, dataset.events):
            raise ValueError("actual evaluation targets differ from the ordered split manifest")
        if not np.array_equal(labels, dataset.store.correct[target_ids]):
            raise ValueError("actual evaluation labels differ from the ordered split manifest")
    progress("computing_metrics")
    metrics = binary_metrics(
        torch.from_numpy(probabilities),
        torch.from_numpy(labels),
        loss=loss_sum / cursor,
    )
    progress("complete")
    return metrics


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation_auc: float,
    control: ValidationAUCEarlyStopping,
) -> dict[str, Any]:
    payload = model.checkpoint()
    payload.update(
        {
            "epoch": epoch,
            "validation_roc_auc": validation_auc,
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": PROJECT_SEED,
            "early_stopping": control.config.as_dict(),
            "early_stopping_state": {
                "best_epoch": control.best_epoch,
                "best_validation_auc": control.best_validation_auc,
                "consecutive_epochs_without_improvement": control.consecutive_epochs_without_improvement,
            },
            "rng_state": capture_rng_state(),
        }
    )
    return payload


def _verify_finite_resume_state(checkpoint: dict[str, Any]) -> None:
    states = [("model", checkpoint["state_dict"])]
    states += [(f"Adam parameter {index}", values)
               for index, values in checkpoint["optimizer_state_dict"]["state"].items()]
    for group, values in states:
        for name, value in values.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                raise ValueError(f"Nonfinite checkpoint state in {group}: {name}")


def train_baseline(
    model_name: str,
    dataset_name: str,
    store: SessionStore,
    output_root: Path,
    *,
    workers: int = 0,
    epoch_ceiling_override: int | None = None,
    resume: bool = False,
    transport: str = "reference",
    early_stopping_patience_override: int | None = None,
    ednet_dkt_dense: bool = False,
) -> dict[str, Any]:
    """Run the protocol, including the explicit DKT/EdNet patience exception."""

    if transport not in ("reference", "packed64"):
        raise ValueError(f"unknown transport: {transport}")
    saved_configuration = (
        json.loads((output_root / "config.json").read_text())
        if resume and (output_root / "config.json").exists()
        else None
    )
    saved_patience = (
        int(saved_configuration["early_stopping_patience"])
        if saved_configuration is not None
        else COMMON_EARLY_STOPPING.patience
    )
    patience = saved_patience if early_stopping_patience_override is None else early_stopping_patience_override
    if saved_configuration is not None and patience < saved_patience:
        raise ValueError("cannot reduce an existing experiment's patience")
    early_stopping = _early_stopping_config(model_name, dataset_name, patience)
    seed_everything()
    if (output_root / "_SUCCESS").exists():
        raise ValueError(f"completed experiment already exists: {output_root}")
    if ednet_dkt_dense:
        if (model_name, dataset_name, transport, patience) != ("dkt", "ednet_kt1", "packed64", 10):
            raise ValueError("EdNet DKT dense mode requires packed64 and the authorized patience of 10")
        if not torch.cuda.is_available():
            raise RuntimeError("EdNet DKT dense mode requires CUDA")
        if (output_root / "_TEST_EVALUATED").exists():
            raise ValueError("final test is already recorded; run independent checkpoint verification")
        if not resume and output_root.exists():
            raise FileExistsError(output_root)
    if dataset_name in ("algebra2005", "assist2012") and not resume and (output_root / "config.json").exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    profile = dict(BASELINE_TRAINING_CONFIGS[model_name])
    epoch_ceiling = int(profile["epoch_ceiling"])
    if epoch_ceiling_override is not None:
        if epoch_ceiling_override <= 0 or "PYTEST_CURRENT_TEST" not in __import__("os").environ:
            raise ValueError("epoch ceiling override is restricted to automated tests")
        epoch_ceiling = epoch_ceiling_override
    history_length = int(profile["history_length"])
    ordered = dataset_name == "assist2009"
    strict_time = dataset_name in ("algebra2005", "assist2012")
    if strict_time and (not torch.cuda.is_available() or not store.strict_timestamp_history):
        raise ValueError("Strict-time benchmark full training requires CUDA and the verified strict-time session store")
    if (ordered or strict_time) and (transport != "packed64" or not torch.cuda.is_available()):
        raise ValueError("This dataset requires packed64 batch transport and CUDA")
    datasets = {
        split: (store.targets(split, history_length) if ordered
                else RollingTargetDataset(store, split, history_length=history_length))
        for split in ("train", "validation", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("an experiment split contains no targets")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    dkt_dense = ednet_dkt_dense or ((ordered or strict_time) and model_name == "dkt")
    model = make_baseline_model(
        model_name,
        num_questions=int(store.metadata["num_questions"]),
        num_skills=int(store.metadata["num_skills"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    fresh_configuration: dict[str, Any] = {
        **profile,
        "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "device": str(device),
        "gpu": gpu_name,
        "workers": workers,
        "transport": transport,
        "transport_block_minibatches": 64 if transport == "packed64" else 1,
        "early_stopping_metric": early_stopping.metric,
        "early_stopping_patience": early_stopping.patience,
        "early_stopping_min_delta": early_stopping.min_delta,
        "early_stopping_strict_improvement": True,
        "reload_best_checkpoint_before_test": True,
        "total_parameters": total_parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "best_epoch": None,
        "best_validation_auc": None,
        "epochs_completed": 0,
    }
    if dkt_dense:
        torch.backends.cudnn.allow_tf32 = False
    if ordered:
        fresh_configuration["dkt_dense_lstm"] = dkt_dense
        fresh_configuration["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        fresh_configuration["dataset_provenance"] = store.provenance
        fresh_configuration["torch_version"] = str(torch.__version__)
        fresh_configuration["cuda_runtime"] = torch.version.cuda
        fresh_configuration["gpu_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    if ednet_dkt_dense:
        fresh_configuration.update(
            dkt_dense_lstm=True,
            cudnn_allow_tf32=False,
            independent_replay_required=True,
            session_store_metadata_sha256=hashlib.sha256((store.root / "metadata.json").read_bytes()).hexdigest(),
            torch_version=str(torch.__version__),
            cuda_runtime=torch.version.cuda,
            gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        )
    if strict_time:
        fresh_configuration.update(dkt_dense_lstm=dkt_dense,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, session_store_metadata_sha256=hashlib.sha256(
            (store.root / "metadata.json").read_bytes()).hexdigest(),
            torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda,
            gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    if epoch_ceiling_override is not None:
        fresh_configuration["test_epoch_ceiling_override"] = epoch_ceiling
    if early_stopping.patience == 10:
        fresh_configuration["early_stopping_protocol_exception"] = "user-authorized DKT/full EdNet patience 10"
    target_counts = {split: len(dataset) for split, dataset in datasets.items()}
    dataset_stats = {
        "dataset": dataset_name,
        "students": int(store.metadata["students"]),
        "interactions": int(store.metadata["interactions"]),
        "questions": int(store.metadata["num_questions"]),
        "skills": int(store.metadata["num_skills"]),
        "sessions": None if ordered else int(store.metadata["sessions"]),
        "target_counts": target_counts,
        "target_ratios": {
            split: count / sum(target_counts.values()) for split, count in target_counts.items()
        },
    }
    metrics_fields = [
        "epoch",
        "split",
        "targets",
        "loss",
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mse",
        "threshold",
    ]
    metrics_path = output_root / "metrics.csv"
    log_path = output_root / "training.jsonl"
    control = ValidationAUCEarlyStopping(early_stopping)
    best_path = output_root / "best_model.pt"
    last_path = output_root / "last_model.pt"
    config_path = output_root / "config.json"
    stats_path = output_root / "dataset_stats.json"
    start_epoch = 1
    epochs_completed = 0
    prior_runtime_seconds = 0.0
    runtime_estimated_before_resume = False
    if resume:
        required = (config_path, stats_path, metrics_path, log_path, best_path, last_path)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ValueError(f"cannot resume; missing artifacts: {missing}")
        configuration = json.loads(config_path.read_text())
        for key in ("model", "dataset", "seed", "batch_size", "history_length"):
            if configuration.get(key) != fresh_configuration.get(key):
                raise ValueError(f"resume configuration mismatch for {key}")
        if strict_time and (configuration.get("session_store_metadata_sha256") !=
                        fresh_configuration["session_store_metadata_sha256"]):
            raise ValueError("Strict-time benchmark resume session store differs")
        if ordered and (configuration.get("dataset_provenance") != store.provenance
                        or configuration.get("dkt_dense_lstm") != dkt_dense):
            raise ValueError("ASSIST2009 dataset provenance differs from the saved experiment")
        if bool(configuration.get("independent_replay_required")) != ednet_dkt_dense:
            raise ValueError("resume execution mode differs from checkpoint run")
        if ednet_dkt_dense and any(configuration.get(k) != fresh_configuration[k] for k in (
                "session_store_metadata_sha256", "dkt_dense_lstm", "cudnn_allow_tf32", "torch_version",
                "cuda_runtime", "gpu")):
            raise ValueError("EdNet DKT replay provenance differs from saved run")
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        if model_name == "dkt" and dataset_name == "ednet_kt1":
            _verify_finite_resume_state(checkpoint)
        if checkpoint.get("seed") != PROJECT_SEED:
            raise ValueError("resume checkpoint seed mismatch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        elif float(configuration.get("dropout", 0.0)) != 0.0:
            raise ValueError("cannot exactly resume stochastic model without RNG state")
        epochs_completed = int(checkpoint["epoch"])
        if epochs_completed != int(configuration["epochs_completed"]):
            raise ValueError("resume epoch mismatch between checkpoint and config")
        epoch_events = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line and json.loads(line).get("event") == "epoch_complete"
        ]
        if not epoch_events or int(epoch_events[-1]["epoch"]) != epochs_completed:
            raise ValueError("resume log does not end at checkpoint epoch")
        state = checkpoint.get("early_stopping_state")
        if state is None:
            state = {
                "best_epoch": configuration["best_epoch"],
                "best_validation_auc": configuration["best_validation_auc"],
                "consecutive_epochs_without_improvement": epoch_events[-1][
                    "consecutive_epochs_without_improvement"
                ],
            }
        control.best_epoch = int(state["best_epoch"])
        control.best_validation_auc = float(state["best_validation_auc"])
        control.consecutive_epochs_without_improvement = int(
            state["consecutive_epochs_without_improvement"]
        )
        start_epoch = epochs_completed + 1
        prior_runtime = configuration.get("runtime_seconds_accumulated")
        if prior_runtime is None:
            prior_runtime_seconds = max(0.0, config_path.stat().st_mtime - stats_path.stat().st_mtime)
            runtime_estimated_before_resume = True
        else:
            prior_runtime_seconds = float(prior_runtime)
        configuration.update(
            {
                "resume_count": int(configuration.get("resume_count", 0)) + 1,
                "resumed_from_epoch": epochs_completed,
                "runtime_before_resume_seconds": prior_runtime_seconds,
                "runtime_before_resume_estimated_from_artifact_mtimes": runtime_estimated_before_resume,
                "previous_transport": configuration.get("transport", "reference"),
                "transport": transport,
                "transport_block_minibatches": 64 if transport == "packed64" else 1,
                "workers": workers,
            }
        )
        configuration["early_stopping_patience"] = early_stopping.patience
        if early_stopping.patience == 10:
            configuration["early_stopping_protocol_exception"] = fresh_configuration["early_stopping_protocol_exception"]
        if saved_patience != early_stopping.patience:
            configuration["early_stopping_patience_previous"] = saved_patience
            configuration["early_stopping_patience_changed_after_epoch"] = epochs_completed
        if epoch_ceiling_override is not None:
            configuration["test_epoch_ceiling_override"] = epoch_ceiling
        _write_json(config_path, configuration)
    else:
        configuration = fresh_configuration
        _write_json(config_path, configuration)
        _write_json(stats_path, dataset_stats)

    started = time.perf_counter()
    metrics_mode = "a" if resume else "w"
    log_mode = "a" if resume else "w"
    with metrics_path.open(metrics_mode, newline="") as metrics_file, log_path.open(log_mode) as log:
        writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
        if not resume:
            writer.writeheader()
        else:
            log.write(
                json.dumps(
                    {
                        "event": "training_resumed",
                        "seed": PROJECT_SEED,
                        "resumed_from_epoch": epochs_completed,
                        "transport": transport,
                        "previous_transport": configuration["previous_transport"],
                        "best_epoch": control.best_epoch,
                        "best_validation_auc": control.best_validation_auc,
                        "consecutive_epochs_without_improvement": control.consecutive_epochs_without_improvement,
                        "early_stopping_patience": early_stopping.patience,
                        "previous_early_stopping_patience": saved_patience,
                    }
                )
                + "\n"
            )
            log.flush()
        for epoch in range(start_epoch, epoch_ceiling + 1):
            if control.should_stop:
                break
            if ordered or strict_time or ednet_dkt_dense:
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics = _run_epoch(
                model,
                datasets["train"],
                batch_size=int(profile["batch_size"]),
                device=device,
                workers=workers,
                epoch=epoch,
                optimizer=optimizer,
                gradient_clip=profile["gradient_clip"],
                transport=transport,
                progress_path=output_root / "progress.json",
                dkt_dense=dkt_dense,
                check_finite_train_state=ednet_dkt_dense or strict_time,
                gpu_sample_path=(output_root / "gpu_activity.json" if (ordered or strict_time or ednet_dkt_dense) and epoch == 1
                                 and len(datasets["train"]) >= 72 * profile["batch_size"] else None),
            )
            validation_metrics = _run_epoch(
                model,
                datasets["validation"],
                batch_size=int(profile["batch_size"]),
                device=device,
                workers=workers,
                epoch=epoch,
                optimizer=None,
                gradient_clip=None,
                transport=transport,
                progress_path=output_root / "progress.json",
                dkt_dense=dkt_dense,
            )
            if ednet_dkt_dense:
                _verify_finite_resume_state({"state_dict": model.state_dict(),
                                             "optimizer_state_dict": optimizer.state_dict()})
            for split, values in (("train", train_metrics), ("validation", validation_metrics)):
                writer.writerow({"epoch": epoch, "split": split, **values})
            metrics_file.flush()
            decision = control.step(epoch, float(validation_metrics["roc_auc"]))
            payload = _checkpoint_payload(
                model,
                optimizer,
                epoch=epoch,
                validation_auc=float(validation_metrics["roc_auc"]),
                control=control,
            )
            torch.save(payload, last_path)
            if decision.improved:
                torch.save(payload, best_path)
            epochs_completed = epoch
            if ordered or strict_time or ednet_dkt_dense:
                print(json.dumps({
                    "event": "epoch_metrics", "model": profile["model"], "epoch": epoch,
                    "train_loss": train_metrics["loss"], "validation_loss": validation_metrics["loss"],
                    "validation_auc": validation_metrics["roc_auc"],
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }), flush=True)
            log.write(
                json.dumps(
                    {
                        "event": "epoch_complete",
                        "seed": PROJECT_SEED,
                        "epoch": epoch,
                        "transport": transport,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "checkpoint_saved": decision.improved,
                        "consecutive_epochs_without_improvement": decision.consecutive_epochs_without_improvement,
                        "early_stop": decision.should_stop,
                        "early_stopping_patience": early_stopping.patience,
                    }
                )
                + "\n"
            )
            log.flush()
            configuration.update(
                {
                    "best_epoch": decision.best_epoch,
                    "best_validation_auc": decision.best_validation_auc,
                    "epochs_completed": epochs_completed,
                    "runtime_seconds_accumulated": prior_runtime_seconds
                    + time.perf_counter()
                    - started,
                }
            )
            _write_json(output_root / "config.json", configuration)
            if decision.should_stop:
                break

        if not best_path.exists() or control.best_epoch is None:
            raise RuntimeError("no finite validation ROC-AUC checkpoint was produced")
        best_model = load_baseline_checkpoint(best_path, map_location=device).to(device).eval()
        test_metrics = _run_epoch(
            best_model,
            datasets["test"],
            batch_size=int(profile["batch_size"]),
            device=device,
            workers=workers,
            epoch=control.best_epoch,
            optimizer=None,
            gradient_clip=None,
            transport=transport,
            progress_path=output_root / "progress.json",
            dkt_dense=dkt_dense,
        )
        writer.writerow({"epoch": control.best_epoch, "split": "test", **test_metrics})
        metrics_file.flush()
        log.write(
            json.dumps(
                {
                    "event": "final_test_from_best_checkpoint",
                    "seed": PROJECT_SEED,
                    "best_epoch": control.best_epoch,
                    "best_validation_auc": control.best_validation_auc,
                    "epochs_completed": epochs_completed,
                    "test": test_metrics,
                }
            )
            + "\n"
        )

    runtime_seconds = prior_runtime_seconds + time.perf_counter() - started
    result = {
        "status": "awaiting_independent_replay" if ednet_dkt_dense else "complete",
        "dataset": dataset_name,
        "model": profile["model"],
        "seed": PROJECT_SEED,
        "patience": early_stopping.patience,
        "min_delta": early_stopping.min_delta,
        "best_epoch": control.best_epoch,
        "best_validation_auc": control.best_validation_auc,
        "epochs_completed": epochs_completed,
        "epoch_ceiling": epoch_ceiling,
        "test": test_metrics,
        "runtime_seconds": runtime_seconds,
        "resume_count": int(configuration.get("resume_count", 0)),
        "runtime_before_resume_estimated_from_artifact_mtimes": runtime_estimated_before_resume,
        "total_parameters": total_parameters,
        "trainable_parameters": configuration["trainable_parameters"],
        "best_checkpoint_reloaded": True,
        "gpu": gpu_name,
        "transport": transport,
    }
    if early_stopping.patience == 10:
        result["early_stopping_protocol_exception"] = fresh_configuration["early_stopping_protocol_exception"]
    _write_json(output_root / "final_results.json", result)
    configuration.update(
        {
            "best_epoch": control.best_epoch,
            "best_validation_auc": control.best_validation_auc,
            "epochs_completed": epochs_completed,
            "runtime_seconds_accumulated": runtime_seconds,
        }
    )
    _write_json(output_root / "config.json", configuration)
    if ednet_dkt_dense:
        (output_root / "_TEST_EVALUATED").write_text("awaiting independent replay\n")
    else:
        (output_root / "_SUCCESS").write_text("complete\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(BASELINE_TRAINING_CONFIGS))
    parser.add_argument("dataset", choices=("assist2017", "junyi", "ednet_kt1", "assist2009", "algebra2005", "assist2012"))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ednet-dkt-dense", action="store_true",
                        help="full EdNet GPU DKT with dense LSTM and independent replay gate")
    parser.add_argument("--transport", choices=("reference", "packed64"), default="reference")
    parser.add_argument("--early-stopping-patience", type=int, choices=(5, 10), default=None,
                        help="explicit DKT/full EdNet exception; saved patience is retained on resume")
    args = parser.parse_args()
    if args.dataset == "assist2009":
        from ktbench.data.assist2009 import Assist2009Store

        store = Assist2009Store(args.store_root)
        if not args.resume and args.output_root.exists():
            raise FileExistsError(args.output_root)
    else:
        store = SessionStore(args.store_root)
    result = train_baseline(
        args.model,
        args.dataset,
        store,
        args.output_root,
        workers=args.workers,
        resume=args.resume,
        transport=args.transport,
        early_stopping_patience_override=args.early_stopping_patience,
        ednet_dkt_dense=args.ednet_dkt_dense,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
