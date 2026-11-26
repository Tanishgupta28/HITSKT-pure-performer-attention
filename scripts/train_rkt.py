#!/usr/bin/env python3
"""Train the approved full RKT performance-only cross-fitted experiment."""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.assist2009 import Assist2009Store
from ktbench.data.rolling_targets import RollingLengthBatchSampler
from ktbench.data.session_store import SessionStore
from ktbench.metrics import binary_metrics
from ktbench.models import PaperFaithfulRKT, RKTConfig
from ktbench.models.rkt import RKT_LABEL, RKT_PHI_LABEL, RKT_PHI_ONLY
from ktbench.rkt.data import OrderedRKTTargets, RKTTargetDataset, collate_rkt
from ktbench.rkt.phi import CrossFittedPhiRepository
from ktbench.rkt.settings import (
    RKT_BATCH_SIZE,
    RKT_EPOCH_CEILING,
    RKT_GRADIENT_CLIP,
    RKT_LEARNING_RATE,
    RKT_WEIGHT_DECAY,
)
from ktbench.training import (
    COMMON_EARLY_STOPPING,
    ValidationAUCEarlyStopping,
    capture_rng_state,
    restore_rng_state,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run_epoch(
    model: PaperFaithfulRKT,
    dataset: RKTTargetDataset,
    repository: CrossFittedPhiRepository,
    *,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    workers: int,
    verify_target_coverage: bool = False,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    model.set_relation_parameters_trainable(training)
    sampler = RollingLengthBatchSampler(
        dataset,
        batch_size=RKT_BATCH_SIZE,
        bucket_size=4096,
        shuffle=training,
        seed=PROJECT_SEED,
    )
    sampler.set_epoch(epoch if training else 0)
    ordered = isinstance(dataset, OrderedRKTTargets)
    if ordered:
        loader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=workers,
                            persistent_workers=workers > 0, pin_memory=True)
    else:
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=partial(collate_rkt, phi_repository=repository),
            num_workers=workers,
            persistent_workers=workers > 0,
        )
    probabilities = np.empty(len(dataset), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.uint8)
    loss_sum = 0.0
    cursor = 0
    target_ids = np.empty(len(dataset), dtype=np.int64) if verify_target_coverage else None
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        batch = batch.to(device, non_blocking=ordered)
        if training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch.target_labels.float()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RKT_GRADIENT_CLIP)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch.target_labels.float()
                )
        count = len(batch.target_labels)
        loss_sum += float(loss.detach().cpu()) * count
        probabilities[cursor : cursor + count] = torch.sigmoid(logits).detach().cpu().numpy()
        labels[cursor : cursor + count] = batch.target_labels.detach().cpu().numpy()
        if target_ids is not None:
            target_ids[cursor : cursor + count] = batch.target_event_id.cpu().numpy()
        cursor += count
        if ordered and (step == 1 or step % 500 == 0):
            print(json.dumps({"epoch": epoch, "split": dataset.split, "targets": cursor,
                              "expected_targets": len(dataset), "elapsed_seconds": time.perf_counter() - started,
                              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)}), flush=True)
    if target_ids is not None and (
        not np.array_equal(np.sort(target_ids), dataset.events if ordered else dataset.store.events_in_split(dataset.split))
        or not np.array_equal(labels, dataset.store.correct[target_ids])
    ):
        raise ValueError("RKT prediction IDs or labels differ from the shared benchmark")
    if cursor != len(dataset):
        raise RuntimeError("RKT target accounting changed during an epoch")
    return binary_metrics(
        torch.from_numpy(probabilities),
        torch.from_numpy(labels),
        loss=loss_sum / cursor,
    )


def train_rkt(
    dataset_name: str,
    store: SessionStore,
    prepared_root: Path,
    output_root: Path,
    *,
    workers: int = 0,
    epoch_ceiling_override: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    seed_everything()
    if (output_root / "_SUCCESS").exists():
        raise ValueError(f"completed experiment already exists: {output_root}")
    if not (prepared_root / "_SUCCESS").exists():
        raise ValueError("full RKT Phi/initialization artifacts are incomplete")
    ordered = dataset_name == "assist2009"
    algebra = dataset_name == "algebra2005"
    if algebra and (not torch.cuda.is_available() or not store.strict_timestamp_history):
        raise ValueError("Algebra2005 full training requires CUDA and the verified strict-time session store")
    if ordered and (not isinstance(store, Assist2009Store) or not torch.cuda.is_available()):
        raise ValueError("ASSIST2009 RKT requires the ordered interaction store and CUDA")
    if (ordered or algebra) and not resume and (output_root / "config.json").exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    epoch_ceiling = RKT_EPOCH_CEILING
    if epoch_ceiling_override is not None:
        if epoch_ceiling_override <= 0 or "PYTEST_CURRENT_TEST" not in os.environ:
            raise ValueError("epoch ceiling override is restricted to automated tests")
        epoch_ceiling = epoch_ceiling_override
    repository = CrossFittedPhiRepository(
        prepared_root, int(store.metadata["num_questions"])
    )
    if int(repository.metadata["cache_students"]) != int(store.metadata["students"]):
        raise ValueError("RKT production training requires a full-student Phi cache")
    if algebra:
        provenance = json.loads((prepared_root / "provenance.json").read_text())
        if provenance.get("session_store_metadata_sha256") != hashlib.sha256(
                (store.root / "metadata.json").read_bytes()).hexdigest():
            raise ValueError("RKT cache does not match the Algebra2005 session store")
    if ordered and repository.metadata.get("dataset_provenance") != store.provenance:
        raise ValueError("Phi cache does not match the shared ASSIST2009 split")
    datasets = {
        split: (OrderedRKTTargets(store, split, repository) if ordered
                else RKTTargetDataset(store, split))
        for split in ("train", "validation", "test")
    }
    if int(repository.metadata["training_targets_cross_fitted"]) != len(datasets["train"]):
        raise ValueError("cross-fitted Phi target count does not match full training targets")
    initial_s = None if ordered else np.load(prepared_root / "initial_s_hours.npy")
    config = RKTConfig(
        num_questions=int(store.metadata["num_questions"]),
        num_students=int(store.metadata["students"]) if ordered else len(initial_s) - 1,
        relation_only=ordered,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    model = PaperFaithfulRKT(config, initial_s).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=RKT_LEARNING_RATE, weight_decay=RKT_WEIGHT_DECAY
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    model_name = RKT_PHI_ONLY if ordered else "RKT"
    label = RKT_PHI_LABEL if ordered else RKT_LABEL
    configuration = {
        "label": label,
        "model": model_name,
        "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "width": 64,
        "heads": 1,
        "dropout": 0.1,
        "history_length": 49,
        "context_length": 50,
        "batch_size": RKT_BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": RKT_LEARNING_RATE,
        "weight_decay": RKT_WEIGHT_DECAY,
        "gradient_clip": RKT_GRADIENT_CLIP,
        "gradient_clip_provenance": "authors' released trainer default; paper unspecified; explicitly approved",
        "epoch_ceiling": epoch_ceiling,
        "early_stopping_metric": COMMON_EARLY_STOPPING.metric,
        "early_stopping_patience": COMMON_EARLY_STOPPING.patience,
        "early_stopping_min_delta": COMMON_EARLY_STOPPING.min_delta,
        "early_stopping_strict_improvement": COMMON_EARLY_STOPPING.strict_improvement,
        "reload_best_checkpoint_before_test": True,
        "phi": repository.metadata,
        "total_parameters": total_parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "best_epoch": None,
        "best_validation_auc": None,
        "epochs_completed": 0,
        "device": str(device),
        "gpu": gpu_name,
        "workers": workers,
    }
    if ordered:
        configuration.update(dataset_provenance=store.provenance, relation_mode="phi_only")
    if algebra:
        configuration.update(session_store_metadata_sha256=hashlib.sha256(
            (store.root / "metadata.json").read_bytes()).hexdigest(),
            torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda,
            gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    if epoch_ceiling_override is not None:
        configuration["test_epoch_ceiling_override"] = epoch_ceiling
    config_path = output_root / "config.json"
    stats_path = output_root / "dataset_stats.json"
    metrics_path = output_root / "metrics.csv"
    log_path = output_root / "training.jsonl"
    best_path = output_root / "best_model.pt"
    last_path = output_root / "last_model.pt"
    start_epoch = 1
    epochs_completed = 0
    prior_runtime_seconds = 0.0
    runtime_estimated_before_resume = False
    resume_state = None
    if resume:
        required = (config_path, stats_path, metrics_path, log_path, best_path, last_path)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ValueError(f"cannot resume; missing artifacts: {missing}")
        stored_configuration = json.loads(config_path.read_text())
        for key in ("model", "dataset", "seed", "batch_size", "history_length"):
            if stored_configuration.get(key) != configuration.get(key):
                raise ValueError(f"resume configuration mismatch for {key}")
        if algebra and any(stored_configuration.get(key) != configuration[key]
                           for key in ("session_store_metadata_sha256", "phi")):
            raise ValueError("Algebra2005 resume session store or Phi cache differs")
        if ordered and any(stored_configuration.get(key) != configuration[key]
                           for key in ("dataset_provenance", "phi", "relation_mode")):
            raise ValueError("RKT resume protocol or Phi cache differs")
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        if checkpoint.get("seed") != PROJECT_SEED:
            raise ValueError("resume checkpoint seed mismatch")
        if "rng_state" not in checkpoint or "early_stopping_state" not in checkpoint:
            raise ValueError("cannot exactly resume RKT without RNG and patience state")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        epochs_completed = int(checkpoint["epoch"])
        if epochs_completed != int(stored_configuration["epochs_completed"]):
            raise ValueError("resume epoch mismatch between checkpoint and config")
        epoch_events = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line and json.loads(line).get("event") == "epoch_complete"
        ]
        if not epoch_events or int(epoch_events[-1]["epoch"]) != epochs_completed:
            raise ValueError("resume log does not end at checkpoint epoch")
        resume_state = checkpoint["early_stopping_state"]
        start_epoch = epochs_completed + 1
        prior_runtime = stored_configuration.get("runtime_seconds_accumulated")
        if prior_runtime is None:
            prior_runtime_seconds = max(
                0.0, config_path.stat().st_mtime - stats_path.stat().st_mtime
            )
            runtime_estimated_before_resume = True
        else:
            prior_runtime_seconds = float(prior_runtime)
        configuration = stored_configuration
        configuration.update(
            resume_count=int(configuration.get("resume_count", 0)) + 1,
            resumed_from_epoch=epochs_completed,
            runtime_before_resume_seconds=prior_runtime_seconds,
            runtime_before_resume_estimated_from_artifact_mtimes=runtime_estimated_before_resume,
        )
        if epoch_ceiling_override is not None:
            configuration["test_epoch_ceiling_override"] = epoch_ceiling
        _write_json(config_path, configuration)
    else:
        _write_json(config_path, configuration)
        targets = {split: len(dataset) for split, dataset in datasets.items()}
        _write_json(
            stats_path,
            {
                "dataset": dataset_name,
                "students": int(store.metadata["students"]),
                "interactions": int(store.metadata["interactions"]),
                "questions": int(store.metadata["num_questions"]),
                "skills": int(store.metadata["num_skills"]),
                "sessions": None if ordered else int(store.metadata["sessions"]),
                "target_counts": targets,
                "target_ratios": {
                    split: count / sum(targets.values())
                    for split, count in targets.items()
                },
            },
        )

    fields = [
        "epoch", "split", "targets", "loss", "roc_auc", "accuracy",
        "precision", "recall", "f1", "mse", "threshold",
    ]
    control = ValidationAUCEarlyStopping()
    if resume_state is not None:
        control.best_epoch = int(resume_state["best_epoch"])
        control.best_validation_auc = float(resume_state["best_validation_auc"])
        control.consecutive_epochs_without_improvement = int(
            resume_state["consecutive_epochs_without_improvement"]
        )
    started = time.perf_counter()
    file_mode = "a" if resume else "w"
    with metrics_path.open(file_mode, newline="") as metrics_file, log_path.open(
        file_mode
    ) as log:
        writer = csv.DictWriter(metrics_file, fieldnames=fields)
        if not resume:
            writer.writeheader()
        else:
            log.write(
                json.dumps(
                    {
                        "event": "training_resumed",
                        "seed": PROJECT_SEED,
                        "resumed_from_epoch": epochs_completed,
                        "best_epoch": control.best_epoch,
                        "best_validation_auc": control.best_validation_auc,
                        "consecutive_epochs_without_improvement": control.consecutive_epochs_without_improvement,
                    }
                ) + "\n"
            )
            log.flush()
        for epoch in range(start_epoch, epoch_ceiling + 1):
            if control.should_stop:
                break
            if algebra:
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics = _run_epoch(
                model, datasets["train"], repository, device=device, epoch=epoch,
                optimizer=optimizer, workers=workers,
            )
            validation_metrics = _run_epoch(
                model, datasets["validation"], repository, device=device, epoch=epoch,
                optimizer=None, workers=workers,
            )
            if algebra:
                print(json.dumps({
                    "event": "epoch_metrics", "epoch": epoch, "train": train_metrics,
                    "validation": validation_metrics,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }), flush=True)
            for split, values in (("train", train_metrics), ("validation", validation_metrics)):
                writer.writerow({"epoch": epoch, "split": split, **values})
            metrics_file.flush()
            decision = control.step(epoch, float(validation_metrics["roc_auc"]))
            payload = model.checkpoint()
            payload.update(
                {
                    "epoch": epoch,
                    "validation_roc_auc": validation_metrics["roc_auc"],
                    "optimizer_state_dict": optimizer.state_dict(),
                    "seed": PROJECT_SEED,
                    "early_stopping": COMMON_EARLY_STOPPING.as_dict(),
                    "early_stopping_state": {
                        "best_epoch": control.best_epoch,
                        "best_validation_auc": control.best_validation_auc,
                        "consecutive_epochs_without_improvement": control.consecutive_epochs_without_improvement,
                    },
                    "rng_state": capture_rng_state(),
                }
            )
            torch.save(payload, last_path)
            if decision.improved:
                torch.save(payload, best_path)
            epochs_completed = epoch
            configuration.update(
                best_epoch=decision.best_epoch,
                best_validation_auc=decision.best_validation_auc,
                epochs_completed=epochs_completed,
                runtime_seconds_accumulated=prior_runtime_seconds
                + time.perf_counter()
                - started,
            )
            _write_json(output_root / "config.json", configuration)
            log.write(
                json.dumps(
                    {
                        "event": "epoch_complete", "seed": PROJECT_SEED,
                        "epoch": epoch, "train": train_metrics,
                        "validation": validation_metrics,
                        "checkpoint_saved": decision.improved,
                        "consecutive_epochs_without_improvement": decision.consecutive_epochs_without_improvement,
                        "early_stop": decision.should_stop,
                    }
                ) + "\n"
            )
            log.flush()
            if ordered:
                print(json.dumps({"epoch": epoch, "train_loss": train_metrics["loss"],
                                  "validation_loss": validation_metrics["loss"],
                                  "validation_auc": validation_metrics["roc_auc"],
                                  "best_epoch": decision.best_epoch}), flush=True)
            if decision.should_stop:
                break
        if not best_path.exists() or control.best_epoch is None:
            raise RuntimeError("no finite validation AUC checkpoint was produced")
        best_model = PaperFaithfulRKT.from_checkpoint(best_path, map_location=device).to(device)
        best_model.eval()
        best_model.set_relation_parameters_trainable(False)
        test_metrics = _run_epoch(
            best_model, datasets["test"], repository, device=device,
            epoch=control.best_epoch, optimizer=None, workers=workers,
        )
        writer.writerow({"epoch": control.best_epoch, "split": "test", **test_metrics})
        log.write(
            json.dumps(
                {
                    "event": "final_test_from_best_checkpoint", "seed": PROJECT_SEED,
                    "best_epoch": control.best_epoch,
                    "best_validation_auc": control.best_validation_auc,
                    "epochs_completed": epochs_completed, "test": test_metrics,
                }
            ) + "\n"
        )
    result = {
        "status": "complete", "label": label, "model": model_name,
        "dataset": dataset_name, "seed": PROJECT_SEED,
        "patience": COMMON_EARLY_STOPPING.patience,
        "min_delta": COMMON_EARLY_STOPPING.min_delta,
        "best_epoch": control.best_epoch,
        "best_validation_auc": control.best_validation_auc,
        "epochs_completed": epochs_completed, "epoch_ceiling": epoch_ceiling,
        "test": test_metrics,
        "runtime_seconds": prior_runtime_seconds + time.perf_counter() - started,
        "resume_count": int(configuration.get("resume_count", 0)),
        "runtime_before_resume_estimated_from_artifact_mtimes": runtime_estimated_before_resume,
        "total_parameters": total_parameters,
        "trainable_parameters": configuration["trainable_parameters"],
        "best_checkpoint_reloaded": True,
        "gpu": gpu_name,
    }
    _write_json(output_root / "final_results.json", result)
    configuration["runtime_seconds_accumulated"] = result["runtime_seconds"]
    _write_json(config_path, configuration)
    (output_root / "_SUCCESS").write_text("complete\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("assist2017", "junyi", "ednet_kt1", "assist2009", "algebra2005"))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    store = (Assist2009Store(args.store_root) if args.dataset == "assist2009"
             else SessionStore(args.store_root))
    result = train_rkt(
        args.dataset, store, args.prepared_root,
        args.output_root, workers=args.workers, resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
