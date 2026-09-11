#!/usr/bin/env python3
"""Train the approved full RKT performance-only cross-fitted experiment."""

from __future__ import annotations

import argparse
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
from ktbench.data.rolling_targets import RollingLengthBatchSampler
from ktbench.data.session_store import SessionStore
from ktbench.metrics import binary_metrics
from ktbench.models import PaperFaithfulRKT, RKTConfig
from ktbench.models.rkt import RKT_LABEL
from ktbench.rkt.data import RKTTargetDataset, collate_rkt
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
    for batch in loader:
        batch = batch.to(device)
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
        cursor += count
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
    datasets = {
        split: RKTTargetDataset(store, split)
        for split in ("train", "validation", "test")
    }
    if int(repository.metadata["training_targets_cross_fitted"]) != len(datasets["train"]):
        raise ValueError("cross-fitted Phi target count does not match full training targets")
    initial_s = np.load(prepared_root / "initial_s_hours.npy")
    config = RKTConfig(
        num_questions=int(store.metadata["num_questions"]),
        num_students=len(initial_s) - 1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    model = PaperFaithfulRKT(config, initial_s).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=RKT_LEARNING_RATE, weight_decay=RKT_WEIGHT_DECAY
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    configuration = {
        "label": RKT_LABEL,
        "model": "RKT",
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
                "sessions": int(store.metadata["sessions"]),
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
            train_metrics = _run_epoch(
                model, datasets["train"], repository, device=device, epoch=epoch,
                optimizer=optimizer, workers=workers,
            )
            validation_metrics = _run_epoch(
                model, datasets["validation"], repository, device=device, epoch=epoch,
                optimizer=None, workers=workers,
            )
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
        "status": "complete", "label": RKT_LABEL, "model": "RKT",
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
    parser.add_argument("dataset", choices=("assist2017", "junyi", "ednet_kt1"))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train_rkt(
        args.dataset, SessionStore(args.store_root), args.prepared_root,
        args.output_root, workers=args.workers, resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
