#!/usr/bin/env python3
"""Train one full rolling DKT, DKVMN, or SAKT experiment."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ktbench.baseline_registry import BASELINE_TRAINING_CONFIGS, make_baseline_model
from ktbench.config import PROJECT_SEED, seed_everything
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
    ValidationAUCEarlyStopping,
    capture_rng_state,
    restore_rng_state,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _loader(
    dataset: RollingTargetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    epoch: int,
    workers: int,
) -> tuple[DataLoader, RollingLengthBatchSampler]:
    sampler = RollingLengthBatchSampler(
        dataset,
        batch_size=batch_size,
        bucket_size=4096,
        shuffle=shuffle,
        seed=PROJECT_SEED,
    )
    sampler.set_epoch(epoch if shuffle else 0)
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
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    loader, _ = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        epoch=epoch,
        workers=workers,
    )
    probabilities = np.empty(len(dataset), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.uint8)
    cursor = 0
    loss_sum = 0.0
    for host_batch in loader:
        batch = host_batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch.target_labels.float(), reduction="mean"
            )
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch.target_labels.float(), reduction="mean"
                )
        count = len(batch.target_labels)
        loss_sum += float(loss.detach().cpu()) * count
        probabilities[cursor : cursor + count] = torch.sigmoid(logits).detach().cpu().numpy()
        labels[cursor : cursor + count] = batch.target_labels.detach().cpu().numpy()
        cursor += count
    if cursor != len(dataset):
        raise RuntimeError(f"evaluated {cursor} targets but expected {len(dataset)}")
    return binary_metrics(
        torch.from_numpy(probabilities),
        torch.from_numpy(labels),
        loss=loss_sum / cursor,
    )


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
            "early_stopping": COMMON_EARLY_STOPPING.as_dict(),
            "early_stopping_state": {
                "best_epoch": control.best_epoch,
                "best_validation_auc": control.best_validation_auc,
                "consecutive_epochs_without_improvement": control.consecutive_epochs_without_improvement,
            },
            "rng_state": capture_rng_state(),
        }
    )
    return payload


def train_baseline(
    model_name: str,
    dataset_name: str,
    store: SessionStore,
    output_root: Path,
    *,
    workers: int = 0,
    epoch_ceiling_override: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the full protocol; the override exists only for automated tests."""

    seed_everything()
    if (output_root / "_SUCCESS").exists():
        raise ValueError(f"completed experiment already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    profile = dict(BASELINE_TRAINING_CONFIGS[model_name])
    epoch_ceiling = int(profile["epoch_ceiling"])
    if epoch_ceiling_override is not None:
        if epoch_ceiling_override <= 0 or "PYTEST_CURRENT_TEST" not in __import__("os").environ:
            raise ValueError("epoch ceiling override is restricted to automated tests")
        epoch_ceiling = epoch_ceiling_override
    history_length = int(profile["history_length"])
    datasets = {
        split: RollingTargetDataset(store, split, history_length=history_length)
        for split in ("train", "validation", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("an experiment split contains no targets")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
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
        "early_stopping_metric": COMMON_EARLY_STOPPING.metric,
        "early_stopping_patience": COMMON_EARLY_STOPPING.patience,
        "early_stopping_min_delta": COMMON_EARLY_STOPPING.min_delta,
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
    if epoch_ceiling_override is not None:
        fresh_configuration["test_epoch_ceiling_override"] = epoch_ceiling
    target_counts = {split: len(dataset) for split, dataset in datasets.items()}
    dataset_stats = {
        "dataset": dataset_name,
        "students": int(store.metadata["students"]),
        "interactions": int(store.metadata["interactions"]),
        "questions": int(store.metadata["num_questions"]),
        "skills": int(store.metadata["num_skills"]),
        "sessions": int(store.metadata["sessions"]),
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
    control = ValidationAUCEarlyStopping()
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
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
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
            }
        )
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
                        "best_epoch": control.best_epoch,
                        "best_validation_auc": control.best_validation_auc,
                        "consecutive_epochs_without_improvement": control.consecutive_epochs_without_improvement,
                    }
                )
                + "\n"
            )
            log.flush()
        for epoch in range(start_epoch, epoch_ceiling + 1):
            train_metrics = _run_epoch(
                model,
                datasets["train"],
                batch_size=int(profile["batch_size"]),
                device=device,
                workers=workers,
                epoch=epoch,
                optimizer=optimizer,
                gradient_clip=profile["gradient_clip"],
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
            )
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
            log.write(
                json.dumps(
                    {
                        "event": "epoch_complete",
                        "seed": PROJECT_SEED,
                        "epoch": epoch,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "checkpoint_saved": decision.improved,
                        "consecutive_epochs_without_improvement": decision.consecutive_epochs_without_improvement,
                        "early_stop": decision.should_stop,
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
        "status": "complete",
        "dataset": dataset_name,
        "model": profile["model"],
        "seed": PROJECT_SEED,
        "patience": COMMON_EARLY_STOPPING.patience,
        "min_delta": COMMON_EARLY_STOPPING.min_delta,
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
    }
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
    (output_root / "_SUCCESS").write_text("complete\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(BASELINE_TRAINING_CONFIGS))
    parser.add_argument("dataset", choices=("assist2017", "junyi", "ednet_kt1"))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train_baseline(
        args.model,
        args.dataset,
        SessionStore(args.store_root),
        args.output_root,
        workers=args.workers,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
