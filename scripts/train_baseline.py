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
from ktbench.training import COMMON_EARLY_STOPPING, ValidationAUCEarlyStopping


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
) -> dict[str, Any]:
    payload = model.checkpoint()
    payload.update(
        {
            "epoch": epoch,
            "validation_roc_auc": validation_auc,
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": PROJECT_SEED,
            "early_stopping": COMMON_EARLY_STOPPING.as_dict(),
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
    configuration: dict[str, Any] = {
        **profile,
        "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "device": str(device),
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
        configuration["test_epoch_ceiling_override"] = epoch_ceiling
    _write_json(output_root / "config.json", configuration)
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
    _write_json(output_root / "dataset_stats.json", dataset_stats)

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
    started = time.perf_counter()
    epochs_completed = 0
    with metrics_path.open("w", newline="") as metrics_file, log_path.open("w") as log:
        writer = csv.DictWriter(metrics_file, fieldnames=metrics_fields)
        writer.writeheader()
        for epoch in range(1, epoch_ceiling + 1):
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

    runtime_seconds = time.perf_counter() - started
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
        "total_parameters": total_parameters,
        "trainable_parameters": configuration["trainable_parameters"],
        "best_checkpoint_reloaded": True,
    }
    _write_json(output_root / "final_results.json", result)
    configuration.update(
        {
            "best_epoch": control.best_epoch,
            "best_validation_auc": control.best_validation_auc,
            "epochs_completed": epochs_completed,
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
    args = parser.parse_args()
    result = train_baseline(
        args.model,
        args.dataset,
        SessionStore(args.store_root),
        args.output_root,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
