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
from ktbench.training import COMMON_EARLY_STOPPING, ValidationAUCEarlyStopping


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
        "workers": workers,
    }
    if epoch_ceiling_override is not None:
        configuration["test_epoch_ceiling_override"] = epoch_ceiling
    _write_json(output_root / "config.json", configuration)
    targets = {split: len(dataset) for split, dataset in datasets.items()}
    _write_json(
        output_root / "dataset_stats.json",
        {
            "dataset": dataset_name,
            "students": int(store.metadata["students"]),
            "interactions": int(store.metadata["interactions"]),
            "questions": int(store.metadata["num_questions"]),
            "skills": int(store.metadata["num_skills"]),
            "sessions": int(store.metadata["sessions"]),
            "target_counts": targets,
            "target_ratios": {
                split: count / sum(targets.values()) for split, count in targets.items()
            },
        },
    )

    fields = [
        "epoch", "split", "targets", "loss", "roc_auc", "accuracy",
        "precision", "recall", "f1", "mse", "threshold",
    ]
    control = ValidationAUCEarlyStopping()
    best_path = output_root / "best_model.pt"
    last_path = output_root / "last_model.pt"
    started = time.perf_counter()
    epochs_completed = 0
    with (output_root / "metrics.csv").open("w", newline="") as metrics_file, (
        output_root / "training.jsonl"
    ).open("w") as log:
        writer = csv.DictWriter(metrics_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epoch_ceiling + 1):
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
        "test": test_metrics, "runtime_seconds": time.perf_counter() - started,
        "total_parameters": total_parameters,
        "trainable_parameters": configuration["trainable_parameters"],
        "best_checkpoint_reloaded": True,
    }
    _write_json(output_root / "final_results.json", result)
    (output_root / "_SUCCESS").write_text("complete\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("assist2017", "junyi", "ednet_kt1"))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    result = train_rkt(
        args.dataset, SessionStore(args.store_root), args.prepared_root,
        args.output_root, workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
