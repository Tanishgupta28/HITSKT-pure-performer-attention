#!/usr/bin/env python3
"""Train one full pure-Performer hierarchical HiTSKT experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.session_store import (
    RollingSessionDataset,
    SessionStore,
    make_session_dataloader,
)
from ktbench.hitskt_registry import HITSKT_COMMON_CONFIG, HITSKT_TRAINING_CONFIGS
from ktbench.metrics import binary_metrics, masked_bce_loss
from ktbench.models import HiTSKT, HiTSKTConfig
from ktbench.training import COMMON_EARLY_STOPPING, ValidationAUCEarlyStopping


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _target_count(dataset: RollingSessionDataset) -> int:
    offsets = dataset.store.session_offsets
    return int(np.sum(offsets[dataset.targets + 1] - offsets[dataset.targets]))


def _run_epoch(
    model: HiTSKT,
    dataset: RollingSessionDataset,
    *,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    workers: int,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    loader, sampler = make_session_dataloader(
        dataset,
        token_budget=int(HITSKT_COMMON_CONFIG["token_budget"]),
        maximum_batch_size=int(HITSKT_COMMON_CONFIG["maximum_batch_size"]),
        bucket_size=int(HITSKT_COMMON_CONFIG["bucket_size"]),
        shuffle=training,
        seed=PROJECT_SEED,
        workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    sampler.set_epoch(epoch if training else 0)
    expected_targets = _target_count(dataset)
    probabilities = np.empty(expected_targets, dtype=np.float32)
    labels = np.empty(expected_targets, dtype=np.uint8)
    cursor = 0
    loss_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = masked_bce_loss(logits, batch.target_labels, batch.target_metric_mask)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batch)
                loss = masked_bce_loss(logits, batch.target_labels, batch.target_metric_mask)
        valid_logits = logits.masked_select(batch.target_metric_mask)
        valid_labels = batch.target_labels.masked_select(batch.target_metric_mask)
        count = len(valid_labels)
        loss_sum += float(loss.detach().cpu()) * count
        probabilities[cursor : cursor + count] = torch.sigmoid(valid_logits).detach().cpu().numpy()
        labels[cursor : cursor + count] = valid_labels.detach().cpu().numpy()
        cursor += count
    if cursor != expected_targets:
        raise RuntimeError("HiTSKT target accounting changed during an epoch")
    return binary_metrics(
        torch.from_numpy(probabilities),
        torch.from_numpy(labels),
        loss=loss_sum / cursor,
    )


def train_hitskt(
    dataset_name: str,
    store: SessionStore,
    output_root: Path,
    *,
    workers: int = 0,
    epoch_ceiling_override: int | None = None,
) -> dict[str, object]:
    seed_everything()
    if dataset_name not in HITSKT_TRAINING_CONFIGS:
        raise ValueError(f"unknown HiTSKT dataset: {dataset_name}")
    if (output_root / "_SUCCESS").exists():
        raise ValueError(f"completed experiment already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    profile = {**HITSKT_COMMON_CONFIG, **HITSKT_TRAINING_CONFIGS[dataset_name]}
    epoch_ceiling = int(profile["epoch_ceiling"])
    if epoch_ceiling_override is not None:
        if epoch_ceiling_override <= 0 or "PYTEST_CURRENT_TEST" not in os.environ:
            raise ValueError("epoch ceiling override is restricted to automated tests")
        epoch_ceiling = epoch_ceiling_override
    datasets = {
        split: RollingSessionDataset(
            store, split, history_sessions=int(profile["history_sessions"])
        )
        for split in ("train", "validation", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("a HiTSKT experiment split contains no target sessions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    model_config = HiTSKTConfig(
        num_questions=int(store.metadata["num_questions"]),
        num_skills=int(store.metadata["num_skills"]),
        width=int(profile["width"]),
        heads=int(profile["heads"]),
        feedforward_width=int(profile["feedforward_width"]),
        action_layers=int(profile["action_layers"]),
        session_layers=int(profile["session_layers"]),
        correct_layers=int(profile["correct_layers"]),
        decoder_layers=int(profile["decoder_layers"]),
        dropout=float(profile["dropout"]),
    )
    model = HiTSKT(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    target_counts = {split: _target_count(dataset) for split, dataset in datasets.items()}
    configuration = {
        **profile,
        "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "device": str(device),
        "gpu": gpu_name,
        "workers": workers,
        "action_truncation": False,
        "action_chunking": False,
        "attention": "pure causal ELU+1 Performer linear attention",
        "epoch_ceiling": epoch_ceiling,
        "early_stopping_metric": COMMON_EARLY_STOPPING.metric,
        "early_stopping_patience": COMMON_EARLY_STOPPING.patience,
        "early_stopping_min_delta": COMMON_EARLY_STOPPING.min_delta,
        "early_stopping_strict_improvement": COMMON_EARLY_STOPPING.strict_improvement,
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
    _write_json(
        output_root / "dataset_stats.json",
        {
            "dataset": dataset_name,
            "students": int(store.metadata["students"]),
            "interactions": int(store.metadata["interactions"]),
            "questions": int(store.metadata["num_questions"]),
            "skills": int(store.metadata["num_skills"]),
            "sessions": int(store.metadata["sessions"]),
            "target_sessions": {split: len(dataset) for split, dataset in datasets.items()},
            "target_counts": target_counts,
            "target_ratios": {
                split: count / sum(target_counts.values())
                for split, count in target_counts.items()
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
                model, datasets["train"], device=device, epoch=epoch,
                optimizer=optimizer, workers=workers,
            )
            validation_metrics = _run_epoch(
                model, datasets["validation"], device=device, epoch=epoch,
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
        best_model = HiTSKT.from_checkpoint(best_path, map_location=device).to(device).eval()
        test_metrics = _run_epoch(
            best_model, datasets["test"], device=device, epoch=control.best_epoch,
            optimizer=None, workers=workers,
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
        "status": "complete", "model": "HiTSKT", "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "patience": COMMON_EARLY_STOPPING.patience,
        "min_delta": COMMON_EARLY_STOPPING.min_delta,
        "best_epoch": control.best_epoch,
        "best_validation_auc": control.best_validation_auc,
        "epochs_completed": epochs_completed, "epoch_ceiling": epoch_ceiling,
        "test": test_metrics, "runtime_seconds": time.perf_counter() - started,
        "total_parameters": total_parameters,
        "trainable_parameters": configuration["trainable_parameters"],
        "best_checkpoint_reloaded": True,
        "gpu": gpu_name,
    }
    _write_json(output_root / "final_results.json", result)
    (output_root / "_SUCCESS").write_text("complete\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(HITSKT_TRAINING_CONFIGS))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    result = train_hitskt(
        args.dataset, SessionStore(args.store_root), args.output_root,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
