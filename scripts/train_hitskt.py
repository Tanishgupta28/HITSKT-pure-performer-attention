#!/usr/bin/env python3
"""Train one full pure-Performer hierarchical HiTSKT experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

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
from ktbench.training import (
    COMMON_EARLY_STOPPING,
    ValidationAUCEarlyStopping,
    capture_rng_state,
    restore_rng_state,
)


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
    resume: bool = False,
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
    fresh_configuration: dict[str, Any] = {
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
        fresh_configuration["test_epoch_ceiling_override"] = epoch_ceiling
    dataset_stats = {
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
    }
    fields = [
        "epoch", "split", "targets", "loss", "roc_auc", "accuracy",
        "precision", "recall", "f1", "mse", "threshold",
    ]
    control = ValidationAUCEarlyStopping()
    best_path = output_root / "best_model.pt"
    last_path = output_root / "last_model.pt"
    config_path = output_root / "config.json"
    stats_path = output_root / "dataset_stats.json"
    metrics_path = output_root / "metrics.csv"
    log_path = output_root / "training.jsonl"
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
        for key in (
            "model", "dataset", "seed", "token_budget", "maximum_batch_size",
            "history_sessions", "width", "heads", "dropout",
        ):
            if configuration.get(key) != fresh_configuration.get(key):
                raise ValueError(f"resume configuration mismatch for {key}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        if checkpoint.get("seed") != PROJECT_SEED:
            raise ValueError("resume checkpoint seed mismatch")
        if "rng_state" not in checkpoint or "early_stopping_state" not in checkpoint:
            raise ValueError("cannot exactly resume HiTSKT without RNG and patience state")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
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
        state = checkpoint["early_stopping_state"]
        control.best_epoch = int(state["best_epoch"])
        control.best_validation_auc = float(state["best_validation_auc"])
        control.consecutive_epochs_without_improvement = int(
            state["consecutive_epochs_without_improvement"]
        )
        start_epoch = epochs_completed + 1
        prior_runtime = configuration.get("runtime_seconds_accumulated")
        if prior_runtime is None:
            prior_runtime_seconds = max(
                0.0, config_path.stat().st_mtime - stats_path.stat().st_mtime
            )
            runtime_estimated_before_resume = True
        else:
            prior_runtime_seconds = float(prior_runtime)
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
        configuration = fresh_configuration
        _write_json(config_path, configuration)
        _write_json(stats_path, dataset_stats)

    started = time.perf_counter()
    file_mode = "a" if resume else "w"
    with metrics_path.open(file_mode, newline="") as metrics_file, log_path.open(file_mode) as log:
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
            _write_json(config_path, configuration)
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
    parser.add_argument("dataset", choices=sorted(HITSKT_TRAINING_CONFIGS))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train_hitskt(
        args.dataset, SessionStore(args.store_root), args.output_root,
        workers=args.workers, resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
