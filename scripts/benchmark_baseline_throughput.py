#!/usr/bin/env python3
"""Measure rolling-baseline training throughput without changing semantics."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.rolling_targets import RollingTargetDataset, collate_rolling_targets
from ktbench.data.session_store import SessionStore
from scripts.smoke_baselines import TRAINING_CONFIGS, _make_model


def _full_history_start(dataset: RollingTargetDataset) -> int:
    for start in range(0, len(dataset), 100_000):
        stop = min(start + 100_000, len(dataset))
        lengths = dataset.history_lengths(np.arange(start, stop, dtype=np.int64))
        matches = np.flatnonzero(lengths == dataset.history_length)
        if len(matches):
            return start + int(matches[0])
    raise RuntimeError("dataset never reaches configured rolling history")


def _benchmark(
    dataset_name: str,
    store: SessionStore,
    model_name: str,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, object]:
    seed_everything()
    profile = TRAINING_CONFIGS[model_name]
    dataset = RollingTargetDataset(
        store, "train", history_length=int(profile["history_length"])
    )
    start = _full_history_start(dataset)
    indices = [(start + offset) % len(dataset) for offset in range(int(profile["batch_size"]))]
    batch = collate_rolling_targets([dataset[index] for index in indices]).to("cuda")
    model = _make_model(model_name, store).cuda().train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch.target_labels.float()
        )
        loss.backward()
        if profile["gradient_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(profile["gradient_clip"])
            )
        optimizer.step()

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(measured_steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    seconds_per_batch = elapsed / measured_steps
    batches = math.ceil(len(dataset) / int(profile["batch_size"]))
    return {
        "dataset": dataset_name,
        "model": profile["model"],
        "seed": PROJECT_SEED,
        "train_targets": len(dataset),
        "batch_size": profile["batch_size"],
        "history_length": profile["history_length"],
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "seconds_per_batch": seconds_per_batch,
        "batches_per_epoch": batches,
        "compute_only_hours_per_epoch": seconds_per_batch * batches / 3600.0,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "estimate_excludes": ["data loading", "validation", "checkpoint I/O"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assist-store", type=Path, required=True)
    parser.add_argument("--junyi-store", type=Path, required=True)
    parser.add_argument("--ednet-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the throughput benchmark")
    stores = {
        "assist2017": SessionStore(args.assist_store),
        "junyi": SessionStore(args.junyi_store),
        "ednet_kt1": SessionStore(args.ednet_store),
    }
    measurements = []
    for dataset_name, store in stores.items():
        for model_name in ("dkt", "dkvmn", "sakt"):
            measurements.append(
                _benchmark(
                    dataset_name,
                    store,
                    model_name,
                    args.warmup_steps,
                    args.measured_steps,
                )
            )
    report = {
        "seed": PROJECT_SEED,
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "method": "repeated full-history forward/loss/backward/optimizer steps using repository batch sizes",
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
