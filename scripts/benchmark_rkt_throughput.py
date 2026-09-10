#!/usr/bin/env python3
"""Benchmark full-cache RKT loading and bounded train/evaluation throughput."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.session_store import SessionStore
from ktbench.models import PaperFaithfulRKT, RKTConfig
from ktbench.rkt.data import RKTTargetDataset, collate_rkt
from ktbench.rkt.phi import CrossFittedPhiRepository
from ktbench.rkt.settings import (
    RKT_BATCH_SIZE,
    RKT_GRADIENT_CLIP,
    RKT_LEARNING_RATE,
    RKT_WEIGHT_DECAY,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _step(
    model: PaperFaithfulRKT,
    batch,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    batch = batch.to(device)
    if optimizer is None:
        with torch.no_grad():
            logits = model(batch)
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch.target_labels.float()
            )
        return
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.target_labels.float()
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), RKT_GRADIENT_CLIP)
    optimizer.step()


def benchmark(
    dataset_name: str,
    store_root: Path,
    prepared_root: Path,
    output_path: Path,
    *,
    measured_batches: int,
) -> dict[str, object]:
    if measured_batches <= 0:
        raise ValueError("measured_batches must be positive")
    seed_everything()
    store = SessionStore(store_root)
    loaded = time.perf_counter()
    repository = CrossFittedPhiRepository(
        prepared_root, int(store.metadata["num_questions"])
    )
    repository_load_seconds = time.perf_counter() - loaded
    if int(repository.metadata["cache_students"]) != int(store.metadata["students"]):
        raise ValueError("throughput benchmark requires a full-student Phi cache")
    train = RKTTargetDataset(store, "train")
    validation = RKTTargetDataset(store, "validation")
    required = (measured_batches + 1) * RKT_BATCH_SIZE
    if required > len(train):
        raise ValueError("training split is too small for requested benchmark")
    generator = np.random.default_rng(PROJECT_SEED)
    indices = generator.choice(len(train), size=required, replace=False)
    batches = [
        indices[start : start + RKT_BATCH_SIZE]
        for start in range(0, len(indices), RKT_BATCH_SIZE)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_s = np.load(prepared_root / "initial_s_hours.npy")
    model = PaperFaithfulRKT(
        RKTConfig(
            num_questions=int(store.metadata["num_questions"]),
            num_students=len(initial_s) - 1,
        ),
        initial_s,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=RKT_LEARNING_RATE, weight_decay=RKT_WEIGHT_DECAY
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def make_batch(raw_indices: np.ndarray):
        return collate_rkt(
            [train[int(index)] for index in raw_indices],
            phi_repository=repository,
        )

    model.train()
    model.set_relation_parameters_trainable(True)
    warmup = make_batch(batches[0])
    _step(model, warmup, device=device, optimizer=optimizer)
    _synchronize(device)
    train_seconds = []
    history_lengths = []
    for raw_indices in batches[1:]:
        started = time.perf_counter()
        batch = make_batch(raw_indices)
        history_lengths.extend(batch.history_mask.sum(dim=1).tolist())
        _step(model, batch, device=device, optimizer=optimizer)
        _synchronize(device)
        train_seconds.append(time.perf_counter() - started)

    model.eval()
    model.set_relation_parameters_trainable(False)
    evaluation_seconds = []
    for raw_indices in batches[1:]:
        started = time.perf_counter()
        batch = make_batch(raw_indices)
        _step(model, batch, device=device, optimizer=None)
        _synchronize(device)
        evaluation_seconds.append(time.perf_counter() - started)

    train_per_target = float(np.sum(train_seconds)) / (measured_batches * RKT_BATCH_SIZE)
    evaluation_per_target = float(np.sum(evaluation_seconds)) / (
        measured_batches * RKT_BATCH_SIZE
    )
    estimated_train_seconds = train_per_target * len(train)
    estimated_validation_seconds = evaluation_per_target * len(validation)
    result: dict[str, object] = {
        "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "full_cache": True,
        "repository_load_seconds": repository_load_seconds,
        "peak_process_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "device": str(device),
        "batch_size": RKT_BATCH_SIZE,
        "measured_batches_per_mode": measured_batches,
        "measured_targets_per_mode": measured_batches * RKT_BATCH_SIZE,
        "sample_history_length": {
            "minimum": int(np.min(history_lengths)),
            "maximum": int(np.max(history_lengths)),
            "mean": float(np.mean(history_lengths)),
        },
        "train_seconds": float(np.sum(train_seconds)),
        "evaluation_seconds": float(np.sum(evaluation_seconds)),
        "train_targets_per_second": 1.0 / train_per_target,
        "evaluation_targets_per_second": 1.0 / evaluation_per_target,
        "estimated_compute_and_collation_seconds": {
            "train_split": estimated_train_seconds,
            "validation_split": estimated_validation_seconds,
            "one_train_plus_validation_epoch": (
                estimated_train_seconds + estimated_validation_seconds
            ),
        },
        "split_targets": {
            "train": len(train),
            "validation": len(validation),
            "test": len(RKTTargetDataset(store, "test")),
        },
        "gradient_clip": RKT_GRADIENT_CLIP,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "estimate_scope": "bounded random targets; includes lookup, collation, transfer, and model step; excludes checkpoint and full-epoch loader setup",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("assist2017", "junyi", "ednet_kt1"))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                args.dataset,
                args.store_root,
                args.prepared_root,
                args.output_path,
                measured_batches=args.batches,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
