#!/usr/bin/env python3
"""Plan real length-bucketed batches and measure HiTSKT CUDA memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ktbench.batching import LengthBucketTokenBatchSampler, collate_hitskt, padded_token_cost
from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.session_store import DatasetShapeSequence, RollingSessionDataset, SessionStore
from ktbench.metrics import masked_bce_loss
from ktbench.models import HiTSKT, HiTSKTConfig


MODEL_CONFIGS = {
    "assist2017": {"width": 256, "heads": 4, "feedforward_width": 2048},
    "junyi": {"width": 128, "heads": 2, "feedforward_width": 1024},
    "ednet_kt1": {"width": 128, "heads": 2, "feedforward_width": 1024},
}
LEGACY_ACTION_CAPACITY = {"assist2017": 63, "junyi": 31, "ednet_kt1": 31}


def _plan_split(
    dataset: RollingSessionDataset, token_budget: int, maximum_batch_size: int
) -> tuple[dict[str, int], list[int]]:
    shapes = DatasetShapeSequence(dataset)
    sampler = LengthBucketTokenBatchSampler(
        shapes,
        token_budget=token_budget,
        max_batch_size=maximum_batch_size,
        bucket_size=512,
        shuffle=False,
    )
    batches = 0
    maximum_tokens = -1
    maximum_size = 0
    singleton_over_budget = 0
    maximum_batch: list[int] = []
    for batch in sampler:
        cost = padded_token_cost([shapes[index] for index in batch])
        batches += 1
        maximum_size = max(maximum_size, len(batch))
        if len(batch) == 1 and cost > token_budget:
            singleton_over_budget += 1
        if cost > maximum_tokens:
            maximum_tokens = cost
            maximum_batch = batch
    return (
        {
            "examples": len(dataset),
            "batches": batches,
            "maximum_batch_size_used": maximum_size,
            "maximum_padded_tokens_used": maximum_tokens,
            "singleton_batches_over_budget": singleton_over_budget,
        },
        maximum_batch,
    )


def _tensor_bytes(batch: object) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in vars(batch).values()
        if isinstance(value, torch.Tensor)
    )


def _cuda_benchmark(
    dataset_name: str,
    dataset: RollingSessionDataset,
    indices: list[int],
    store: SessionStore,
) -> dict[str, float | int | str]:
    if not torch.cuda.is_available():
        return {"status": "CUDA unavailable"}
    examples = [dataset[index] for index in indices]
    batch = collate_hitskt(
        examples,
        num_questions=int(store.metadata["num_questions"]),
        num_skills=int(store.metadata["num_skills"]),
    )
    config = HiTSKTConfig(
        int(store.metadata["num_questions"]),
        int(store.metadata["num_skills"]),
        dropout=0.1,
        **MODEL_CONFIGS[dataset_name],
    )
    model = HiTSKT(config).cuda().train()
    cuda_batch = batch.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits = model(cuda_batch)
    loss = masked_bce_loss(logits, cuda_batch.target_labels, cuda_batch.target_metric_mask)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "status": "complete",
        "batch_size": len(indices),
        "padded_tokens": batch.padded_tokens,
        "supervised_targets": int(batch.target_metric_mask.sum().item()),
        "host_batch_tensor_bytes": _tensor_bytes(batch),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "forward_backward_seconds": elapsed,
        "loss": float(loss.detach().cpu()),
    }


def benchmark_dataset(
    dataset_name: str,
    store_root: Path,
    token_budget: int,
    maximum_batch_size: int,
) -> dict[str, object]:
    store = SessionStore(store_root)
    split_reports: dict[str, dict[str, int]] = {}
    largest: tuple[int, RollingSessionDataset | None, list[int]] = (-1, None, [])
    for split in ("train", "validation", "test"):
        dataset = RollingSessionDataset(store, split, history_sessions=15)
        report, batch = _plan_split(dataset, token_budget, maximum_batch_size)
        split_reports[split] = report
        if report["maximum_padded_tokens_used"] > largest[0]:
            largest = (report["maximum_padded_tokens_used"], dataset, batch)
    largest_dataset = largest[1]
    assert largest_dataset is not None
    benchmark = _cuda_benchmark(dataset_name, largest_dataset, largest[2], store)

    batch_size = int(benchmark.get("batch_size", 0))
    dataset_maximum = int(store.metadata["maximum_session_length"]) + 1
    global_complete_tokens = (
        batch_size * 15 * dataset_maximum
        + batch_size * dataset_maximum
        + batch_size * 16
    )
    legacy_capacity = LEGACY_ACTION_CAPACITY[dataset_name] + 1
    legacy_fixed_tokens = batch_size * (16 * legacy_capacity + 16)
    return {
        "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "store": str(store_root),
        "history_sessions": 15,
        "configured_token_budget": token_budget,
        "configured_maximum_batch_size": maximum_batch_size,
        "action_truncation": False,
        "action_chunking": False,
        "split_plans": split_reports,
        "largest_planned_batch_cuda": benchmark,
        "same_batch_global_complete_padding_tokens": global_complete_tokens,
        "legacy_fixed_truncating_tokens": legacy_fixed_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("--token-budget", type=int, default=32_768)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seed_everything()
    report = benchmark_dataset(
        args.dataset, args.store_root, args.token_budget, args.max_batch_size
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
