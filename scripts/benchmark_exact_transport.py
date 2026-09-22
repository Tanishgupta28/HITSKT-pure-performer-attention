#!/usr/bin/env python3
"""Isolated A/B test of unchanged FP32 SAKT steps with faster data transport.

Never writes to a production experiment, never signals its trainer, and never
changes batch boundaries, model arithmetic, optimizer, or RNG policy.
"""

import argparse
import copy
import hashlib
import itertools
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.fast_rolling import BatchPlanDataset
from ktbench.data.rolling_targets import RollingTargetDataset, RollingLengthBatchSampler, collate_rolling_targets
from ktbench.data.session_store import SessionStore
from ktbench.metrics import binary_metrics
from ktbench.models import load_baseline_checkpoint
from ktbench.training import restore_rng_state, capture_rng_state


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def run_variant(args, dataset, plan, checkpoint, variant, training):
    model = load_baseline_checkpoint(args.checkpoint, map_location="cpu").cuda()
    model.train(training)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if variant == "reference":
        loader = DataLoader(dataset, batch_sampler=plan, collate_fn=collate_rolling_targets,
                            num_workers=args.workers, pin_memory=True)
    else:
        packed = variant == "packed_deferred"
        source = BatchPlanDataset(dataset, plan, args.block_size if packed else 1, packed)
        loader = DataLoader(source, batch_size=None, num_workers=args.workers, pin_memory=True)
    restore_rng_state(checkpoint["rng_state"])
    iterator = iter(loader)  # Same one main-process RNG draw in every variant.

    def batches():
        for item in iterator:
            if variant == "packed_deferred":
                yield from item.to_batches("cuda")
            else:
                yield item.to("cuda")

    records, counts, flushed = [], [], []

    def flush():
        if records:
            flushed.append((torch.stack([r[0] for r in records]).cpu().numpy(),
                            torch.cat([r[1] for r in records]).cpu().numpy(),
                            torch.cat([r[2] for r in records]).cpu().numpy()))
            records.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()
    measured_started = None
    for step, batch in enumerate(batches()):
        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch.target_labels.float(), reduction="mean")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        count = len(batch.target_labels)
        probabilities = torch.sigmoid(logits).detach()
        if variant == "packed_deferred":
            records.append((loss.detach(), probabilities, batch.target_labels))
        else:
            # Match the production loop's individual synchronizing transfers.
            records.append((float(loss.detach().cpu()), probabilities.cpu().numpy(),
                            batch.target_labels.cpu().numpy()))
        counts.append(count)
        if variant == "packed_deferred" and len(records) == args.block_size:
            flush()
        if step + 1 == args.warmup:
            torch.cuda.synchronize()
            measured_started = time.perf_counter()
    if variant == "packed_deferred":
        flush()
        losses = np.concatenate([r[0] for r in flushed])
        probabilities = np.concatenate([r[1] for r in flushed])
        labels = np.concatenate([r[2] for r in flushed])
    else:
        losses = np.asarray([r[0] for r in records], dtype=np.float32)
        probabilities = np.concatenate([r[1] for r in records])
        labels = np.concatenate([r[2] for r in records])
    torch.cuda.synchronize()
    ended = time.perf_counter()
    # Same sequential Python float summation, not a different GPU reduction.
    loss_sum = 0.0
    for loss, count in zip(losses, counts, strict=True):
        loss_sum += float(loss) * count
    metrics = binary_metrics(torch.from_numpy(probabilities), torch.from_numpy(labels),
                             loss=loss_sum / sum(counts))
    state = {
        "model": cpu_tree(model.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "rng": capture_rng_state(),
        "losses": torch.from_numpy(losses),
        "probabilities": torch.from_numpy(probabilities),
        "labels": torch.from_numpy(labels),
        "metrics": metrics,
    }
    result = {"variant": variant, "training": training,
              "measured_seconds": ended - measured_started,
              "total_seconds": ended - total_started,
              "measured_steps": args.steps, "warmup_steps": args.warmup,
              "targets": sum(counts),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    del iterator, loader, model, optimizer, records
    return result, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("store", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=64)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--repetitions", type=int, default=3)
    args = p.parse_args()
    if args.steps <= 0 or args.warmup < 1 or args.block_size < 1:
        p.error("steps, warmup, block size must be positive")
    if args.output.exists():
        p.error("refusing to overwrite a benchmark report")
    if args.checkpoint.name != "transport_benchmark_snapshot.pt":
        p.error("use a separate transport_benchmark_snapshot.pt copy, not a live checkpoint")
    seed_everything()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["model"] == "SAKT" and checkpoint["seed"] == PROJECT_SEED
    store = SessionStore(args.store)
    report = {"seed": PROJECT_SEED, "checkpoint_epoch": checkpoint["epoch"],
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "device": torch.cuda.get_device_name(0), "pytorch": torch.__version__,
              "batch_size": 10, "history_length": 99, "block_size": args.block_size,
              "workers": args.workers, "measurements": [],
              "limitations": ["short-window benchmark, not full-epoch timing",
                              "live trainer shares GPU/CPU; contention affects absolute times",
                              "production runner untouched; not approved for deployment"]}
    for training in (True, False):
        dataset = RollingTargetDataset(store, "train" if training else "validation", history_length=99)
        sampler = RollingLengthBatchSampler(dataset, batch_size=10, bucket_size=4096,
                                            shuffle=training, seed=PROJECT_SEED)
        sampler.set_epoch(checkpoint["epoch"] + 1 if training else 0)
        plan = list(itertools.islice(sampler, args.steps + args.warmup))
        assert len(plan) == args.steps + args.warmup
        # Explicitly prove every transported field, including target IDs, before timing.
        from ktbench.data.fast_rolling import vectorized_batch, pack_batches
        for indices in plan:
            expected = collate_rolling_targets([dataset[i] for i in indices])
            candidate = vectorized_batch(dataset, indices)
            restored = pack_batches([candidate]).to_batches("cpu")[0]
            assert exact(vars(expected), vars(candidate)) and exact(vars(expected), vars(restored))
        reference = None
        variants = ["reference", "vectorized", "packed_deferred"]
        for repeat in range(args.repetitions):
            order = variants if repeat % 2 == 0 else variants[::-1]
            for variant in order:
                result, state = run_variant(args, dataset, plan, checkpoint, variant, training)
                if reference is None:
                    assert variant == "reference"
                    reference = state
                result["repeat"] = repeat
                result["exact_checks"] = {k: exact(reference[k], state[k]) for k in reference}
                result["all_exact"] = all(result["exact_checks"].values())
                report["measurements"].append(result)
                print(json.dumps(result), flush=True)
    report["summary"] = []
    for training in (True, False):
        medians = {v: statistics.median(x["measured_seconds"] for x in report["measurements"]
                                       if x["training"] == training and x["variant"] == v)
                   for v in variants}
        report["summary"].append({"training": training, "median_seconds": medians,
                                  "speedup": {v: medians["reference"] / t for v, t in medians.items()}})
    report["all_exact"] = all(x["all_exact"] for x in report["measurements"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_exact": report["all_exact"], "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
