#!/usr/bin/env python3
"""Run bounded real-data prediction checks and CUDA optimizer smoke steps."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ktbench.baseline_registry import BASELINE_TRAINING_CONFIGS, make_baseline_model
from ktbench.batching import collate_hitskt
from ktbench.config import seed_everything
from ktbench.data.algebra2005 import AlgebraSessionTargets
from ktbench.data.fast_rolling import vectorized_batch
from ktbench.data.rolling_targets import RollingTargetDataset
from ktbench.data.session_store import SessionStore
from ktbench.hitskt_registry import HITSKT_TRAINING_CONFIGS
from ktbench.metrics import masked_bce_loss
from ktbench.models import HiTSKT, HiTSKTConfig, PaperFaithfulRKT, RKTConfig
from ktbench.rkt.data import RKTTargetDataset, collate_rkt
from ktbench.rkt.phi import CrossFittedPhiRepository


def batches(store, prepared):
    metadata = store.metadata
    vocabulary = {name: metadata[name] for name in ("num_questions", "num_skills")}
    for name, profile in BASELINE_TRAINING_CONFIGS.items():
        dataset = RollingTargetDataset(store, "train", history_length=profile["history_length"])
        yield name, make_baseline_model(name, **vocabulary), vectorized_batch(dataset, [0, 1, 210, 211]), None
    initial = np.load(prepared / "initial_s_hours.npy")
    repository = CrossFittedPhiRepository(prepared, vocabulary["num_questions"])
    dataset = RKTTargetDataset(store, "train")
    batch = collate_rkt([dataset[i] for i in [0, 1, 210, 211]], phi_repository=repository)
    if not torch.all(batch.delta_hours[batch.history_mask] > 0):
        raise RuntimeError("RKT smoke has an invalid elapsed-time difference")
    yield "rkt", PaperFaithfulRKT(RKTConfig(vocabulary["num_questions"], len(initial) - 1), initial), batch, None

    dataset = AlgebraSessionTargets(store, "train")
    examples = [dataset[i] for i in [0, 1, len(dataset.targets)]]
    batch = collate_hitskt(examples, **vocabulary)
    target = examples[-1]
    flipped = np.asarray(target.target.correct).copy()
    flipped[-1] ^= 1
    variant = collate_hitskt([*examples[:-1], replace(target, target=replace(target.target, correct=flipped))],
                            **vocabulary)
    config = HITSKT_TRAINING_CONFIGS["algebra2005"]
    model = HiTSKT(HiTSKTConfig(**vocabulary, width=config["width"], heads=config["heads"],
                               feedforward_width=config["feedforward_width"]))
    yield "hitskt", model, batch, variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--cpu-forward-only", action="store_true")
    args = parser.parse_args()
    if not args.cpu_forward_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the optimizer smoke")
    device = torch.device("cpu" if args.cpu_forward_only else "cuda")
    seed_everything()
    for name, model, batch, variant in batches(SessionStore(args.store), args.prepared):
        model = model.to(device).eval()
        batch = batch.to(device)
        if variant is None:
            variant = replace(batch, target_labels=1 - batch.target_labels)
        else:
            variant = variant.to(device)
        forward = model.forward_dense if name == "dkt" else model
        if name == "dkt":
            torch.backends.cudnn.allow_tf32 = False
        with torch.no_grad():
            prediction = forward(batch)
            changed = forward(variant)
            if name == "dkt":
                torch.testing.assert_close(prediction, model(batch), atol=5e-5, rtol=1e-3)
            if name == "hitskt":
                changed, prediction = (tensor.masked_select(batch.target_metric_mask)
                                       for tensor in (changed, prediction))
            torch.testing.assert_close(changed, prediction, atol=0, rtol=0)
        result = {
            "model": name, "device": str(device), "target_outcome_invariance": True,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"nonfinite logits for {name}")
        if device.type == "cuda":
            model.train()
            optimizer = torch.optim.Adam(model.parameters())
            start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.reset_peak_memory_stats()
            start.record()
            logits = forward(batch)
            loss = (masked_bce_loss(logits, batch.target_labels, batch.target_metric_mask)
                    if name == "hitskt" else
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.target_labels.float()))
            loss.backward()
            optimizer.step()
            stop.record()
            stop.synchronize()
            if not logits.is_cuda or not torch.isfinite(loss) or not all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"invalid CUDA optimizer smoke for {name}")
            result.update(cuda_step_ms=start.elapsed_time(stop), loss=loss.item(),
                          peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                          gpu=torch.cuda.get_device_name())
            del optimizer, logits, loss
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
