#!/usr/bin/env python3
"""Exercise CUDA training and validation on a bounded ASSIST2009 subset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from ktbench.batching import Session, SessionExample, collate_hitskt
from ktbench.metrics import masked_bce_loss
from ktbench.models import HiTSKT, HiTSKTConfig, PaperFaithfulRKT, RKTConfig
from ktbench.rkt.data import RKTBatch
from ktbench.baseline_registry import BASELINE_TRAINING_CONFIGS, make_baseline_model
from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.assist2009 import Assist2009Store
from ktbench.models import load_baseline_checkpoint
from scripts.train_baseline import _checkpoint_payload, _run_epoch
from ktbench.training import COMMON_EARLY_STOPPING, ValidationAUCEarlyStopping


def smoke(name, store, output):
    seed_everything()
    device = torch.device("cuda")
    settings = BASELINE_TRAINING_CONFIGS[name]
    model = make_baseline_model(name, num_questions=store.metadata["num_questions"],
                                num_skills=store.metadata["num_skills"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    train = store.targets("train", settings["history_length"])
    valid = store.targets("validation", settings["history_length"])
    train_indices = np.linspace(0, len(train) - 1, settings["batch_size"], dtype=int)
    valid.events = valid.events[np.linspace(0, len(valid) - 1, 128, dtype=int)]
    batch = train.vectorized_batch(train_indices).to(device)
    if name == "dkt":
        torch.backends.cudnn.allow_tf32 = False
        mixed_indices = np.r_[0, train_indices[1:]]
        comparison = train.vectorized_batch(mixed_indices).to(device)
        reference = model(comparison)
        accelerated = model.forward_dense(comparison)
        torch.testing.assert_close(accelerated, reference, atol=5e-5, rtol=1e-3)
        parameters = tuple(model.parameters())
        for actual, expected in zip(torch.autograd.grad(accelerated.sum(), parameters),
                                    torch.autograd.grad(reference.sum(), parameters), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
    forward = model.forward_dense if name == "dkt" else model
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        logits = forward(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch.target_labels.float())
        loss.backward()
        if settings["gradient_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
        optimizer.step()
        torch.cuda.synchronize()
    kernels = [e for e in trace.events() if e.device_type == torch.autograd.DeviceType.CUDA
               and not e.name.startswith(("[CUDA memcpy", "[CUDA memset", "Memcpy", "Memset"))]
    if not logits.is_cuda or not kernels or not torch.isfinite(loss):
        raise RuntimeError(f"{name}: CUDA smoke did not record a valid training step")
    common = dict(batch_size=settings["batch_size"], device=device, workers=0,
                  epoch=1, optimizer=None, gradient_clip=None, transport="packed64",
                  dkt_dense=name == "dkt")
    metrics = _run_epoch(model, valid, **common)
    checkpoint = output / "smoke_model.pt"
    control = ValidationAUCEarlyStopping(COMMON_EARLY_STOPPING)
    control.step(1, metrics["roc_auc"])
    torch.save(_checkpoint_payload(model, optimizer, epoch=1,
                                   validation_auc=metrics["roc_auc"], control=control), checkpoint)
    replay = _run_epoch(load_baseline_checkpoint(checkpoint, map_location=device).to(device),
                        valid, **common)
    if replay != metrics:
        raise RuntimeError(f"{name}: CUDA smoke checkpoint replay differs")
    result = {
        "status": "smoke_complete", "model": settings["model"], "dataset": "assist2009",
        "seed": PROJECT_SEED, "epoch": 1, "train_loss": float(loss.detach()),
        "validation": metrics, "cuda_kernels": len(kernels),
        "summed_cuda_kernel_duration_us": sum(e.time_range.elapsed_us() for e in kernels),
        "gpu": torch.cuda.get_device_name(device), "provenance": store.provenance,
    }
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def smoke_reference_models(output):
    """CUDA plumbing checks on explicit unit-test inputs for time-dependent models."""
    seed_everything()
    device = torch.device("cuda")
    history = Session([1, 2], [1, 1], [0, 1])
    next_session = Session([3, 4], [2, 2], [1, 0])
    hi_batch = collate_hitskt([SessionExample([history], next_session, 0)],
                              num_questions=4, num_skills=2).to(device)
    hitskt = HiTSKT(HiTSKTConfig(4, 2, width=256, heads=4, feedforward_width=2048))
    rkt_batch = RKTBatch(
        student_id=torch.tensor([1, 2]), history_question=torch.tensor([[1, 2], [1, 3]]),
        history_correct=torch.tensor([[0, 1], [1, 0]]),
        history_position=torch.tensor([[1, 2], [1, 2]]),
        history_mask=torch.ones((2, 2), dtype=torch.bool),
        delta_hours=torch.tensor([[2.0, 1.0], [1.75, 0.25]]),
        phi=torch.tensor([[0.0, 0.2], [0.0, -0.1]]), target_question=torch.tensor([3, 4]),
        target_labels=torch.tensor([1, 0]), target_event_id=torch.tensor([2, 5]),
        split=torch.tensor([0, 0]),
    ).to(device)
    rkt = PaperFaithfulRKT(RKTConfig(4, 2), torch.ones(3))
    output.mkdir(parents=True, exist_ok=True)
    for name, model, batch, learning_rate in (("HiTSKT", hitskt, hi_batch, 5e-5),
                                               ("RKT", rkt, rkt_batch, 1e-3)):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        logits = model(batch)
        if name == "HiTSKT":
            loss = masked_bce_loss(logits, batch.target_labels, batch.target_metric_mask)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, batch.target_labels.float())
        loss.backward()
        optimizer.step()
        end.record()
        end.synchronize()
        checkpoint = output / f"{name.lower()}_fixture.pt"
        torch.save(model.checkpoint(), checkpoint)
        restored = type(model).from_checkpoint(checkpoint, map_location=device).to(device).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(batch), model.eval()(batch), atol=0, rtol=0)
        if not logits.is_cuda or not torch.isfinite(loss):
            raise RuntimeError(f"{name} CUDA fixture failed")
        print(json.dumps({
            "status": "fixture_smoke_complete", "model": name,
            "dataset": "synthetic_unit_test_fixture", "epoch": 1,
            "training_loss": float(loss.detach()), "cuda_step_elapsed_ms": start.elapsed_time(end),
            "gpu": torch.cuda.get_device_name(device), "checkpoint_round_trip": True,
        }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("experiments", type=Path)
    parser.add_argument("--reference-fixtures", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ASSIST2009 smoke requires CUDA")
    if args.reference_fixtures:
        smoke_reference_models(args.experiments)
        return
    store = Assist2009Store(args.store)
    for name in BASELINE_TRAINING_CONFIGS:
        output = args.experiments / name / "assist2009" / "smoke"
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps(smoke(name, store, output)), flush=True)


if __name__ == "__main__":
    main()
