#!/usr/bin/env python3
"""Run the bounded pre-training RKT smoke gate on prepared artifacts."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.session_store import SessionStore
from ktbench.metrics import binary_metrics
from ktbench.models import PaperFaithfulRKT, RKTConfig
from ktbench.models.rkt import RKT_LABEL
from ktbench.rkt.data import RKTTargetDataset, collate_rkt
from ktbench.rkt.phi import CrossFittedPhiRepository


def _evaluate(
    model: PaperFaithfulRKT,
    dataset: RKTTargetDataset,
    repository: CrossFittedPhiRepository,
    device: torch.device,
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=partial(collate_rkt, phi_repository=repository),
    )
    probabilities = []
    labels = []
    loss_sum = 0.0
    model.eval()
    model.set_relation_parameters_trainable(False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss_sum += float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch.target_labels.float(), reduction="sum"
                ).item()
            )
            probabilities.append(torch.sigmoid(logits).cpu())
            labels.append(batch.target_labels.cpu())
    all_probabilities = torch.cat(probabilities)
    all_labels = torch.cat(labels)
    return binary_metrics(
        all_probabilities,
        all_labels,
        loss=loss_sum / len(all_labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("store_root", type=Path)
    parser.add_argument("prepared_root", type=Path)
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    seed_everything()
    store = SessionStore(args.store_root)
    selected_path = args.prepared_root / "selected_students.csv"
    selected = None
    if selected_path.exists():
        selected = pd.read_csv(selected_path)["student_id"].astype(int).tolist()
    repository = CrossFittedPhiRepository(
        args.prepared_root, int(store.metadata["num_questions"])
    )
    datasets = {
        split: RKTTargetDataset(store, split, selected_students=selected)
        for split in ("train", "validation", "test")
    }
    all_target_ids = [
        example.target_event_id for dataset in datasets.values() for example in dataset
    ]
    if len(all_target_ids) != len(set(all_target_ids)):
        raise RuntimeError("RKT target duplication detected")
    if repository.metadata["training_targets_cross_fitted"] != len(datasets["train"]):
        raise RuntimeError("cross-fitted Phi targets do not match training targets")

    initial_s = np.load(args.prepared_root / "initial_s_hours.npy")
    config = RKTConfig(
        num_questions=int(store.metadata["num_questions"]),
        num_students=len(initial_s) - 1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperFaithfulRKT(config, initial_s).to(device).train()
    model.set_relation_parameters_trainable(True)
    training_trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    generator = np.random.default_rng(PROJECT_SEED)
    sample_size = min(128, len(datasets["train"]))
    train_indices = np.sort(generator.choice(len(datasets["train"]), sample_size, replace=False))
    train_batch = collate_rkt(
        [datasets["train"][int(index)] for index in train_indices],
        phi_repository=repository,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    optimizer.zero_grad(set_to_none=True)
    logits = model(train_batch)
    train_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, train_batch.target_labels.float()
    )
    train_loss.backward()
    optimizer.step()

    args.experiment_root.mkdir(parents=True, exist_ok=True)
    cache = args.experiment_root / "cache"
    cache.mkdir(exist_ok=True)
    checkpoint = cache / "smoke_model.pt"
    torch.save(model.checkpoint(), checkpoint)
    model.eval()
    model.set_relation_parameters_trainable(False)
    restored = PaperFaithfulRKT.from_checkpoint(checkpoint, map_location=device).to(device).eval()
    restored.set_relation_parameters_trainable(False)
    with torch.no_grad():
        if not torch.equal(model(train_batch), restored(train_batch)):
            raise RuntimeError("RKT checkpoint round trip changed predictions")

    metrics = {
        split: _evaluate(restored, dataset, repository, device)
        for split, dataset in datasets.items()
        if split != "train"
    }
    parameter_count = sum(parameter.numel() for parameter in restored.parameters())
    configuration = {
        "label": RKT_LABEL,
        "seed": PROJECT_SEED,
        "width": 64,
        "heads": 1,
        "dropout": 0.1,
        "history_interactions": 49,
        "maximum_context_including_target": 50,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "batch_size": 128,
        "device": str(device),
        "parameters": parameter_count,
        "training_trainable_parameters": training_trainable_parameter_count,
        "evaluation_trainable_parameters": sum(
            parameter.numel() for parameter in restored.parameters() if parameter.requires_grad
        ),
        "phi": "directed raw performance Phi; 5-fold student cross-fit for train; all-train for validation/test",
        "timestamp": "processed milliseconds converted to hours by division by 3,600,000",
    }
    (args.experiment_root / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "status": "smoke_complete",
        "seed": PROJECT_SEED,
        "train_step_loss": float(train_loss.detach().cpu()),
        "tensor_shapes": {
            "history_question": list(train_batch.history_question.shape),
            "phi": list(train_batch.phi.shape),
            "logits": list(logits.shape),
        },
        "targets": {split: len(dataset) for split, dataset in datasets.items()},
        "target_ids_unique": True,
        "checkpoint_round_trip": True,
        "relation_parameters_frozen_for_evaluation": True,
        "metrics": metrics,
    }
    (args.experiment_root / "smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (args.experiment_root / "training.jsonl").open("w") as log:
        log.write(json.dumps({"event": "smoke_train_step", "seed": PROJECT_SEED, "loss": result["train_step_loss"]}) + "\n")
        log.write(json.dumps({"event": "smoke_evaluation", "seed": PROJECT_SEED, "metrics": metrics}) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
