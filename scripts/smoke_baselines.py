#!/usr/bin/env python3
"""Run real-data rolling-target smoke gates for DKT, DKVMN, and SAKT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.rolling_targets import RollingTargetDataset, collate_rolling_targets
from ktbench.data.session_store import SessionStore
from ktbench.metrics import binary_metrics
from ktbench.models import (
    DKT,
    DKVMN,
    SAKT,
    DKTConfig,
    DKVMNConfig,
    SAKTConfig,
    load_baseline_checkpoint,
)


TRAINING_CONFIGS: dict[str, dict[str, Any]] = {
    "dkt": {
        "model": "DKT",
        "context_length": 200,
        "history_length": 199,
        "batch_size": 20,
        "optimizer": "Adam",
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "repository_epochs": 200,
        "gradient_clip": None,
        "dropout": 0.1,
        "embedding_size": 64,
        "hidden_size": 64,
        "layers": 1,
        "effective_recurrent_dropout": 0.0,
    },
    "dkvmn": {
        "model": "DKVMN",
        "context_length": 200,
        "history_length": 199,
        "batch_size": 32,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "repository_epochs": 100,
        "gradient_clip": 50.0,
        "dropout": 0.0,
        "question_embedding_size": 50,
        "interaction_embedding_size": 100,
        "memory_size": 20,
        "key_state_size": 50,
        "value_state_size": 100,
        "final_size": 50,
    },
    "sakt": {
        "model": "SAKT",
        "context_length": 100,
        "history_length": 99,
        "batch_size": 10,
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "repository_epochs": 300,
        "gradient_clip": 10.0,
        "dropout": 0.2,
        "width": 200,
        "layers": 1,
        "heads": 5,
        "maximum_relative_position": 10,
    },
}


def _selected_students(store: SessionStore, count: int) -> list[int]:
    student_ids = store.session_student[store.student_offsets[:-1]].astype(np.int32)
    event_offsets = store.session_offsets[store.student_offsets]
    interaction_counts = np.diff(event_offsets)
    longest = int(student_ids[int(np.argmax(interaction_counts))])
    generator = np.random.default_rng(PROJECT_SEED)
    remaining = student_ids[student_ids != longest]
    chosen = generator.choice(remaining, min(count - 1, len(remaining)), replace=False)
    return sorted([longest, *(int(value) for value in chosen)])


def _maximum_history_index(dataset: RollingTargetDataset) -> tuple[int, int]:
    best_index = 0
    best_length = -1
    for start in range(0, len(dataset), 100_000):
        stop = min(start + 100_000, len(dataset))
        lengths = dataset.history_lengths(np.arange(start, stop, dtype=np.int64))
        local = int(np.argmax(lengths))
        if int(lengths[local]) > best_length:
            best_index = start + local
            best_length = int(lengths[local])
    return best_index, best_length


def _indices(dataset: RollingTargetDataset, size: int) -> tuple[np.ndarray, int]:
    maximum_index, maximum_length = _maximum_history_index(dataset)
    fixed = {0, maximum_index}
    generator = np.random.default_rng(PROJECT_SEED)
    available = np.setdiff1d(np.arange(len(dataset), dtype=np.int64), list(fixed))
    extra = generator.choice(
        available, min(max(0, size - len(fixed)), len(available)), replace=False
    )
    return np.asarray(sorted(fixed | set(int(value) for value in extra))), maximum_length


def _make_model(name: str, store: SessionStore) -> torch.nn.Module:
    skills = int(store.metadata["num_skills"])
    questions = int(store.metadata["num_questions"])
    if name == "dkt":
        return DKT(DKTConfig(num_skills=skills))
    if name == "dkvmn":
        return DKVMN(DKVMNConfig(num_skills=skills))
    if name == "sakt":
        return SAKT(SAKTConfig(num_questions=questions, num_skills=skills))
    raise ValueError(name)


def _split_counts(store: SessionStore) -> dict[str, int]:
    lengths = np.diff(store.session_offsets)
    return {
        name: int(lengths[store.session_split == split].sum())
        for name, split in (("train", 0), ("validation", 1), ("test", 2))
    }


def smoke_model(
    name: str,
    dataset_name: str,
    store: SessionStore,
    output_root: Path,
    selected_students: list[int],
) -> dict[str, Any]:
    seed_everything()
    profile = dict(TRAINING_CONFIGS[name])
    history_length = int(profile["history_length"])
    datasets = {
        split: RollingTargetDataset(
            store,
            split,
            history_length=history_length,
            selected_students=selected_students,
        )
        for split in ("train", "validation", "test")
    }
    selected_counts = {split: len(dataset) for split, dataset in datasets.items()}
    if any(value == 0 for value in selected_counts.values()):
        raise RuntimeError("smoke student selection produced an empty split")
    split_sessions = [set(dataset.sessions.tolist()) for dataset in datasets.values()]
    if any(split_sessions[left] & split_sessions[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("a session occurs in more than one target split")
    full_counts = _split_counts(store)
    if sum(full_counts.values()) != int(store.metadata["interactions"]):
        raise RuntimeError("full target counts do not reconcile with session store")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(name, store).to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    train_indices, maximum_history = _indices(
        datasets["train"], int(profile["batch_size"])
    )
    train_batch = collate_rolling_targets(
        [datasets["train"][int(index)] for index in train_indices]
    ).to(device)
    if maximum_history != history_length:
        raise RuntimeError("selected real-data smoke did not exercise the history cap")
    optimizer.zero_grad(set_to_none=True)
    logits = model(train_batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, train_batch.target_labels.float()
    )
    loss.backward()
    clip = profile["gradient_clip"]
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
    optimizer.step()

    output_root.mkdir(parents=True, exist_ok=True)
    cache = output_root / "cache"
    cache.mkdir(exist_ok=True)
    checkpoint_path = cache / "smoke_model.pt"
    torch.save(model.checkpoint(), checkpoint_path)
    model.eval()
    restored = load_baseline_checkpoint(checkpoint_path, map_location=device).to(device).eval()
    with torch.no_grad():
        expected = model(train_batch)
        actual = restored(train_batch)
    if not torch.equal(expected, actual):
        raise RuntimeError("strict checkpoint round trip changed predictions")

    smoke_metrics = {}
    for split in ("validation", "test"):
        indices, _ = _indices(datasets[split], int(profile["batch_size"]))
        batch = collate_rolling_targets(
            [datasets[split][int(index)] for index in indices]
        ).to(device)
        with torch.no_grad():
            split_logits = restored(batch)
            split_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                split_logits, batch.target_labels.float()
            )
        smoke_metrics[split] = binary_metrics(
            torch.sigmoid(split_logits),
            batch.target_labels,
            loss=float(split_loss.detach().cpu()),
        )

    total_parameters = sum(parameter.numel() for parameter in restored.parameters())
    configuration = {
        **profile,
        "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "device": str(device),
        "target_construction": "rolling target-once; strictly prior chronological history",
        "padding": "dynamic per batch; masked from model state, loss, and metrics",
        "checkpoint_selection": "validation ROC-AUC for full experiments",
        "early_stopping_patience": None,
        "early_stopping_note": "not applied in one-step smoke; full-run value requires explicit protocol",
        "epochs_completed": 1,
        "best_epoch": 1,
        "total_parameters": total_parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in restored.parameters() if parameter.requires_grad
        ),
    }
    result = {
        "status": "smoke_complete",
        "model": profile["model"],
        "dataset": dataset_name,
        "seed": PROJECT_SEED,
        "selected_students": selected_students,
        "full_target_counts": full_counts,
        "selected_target_counts": selected_counts,
        "targets_unique": True,
        "chronological_past_only": True,
        "maximum_history_exercised": maximum_history,
        "dynamic_batch_history_width": int(train_batch.history_question.shape[1]),
        "train_step_loss": float(loss.detach().cpu()),
        "checkpoint_round_trip": True,
        "padding_excluded": True,
        "metrics": smoke_metrics,
    }
    dataset_stats = {
        "dataset": dataset_name,
        "students": int(store.metadata["students"]),
        "interactions": int(store.metadata["interactions"]),
        "questions": int(store.metadata["num_questions"]),
        "skills": int(store.metadata["num_skills"]),
        "sessions": int(store.metadata["sessions"]),
        "full_target_counts": full_counts,
        "selected_smoke_students": len(selected_students),
        "selected_target_counts": selected_counts,
    }
    (output_root / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "dataset_stats.json").write_text(
        json.dumps(dataset_stats, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (output_root / "training.jsonl").open("w") as log:
        log.write(
            json.dumps(
                {
                    "event": "smoke_train_step",
                    "model": profile["model"],
                    "dataset": dataset_name,
                    "seed": PROJECT_SEED,
                    "epoch": 1,
                    "loss": result["train_step_loss"],
                }
            )
            + "\n"
        )
        log.write(
            json.dumps(
                {
                    "event": "smoke_evaluation",
                    "model": profile["model"],
                    "dataset": dataset_name,
                    "seed": PROJECT_SEED,
                    "metrics": smoke_metrics,
                }
            )
            + "\n"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("assist2017", "junyi", "ednet_kt1"))
    parser.add_argument("store_root", type=Path)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--models", nargs="+", choices=sorted(TRAINING_CONFIGS), default=sorted(TRAINING_CONFIGS))
    parser.add_argument("--students", type=int, default=4)
    args = parser.parse_args()
    if args.students < 2:
        raise ValueError("at least two smoke students are required")
    seed_everything()
    store = SessionStore(args.store_root)
    selected = _selected_students(store, args.students)
    reports = {}
    for name in args.models:
        destination = args.experiment_root / name / args.dataset / "smoke"
        reports[name] = smoke_model(name, args.dataset, store, destination, selected)
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
