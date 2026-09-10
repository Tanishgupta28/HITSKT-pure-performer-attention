#!/usr/bin/env python3
"""Generate master result files only from completed experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


MODELS = ("DKT", "DKVMN", "SAKT", "RKT", "HiTSKT")
DATASETS = ("assist2017", "junyi", "ednet_kt1")
FIELDS = (
    "dataset",
    "model",
    "students",
    "interactions",
    "questions",
    "skills",
    "train_examples",
    "validation_examples",
    "test_examples",
    "train_ratio",
    "validation_ratio",
    "test_ratio",
    "sequence_length",
    "history_length",
    "optimizer",
    "learning_rate",
    "batch_size",
    "dropout",
    "patience",
    "min_delta",
    "epoch_ceiling",
    "epochs_completed",
    "best_epoch",
    "best_validation_auc",
    "test_loss",
    "test_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mse",
    "total_parameters",
    "trainable_parameters",
    "runtime",
    "GPU",
    "seed",
    "experiment_directory",
)


def _gpu_name(config: dict[str, Any], result: dict[str, Any]) -> str:
    persisted = result.get("gpu", config.get("gpu"))
    if persisted:
        return str(persisted)
    if str(config.get("device", "")).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device("cuda"))
    return str(config.get("device", "unknown"))


def _row(root: Path) -> dict[str, Any]:
    config = json.loads((root / "config.json").read_text())
    stats = json.loads((root / "dataset_stats.json").read_text())
    result = json.loads((root / "final_results.json").read_text())
    if result.get("status") != "complete" or not (root / "_SUCCESS").exists():
        raise ValueError(f"experiment is not complete: {root}")
    targets = stats["target_counts"]
    ratios = stats["target_ratios"]
    sequence_length = config.get(
        "context_length", config.get("context_sessions_including_target")
    )
    test = result["test"]
    return {
        "dataset": result["dataset"],
        "model": result["model"],
        "students": stats["students"],
        "interactions": stats["interactions"],
        "questions": stats["questions"],
        "skills": stats["skills"],
        "train_examples": targets["train"],
        "validation_examples": targets["validation"],
        "test_examples": targets["test"],
        "train_ratio": ratios["train"],
        "validation_ratio": ratios["validation"],
        "test_ratio": ratios["test"],
        "sequence_length": sequence_length,
        "history_length": config.get("history_length", config.get("history_sessions")),
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "batch_size": config.get("batch_size", config.get("maximum_batch_size")),
        "dropout": config["dropout"],
        "patience": result["patience"],
        "min_delta": result["min_delta"],
        "epoch_ceiling": result["epoch_ceiling"],
        "epochs_completed": result["epochs_completed"],
        "best_epoch": result["best_epoch"],
        "best_validation_auc": result["best_validation_auc"],
        "test_loss": test["loss"],
        "test_auc": test["roc_auc"],
        "test_accuracy": test["accuracy"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "test_mse": test["mse"],
        "total_parameters": result["total_parameters"],
        "trainable_parameters": result["trainable_parameters"],
        "runtime": result["runtime_seconds"],
        "GPU": _gpu_name(config, result),
        "seed": result["seed"],
        "experiment_directory": str(root),
    }


def aggregate(experiments_root: Path, output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            root = experiments_root / model.lower() / dataset / "full"
            if (root / "_SUCCESS").exists():
                rows.append(_row(root))
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "final_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "final_results.json").write_text(
        json.dumps(
            {
                "completed_experiments": len(rows),
                "expected_experiments": len(MODELS) * len(DATASETS),
                "source": "completed per-experiment artifacts only; no manually entered metrics",
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return rows


if __name__ == "__main__":
    aggregate(Path("experiments"), Path("reports"))
