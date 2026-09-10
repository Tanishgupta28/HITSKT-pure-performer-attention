import csv
import json
from pathlib import Path

from scripts.aggregate_results import aggregate


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_aggregate_uses_only_completed_full_artifacts(tmp_path: Path) -> None:
    full = tmp_path / "experiments" / "rkt" / "assist2017" / "full"
    full.mkdir(parents=True)
    _write_json(
        full / "config.json",
        {
            "context_length": 50,
            "history_length": 49,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 128,
            "dropout": 0.1,
            "device": "cuda",
            "gpu": "fixture GPU",
        },
    )
    _write_json(
        full / "dataset_stats.json",
        {
            "students": 2,
            "interactions": 30,
            "questions": 5,
            "skills": 3,
            "target_counts": {"train": 18, "validation": 6, "test": 6},
            "target_ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
        },
    )
    _write_json(
        full / "final_results.json",
        {
            "status": "complete",
            "dataset": "assist2017",
            "model": "RKT",
            "seed": 42,
            "patience": 5,
            "min_delta": 0.0,
            "epoch_ceiling": 300,
            "epochs_completed": 7,
            "best_epoch": 2,
            "best_validation_auc": 0.75,
            "runtime_seconds": 12.0,
            "total_parameters": 10,
            "trainable_parameters": 10,
            "test": {
                "loss": 0.5,
                "roc_auc": 0.7,
                "accuracy": 0.6,
                "precision": 0.5,
                "recall": 0.4,
                "f1": 0.45,
                "mse": 0.2,
            },
        },
    )
    (full / "_SUCCESS").write_text("complete\n")

    # A smoke artifact must never appear in the master scientific results.
    smoke = tmp_path / "experiments" / "rkt" / "junyi" / "smoke"
    smoke.mkdir(parents=True)
    (smoke / "_SUCCESS").write_text("complete\n")

    rows = aggregate(tmp_path / "experiments", tmp_path / "reports")
    assert len(rows) == 1
    assert rows[0]["test_examples"] == 6
    assert rows[0]["GPU"] == "fixture GPU"
    with (tmp_path / "reports" / "final_results.csv").open() as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == 1
    assert persisted[0]["model"] == "RKT"
    summary = json.loads((tmp_path / "reports" / "final_results.json").read_text())
    assert summary["completed_experiments"] == 1
    assert summary["expected_experiments"] == 15
