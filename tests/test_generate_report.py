import csv
import json
from pathlib import Path

import pytest

from scripts.aggregate_results import DATASETS, MODELS
from scripts.generate_report import generate_report


def _fixture(root: Path, model: str, dataset: str) -> Path:
    """Synthetic artifact fixture only; never a scientific benchmark result."""
    path = root / model.lower() / dataset / "full"
    path.mkdir(parents=True)
    test = dict(loss=0.5, roc_auc=0.7, accuracy=0.6, precision=0.5,
                recall=0.4, f1=0.45, mse=0.2)
    objects = {
        "config.json": dict(context_length=50, history_length=49,
                            optimizer="Adam", learning_rate=0.001,
                            batch_size=128, dropout=0.1, device="cpu", gpu="fixture"),
        "dataset_stats.json": dict(students=2, interactions=30, questions=5, skills=3,
                                   target_counts=dict(train=18, validation=6, test=6),
                                   target_ratios=dict(train=0.6, validation=0.2, test=0.2)),
        "final_results.json": dict(status="complete", model=model, dataset=dataset,
                                   seed=42, patience=5, min_delta=0.0,
                                   epoch_ceiling=2, epochs_completed=2, best_epoch=2,
                                   best_validation_auc=0.8, runtime_seconds=1.0,
                                   total_parameters=10, trainable_parameters=10, test=test),
    }
    for name, value in objects.items():
        (path / name).write_text(json.dumps(value))
    with (path / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "split", "targets", *test))
        writer.writeheader()
        for epoch in (1, 2):
            for split in ("train", "validation"):
                writer.writerow(dict(epoch=epoch, split=split, targets=6,
                                     **{**test, "roc_auc": 0.6 if epoch == 1 else 0.8}))
        writer.writerow(dict(epoch=2, split="test", targets=6, **test))
    (path / "_SUCCESS").write_text("complete\n")
    return path


def test_final_report_refuses_partial_without_writing(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    _fixture(experiments, "RKT", "assist2017")
    destination = tmp_path / "reports"
    with pytest.raises(ValueError, match="requires 15 completed runs; found 1"):
        generate_report(experiments, destination)
    assert not destination.exists()


def test_report_tables_plots_and_hashes_use_full_artifacts(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    for model in MODELS:
        for dataset in DATASETS:
            _fixture(experiments, model, dataset)
    # An unrelated smoke result must not enter the report.
    smoke = experiments / "rkt" / "junyi" / "smoke"
    smoke.mkdir()
    (smoke / "_SUCCESS").write_text("complete\n")
    output = tmp_path / "reports"
    manifest = generate_report(experiments, output)
    assert manifest["status"] == "complete"
    assert manifest["completed_experiments"] == 15
    assert len(manifest["source_sha256"]) == 45
    assert len(list((output / "figures").glob("*.png"))) == 18
    assert len(list((output / "figures").glob("*.svg"))) == 18
    with (output / "tables" / "auc.csv").open(newline="") as handle:
        table = list(csv.reader(handle))
    assert table[1] == ["DKT", "0.7", "0.7", "0.7"]
    text = (output / "comparison.md").read_text()
    assert "INCOMPLETE" not in text
    assert "0.700000" in text
    assert "not model contexts or hyperparameters" in text
    assert len(list((output / "tables").glob("*.csv"))) == 7


def test_inconsistent_metrics_are_rejected_before_report_writes(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    path = _fixture(experiments, "RKT", "assist2017")
    result = json.loads((path / "final_results.json").read_text())
    result["test"]["roc_auc"] = 0.99
    (path / "final_results.json").write_text(json.dumps(result))
    output = tmp_path / "reports"
    with pytest.raises(ValueError, match="test metric mismatch"):
        generate_report(experiments, output, allow_partial=True)
    assert not output.exists()
