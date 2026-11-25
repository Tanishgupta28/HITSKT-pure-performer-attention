#!/usr/bin/env python3
"""Publish only independently verified ASSIST2009 full runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from ktbench.data.flat import sha256_file
from ktbench.models.rkt import RKT_PHI_ONLY
from scripts.aggregate_results import FIELDS, MODELS, _row


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def verify_epoch_artifacts(run, result, config, replay):
    with (run / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = range(1, result["epochs_completed"] + 1)
    expected = [(epoch, split) for epoch in epochs for split in ("train", "validation")]
    expected.append((result["best_epoch"], "test"))
    if [(int(row["epoch"]), row["split"]) for row in rows] != expected:
        raise ValueError(f"unexpected or duplicate epoch observations: {run}")
    stats = read_json(run / "dataset_stats.json")
    for row in rows:
        if int(row["targets"]) != stats["target_counts"][row["split"]]:
            raise ValueError(f"epoch observation has wrong target count: {run}")
        if not all(math.isfinite(float(value)) for key, value in row.items() if key != "split"):
            raise ValueError(f"nonfinite epoch observation: {run}")
    validation = [row for row in rows if row["split"] == "validation"]
    best = max(validation, key=lambda row: float(row["roc_auc"]))
    if ((int(best["epoch"]), float(best["roc_auc"])) !=
            (result["best_epoch"], result["best_validation_auc"])):
        raise ValueError(f"the best epoch disagrees with the validation observations: {run}")
    if not replay.get("validation_and_test_target_ids_verified"):
        raise ValueError(f"replay did not verify held-out event IDs: {run}")
    if any(float(best[key]) != value for key, value in replay["validation"].items()):
        raise ValueError(f"validation replay differs from best epoch metrics: {run}")
    if any(float(rows[-1][key]) != value for key, value in result["test"].items()):
        raise ValueError(f"test observations differ from the final outcome: {run}")
    last = torch.load(run / "last_model.pt", map_location="cpu", weights_only=True)
    if (last["epoch"] != result["epochs_completed"] or last["seed"] != result["seed"]
            or not {"optimizer_state_dict", "rng_state", "early_stopping_state"}.issubset(last)
            or config["epochs_completed"] != result["epochs_completed"]):
        raise ValueError(f"last checkpoint cannot substantiate the completed epoch: {run}")


def collect(experiments):
    rows, statuses = [], {}
    provenance = None
    for canonical in MODELS:
        model = RKT_PHI_ONLY if canonical == "RKT" else canonical
        run = experiments / canonical.lower() / "assist2009" / "full"
        replay_path = run / "independent_evaluation.json"
        if not (run / "_SUCCESS").is_file() or not replay_path.is_file():
            statuses[model] = {
                "status": "pending",
                "reason": "Full training or independent replay is pending.",
            }
            continue
        result = read_json(run / "final_results.json")
        config = read_json(run / "config.json")
        replay = read_json(replay_path)
        current = config["dataset_provenance"]
        if provenance is None:
            provenance = current
        if (current != provenance or replay.get("dataset_provenance") != current
                or replay.get("test") != result["test"] or not replay.get("verified")
                or replay.get("checkpoint_sha256") != sha256_file(run / "best_model.pt")
                or replay.get("total_parameters") != result["total_parameters"]
                or replay.get("trainable_parameters") != result["trainable_parameters"]
                or (result["dataset"], result["model"], result["seed"]) != ("assist2009", model, 42)):
            raise ValueError(f"mismatched completed artifacts: {run}")
        verify_epoch_artifacts(run, result, config, replay)
        rows.append(_row(run))
        statuses[model] = {"status": "verified_complete"}
        for key in ("session_protocol", "relation_mode"):
            if key in config:
                statuses[model][key] = config[key]
    return rows, statuses


def main():
    reports = Path("reports")
    historical_csv = (reports / "final_results.csv").read_bytes()
    historical = read_json(reports / "final_results.json")
    rows, statuses = collect(Path("experiments"))
    path = reports / "assist2009_final_results.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    header, _, new_rows = path.read_bytes().partition(b"\n")
    if historical_csv.splitlines()[0] != header:
        raise ValueError("historical CSV schema differs from ASSIST2009 export")
    separator = b"" if historical_csv.endswith(b"\n") else b"\n"
    (reports / "combined_benchmark_results.csv").write_bytes(historical_csv + separator + new_rows)
    write_json(reports / "assist2009_final_results.json", {
        "completed_experiments": len(rows), "expected_experiments": len(MODELS),
        "source": "independently verified full-run ASSIST2009 checkpoints",
        "protocol": "assist2009_ordered_interactions_60_20_20_v1",
        "model_status": statuses, "results": rows,
    })
    write_json(reports / "combined_benchmark_results.json", {
        **historical, "completed_experiments": historical["completed_experiments"] + len(rows),
        "expected_experiments": historical["expected_experiments"] + len(MODELS),
        "source": "historical benchmark results and independently verified full ASSIST2009 runs",
        "assist2009_model_status": statuses, "results": historical["results"] + rows,
    })
    print(f"Exported {len(rows)} verified ASSIST2009 experiments.")


if __name__ == "__main__":
    main()
