#!/usr/bin/env python3
"""Publish Algebra rows only after exact independent checkpoint verification."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

from ktbench.data.flat import sha256_file
from scripts.aggregate_assist2009 import verify_epoch_artifacts
from scripts.aggregate_results import FIELDS, MODELS, _row

DATASET = "algebra2005"
METRICS = {"test_loss": ("Loss", min), "test_auc": ("ROC-AUC", max),
           "test_accuracy": ("Accuracy", max), "test_precision": ("Precision", max),
           "test_recall": ("Recall", max), "test_f1": ("F1", max), "test_mse": ("MSE", min)}


def read_json(path):
    return json.loads(path.read_text())


def write_bytes(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def collect(dataset=DATASET):
    data = Path("reports/datasets") / dataset
    verification = read_json(data / "data_verification.json")
    summary = read_json(data / "summary.json")
    hashes = verification["source_target_manifests"]
    rows = []
    for model in MODELS:
        run = Path("experiments", model.lower(), dataset, "full")
        config, result, replay, stats = (read_json(run / name) for name in (
            "config.json", "final_results.json", "independent_evaluation.json", "dataset_stats.json"))
        if (not (run / "_SUCCESS").is_file() or not replay["verified"]
                or (result["dataset"], result["model"], result["seed"]) != (dataset, model, 42)
                or config.get("test_epoch_ceiling_override") is not None
                or (result["patience"], result["min_delta"], config["device"]) != (5, 0.0, "cuda")
                or config["session_store_metadata_sha256"] != verification["session_store_metadata_sha256"]
                or replay["session_store_metadata_sha256"] != verification["session_store_metadata_sha256"]
                or replay["checkpoint_sha256"] != sha256_file(run / "best_model.pt")
                or stats["target_counts"] != {split: value["targets"] for split, value in hashes.items()}
                or stats["target_counts"] != summary["split_interactions"]
                or replay["test"] != result["test"]
                or any(replay[key] != result[key] for key in ("total_parameters", "trainable_parameters"))):
            raise ValueError(f"unverified or mismatched full run: {run}")
        verify_epoch_artifacts(run, result, config, replay)
        log = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines() if line]
        if [event["epoch"] for event in log if event["event"] == "epoch_complete"] != list(
                range(1, result["epochs_completed"] + 1)):
            raise ValueError(f"missing epoch log records: {run}")
        rows.append(_row(run))
    return rows, verification


def update_combined(rows):
    csv_path = Path("reports/combined_benchmark_results.csv")
    json_path = Path("reports/combined_benchmark_results.json")
    existing = csv_path.read_bytes()
    combined = read_json(json_path)
    parsed = list(csv.DictReader(io.StringIO(existing.decode())))
    if [(r["dataset"], r["model"]) for r in parsed] != [
            (r["dataset"], r["model"]) for r in combined["results"]]:
        raise ValueError("combined CSV and JSON benchmark identities differ")
    new = csv_bytes(rows)
    if existing.splitlines()[0] != new.splitlines()[0]:
        raise ValueError("combined benchmark header differs from the canonical CSV schema")
    if any(row["dataset"] == DATASET for row in parsed):
        if [row for row in combined["results"] if row["dataset"] == DATASET] != rows:
            raise ValueError("the combined reports already contain different Algebra results")
        return
    combined["results"].extend(rows)
    combined["completed_experiments"] += len(rows)
    combined["expected_experiments"] += len(MODELS)
    combined["algebra2005_model_status"] = {model: {"status": "verified_complete"} for model in MODELS}
    combined["source"] = combined["source"] + " and independently verified full Algebra2005 runs"
    write_bytes(csv_path, existing + (b"" if existing.endswith(b"\n") else b"\n") + new.split(b"\n", 1)[1])
    write_json(json_path, combined)


def write_report(rows, dataset=DATASET, *, comparison=None):
    ranked = sorted(rows, key=lambda row: -row["test_auc"])
    best = {field: selector(row[field] for row in rows) for field, (_, selector) in METRICS.items()}
    lines = [
        "All five full GPU runs completed and independent checkpoint replay matched the recorded metrics.",
        "",
        "| Model | Best / completed epoch | Validation ROC-AUC | Loss ↓ | ROC-AUC ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ | MSE ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        values = [row["model"], f'{row["best_epoch"]} / {row["epochs_completed"]}',
                  f'{row["best_validation_auc"]:.6f}']
        for field in METRICS:
            value = f"{row[field]:.6f}"
            values.append(f"**{value}**" if row[field] == best[field] else value)
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "### Metric leaders", ""]
    for field, (label, _) in METRICS.items():
        leaders = ", ".join(row["model"] for row in ranked if row[field] == best[field])
        lines.append(f"- **{label}** — {leaders} ({best[field]:.6f})")
    lines += ["", "### Parameter counts and runtime", "",
              "| Model | Total parameters | Trainable parameters | Runtime (seconds) | GPU |",
              "| --- | ---: | ---: | ---: | --- |"]
    for row in ranked:
        if not math.isfinite(row["runtime"]) or row["runtime"] <= 0:
            raise ValueError("invalid full training runtime")
        lines.append(f'| {row["model"]} | {row["total_parameters"]:,} | {row["trainable_parameters"]:,} | '
                     f'{row["runtime"]:.2f} | {row["GPU"]} |')
    leaders = min(ranked, key=lambda row: row["runtime"]), min(ranked, key=lambda row: row["trainable_parameters"])
    lines += ["", "### Comparison", "",
              f'{ranked[0]["model"]} had the highest observed test ROC-AUC ({ranked[0]["test_auc"]:.6f}). '
              f'{leaders[0]["model"]} had the shortest recorded wall time. '
              f'{leaders[1]["model"]} had the fewest trainable parameters.', ""]
    hitskt = next(row for row in ranked if row["model"] == "HiTSKT")
    for rank, row in enumerate(ranked, 1):
        if row["model"] == "HiTSKT":
            lines.append(f'- **HiTSKT** ranked {rank} by ROC-AUC and recorded {row["test_f1"]:.6f} F1.')
        else:
            gap = row["test_auc"] - hitskt["test_auc"]
            lines.append(f'- **{row["model"]}** ranked {rank} by ROC-AUC, with a signed difference of '
                         f'{gap:+.6f} from HiTSKT. Its F1 was {row["test_f1"]:.6f}.')
    if comparison:
        lines += ["", comparison]
    lines += ["", "These ranks describe a single seeded run on one cohort. The held-out partitions are later "
              "sessions from the same students. The benchmark does not estimate uncertainty across seeds or "
              "generalization to unseen students.", "",
              f"[Result CSV]({dataset}_final_results.csv) · [Result JSON]({dataset}_final_results.json) · "
              f"[Validation loss](../report_graphs/{dataset}_loss_vs_epoch.png) · "
              f"[Validation ROC-AUC](../report_graphs/{dataset}_validation_auc_vs_epoch.png) · "
              f"[Trainable parameters](../report_graphs/{dataset}_parameter_comparison.png)", ""]
    path = Path("reports") / f"{dataset.upper()}_EXPERIMENT_REPORT.md"
    before, rest = path.read_text().split("<!-- RESULTS_START -->\n", 1)
    after = rest.split("<!-- RESULTS_END -->", 1)[1]
    write_bytes(path, (before + "<!-- RESULTS_START -->\n" + "\n".join(lines) +
                       "<!-- RESULTS_END -->" + after).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--skip-combined", action="store_true")
    args = parser.parse_args()
    if args.combined_only:
        rows = read_json(Path("reports/algebra2005_final_results.json"))["results"]
        update_combined(rows)
        return
    rows, verification = collect()
    if Path("reports/final_results.csv").read_bytes().splitlines()[0] != csv_bytes(rows).splitlines()[0]:
        raise ValueError("Algebra export does not match the historical CSV schema")
    write_bytes(Path("reports/algebra2005_final_results.csv"), csv_bytes(rows))
    write_json(Path("reports/algebra2005_final_results.json"), {
        "completed_experiments": len(rows), "expected_experiments": len(MODELS),
        "source": "independently replayed full GPU checkpoints",
        "source_target_manifests": verification["source_target_manifests"], "results": rows,
    })
    write_report(rows)
    if not args.skip_combined:
        update_combined(rows)


if __name__ == "__main__":
    main()
