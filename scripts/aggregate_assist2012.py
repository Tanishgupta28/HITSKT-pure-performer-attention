"""Export ASSIST2012 only from all five verified full CUDA checkpoints."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from scripts.aggregate_algebra2005 import collect, csv_bytes, read_json, write_bytes, write_json, write_report
from scripts.aggregate_results import MODELS

DATASET = "assist2012"
REPORTS = Path("reports")


def append_to_index(stem, rows):
    csv_path = REPORTS / f"{stem}.csv"
    json_path = REPORTS / f"{stem}.json"
    content = csv_path.read_bytes()
    index = read_json(json_path)
    keys = lambda records: [(record["dataset"], record["model"]) for record in records]
    if keys(csv.DictReader(io.StringIO(content.decode()))) != keys(index["results"]):
        raise ValueError(f"{stem} CSV and JSON have different experiment identities")
    new = csv_bytes(rows)
    if content.splitlines()[0] != new.splitlines()[0]:
        raise ValueError("canonical report CSV headers differ")
    present = [row for row in index["results"] if row["dataset"] == DATASET]
    if present:
        if present != rows:
            raise ValueError(f"{stem} already contains a different ASSIST2012 result")
        return
    index["results"].extend(rows)
    index["completed_experiments"] += len(rows)
    index["expected_experiments"] += len(MODELS)
    if "source" in index:
        index["source"] += " and independently verified ASSIST2012 runs"
    write_bytes(csv_path, content.rstrip(b"\n") + b"\n" + new.split(b"\n", 1)[1])
    write_json(json_path, index)


def main():
    rows, verification = collect(DATASET)
    csv = csv_bytes(rows)
    if csv.splitlines()[0] != (REPORTS / "final_results.csv").read_bytes().splitlines()[0]:
        raise ValueError("ASSIST2012 CSV schema differs from the benchmark schema")
    write_bytes(REPORTS / "assist2012_final_results.csv", csv)
    write_json(REPORTS / "assist2012_final_results.json", {
        "completed_experiments": len(rows), "expected_experiments": len(MODELS),
        "source": "independently replayed full GPU checkpoints",
        "source_target_manifests": verification["source_target_manifests"], "results": rows,
    })
    hitskt = next(row for row in rows if row["model"] == "HiTSKT")
    entries = [(DATASET, hitskt["test_auc"])]
    for prior in ("assist2009", "algebra2005"):
        values = read_json(REPORTS / f"{prior}_final_results.json")["results"]
        entries.append((prior, next(r["test_auc"] for r in values if r["model"] == "HiTSKT")))
    aucs = ", ".join(f"{name.upper()} {auc:.6f}" for name, auc in entries)
    write_report(rows, DATASET, comparison=f"HiTSKT test ROC-AUC by dataset was {aucs}. "
                 "The comparisons are descriptive because the cohorts, clocks, and splits differ.")


if __name__ == "__main__":
    main()
