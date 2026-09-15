#!/usr/bin/env python3
"""Build numerical tables and plots only from completed full benchmark runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ktbench.config import PROJECT_SEED
from scripts.aggregate_results import DATASETS, MODELS, aggregate, collect_results


METRICS = {
    "auc": ("ROC-AUC", "test_auc"),
    "accuracy": ("Accuracy", "test_accuracy"),
    "precision": ("Precision", "test_precision"),
    "recall": ("Recall", "test_recall"),
    "f1": ("F1", "test_f1"),
    "mse": ("MSE", "test_mse"),
    "loss": ("Loss", "test_loss"),
}
DISPLAY_DATASETS = ("ASSIST2017", "Junyi", "EdNet-KT1")


def _read_curves(row: dict[str, Any]) -> list[dict[str, str]]:
    path = Path(row["experiment_directory"]) / "metrics.csv"
    with path.open(newline="") as handle:
        curves = list(csv.DictReader(handle))
    for split in ("train", "validation"):
        epochs = [int(x["epoch"]) for x in curves if x["split"] == split]
        if epochs != list(range(1, int(row["epochs_completed"]) + 1)):
            raise ValueError(f"non-contiguous or duplicate {split} epoch records: {path}")
    targets = [x for x in curves if x["split"] == "test"]
    if len(targets) != 1 or int(targets[0]["epoch"]) != int(row["best_epoch"]):
        raise ValueError(f"test record does not use the single best checkpoint: {path}")
    for _, (_, key) in METRICS.items():
        csv_key = "roc_auc" if key == "test_auc" else key.removeprefix("test_")
        if float(targets[0][csv_key]) != float(row[key]):
            raise ValueError(f"test metric mismatch for {key}: {path}")
    validation = [x for x in curves if x["split"] == "validation"]
    best = max(validation, key=lambda x: float(x["roc_auc"]))
    if (
        int(best["epoch"]) != int(row["best_epoch"])
        or float(best["roc_auc"]) != float(row["best_validation_auc"])
    ):
        raise ValueError(f"best epoch/AUC mismatch: {path}")
    return curves


def generate_report(
    experiments_root: Path,
    output_root: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate sources before writing; missing values are never synthesized."""
    rows = collect_results(experiments_root)
    expected = len(MODELS) * len(DATASETS)
    if len(rows) != expected and not allow_partial:
        raise ValueError(f"final report requires {expected} completed runs; found {len(rows)}")
    if not rows:
        raise ValueError("no completed full experiments to report")
    if any(int(row["seed"]) != PROJECT_SEED for row in rows):
        raise ValueError("primary benchmark contains a non-project seed")
    lookup = {(row["model"], row["dataset"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate model/dataset results")
    curves = {key: _read_curves(row) for key, row in lookup.items()}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate(experiments_root, output_root)
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    partial = len(rows) != expected
    document = [
        "# Benchmark comparison" + (" — INCOMPLETE" if partial else ""),
        "",
        f"Completed full experiments: {len(rows)}/{expected}. Seed: {PROJECT_SEED}.",
        "",
        "All values come from completed full-run best-checkpoint test evaluations. "
        "Display values are rounded to six decimals; table CSVs retain source precision. "
        "Missing experiments remain blank (—); no values are estimated.",
        "",
        "Evaluation is standardized, not model contexts or hyperparameters: "
        "DKT/DKVMN use 200 interactions; SAKT uses 100; RKT uses 50; "
        "HiTSKT uses up to 15 complete prior sessions plus its target session. "
        "Baselines retain their established repository profiles. "
        "RKT is the paper-faithful performance-only Phi-relation variant with "
        "5-fold student-level cross-fitting, not the full text-relation method.",
        "",
    ]
    for name, (label, metric) in METRICS.items():
        document.extend([
            f"## {label}", "",
            "| Model | " + " | ".join(DISPLAY_DATASETS) + " |",
            "| --- | --- | --- | --- |",
        ])
        with (tables / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("Model", *DISPLAY_DATASETS))
            for model in MODELS:
                values = [lookup.get((model, dataset), {}).get(metric) for dataset in DATASETS]
                writer.writerow((model, *("" if x is None else x for x in values)))
                document.append("| " + " | ".join(
                    (model, *("—" if x is None else f"{x:.6f}" for x in values))
                ) + " |")
        document.append("")

    for (model, dataset), values in curves.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, metric, label in zip(axes, ("loss", "roc_auc"), ("Loss", "ROC-AUC")):
            for split in ("train", "validation"):
                selected = [x for x in values if x["split"] == split]
                ax.plot([int(x["epoch"]) for x in selected],
                        [float(x[metric]) for x in selected], label=split)
            ax.axvline(lookup[(model, dataset)]["best_epoch"], color="gray",
                       linestyle="--", label="best validation-AUC epoch")
            ax.set(xlabel="Epoch", ylabel=label)
            ax.legend(fontsize=7)
        fig.suptitle(f"{model} / {dataset} — seed {PROJECT_SEED}")
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(figures / f"{model.lower()}_{dataset}_curves.{extension}", dpi=150)
        plt.close(fig)

    for name in ("auc", "accuracy", "f1"):
        label, metric = METRICS[name]
        fig, ax = plt.subplots(figsize=(8, 4))
        for index, (dataset, display) in enumerate(zip(DATASETS, DISPLAY_DATASETS)):
            positions = [i for i, model in enumerate(MODELS) if (model, dataset) in lookup]
            ax.bar([i + (index - 1) * 0.24 for i in positions],
                   [lookup[(MODELS[i], dataset)][metric] for i in positions],
                   width=0.24, label=display)
        ax.set_xticks(range(len(MODELS)), MODELS)
        ax.set(ylim=(0, 1), ylabel=f"Test {label}",
               title=f"{label} — seed {PROJECT_SEED}" + (" (incomplete)" if partial else ""))
        ax.legend()
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(figures / f"comparison_{name}.{extension}", dpi=150)
        plt.close(fig)

    (output_root / "comparison.md").write_text("\n".join(document), encoding="utf-8")
    manifest = {
        "status": "incomplete" if partial else "complete",
        "completed_experiments": len(rows),
        "expected_experiments": expected,
        "seed": PROJECT_SEED,
        "source": "canonical completed full-run artifacts only",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for row in rows
            for name in ("config.json", "final_results.json", "metrics.csv")
            for path in (Path(row["experiment_directory"]) / name,)
        },
    }
    (output_root / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-root", type=Path, default=Path("reports"))
    parser.add_argument("--allow-partial", action="store_true",
                        help="explicitly label incomplete reports; never fill missing values")
    args = parser.parse_args()
    print(json.dumps(generate_report(args.experiments_root, args.output_root,
                                     allow_partial=args.allow_partial), indent=2))


if __name__ == "__main__":
    main()
