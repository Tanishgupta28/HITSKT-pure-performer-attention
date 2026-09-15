#!/usr/bin/env python3
"""Independently replay the best checkpoint of a completed full experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from ktbench.config import PROJECT_SEED, seed_everything
from ktbench.data.rolling_targets import RollingTargetDataset
from ktbench.data.session_store import RollingSessionDataset, SessionStore
from ktbench.models import HiTSKT, PaperFaithfulRKT, load_baseline_checkpoint


def evaluate_checkpoint(
    experiment_root: Path,
    store: SessionStore,
    *,
    prepared_root: Path | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    """Read-only verification; exact metric mismatches fail instead of rounding."""
    if not (experiment_root / "_SUCCESS").exists():
        raise ValueError("independent replay requires a completed experiment")
    config = json.loads((experiment_root / "config.json").read_text())
    result = json.loads((experiment_root / "final_results.json").read_text())
    stats = json.loads((experiment_root / "dataset_stats.json").read_text())
    if result.get("status") != "complete" or not result.get("best_checkpoint_reloaded"):
        raise ValueError("experiment does not record a completed best-checkpoint test")
    if config.get("seed") != PROJECT_SEED or result.get("seed") != PROJECT_SEED:
        raise ValueError("experiment seed does not match primary benchmark seed")
    for stat, metadata in (("students", "students"), ("interactions", "interactions"),
                           ("questions", "num_questions"), ("skills", "num_skills")):
        if int(stats[stat]) != int(store.metadata[metadata]):
            raise ValueError(f"session store mismatch for {stat}")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = experiment_root / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        int(checkpoint["epoch"]) != int(result["best_epoch"])
        or float(checkpoint["validation_roc_auc"]) != float(result["best_validation_auc"])
    ):
        raise ValueError("best checkpoint epoch/AUC does not match result")
    del checkpoint
    common = dict(device=device, epoch=int(result["best_epoch"]),
                  optimizer=None, workers=workers)
    started = time.perf_counter()
    model_name = result["model"]
    if model_name == "HiTSKT":
        from ktbench.hitskt_registry import HITSKT_COMMON_CONFIG
        from scripts.train_hitskt import _run_epoch

        for key in ("token_budget", "maximum_batch_size", "bucket_size"):
            if config[key] != HITSKT_COMMON_CONFIG[key]:
                raise ValueError(f"active HiTSKT batching differs from stored {key}")
        dataset = RollingSessionDataset(store, "test", history_sessions=config["history_sessions"])
        model = HiTSKT.from_checkpoint(checkpoint_path, map_location=device).to(device).eval()
        metrics = _run_epoch(model, dataset, **common)
    elif model_name == "RKT":
        from ktbench.rkt.data import RKTTargetDataset
        from ktbench.rkt.phi import CrossFittedPhiRepository
        from ktbench.rkt.settings import RKT_BATCH_SIZE
        from scripts.train_rkt import _run_epoch

        if prepared_root is None or not (prepared_root / "_SUCCESS").exists():
            raise ValueError("RKT replay requires the completed training-only Phi cache")
        if config["batch_size"] != RKT_BATCH_SIZE:
            raise ValueError("active RKT batch size differs from stored config")
        repository = CrossFittedPhiRepository(prepared_root, int(store.metadata["num_questions"]))
        if repository.metadata != config["phi"]:
            raise ValueError("RKT Phi cache metadata differs from training config")
        dataset = RKTTargetDataset(store, "test")
        model = PaperFaithfulRKT.from_checkpoint(checkpoint_path, map_location=device).to(device).eval()
        model.set_relation_parameters_trainable(False)
        metrics = _run_epoch(model, dataset, repository, **common)
    elif model_name in ("DKT", "DKVMN", "SAKT"):
        from scripts.train_baseline import _run_epoch

        dataset = RollingTargetDataset(store, "test", history_length=config["history_length"])
        model = load_baseline_checkpoint(checkpoint_path, map_location=device).to(device).eval()
        metrics = _run_epoch(model, dataset, batch_size=config["batch_size"],
                             gradient_clip=None, **common)
    else:
        raise ValueError(f"unsupported benchmark model: {model_name}")
    if metrics["targets"] != stats["target_counts"]["test"]:
        raise ValueError("independent test target count differs from stored data stats")
    differences = {key: {"stored": value, "replayed": metrics.get(key)}
                   for key, value in result["test"].items() if metrics.get(key) != value}
    if differences:
        raise ValueError(f"independent replay mismatch: {differences}")
    return {
        "verified": True,
        "verification": "fresh model loaded from best checkpoint; exact equality of all test metrics",
        "model": model_name,
        "dataset": result["dataset"],
        "seed": PROJECT_SEED,
        "best_epoch": result["best_epoch"],
        "test": metrics,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "session_store_metadata_sha256": hashlib.sha256((store.root / "metadata.json").read_bytes()).hexdigest(),
        "runtime_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("store_root", type=Path)
    parser.add_argument("--prepared-root", type=Path, help="required for RKT training-only Phi cache")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, help="persist successful verification as JSON")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.experiment_root, SessionStore(args.store_root),
                                 prepared_root=args.prepared_root, workers=args.workers)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
