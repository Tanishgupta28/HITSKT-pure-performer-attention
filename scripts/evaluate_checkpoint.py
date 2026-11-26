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
from ktbench.models.rkt import RKT_PHI_ONLY


def evaluate_checkpoint(
    experiment_root: Path,
    store: SessionStore,
    *,
    prepared_root: Path | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    """Read-only verification; exact metric mismatches fail instead of rounding."""
    if not any((experiment_root / marker).is_file() for marker in ("_SUCCESS", "_TEST_EVALUATED")):
        raise ValueError("independent replay requires a fully evaluated experiment")
    config = json.loads((experiment_root / "config.json").read_text())
    result = json.loads((experiment_root / "final_results.json").read_text())
    stats = json.loads((experiment_root / "dataset_stats.json").read_text())
    gated = config.get("independent_replay_required") is True
    pending = gated and (experiment_root / "_TEST_EVALUATED").is_file()
    complete = (experiment_root / "_SUCCESS").is_file()
    if not ((complete and result.get("status") == "complete") or
            (pending and result.get("status") == "awaiting_independent_replay")):
        raise ValueError("independent replay requires a fully evaluated experiment")
    if not result.get("best_checkpoint_reloaded"):
        raise ValueError("experiment does not record a completed best-checkpoint test")
    if config.get("seed") != PROJECT_SEED or result.get("seed") != PROJECT_SEED:
        raise ValueError("experiment seed does not match primary benchmark seed")
    for stat, metadata in (("students", "students"), ("interactions", "interactions"),
                           ("questions", "num_questions"), ("skills", "num_skills")):
        if int(stats[stat]) != int(store.metadata[metadata]):
            raise ValueError(f"session store mismatch for {stat}")
    ordered = config["dataset"] == "assist2009"
    algebra = config["dataset"] == "algebra2005"
    if algebra:
        store_hash = hashlib.sha256((store.root / "metadata.json").read_bytes()).hexdigest()
        if (not torch.cuda.is_available() or not store.strict_timestamp_history
                or store_hash != config["session_store_metadata_sha256"]):
            raise ValueError("Algebra2005 replay requires CUDA and the original strict-time session store")
        if "cudnn_allow_tf32" in config:
            torch.backends.cudnn.allow_tf32 = config["cudnn_allow_tf32"]
    if ordered:
        if not torch.cuda.is_available():
            raise ValueError("ASSIST2009 verification requires CUDA")
        if "cudnn_allow_tf32" in config:
            torch.backends.cudnn.allow_tf32 = config["cudnn_allow_tf32"]
        if config.get("dataset_provenance") != store.provenance:
            raise ValueError("ASSIST2009 replay dataset differs from training")
    if gated:
        if (config["model"], config["dataset"]) != ("DKT", "ednet_kt1") or not torch.cuda.is_available():
            raise ValueError("gated EdNet DKT replay requires CUDA")
        store_hash = hashlib.sha256((store.root / "metadata.json").read_bytes()).hexdigest()
        if store_hash != config["session_store_metadata_sha256"]:
            raise ValueError("EdNet session store differs from training")
        torch.backends.cudnn.allow_tf32 = config["cudnn_allow_tf32"]
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
        dataset_class = RollingSessionDataset
        if ordered:
            from ktbench.data.assist2009_sessions import AssignmentTargets

            if (config["session_offsets_sha256"] != store.session_offsets_sha256
                    or config["session_protocol"] != store.metadata["session_protocol"]):
                raise ValueError("HiTSKT assignment boundary provenance differs from training")
            dataset_class = AssignmentTargets
        if algebra:
            from ktbench.data.algebra2005 import AlgebraSessionTargets

            dataset_class = AlgebraSessionTargets
        model = HiTSKT.from_checkpoint(checkpoint_path, map_location=device).to(device).eval()
        dataset = dataset_class(store, "test", history_sessions=config["history_sessions"])
        metrics = _run_epoch(model, dataset, verify_target_coverage=ordered or algebra, **common)
        if ordered or algebra:
            validation_set = dataset_class(store, "validation", history_sessions=config["history_sessions"])
            validation = _run_epoch(model, validation_set, verify_target_coverage=True, **common)
    elif model_name in ("RKT", RKT_PHI_ONLY):
        from ktbench.rkt.data import OrderedRKTTargets, RKTTargetDataset
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
        model = PaperFaithfulRKT.from_checkpoint(checkpoint_path, map_location=device).to(device).eval()
        if model.config.relation_only != (model_name == RKT_PHI_ONLY) or ordered != model.config.relation_only:
            raise ValueError("RKT checkpoint relation mode differs from this experiment")
        if ordered:
            dataset = OrderedRKTTargets(store, "test", repository)
            metrics = _run_epoch(model, dataset, repository, verify_target_coverage=True, **common)
            validation_set = OrderedRKTTargets(store, "validation", repository)
            validation = _run_epoch(model, validation_set, repository, verify_target_coverage=True, **common)
            model.set_relation_parameters_trainable(True)
        else:
            dataset = RKTTargetDataset(store, "test")
            metrics = _run_epoch(model, dataset, repository, verify_target_coverage=algebra, **common)
            if algebra:
                validation_set = RKTTargetDataset(store, "validation")
                validation = _run_epoch(model, validation_set, repository, verify_target_coverage=True, **common)
                model.set_relation_parameters_trainable(True)
    elif model_name in ("DKT", "DKVMN", "SAKT"):
        from scripts.train_baseline import _run_epoch

        dataset = (store.targets("test", config["history_length"]) if ordered
                   else RollingTargetDataset(store, "test", history_length=config["history_length"]))
        model = load_baseline_checkpoint(checkpoint_path, map_location=device).to(device).eval()
        settings = dict(batch_size=config["batch_size"], gradient_clip=None,
                        transport=config["transport"] if ordered or gated or algebra else "reference",
                        dkt_dense=config.get("dkt_dense_lstm", False),
                        verify_target_coverage=ordered or gated, **common)
        metrics = _run_epoch(model, dataset, **settings)
        if ordered or gated or algebra:
            validation_set = (store.targets("validation", config["history_length"]) if ordered
                              else RollingTargetDataset(store, "validation", history_length=config["history_length"]))
            validation = _run_epoch(model, validation_set, **settings)
    else:
        raise ValueError(f"unsupported benchmark model: {model_name}")
    if metrics["targets"] != stats["target_counts"]["test"]:
        raise ValueError("independent test target count differs from stored data stats")
    if ordered or gated or algebra:
        if validation["roc_auc"] != result["best_validation_auc"]:
            raise ValueError("independent replay validation ROC-AUC differs from best record")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for key, count in (("total_parameters", total), ("trainable_parameters", trainable)):
            if config[key] != count or result[key] != count:
                raise ValueError(f"independent {key} count differs from run artifacts")
    differences = {key: {"stored": value, "replayed": metrics.get(key)}
                   for key, value in result["test"].items() if metrics.get(key) != value}
    if differences:
        raise ValueError(f"independent replay mismatch: {differences}")
    return {
        **({"dataset_provenance": store.provenance, "total_parameters": total,
            "trainable_parameters": trainable, "validation": validation,
            "validation_and_test_target_ids_verified": True} if ordered else {}),
        **({"total_parameters": total, "trainable_parameters": trainable,
            "validation": validation, "validation_and_test_target_ids_verified": True} if gated or algebra else {}),
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
    parser.add_argument("--assignment-root", type=Path)
    parser.add_argument("--output", type=Path, help="persist successful verification as JSON")
    args = parser.parse_args()
    config = json.loads((args.experiment_root / "config.json").read_text())
    if config["dataset"] == "assist2009":
        from ktbench.data.assist2009 import Assist2009Store

        store = Assist2009Store(args.store_root)
        if config["model"] == "HiTSKT":
            from ktbench.data.assist2009_sessions import AssignmentRunStore

            if args.assignment_root is None:
                parser.error("ASSIST2009 HiTSKT replay requires --assignment-root")
            store = AssignmentRunStore.open(store, args.assignment_root)
    else:
        store = SessionStore(args.store_root)
    result = evaluate_checkpoint(args.experiment_root, store,
                                 prepared_root=args.prepared_root, workers=args.workers)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
