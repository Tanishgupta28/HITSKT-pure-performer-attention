"""Audit ASSIST2012 target identity and historical HiTSKT architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from ktbench.models import HiTSKT, HiTSKTConfig
from scripts.verify_algebra2005_data import verify

SOURCES = ("ktbench/models/hitskt.py", "ktbench/models/performer.py")
REVISIONS = {"assist2017": "d3ff9ec", "junyi": "d6c0ff4", "ednet_kt1": "d499585"}


def check_output(*args):
    return subprocess.check_output(["git", *args])


def architecture():
    source = {name: Path(name).read_bytes() for name in SOURCES}
    runs = {}
    for dataset, revision in REVISIONS.items():
        commit = check_output("rev-parse", revision).decode().strip()
        for name, current in source.items():
            if check_output("show", f"{commit}:{name}") != current:
                raise ValueError(f"historical HiTSKT architecture differs in {name} at {commit}")
        run = Path("experiments/hitskt", dataset, "full")
        config = json.loads((run / "config.json").read_text())
        checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=True)
        model = HiTSKT(HiTSKTConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        total = sum(parameter.numel() for parameter in model.parameters())
        if (total != config["total_parameters"] or checkpoint["epoch"] != config["best_epoch"]
                or checkpoint["validation_roc_auc"] != config["best_validation_auc"]):
            raise ValueError(f"historical HiTSKT checkpoint identity mismatch for {dataset}")
        runs[dataset] = {"commit": commit, "checkpoint_format": checkpoint["format_version"],
                         "config": checkpoint["config"], "best_epoch": checkpoint["epoch"],
                         "total_parameters": total}
    return {"verified": True, "source_main_commit": check_output("rev-parse", "HEAD").decode().strip(),
            "runner": "scripts/train_hitskt.py", "model": "ktbench.models.hitskt.HiTSKT",
            "attention": "deterministic ELU+1 feature map and causal prefix sums",
            "sha256": {name: hashlib.sha256(value).hexdigest() for name, value in source.items()},
            "historical_checkpoints": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed", type=Path, nargs="?")
    parser.add_argument("--architecture", action="store_true")
    args = parser.parse_args()
    evidence = architecture() if args.architecture else verify(args.processed, source_id="problem_log_id")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
