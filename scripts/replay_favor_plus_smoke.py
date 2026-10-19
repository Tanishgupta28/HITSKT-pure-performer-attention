#!/usr/bin/env python3
"""Read-only fresh CPU-process replay of synthetic smoke checkpoint/logits."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ktbench.batching import Session,SessionExample,collate_hitskt
from ktbench.favor_plus_pipeline import sha256
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root",type=Path)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA must be hidden before process start")
    torch.set_num_threads(1)
    root = args.run_root
    manifest = json.loads((root/"smoke_manifest.json").read_text())
    if manifest["kind"] != "CPU_synthetic_smoke_only": raise ValueError("not a synthetic smoke run")
    assert sha256(root/"smoke_model.pt") == manifest["checkpoint_sha256"]
    assert sha256(root/"smoke_logits.npy") == manifest["logits_sha256"]
    examples = [SessionExample([Session([1,2,3],[1,2,3],[0,1,0])]*h,
        Session(list(range(1,n+1)),[1]*n,[i%2 for i in range(n)]),0) for n,h in ((7,2),(3,1))]
    batch = collate_hitskt(examples,num_questions=20,num_skills=5)
    net = HiTSKTFavorPlus.from_checkpoint(root/"smoke_model.pt").eval()
    with torch.no_grad(): logits = net(batch).numpy()
    saved = np.load(root/"smoke_logits.npy",allow_pickle=False)
    assert np.isfinite(logits).all() and np.array_equal(logits,saved)
    print(json.dumps(dict(status="PASS",scope="fresh CPU process; synthetic only",
                          dataset_profile=manifest["dataset_profile"],
                          max_abs_logit_difference=float(np.max(np.abs(logits-saved))),
                          checkpoint_sha256=manifest["checkpoint_sha256"])))


if __name__ == "__main__": main()
