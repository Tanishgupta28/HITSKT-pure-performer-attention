#!/usr/bin/env python3
"""Bounded synthetic pilot, separate from full training. Do not run on busy MIG.

CUDA requires an explicit independent-device confirmation after obtaining an
available device. This flag is an operator assertion, not automatic proof of
availability. No dataset read, implicit CUDA selection or full experiment.
"""
import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ktbench.batching import Session,SessionExample,collate_hitskt
from ktbench.config import seed_everything
from ktbench.favor_plus_pipeline import fresh_run,model_config,profile_for,runtime,sha256,source_identity,write_json
from ktbench.hitskt_registry import HITSKT_TRAINING_CONFIGS
from ktbench.metrics import masked_bce_loss
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus,VARIANT


def pilot(dataset,run_id,*,device,length=257,steps=2,independent_gpu_confirmed=False):
    if not (1 <= length <= 13080 and 1 <= steps <= 5):
        raise ValueError("pilot bounded to length 1..13080 and 1..5 steps")
    dev = torch.device(device)
    if dev.type not in ("cpu","cuda"):
        raise ValueError("pilot supports explicit CPU or CUDA device only")
    if dev.type == "cuda" and not independent_gpu_confirmed:
        raise ValueError("independent GPU availability must be established before any CUDA allocation")
    seed_everything()
    out = fresh_run(dataset,run_id)
    config = model_config(dataset,20,5)
    favor = FavorPlusConfig()
    net = HiTSKTFavorPlus(config,favor).to(dev)
    session = lambda n:Session([i%20+1 for i in range(n)],[i%5+1 for i in range(n)],[i%2 for i in range(n)])
    # Fixed B=1, history length <=33. Pilot sizes are synthetic diagnostics,
    # explicitly distinct from the unchanged full-training batching profile.
    batch = collate_hitskt([SessionExample([session(min(length,33))],session(length),0)],
                          num_questions=20,num_skills=5).to(dev)
    profile = profile_for(dataset)
    optimizer = torch.optim.Adam(net.parameters(),lr=profile["learning_rate"],weight_decay=profile["weight_decay"])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter(); losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_bce_loss(net(batch),batch.target_labels,batch.target_metric_mask)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise FloatingPointError("nonfinite pilot loss/gradients")
        optimizer.step(); losses.append(float(loss.detach().cpu()))
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    elapsed = time.perf_counter()-started
    net.eval()
    with torch.no_grad(): before = net(batch)
    torch.save(net.checkpoint(),out/"pilot_model.pt")
    restored = HiTSKTFavorPlus.from_checkpoint(out/"pilot_model.pt").to(dev).eval()
    with torch.no_grad(): after = restored(batch)
    delta = float((before-after).abs().max().cpu())
    if delta != 0: raise RuntimeError("pilot checkpoint replay changed logits")
    result = dict(status="PASS",model=VARIANT,kind="bounded_synthetic_pilot_only",dataset_profile=dataset,
        length=length,batch_size=1,steps=steps,losses=losses,elapsed_seconds=elapsed,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
        checkpoint_sha256=sha256(out/"pilot_model.pt"),checkpoint_max_abs_logit_difference=delta,
        favor_config=asdict(favor),model_config=asdict(config),runtime=runtime(dev),source=source_identity(),
        pilot_script_sha256=sha256(Path(__file__)),
        limitations="synthetic reduced vocabulary/B=1; no dataset AUC or full training")
    write_json(out/"pilot_manifest.json",result)
    (out/"_PILOT_COMPLETE").write_text("synthetic diagnostic only\n")
    return result


def main():
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",choices=HITSKT_TRAINING_CONFIGS)
    p.add_argument("--run-id",required=True)
    p.add_argument("--device",required=True)
    p.add_argument("--length",type=int,default=257)
    p.add_argument("--steps",type=int,default=2)
    p.add_argument("--independent-gpu-confirmed",action="store_true")
    args = p.parse_args()
    print(json.dumps(pilot(args.dataset,args.run_id,device=args.device,length=args.length,
        steps=args.steps,independent_gpu_confirmed=args.independent_gpu_confirmed),indent=2))


if __name__ == "__main__": main()
