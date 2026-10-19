#!/usr/bin/env python3
"""Bounded CPU numerical/approximation/memory diagnostics; no dataset training.

Run each benchmark in a separate process for an attributable peak RSS. The
parent executes them serially. Defaults cover the original maximum action
lengths plus EOS, using production head width and FAVOR+ feature count at B=1.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

from ktbench.favor_plus_pipeline import sha256,source_identity,write_json,runtime
from ktbench.models.favor_plus import causal_favor_attention,orthogonal_gaussian_projection


def benchmark(length: int, chunk: int, *, backward: bool = True) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("diagnostic requires CUDA_VISIBLE_DEVICES='' before process start")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    b,h,d,m = 1,2,64,128
    tensors = [torch.randn(b,h,length,d,requires_grad=backward)*.4 for _ in range(3)]
    tensors = [t.detach().requires_grad_(backward) for t in tensors]
    projection = orthogonal_gaussian_projection(h,m,d,42).float()
    mask = torch.ones(b,length,dtype=torch.bool)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    started = time.perf_counter()
    out = causal_favor_attention(*tensors,projection,query_mask=mask,key_mask=mask,chunk_size=chunk)
    forward_seconds = time.perf_counter()-started
    backward_seconds = None
    if backward:
        started = time.perf_counter()
        out.square().mean().backward()
        backward_seconds = time.perf_counter()-started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    finite = bool(torch.isfinite(out).all() and all(
        t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)) if backward else bool(torch.isfinite(out).all())
    if not finite: raise FloatingPointError("nonfinite long-sequence diagnostic")
    return dict(status="PASS",kind="synthetic_attention_only",length_including_eos=length,
                batch=b,heads=h,head_width=d,features=m,chunk_size=chunk,
                forward_seconds=forward_seconds,backward_seconds=backward_seconds,
                peak_rss_bytes=peak,peak_rss_increment_bytes=peak-baseline,
                naive_full_outer_prefix_bytes=b*h*length*m*d*4,
                finite_outputs_and_gradients=finite,runtime=runtime(torch.device("cpu")))


def approximation() -> list[dict]:
    """Prespecified norm/feature/seed grid. No hyperparameter selection on data."""
    torch.manual_seed(931)
    base_q,base_k = [torch.randn(1,1,24,8,dtype=torch.float64) for _ in range(2)]
    v = torch.randn(1,1,24,8,dtype=torch.float64)
    mask = torch.ones(1,24,dtype=torch.bool)
    results = []
    for scale in (.35,1.,3.):
        q,k = base_q*scale,base_k*scale
        score = q@k.transpose(-1,-2)/8**.5
        score.masked_fill_(~torch.ones(24,24,dtype=torch.bool).tril(),-torch.inf)
        exact = score.softmax(-1)@v
        for m in (8,32,128,256):
            errors = []
            for seed in range(8):
                omega = orthogonal_gaussian_projection(1,m,8,seed)
                out = causal_favor_attention(q,k,v,omega,query_mask=mask,key_mask=mask,chunk_size=16)
                errors.append(dict(seed=seed,mse=float((out-exact).square().mean()),
                                   max_abs=float((out-exact).abs().max())))
            results.append(dict(input_scale=scale,features=m,
                mean_mse=sum(e["mse"] for e in errors)/len(errors),per_seed=errors))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    p.add_argument("--length",type=int)
    p.add_argument("--chunk-size",type=int,default=32)
    args = p.parse_args()
    if args.length is not None:
        print(json.dumps(benchmark(args.length,args.chunk_size)))
        return
    if args.output is None: p.error("--output is required")
    results = []
    for length,chunk in ((939,32),(3925,32),(13081,16),(13081,32),(13081,64)):
        child = subprocess.run([sys.executable,__file__,"--length",str(length),"--chunk-size",str(chunk)],
                               check=True,text=True,capture_output=True,timeout=180)
        result = json.loads(child.stdout)
        results.append(result)
        print(json.dumps({k:result[k] for k in ("length_including_eos","chunk_size","forward_seconds","backward_seconds","peak_rss_bytes")}),flush=True)
    write_json(args.output,dict(status="PASS",scope="CPU synthetic, not GPU or dataset training",
        long_sequences=results,approximation=approximation(),source=source_identity(),
        limitations="B=1 attention only; full-model/batch/GPU performance still needs a bounded pilot"))


if __name__ == "__main__": main()
