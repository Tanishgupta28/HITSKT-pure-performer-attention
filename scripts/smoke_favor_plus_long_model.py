#!/usr/bin/env python3
"""Bounded CPU full-hierarchy maximum-session synthetic check; no real training."""
import argparse
import os
import resource
import time

import torch

from ktbench.batching import Session,SessionExample,collate_hitskt
from ktbench.config import seed_everything
from ktbench.favor_plus_pipeline import model_config,runtime,source_identity,write_json
from ktbench.metrics import masked_bce_loss
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus


def session(length):
    return Session([i%20+1 for i in range(length)],[i%5+1 for i in range(length)],
                   [i%2 for i in range(length)])


def main():
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA must be hidden before process start")
    torch.set_num_threads(1)
    seed_everything()
    net = HiTSKTFavorPlus(model_config("ednet_kt1",20,5),FavorPlusConfig()).eval()
    examples = [SessionExample([session(21),session(4)],session(13080),0),
                SessionExample([session(3)],session(17),0)]
    batch = collate_hitskt(examples,num_questions=20,num_skills=5)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    start = time.perf_counter(); logits = net(batch); forward = time.perf_counter()-start
    loss = masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask)
    start = time.perf_counter(); loss.backward(); backward = time.perf_counter()-start
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    with torch.no_grad():
        solo = net(collate_hitskt([examples[1]],num_questions=20,num_skills=5))
    difference = float((logits.detach()[1,:17]-solo[0,:17]).abs().max())
    assert difference < 3e-6
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write_json(args.output,dict(status="PASS",kind="CPU_synthetic_full_model_no_optimizer_step",
        dataset_profile="ednet_kt1",target_lengths=[13080,17],history_lengths=[[21,4],[3]],
        features=128,chunk_size=32,forward_seconds=forward,backward_seconds=backward,
        peak_rss_bytes=peak,peak_rss_increment_bytes=peak-before,
        supervised_targets=int(batch.target_metric_mask.sum()),finite_outputs_and_gradients=True,
        companion_padding_max_abs_logit_difference=difference,
        runtime=runtime(torch.device("cpu")),source=source_identity(),
        limitations="CPU synthetic reduced vocabulary; not maximum history length/batch or CUDA"))
    print("PASS full model: targets=13097; finite forward/backward; peak RSS bytes:",peak)


if __name__ == "__main__": main()
