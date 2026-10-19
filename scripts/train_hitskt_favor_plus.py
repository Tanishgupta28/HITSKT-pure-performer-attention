#!/usr/bin/env python3
"""Separate hitskt_favor_plus_v1 runner. Defaults to bounded CPU synthetic smoke."""
import argparse
import json
from pathlib import Path

from ktbench.favor_plus_pipeline import smoke,train
from ktbench.hitskt_registry import HITSKT_TRAINING_CONFIGS
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.data.session_store import SessionStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",choices=HITSKT_TRAINING_CONFIGS)
    p.add_argument("--run-id",required=True)
    p.add_argument("--mode",choices=("smoke","full"),default="smoke")
    p.add_argument("--device",default="cpu",help="explicit device; never automatically selects CUDA")
    p.add_argument("--store",type=Path)
    p.add_argument("--features",type=int,default=128)
    p.add_argument("--projection-seed",type=int,default=42)
    p.add_argument("--chunk-size",type=int,default=32)
    p.add_argument("--accumulation-dtype",choices=("float32","float64"),default="float32")
    p.add_argument("--smoke-steps",type=int,default=2)
    p.add_argument("--allow-full-training",action="store_true")
    args = p.parse_args()
    favor = FavorPlusConfig(features=args.features,projection_seed=args.projection_seed,
                            chunk_size=args.chunk_size,accumulation_dtype=args.accumulation_dtype)
    if args.mode == "smoke":
        if args.device != "cpu": p.error("current smoke mode is CPU-only")
        result = smoke(args.dataset,args.run_id,favor=favor,steps=args.smoke_steps)
    else:
        if not args.allow_full_training or args.store is None:
            p.error("full training needs --allow-full-training and --store after a separate decision")
        result = train(args.dataset,SessionStore(args.store),args.run_id,device=args.device,
                       favor=favor,allow_full_training=args.allow_full_training)
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
