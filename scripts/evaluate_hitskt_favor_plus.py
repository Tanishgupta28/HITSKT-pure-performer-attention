#!/usr/bin/env python3
"""Fresh tagged FAVOR+ checkpoint replay, separately hashed predictions/metrics."""
import argparse
import json
from pathlib import Path

from ktbench.data.session_store import SessionStore
from ktbench.favor_plus_pipeline import evaluate
from ktbench.hitskt_registry import HITSKT_TRAINING_CONFIGS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",choices=HITSKT_TRAINING_CONFIGS)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("store",type=Path)
    p.add_argument("--run-id",required=True)
    p.add_argument("--device",default="cpu")
    args = p.parse_args()
    result = evaluate(args.checkpoint,SessionStore(args.store),args.dataset,args.run_id,device=args.device)
    print(json.dumps(result,indent=2))
    if result["reproduction_status"] == "FAIL": raise SystemExit(1)


if __name__ == "__main__": main()
