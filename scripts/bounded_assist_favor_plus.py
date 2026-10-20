#!/usr/bin/env python3
"""Explicit CPU review or one externally approved CUDA milestone; no automatic chaining."""
import argparse
import json
from pathlib import Path
from ktbench.favor_plus_bounded_assist import prepare_review, execute_stage

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    cpu=sub.add_parser('prepare-review');cpu.add_argument('--run-id',required=True)
    gpu=sub.add_parser('stage');gpu.add_argument('--run-id',required=True)
    gpu.add_argument('--plan',type=Path,required=True)
    gpu.add_argument('--allocation',type=Path,required=True)
    gpu.add_argument('--to',type=int,choices=[20,50,100],required=True)
    gpu.add_argument('--previous-checkpoint',type=Path)
    gpu.add_argument('--execute-cuda',action='store_true',required=True)
    args=parser.parse_args()
    if args.command=='prepare-review':
        result=prepare_review(args.run_id)
    else:
        result=execute_stage(args.plan,json.loads(args.allocation.read_text()),args.run_id,args.to,args.previous_checkpoint)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
