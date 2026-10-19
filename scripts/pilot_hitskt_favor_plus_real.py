#!/usr/bin/env python3
"""CPU prepare or separately approved CUDA execute of the real ASSIST2017 pilot.

No implicit device selection, CPU training, resume, epoch loop or test evaluation.
"""
import argparse
import json
from pathlib import Path

from ktbench.favor_plus_real_pilot import execute, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("prepare", help="CPU-only data/hash/collation preflight; zero updates")
    cpu.add_argument("--run-id", required=True)
    cpu.add_argument("--store", type=Path)
    cpu.add_argument("--steps", type=int, default=5)
    gpu = commands.add_parser("execute", help="NOT authorized by CPU preparation; requires new user approval")
    gpu.add_argument("--prepared-plan", type=Path, required=True)
    gpu.add_argument("--allocation-record", type=Path, required=True)
    gpu.add_argument("--cuda-validation", type=Path, required=True)
    gpu.add_argument("--run-id", required=True)
    gpu.add_argument("--execute-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare(args.run_id, store_path=args.store, steps=args.steps)
        print(json.dumps({"status": "PREPARED_NOT_RUN", "prepared_plan": str(path),
                          "optimizer_updates": 0, "cuda_status": "NOT RUN"}, indent=2))
    else:
        allocation = json.loads(args.allocation_record.read_text())
        print(json.dumps(execute(args.prepared_plan, allocation, args.cuda_validation, args.run_id), indent=2))


if __name__ == "__main__":
    main()
