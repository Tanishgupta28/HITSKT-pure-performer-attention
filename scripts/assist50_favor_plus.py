#!/usr/bin/env python3
"""Exact20→50 continuation only; no fresh training or arbitrary step endpoints."""
import argparse
import json
from pathlib import Path
from ktbench.favor_plus_assist50 import prepare,execute,verify
from ktbench.favor_plus_continuation_recovery import KIND
from ktbench.favor_plus_pipeline import EXPERIMENT_ROOT,sha256,write_json
from ktbench.favor_plus_resource_telemetry import ResourceConflict


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    cpu=sub.add_parser('prepare');cpu.add_argument('--run-id',required=True)
    for name in ('stage','verify-cuda'):
        s=sub.add_parser(name);s.add_argument('--plan',type=Path,required=True)
        s.add_argument('--allocation',type=Path,required=True)
        s.add_argument('--update',type=int,choices=(30,40,50),required=True)
        s.add_argument('--execute-cuda',action='store_true',required=True)
    a=p.parse_args()
    try:
        result=prepare(a.run_id) if a.action=='prepare' else (execute if a.action=='stage' else verify)(
            a.plan,json.loads(a.allocation.read_text()),a.update)
    except ResourceConflict as exc:
        failure=dict(status='FAIL',failure_type='ResourceConflict',message=str(exc),conflict_snapshot=exc.snapshot,
            cuda_calls_by_handler=0,no_automatic_restart=True)
        if a.plan.resolve().is_relative_to(EXPERIMENT_ROOT.resolve()):
            root=a.plan.parent;plan=json.loads(a.plan.read_text())
            if plan.get('kind')==KIND and not (root/'study_failure.json').exists():
                failure['published_checkpoints']=[dict(path=str(q),sha256=sha256(q)) for q in sorted((root/'checkpoints').glob('checkpoint_*.pt'))]
                write_json(root/'study_failure.json',failure)
        print(json.dumps(failure,indent=2),flush=True);raise
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
