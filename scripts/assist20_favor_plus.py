#!/usr/bin/env python3
"""Prepare, run one planned checkpoint stage, or verify recovery without updates."""
import argparse
import json
from pathlib import Path
from ktbench.favor_plus_assist20 import prepare,execute,verify
from ktbench.favor_plus_resource_telemetry import ResourceConflict


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    cpu=sub.add_parser('prepare');cpu.add_argument('--run-id',required=True)
    for name in ('stage','verify-cuda'):
        s=sub.add_parser(name);s.add_argument('--plan',type=Path,required=True)
        s.add_argument('--allocation',type=Path,required=True)
        s.add_argument('--update',type=int,choices=[5,10,20],required=True)
        s.add_argument('--execute-cuda',action='store_true',required=True)
    a=p.parse_args()
    try:
        if a.action=='prepare':r=prepare(a.run_id)
        else:
            record=json.loads(a.allocation.read_text())
            r=(execute if a.action=='stage' else verify)(a.plan,record,a.update)
    except ResourceConflict as exc:
        from ktbench.favor_plus_pilot_recovery import KIND
        from ktbench.favor_plus_real_pilot import host_memory
        from ktbench.favor_plus_pipeline import sha256,write_json,EXPERIMENT_ROOT
        failure=dict(status='FAIL',failure_type='ResourceConflict',message=str(exc),
            conflict_snapshot=exc.snapshot,host_memory=host_memory(),cuda_calls_by_handler=0,
            note='no automatic restart; pre-initialization admission or existing saved failure')
        print(json.dumps(failure,indent=2),flush=True)
        if a.plan.resolve().is_relative_to(EXPERIMENT_ROOT.resolve()):
            plan=json.loads(a.plan.read_text());root=a.plan.parent
            if plan.get('kind')==KIND and not (root/'study_failure.json').exists():
                failure['published_checkpoints']=[dict(path=str(p),sha256=sha256(p))
                    for p in sorted((root/'checkpoints').glob('checkpoint_*.pt'))]
                write_json(root/'study_failure.json',failure)
        raise
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
