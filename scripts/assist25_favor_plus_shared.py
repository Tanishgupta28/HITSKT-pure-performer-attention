#!/usr/bin/env python3
"""Five-update shared CUDA pilot only. Never accepts26/30/40/50 endpoints."""
import argparse
import json
from pathlib import Path
from ktbench.favor_plus_pipeline import EXPERIMENT_ROOT,sha256,write_json
from ktbench.favor_plus_shared25 import PLAN_KIND,prepare,execute,verify


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    cpu=sub.add_parser('prepare');cpu.add_argument('--run-id',required=True)
    for action in ('stage','verify-cuda'):
        p=sub.add_parser(action);p.add_argument('--plan',type=Path,required=True)
        p.add_argument('--shared-record',type=Path,required=True);p.add_argument('--update',type=int,choices=(25,),required=True)
        p.add_argument('--execute-cuda',action='store_true',required=True)
    args=parser.parse_args()
    try:
        result=prepare(args.run_id) if args.action=='prepare' else (execute if args.action=='stage' else verify)(
            args.plan,json.loads(args.shared_record.read_text()),args.update)
    except BaseException as exc:
        if hasattr(args,'plan') and args.plan.resolve().is_relative_to(EXPERIMENT_ROOT.resolve()):
            root=args.plan.parent
            if (root.is_dir() and (root/'prepared_plan.json').exists() and not (root/'study_failure.json').exists()
                    and not (root/'study_result.json').exists() and json.loads((root/'prepared_plan.json').read_text()).get('kind')==PLAN_KIND):
                write_json(root/'study_failure.json',dict(status='FAIL',failure_type=type(exc).__name__,message=str(exc),
                    resource_snapshot=getattr(exc,'snapshot',None),no_automatic_restart=True,cuda_calls_by_handler=0,
                    published_checkpoints=[dict(path=str(p),sha256=sha256(p)) for p in sorted((root/'checkpoints').glob('*.pt'))]))
        raise
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
