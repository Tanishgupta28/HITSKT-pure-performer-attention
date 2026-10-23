#!/usr/bin/env python3
"""Full ASSIST-only epoch run. No automatic test evaluation or budget approval."""
import argparse
import json
from pathlib import Path
from ktbench.favor_plus_assist_epochs import prepare,profile,execute_epoch,verify_epoch,validate_best


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--run-id',required=True)
    for action in ('profile','train-epoch','verify-epoch','validate-best'):
        p=sub.add_parser(action);p.add_argument('--plan',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
        p.add_argument('--execute-cuda',action='store_true',required=True)
        if action in ('train-epoch','verify-epoch'):p.add_argument('--epoch',type=int,choices=range(1,11),required=True)
    args=parser.parse_args()
    if args.action=='prepare':result=prepare(args.run_id)
    else:
        record=json.loads(args.record.read_text())
        if args.action=='profile':result=profile(args.plan,record)
        elif args.action=='validate-best':result=validate_best(args.plan,record)
        else:result=(execute_epoch if args.action=='train-epoch' else verify_epoch)(args.plan,record,args.epoch)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
