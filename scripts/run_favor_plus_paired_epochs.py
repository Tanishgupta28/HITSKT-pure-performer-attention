#!/usr/bin/env python3
"""Two-arm driver. No automatic campaign, test evaluation or approval fallback."""
import argparse,json
from pathlib import Path
from ktbench.favor_plus_paired_epochs import prepare,profile,execute_epoch,verify_epoch,validate_best,campaign,compare_results,ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--run-id',required=True)
    q=sub.add_parser('compare');q.add_argument('--plan',type=Path,required=True)
    q=sub.add_parser('campaign');q.add_argument('--plan',type=Path,required=True);q.add_argument('--record',type=Path,required=True);q.add_argument('--execute-cuda',action='store_true',required=True)
    for action in ('profile','train-epoch','verify-epoch','validate-best'):
        q=sub.add_parser(action);q.add_argument('--plan',type=Path,required=True);q.add_argument('--record',type=Path,required=True);q.add_argument('--arm',choices=ARMS,required=True);q.add_argument('--execute-cuda',action='store_true',required=True)
        if action in ('train-epoch','verify-epoch'):q.add_argument('--epoch',type=int,choices=range(1,9),required=True)
    a=p.parse_args()
    if a.action=='prepare':result=prepare(a.run_id)
    elif a.action=='compare':result=compare_results(a.plan)
    elif a.action=='campaign':result=campaign(a.plan,a.record)
    else:
        record=json.loads(a.record.read_text())
        if a.action=='profile':result=profile(a.plan,record,a.arm)
        elif a.action=='validate-best':result=validate_best(a.plan,record,a.arm)
        else:result=(execute_epoch if a.action=='train-epoch' else verify_epoch)(a.plan,record,a.arm,a.epoch)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
