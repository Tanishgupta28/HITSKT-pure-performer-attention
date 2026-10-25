#!/usr/bin/env python3
"""Bounded zero-update timing; no training/optimizer action."""
import argparse,json
from pathlib import Path
from ktbench.favor_plus_pair_profile_budget import prepare,profile

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','profile'));p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--record',type=Path);p.add_argument('--arm',choices=('orf128','orf512'));p.add_argument('--execute-cuda',action='store_true');a=p.parse_args()
    if a.action=='prepare':
        if not a.out:p.error('output required')
        result=prepare(a.plan,a.out)
    else:
        if not a.record or not a.arm or not a.execute_cuda:p.error('actual scoped record/arm/CUDA flag required')
        result=profile(a.plan,json.loads(a.record.read_text()),a.arm)
    print(json.dumps(result,indent=2))
