#!/usr/bin/env python3
"""Prepare an unapproved budget template or run an actually approved7h parent."""
import argparse,json
from pathlib import Path
from ktbench.favor_plus_pair_long_window import template,run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('template','campaign'));p.add_argument('--plan',type=Path,required=True);p.add_argument('--scope',type=Path,required=True);p.add_argument('--execute-training',action='store_true');a=p.parse_args()
    if a.action=='template':result=template(a.plan,a.scope)
    else:
        if not a.execute_training:p.error('actual newtrainingapproval and explicittrainingflag required')
        result=run(a.plan,a.scope)
    print(json.dumps(result,indent=2))
