#!/usr/bin/env python3
"""Prepare a NEW common-batch plan only. No GPU execution option."""
import argparse,json
from pathlib import Path
from ktbench.favor_plus_paired_rebatch import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-plan',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--maximum-batch-size',type=int,default=8)
    a=p.parse_args();print(json.dumps(prepare(a.parent_plan,a.run_id,a.maximum_batch_size),indent=2))
