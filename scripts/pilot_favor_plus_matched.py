#!/usr/bin/env python3
"""Prepare or explicitly approved matched20 comparison; no approval fallback."""
import argparse
from pathlib import Path
from ktbench.favor_plus_matched_pilot import prepare,execute,recover,ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','train','recover'));p.add_argument('--arm',choices=ARMS);p.add_argument('--record',type=Path);p.add_argument('--execute-optimizer',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        if not a.arm or not a.record or not a.execute_optimizer:p.error('new actual approval record and explicit execution flag required')
        (execute if a.action=='train' else recover)(a.arm,a.record)


if __name__=='__main__':main()
