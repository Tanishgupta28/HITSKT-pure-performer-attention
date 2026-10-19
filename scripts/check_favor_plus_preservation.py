#!/usr/bin/env python3
"""Read-only preservation checks for the isolated FAVOR+ research branch."""
import argparse
import hashlib
from pathlib import Path
import subprocess
from scripts.diagnose_hitskt_common import original_inventory,sha,write,read,now

BASE='f465d84838e5f8b1c637736e362a945c20cfde10'
OUT=Path('reports/favor_plus_v1')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['baseline','handoff'],required=True);args=parser.parse_args()
    originals=original_inventory();prior={}
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines()
    for name in names:
        if name not in originals and name.startswith(('reports/','ktbench/','scripts/','tests/')):
            prior[name]=hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+name])).hexdigest()
    failures=[]
    for kind,expected in [('original_research',originals),('prior_source_and_verification',prior)]:
        for name,value in expected.items():
            actual=sha(name)
            if actual!=value:failures.append(dict(kind=kind,path=name,expected=value,actual=actual))
    unrelated={p:sha(p) for p in ['README.md','experiments/sakt/ednet_kt1/full/independent_evaluation.console.log']}
    if args.stage=='handoff':assert unrelated==read(OUT/'preservation_baseline.json')['unrelated_files']
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    assert branch=='research/hitskt-favor-plus-v1'
    write(OUT/f'preservation_{args.stage}.json',dict(utc=now(),status='PASS' if not failures else 'FAIL',
        base_commit=BASE,branch=branch,original_count=len(originals),prior_protected_count=len(prior),
        originals=originals,prior_source_and_verification=prior,failures=failures,unrelated_files=unrelated,
        implementation_sha256=sha(__file__)))
    assert not failures
    print('PASS:',len(originals),'original research files;',len(prior),'prior implementation/verification files')


if __name__=='__main__':main()
