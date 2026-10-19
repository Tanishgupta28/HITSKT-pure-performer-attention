#!/usr/bin/env python3
"""Preserve original research and all prior verification evidence from e1065c9."""
import argparse
import hashlib
from pathlib import Path
import subprocess
from scripts.diagnose_hitskt_common import original_inventory,read,sha,write,now

BASE='e1065c9c6215f9c676b43e33ee0d5f689bbe3fbb'
OUT=Path('reports/supervisor_verification')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['initial','final'],required=True);args=parser.parse_args()
    expected=original_inventory();prior={}
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines()
    for name in names:
        protect=(name.startswith(('reports/independent_correctness_audit/','reports/reproducibility_diagnostics/','reports/HITSKT_',
            'scripts/audit_','scripts/diagnose_hitskt_','scripts/generate_independent_correctness_audit',
            'scripts/generate_hitskt_diagnostic_reports','scripts/verify_hitskt_diagnostic_evidence')) or name in [
                'tests/test_independent_hitskt_audit.py','tests/test_audit_tool_oracles.py','tests/test_reproducibility_diagnostics.py'])
        if protect:prior[name]=hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+name])).hexdigest()
    failures=[]
    for kind,mapping in [('original',expected),('prior_verification',prior)]:
        for name,value in mapping.items():
            actual=sha(name)
            if actual!=value:failures.append(dict(kind=kind,path=name,expected=value,actual=actual))
    unrelated={p:sha(p) for p in ['README.md','experiments/sakt/ednet_kt1/full/independent_evaluation.console.log']}
    if args.stage=='final':assert unrelated==read(OUT/'preservation_initial.json')['unrelated_files']
    write(OUT/f'preservation_{args.stage}.json',dict(utc=now(),status='PASS' if not failures else 'FAIL',base_commit=BASE,
        original_count=len(expected),prior_verification_count=len(prior),original_expected=expected,prior_expected=prior,
        failures=failures,unrelated_files=unrelated,implementation_sha256=sha(__file__)))
    assert not failures
    print('PASS:',len(expected),'original research files;',len(prior),'prior verification files')


if __name__=='__main__':main()
