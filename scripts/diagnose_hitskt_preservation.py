#!/usr/bin/env python3
"""Verify 375 original files plus every prior audit report/script/test against Git."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
from scripts.diagnose_hitskt_common import BASE,OUT,now,original_inventory,sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['initial','final'],required=True)
    args=parser.parse_args()
    try:Path('/proc/self/oom_score_adj').write_text('1000')
    except PermissionError:pass
    expected=original_inventory();previous={}
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines()
    for name in files:
        if (name.startswith('reports/independent_correctness_audit/') or name=='reports/HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md'
            or name.startswith('scripts/audit_') or name=='scripts/generate_independent_correctness_audit.py'
            or name in ['tests/test_independent_hitskt_audit.py','tests/test_audit_tool_oracles.py']):
            data=subprocess.check_output(['git','show',BASE+':'+name])
            previous[name]=hashlib.sha256(data).hexdigest()
    failures=[]
    for kind,items in [('original',expected),('previous_audit',previous)]:
        for name,value in sorted(items.items()):
            actual=sha(name)
            if actual!=value:failures.append(dict(kind=kind,path=name,expected=value,actual=actual))
    write(OUT/f'preservation_{args.stage}.json',dict(status='PASS' if not failures else 'FAIL',
        utc=now(),audit_base=BASE,original_files=len(expected),previous_audit_files=len(previous),
        original_expected=expected,previous_audit_expected=previous,failures=failures,
        unrelated_readme_sha256=sha('README.md'),unrelated_sakt_log_sha256=sha('experiments/sakt/ednet_kt1/full/independent_evaluation.console.log'),
        implementation_sha256=sha(__file__)))
    print(args.stage,'PASS' if not failures else 'FAIL',len(expected),'originals;',len(previous),'previous audit files',flush=True)
    assert not failures


if __name__=='__main__':main()
