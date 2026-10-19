#!/usr/bin/env python3
"""Read-only preservation against 31886f6 plus captured original artifact hashes."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.diagnose_hitskt_common import original_inventory,sha,write,now

BASE='31886f6ceab1dbcb0243b40468855256ba8f1f5c'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();original=original_inventory()
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines()
    prior={name:hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+name])).hexdigest()
           for name in names if name not in original and name.startswith(('ktbench/','scripts/','tests/','reports/','experiments/hitskt_favor_plus_v1/'))}
    # Existing new-variant binary smoke artifacts were deliberately local-only.
    for dataset in ('assist2017','junyi','ednet_kt1'):
        root=Path('experiments/hitskt_favor_plus_v1')/dataset/'cpu_smoke_20261019_v1'
        manifest=json.loads((root/'smoke_manifest.json').read_text())
        prior[str(root/'smoke_model.pt')]=manifest['checkpoint_sha256']
        prior[str(root/'smoke_logits.npy')]=manifest['logits_sha256']
    unrelated=json.loads(Path('reports/favor_plus_v1/implementation_v1/preservation_handoff.json').read_text())['unrelated_files']
    failures=[]
    for kind,group in (('original',original),('prior_source_evidence_smokes',prior),('unrelated',unrelated)):
        for name,expected in group.items():
            actual=sha(name)
            if actual!=expected:failures.append(dict(kind=kind,path=name,expected=expected,actual=actual))
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    assert branch=='research/hitskt-favor-plus-v1'
    write(args.output,dict(status='PASS' if not failures else 'FAIL',utc=now(),base=BASE,
        branch=branch,original_count=len(original),prior_count=len(prior),originals=original,
        prior=prior,unrelated=unrelated,failures=failures,script_sha256=sha(__file__)))
    assert not failures
    print('PASS',len(original),'original and',len(prior),'prior source/evidence/smoke files preserved')


if __name__=='__main__':main()
