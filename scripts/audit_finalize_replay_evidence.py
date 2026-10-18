#!/usr/bin/env python3
"""Attach actual supplemental worker checks to a completed audit-only manifest.

Preserve its emitted version. No predictions, metrics or experiment files change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['junyi','ednet_kt1'])
    args=parser.parse_args()
    root=Path('reports/independent_correctness_audit')
    path=root/f'{args.dataset}_evaluation_guarded_manifest.json'
    backup=path.with_name(path.stem+'_before_annotation.json')
    supplement=root/f'{args.dataset}_guarded_worker_direct_crosscheck.json'
    value=json.loads(path.read_text())
    check=json.loads(supplement.read_text())
    assert value['status'] in ['PASS','FAIL'] and 'recalculated_metrics' in value
    assert value['all_supervised_test_targets_once']
    assert value['recalculated_metrics']['targets']==value['expected_targets']
    assert check['status']=='PASS' and len(check['checks'])==3
    assert check['checkpoint_sha256']==value['historical_artifact_sha256']['best_model.pt']
    assert not backup.exists() and 'postprocessing' not in value
    backup.write_bytes(path.read_bytes())
    # The emitted zero was never measured in the resumed main-loop branch.
    assert value['preserved_prefix']['batches']>3
    value['first_three_batch_worker_direct_max_difference']=None
    value['main_loop_worker_direct_probe_performed']=False
    value['worker_direct_crosscheck']={
        'status':'PASS','artifact':str(supplement),'sha256':sha(supplement),
        'scope':'First three newly evaluated batches after preserved prefix; separate fresh process',
        'maximum_absolute_logit_difference':max(r['maximum_absolute_logit_difference'] for r in check['checks'])}
    if 'reason' in value:
        value['precheck_interruption_reason']=value.pop('reason')
    value['postprocessing']={
        'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'emitted_manifest_preserved':str(backup),'emitted_manifest_sha256':sha(backup),
        'changes':'Clarify unperformed default parity field and inherited interruption reason; link actual supplemental probe',
        'predictions_and_metrics_unchanged':True}
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(args.dataset,value['status'],'supplemental probe PASS; metrics unchanged')


if __name__=='__main__':main()
