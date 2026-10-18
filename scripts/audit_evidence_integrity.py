#!/usr/bin/env python3
"""Verify audit-source bindings, full prediction files and passing-test logs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    evidence=Path('reports/independent_correctness_audit')
    source_paths=(list(Path('scripts').glob('audit_*.py'))+
                  [Path('scripts/generate_independent_correctness_audit.py')]+
                  list(evidence.glob('*_source.py')))
    by_hash={}
    for path in source_paths:by_hash.setdefault(sha(path),[]).append(str(path))
    bindings=[]
    for path in sorted(evidence.glob('*.json')):
        value=json.loads(path.read_text())
        for key in ['implementation_sha256','evaluator_sha256','parallel_evaluator_sha256','source_sha256']:
            if key in value:
                assert value[key] in by_hash,(path,key,value[key])
                bindings.append(dict(artifact=str(path),field=key,sha256=value[key],matching_sources=by_hash[value[key]]))
        if 'postprocessing' in value:
            post=value['postprocessing']
            assert post['source_sha256'] in by_hash
            assert sha(post['emitted_manifest_preserved'])==post['emitted_manifest_sha256']
        if 'precheck_manifest_sha256' in value:
            # All continuations used the preserved first precheck manifest.
            precheck=evidence/f"{value['dataset']}_evaluation_manifest.json"
            assert sha(precheck)==value['precheck_manifest_sha256'],path
    predictions=[]
    for dataset in ['assist2017','junyi','ednet_kt1']:
        suffix='evaluation_manifest.json' if dataset=='assist2017' else 'evaluation_guarded_manifest.json'
        value=json.loads((evidence/f'{dataset}_{suffix}').read_text())
        assert value['status'] in ['PASS','FAIL'] and 'recalculated_metrics' in value
        assert value['recalculated_metrics']['targets']==value['expected_targets']
        assert value['all_supervised_test_targets_once']
        root=Path(value['prediction_artifacts_local'])
        for filename,expected in value['prediction_sha256'].items():
            path=root/filename
            assert sha(path)==expected,path
            predictions.append(dict(dataset=dataset,path=str(path),sha256=expected))
    tests=json.loads((evidence/'test_summary.json').read_text())
    for row in tests['new_test_artifacts']:
        assert sha(evidence/row['file'])==row['sha256']
    assert tests['original_tests_passed']+tests['independent_new_cases_passed']==133
    result=dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        source_bindings=bindings,prediction_files=predictions,
        test_summary_hashes_match=True,original_implementation_separately_checked='preservation_verification.json')
    (evidence/'evidence_integrity.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS:',len(bindings),'source bindings;',len(predictions),'full prediction files; passing-test artifacts')


if __name__=='__main__':main()
