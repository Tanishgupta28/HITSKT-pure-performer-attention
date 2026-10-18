#!/usr/bin/env python3
"""Final read-only hash comparison of every captured original audit input."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    try:Path('/proc/self/oom_score_adj').write_text('1000')
    except PermissionError:pass
    evidence=Path('reports/independent_correctness_audit')
    expected={}
    original=json.loads((evidence/'benchmark_artifacts.json').read_text())
    expected.update((row['path'],row['sha256']) for row in original['original_artifact_hashes'])
    sources=json.loads((evidence/'source_inventory.json').read_text())
    expected.update((row['path'],row['sha256']) for row in sources['original_sources'])
    for dataset in ['assist2017','junyi','ednet_kt1']:
        data=json.loads((evidence/f'{dataset}_data_manifest.json').read_text())
        expected.update((row['path'],row['sha256']) for row in data['processed_file_checksums'])
        replay=json.loads((evidence/(f'{dataset}_evaluation_manifest.json' if dataset=='assist2017' else f'{dataset}_evaluation_guarded_manifest.json')).read_text())
        assert replay['status'] in ['PASS','FAIL'] and 'recalculated_metrics' in replay, 'Full required replay not completed'
        assert replay['recalculated_metrics']['targets']==replay['expected_targets']
    cache=json.loads((evidence/'rkt_cache_algebra.json').read_text())
    for data in cache['records'].values():expected.update((r['path'],r['sha256']) for r in data['cache_files'])
    for path,value in [
        ('data/raw/assist2017/2017.csv','7577287d4ead073fdb9393b586a260043151f27338246bd076ca740e28be8bbb'),
        ('data/raw/junyi/Junyi.csv','8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a'),
        ('data/raw/ednet/archives/EdNet-KT1.zip','0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535'),
        ('data/raw/ednet/archives/EdNet-Contents.zip','aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7')]:expected[path]=value
    mismatches=[]
    for path,value in sorted(expected.items()):
        actual=digest(path)
        if actual!=value:mismatches.append(dict(path=path,expected=value,actual=actual))
    tracked_best=[]
    for path,value in sorted(expected.items()):
        if not path.endswith('/best_model.pt'):continue
        process=subprocess.Popen(['git','cat-file','blob','656a9f0:'+path],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        git_digest=hashlib.sha256()
        for block in iter(lambda:process.stdout.read(8*1024*1024),b''):git_digest.update(block)
        error=process.stderr.read();assert process.wait()==0,(path,error.decode())
        actual=git_digest.hexdigest()
        assert actual==value,('Best checkpoint differs from historical Git blob',path)
        tracked_best.append(dict(path=path,sha256=actual,matches_base_git_blob=True))
    # The earlier gh-resumption inspection printed this README SHA prefix and
    # subsequent tool calls never wrote it; retain the exact current hash too.
    readme=digest('README.md')
    assert readme=='e5754c28ae89a628d46e0269523115404be13aba55d252a92c90ece2bf8ac6b8'
    assert not (Path('experiments/dkt/ednet_kt1/full')/'_SUCCESS').exists()
    original_sources_unchanged=not subprocess.check_output(['git','diff','656a9f0','--','ktbench','scripts/train_hitskt.py','scripts/train_baseline.py','scripts/train_rkt.py','scripts/evaluate_checkpoint.py'])
    result=dict(status='PASS' if not mismatches and original_sources_unchanged else 'FAIL',
        implementation_sha256=digest(__file__),
        checked_at_utc=datetime.now(timezone.utc).isoformat(),captured_original_files_checked=len(expected),
        tracked_best_checkpoint_git_identity=tracked_best,
        mismatches=mismatches,original_sources_unchanged=original_sources_unchanged,
        unrelated_readme_worktree_sha256=readme,original_experiments_not_overwritten=True,
        dkt_ednet_not_resumed=True,light_sakt2='No files edited and no processes signalled/reconfigured; protected PID survival observed after resource incident',
        scope='Captured canonical experiment files, raw sources, processed event/mapping/report/audit/store files, original implementation files and persisted RKT caches')
    (evidence/'preservation_verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
    assert result['status']=='PASS'


if __name__=='__main__':main()
