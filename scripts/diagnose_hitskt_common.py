"""Read-only helpers for the follow-up diagnostics; no historical outputs written."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE='5625478ab0be344b145d2813939a71a0f6e49924'
OLD=Path('reports/independent_correctness_audit')
OUT=Path('reports/reproducibility_diagnostics')
DATASETS=('assist2017','junyi','ednet_kt1')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def now():return datetime.now(timezone.utc).isoformat()


def old_evaluation(dataset):
    suffix='evaluation_manifest.json' if dataset=='assist2017' else 'evaluation_guarded_manifest.json'
    return read(OLD/f'{dataset}_{suffix}')


def original_inventory():
    expected={r['path']:r['sha256'] for r in read(OLD/'benchmark_artifacts.json')['original_artifact_hashes']}
    expected.update((r['path'],r['sha256']) for r in read(OLD/'source_inventory.json')['original_sources'])
    for dataset in DATASETS:
        expected.update((r['path'],r['sha256']) for r in read(OLD/f'{dataset}_data_manifest.json')['processed_file_checksums'])
    for value in read(OLD/'rkt_cache_algebra.json')['records'].values():
        expected.update((r['path'],r['sha256']) for r in value['cache_files'])
    for path,value in [
        ('data/raw/assist2017/2017.csv','7577287d4ead073fdb9393b586a260043151f27338246bd076ca740e28be8bbb'),
        ('data/raw/junyi/Junyi.csv','8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a'),
        ('data/raw/ednet/archives/EdNet-KT1.zip','0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535'),
        ('data/raw/ednet/archives/EdNet-Contents.zip','aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7')]:expected[path]=value
    assert len(expected)==375
    return expected
