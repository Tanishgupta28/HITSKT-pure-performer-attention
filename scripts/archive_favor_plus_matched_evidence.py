#!/usr/bin/env python3
"""Publish only new non-checkpoint receipts, with per-entry verification."""
import hashlib,io,json,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/favor_plus_v1/four_arm_cuda_comparison_20261024_v1'
NAMES=('four_arm_cuda_readiness_20261024_v1','four_arm_cuda_readiness_fp64_20261024_v2','matched_four_arm20_fp64_20261024_v1')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    archive=OUT/'NEW_RUN_RECEIPTS.tar.gz'
    if archive.exists():raise FileExistsError('never overwrite evidence archive')
    paths=[];checkpoints=[]
    for name in NAMES:
        for path in sorted((ROOT/'experiments/hitskt_favor_plus_v1/assist2017'/name).rglob('*')):
            if path.is_file():
                row=dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path))
                if path.suffix=='.pt':checkpoints.append(row)
                else:paths.append(row)
    with tarfile.open(archive,'w:gz') as tar:
        for row in paths:
            data=(ROOT/row['path']).read_bytes();entry=tarfile.TarInfo(row['path']);entry.size=len(data);entry.mode=0o644;entry.mtime=0
            tar.addfile(entry,io.BytesIO(data))
    with tarfile.open(archive,'r:gz') as tar:
        assert len(tar.getmembers())==len(paths)
        for row in paths:assert hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest()==row['sha256']
    value=dict(status='PASS_ALL_ENTRIES_ROUNDTRIP_SHA256',archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,entries=paths,local_checkpoint_binaries_not_published=checkpoints,
        initial_qkv='56 source arrays remain local; identities sealed in prepared plans and prior RAW_ARRAY_MANIFEST.json',optimizer_updates=0,training_approval='PENDING_EXPLICIT_USER_RESPONSE')
    (OUT/'ARCHIVE_MANIFEST.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(entries=len(paths),local_checkpoint_binaries=len(checkpoints),archive_bytes=archive.stat().st_size,status=value['status'])))


if __name__=='__main__':main()
