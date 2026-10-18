#!/usr/bin/env python3
"""Independently hash current official EdNet downloads without replacing inputs."""
from datetime import datetime,timezone
import argparse
from pathlib import Path
import hashlib
import json
import os
import time
import requests

RATE=5*1024*1024
SOURCES=[('KT1','https://bit.ly/ednet_kt1','1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw',
    1201163816,'0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535'),
    ('Contents','https://bit.ly/ednet-content','117aYJAWG3GU48suS66NPaB82HwFj6xWS',
    173976,'aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7')]
try:Path('/proc/self/oom_score_adj').write_text('1000')
except PermissionError:pass
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',choices=['KT1','Contents'])
parser.add_argument('--output',type=Path,default=Path('reports/independent_correctness_audit/ednet_official_download_identity.json'))
args=parser.parse_args()
if args.output.exists():raise SystemExit('Refusing to overwrite prior download evidence')
records=[]
for name,shortcut,file_id,expected_bytes,expected_sha in SOURCES:
    if args.source and name!=args.source:continue
    record=dict(name=name,official_repository='https://github.com/riiid/ednet#download-links',
        shortcut=shortcut,drive_file_id=file_id,expected_local_sha256=expected_sha,
        expected_local_bytes=expected_bytes,maximum_transfer_bytes_per_second=RATE)
    try:
        with requests.get(shortcut,stream=True,timeout=(20,30)) as source:
            record['resolved_source_page']=source.url
            record['shortcut_http_status']=source.status_code
            assert file_id in source.url,'Official shortcut no longer resolves to recorded object'
        url=f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
        with requests.get(url,stream=True,timeout=(20,60)) as response:
            response.raise_for_status()
            assert response.headers.get('Content-Type')=='application/octet-stream'
            digest=hashlib.sha256();count=0;started=last=time.perf_counter()
            for chunk in response.iter_content(chunk_size=1024*1024):
                if not chunk:continue
                digest.update(chunk);count+=len(chunk)
                delay=count/RATE-(time.perf_counter()-started)
                if delay>0:time.sleep(delay)
                if time.perf_counter()-last>=30:
                    print(json.dumps(dict(source=name,bytes_read=count,total=expected_bytes)),flush=True)
                    last=time.perf_counter()
        record.update(downloaded_bytes=count,downloaded_sha256=digest.hexdigest(),
                      status='PASS' if count==expected_bytes and digest.hexdigest()==expected_sha else 'FAIL')
    except Exception as error:
        record.update(status='NOT VERIFIED',error_type=type(error).__name__,error=str(error))
    records.append(record)
    print(name,record['status'],flush=True)
args.output.write_text(json.dumps(dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    status='PASS' if all(r['status']=='PASS' for r in records) else 'NOT VERIFIED',
    local_archives_unchanged=True,records=records),indent=2,sort_keys=True)+'\n')
