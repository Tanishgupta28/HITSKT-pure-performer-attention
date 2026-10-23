"""Finite admission/release metadata; existing owner permission, no CUDA calls."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

from ktbench.favor_plus_assist50 import ACTION, admit, load_plan
from ktbench.favor_plus_gpu_access import CUDA_VALIDATION
from ktbench.favor_plus_resource_telemetry import device_snapshot, nvml_snapshot, LIGHT_HOLD, DEVICE_EXPOSURE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = ROOT / 'experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261023_v2/prepared_plan.json'
mode = sys.argv[1]
if mode not in ('admit', 'release'):
    raise ValueError('admit or release only')
record_path = HERE / 'allocation_record.json'
if mode == 'admit':
    _, contract, _, _ = load_plan(PLAN)
    record = json.loads((HERE.parent / 'assist20_recovery_20261020_v1/allocation_record.json').read_text())
    now = dt.datetime.now(dt.timezone.utc)
    record.update(job_id='local-assist50-continuation-20261023-v1', starts_utc=now.isoformat(),
        expires_utc=(now + dt.timedelta(hours=2)).isoformat(),
        owner_confirmation_evidence=str(HERE / 'CURRENT_ADMISSION_ASSESSMENT.md'),
        conflict_evidence=str(HERE / 'resource_admission.json'))
    record['continuation_approval'] = contract['user_approval']
    record['user_approval'] = dict(approved=True, evidence=str(HERE / 'CURRENT_ADMISSION_ASSESSMENT.md'),
                                   actions=[CUDA_VALIDATION, ACTION])
    with record_path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    uuid, snapshot = admit(record)
else:
    record = json.loads(record_path.read_text())
    uuid, snapshot = record['resource_uuid'], device_snapshot()
memory = nvml_snapshot(uuid)
base = Path('/sys/fs/cgroup/memory')
cgroup = {name: (base/name).read_text().strip() for name in
          ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.oom_control')}
result = dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), resource_uuid=uuid,
    allocation_authority='existing explicit owner-confirmed allocation/transfer and renewed bounded scope; no scheduler lease claimed',
    snapshot=snapshot, nvml=memory, host_cgroup=cgroup, cuda_initialized=False,
    light_hold_status=json.loads(LIGHT_HOLD.read_text())['status'],
    light_hold_sha256=hashlib.sha256(LIGHT_HOLD.read_bytes()).hexdigest(),
    exposed_devices=sorted(p.name for p in DEVICE_EXPOSURE.iterdir()))
result['status'] = 'PASS' if (not(snapshot['conflicts'] or snapshot['inaccessible_fd_pids'] or snapshot['light_processes'])
    and memory.get('compute_processes') == [] and memory['free_bytes'] >= 10*1024**3
    and int(cgroup['memory.limit_in_bytes'])-int(cgroup['memory.usage_in_bytes']) >= 2*1024**3) else 'STOP'
with (HERE / ('resource_admission.json' if mode == 'admit' else 'resource_release.json')).open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result, indent=2))
if result['status'] != 'PASS':
    raise SystemExit(1)
