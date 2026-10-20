"""One synchronous stdlib/NVML allocation observation; no PyTorch/CUDA context."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from ktbench.favor_plus_resource_telemetry import (admit,nvml_snapshot,trusted_binaries,
    PILOT_ACTION,LIGHT_HOLD,DEVICE_EXPOSURE,ResourceConflict)
HERE=Path(__file__).resolve().parent
record_path=HERE/'allocation_record.json'
# A release observation uses no authorization refresh or experiment launch.
if len(sys.argv)>1 and sys.argv[1]=='release':
 from ktbench.favor_plus_resource_telemetry import device_snapshot
 record=json.loads(record_path.read_text());snapshot=device_snapshot();uuid=record['resource_uuid']
else:
 now=dt.datetime.now(dt.timezone.utc)
 old=json.loads((HERE.parent/'assist_bounded_v1/allocation_record.json').read_text())
 old.update(job_id='local-assist20-recovery-20261020-v1',starts_utc=now.isoformat(),
  expires_utc=(now+dt.timedelta(hours=2)).isoformat(),
  owner_confirmation_evidence=str(HERE/'OWNER_AND_STUDY_AUTHORIZATION.md'),
  conflict_evidence=str(HERE/'resource_admission.json'))
 old['user_approval']=dict(approved=True,evidence=str(HERE/'OWNER_AND_STUDY_AUTHORIZATION.md'),
  actions=['favor_plus_cuda_validation',PILOT_ACTION,'favor_plus_assist2017_checkpoint_recovery_zero_updates'])
 # Fresh operator execution record, not an invented scheduler lease.
 with record_path.open('x') as f:json.dump(old,f,indent=2);f.write('\n')
 record=old
 try:uuid,snapshot=admit(record,PILOT_ACTION)
 except ResourceConflict as exc:
  with (HERE/'resource_admission.json').open('x') as f:json.dump(dict(status='STOP',snapshot=exc.snapshot,cuda_initialized=False),f,indent=2)
  raise
memory=nvml_snapshot(uuid)
base=Path('/sys/fs/cgroup/memory')
cgroup={name:(base/name).read_text().strip() for name in ('memory.limit_in_bytes','memory.usage_in_bytes','memory.oom_control')}
inventory=subprocess.run(['nvidia-smi','-L'],text=True,capture_output=True,timeout=20)
result=dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),resource_uuid=uuid,
 source='existing owner-confirmed research allocation; new operator window2h, not scheduler expiry',
 cuda_initialized=False,snapshot=snapshot,nvml=memory,host_cgroup=cgroup,
 light_hold_status=json.loads(LIGHT_HOLD.read_text())['status'],
 light_hold_sha256=hashlib.sha256(LIGHT_HOLD.read_bytes()).hexdigest(),
 device_exposure=sorted(p.name for p in DEVICE_EXPOSURE.iterdir()),trusted_monitor_binaries=trusted_binaries(),
 inventory=dict(returncode=inventory.returncode,stdout=inventory.stdout.strip(),stderr=inventory.stderr.strip()),
 disk_free_bytes=shutil.disk_usage('/workspace').free,
 visibility='PID namespace/NVML only; no scheduler metadata; historical Kubernetes403 is visibility limitation, owner attestation is permission')
result['status']='PASS' if (not snapshot['conflicts'] and not snapshot['inaccessible_fd_pids']
 and not snapshot['light_processes'] and not memory.get('compute_processes')
 and memory['free_bytes']>10*1024**3
 and int(cgroup['memory.limit_in_bytes'])-int(cgroup['memory.usage_in_bytes'])>2*1024**3) else 'STOP'
name='resource_release.json' if len(sys.argv)>1 and sys.argv[1]=='release' else 'resource_admission.json'
with (HERE/name).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
