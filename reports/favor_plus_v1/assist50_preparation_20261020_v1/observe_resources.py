"""One read-only observation, no allocation mutation, polling or CUDA context."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from ktbench.favor_plus_resource_telemetry import device_snapshot,nvml_snapshot,LIGHT_HOLD,DEVICE_EXPOSURE

HERE=Path(__file__).resolve().parent
UUID='MIG-d2fcfd79-720e-5218-82c9-0a132be68920'
snapshot=device_snapshot()
memory=nvml_snapshot(UUID)
inventory=subprocess.run(['nvidia-smi','-L'],text=True,capture_output=True,timeout=20)
hold=json.loads(LIGHT_HOLD.read_text())
base=Path('/sys/fs/cgroup/memory')
cgroup={name:(base/name).read_text().strip() for name in
    ('memory.limit_in_bytes','memory.usage_in_bytes','memory.oom_control')}
exposure=sorted(p.name for p in DEVICE_EXPOSURE.iterdir())
conflict_free=(not snapshot['conflicts'] and not snapshot['inaccessible_fd_pids']
    and not snapshot['light_processes'] and memory.get('compute_processes')==[])
result=dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),resource_uuid=UUID,
    snapshot=snapshot,nvml=memory,inventory=dict(returncode=inventory.returncode,
        stdout=inventory.stdout.strip(),stderr=inventory.stderr.strip()),
    device_exposure=exposure,host_cgroup=cgroup,workspace_free_bytes=shutil.disk_usage('/workspace').free,
    light_hold_status=hold['status'],light_hold_sha256=hashlib.sha256(LIGHT_HOLD.read_bytes()).hexdigest(),
    visible_conflict_free=conflict_free,
    resource_sufficient_now=(memory['free_bytes']>=10*1024**3 and
        int(cgroup['memory.limit_in_bytes'])-int(cgroup['memory.usage_in_bytes'])>=2*1024**3),
    allocation_authority='Existing explicit allocation-owner confirmation/transfer in conversation, preserved in assist20 OWNER_AND_STUDY_AUTHORIZATION.md; no new external scheduler evidence',
    separate_new_gpu_allocation_confirmed=False,
    new_continuation_approval='PENDING; old20 scope cannot authorize30 additional updates',
    cuda_initialized=False,signals_sent=0,research_jobs_modified=0,
    limitation='Finite visible-PID/NVML snapshot, not a scheduler reservation guarantee or future availability. Do not treat display monitors as compute. Fresh admission and scoped user approval required before any CUDA.')
with (HERE/'RESOURCE_OBSERVATION.json').open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
