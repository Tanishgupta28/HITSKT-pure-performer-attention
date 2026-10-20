"""Finite metadata/NVML admission/release; no CUDA API or background watcher."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from ktbench.favor_plus_assist50 import ACTION,admit,load_plan
from ktbench.favor_plus_gpu_access import CUDA_VALIDATION
from ktbench.favor_plus_resource_telemetry import device_snapshot,nvml_snapshot,LIGHT_HOLD,DEVICE_EXPOSURE

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PLAN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261020_v2/prepared_plan.json'
mode=sys.argv[1] if len(sys.argv)>1 else 'admit'
if mode not in ('admit','release'):raise ValueError('admit or release only')
record_path=HERE/'allocation_record_v2.json'
if mode=='admit':
    # This operator execution window records existing owner permission and new
    # actual scope approval; it is not an allocation/scheduler command or lease.
    plan,contract,_,_=load_plan(PLAN)
    old=json.loads((HERE.parent/'assist20_recovery_20261020_v1/allocation_record.json').read_text())
    now=dt.datetime.now(dt.timezone.utc)
    old.update(job_id='local-assist50-continuation-20261020-v2',starts_utc=now.isoformat(),
        expires_utc=(now+dt.timedelta(hours=2)).isoformat(),
        owner_confirmation_evidence=str(HERE/'OWNER_AND_CONTINUATION_AUTHORIZATION.md'),
        conflict_evidence=str(HERE/'resource_admission_v2.json'))
    old['continuation_approval']=contract['user_approval']
    old['user_approval']=dict(approved=True,evidence=str(HERE/'OWNER_AND_CONTINUATION_AUTHORIZATION.md'),actions=[CUDA_VALIDATION,ACTION])
    with record_path.open('x') as f:json.dump(old,f,indent=2);f.write('\n')
    record=old;uuid,snapshot=admit(record)
else:
    record=json.loads(record_path.read_text());uuid=record['resource_uuid'];snapshot=device_snapshot()
memory=nvml_snapshot(uuid);base=Path('/sys/fs/cgroup/memory')
cgroup={name:(base/name).read_text().strip() for name in ('memory.limit_in_bytes','memory.usage_in_bytes','memory.oom_control')}
inventory=subprocess.run(['nvidia-smi','-L'],text=True,capture_output=True,timeout=20)
result=dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),resource_uuid=uuid,
    allocation_authority='existing explicit owner-confirmed research allocation/transfer; actual scoped30-extra-update approval; no scheduler/admin lease claimed',
    cuda_initialized=False,snapshot=snapshot,nvml=memory,host_cgroup=cgroup,
    light_hold_status=json.loads(LIGHT_HOLD.read_text())['status'],
    light_hold_sha256=hashlib.sha256(LIGHT_HOLD.read_bytes()).hexdigest(),
    device_exposure=sorted(p.name for p in DEVICE_EXPOSURE.iterdir()),
    inventory=dict(returncode=inventory.returncode,stdout=inventory.stdout.strip(),stderr=inventory.stderr.strip()),
    scope='PID namespace/NVML/exposure and preserved allocation-owner authority; no claim of independent scheduler visibility')
result['status']='PASS' if not(snapshot['conflicts'] or snapshot['inaccessible_fd_pids'] or snapshot['light_processes']) and memory.get('compute_processes')==[] and memory['free_bytes']>=10*1024**3 and int(cgroup['memory.limit_in_bytes'])-int(cgroup['memory.usage_in_bytes'])>=2*1024**3 else 'STOP'
with (HERE/('resource_admission_v2.json' if mode=='admit' else 'resource_release_v2.json')).open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
