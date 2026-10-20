"""One synchronous read-only admission/release observation; never imports torch."""
import ctypes
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('light_snapshot','/workspace/LightSAKT2/scripts/capture_priority_handoff.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
processes=module.process_snapshot(Path('/workspace/LightSAKT2'))
from ktbench.favor_plus_gpu_access import conflicting_device_processes
conflicts=conflicting_device_processes()
hold_path=Path('/workspace/LightSAKT2/experiments/USER_TRAINING_HOLD.json')
hold=json.loads(hold_path.read_text())
lib=ctypes.CDLL('libnvidia-ml.so.1')
def check(result):
    if result:raise RuntimeError(f'NVML error{result}')
class Memory(ctypes.Structure):
    _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
uuid='MIG-d2fcfd79-720e-5218-82c9-0a132be68920'
check(lib.nvmlInit_v2())
try:
    handle=ctypes.c_void_p();check(lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(handle)))
    memory=Memory();check(lib.nvmlDeviceGetMemoryInfo(handle,ctypes.byref(memory)))
finally:check(lib.nvmlShutdown())
inv=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=20)
base=Path('/sys/fs/cgroup/memory')
cgroup={name:(base/name).read_text().strip() for name in ('memory.limit_in_bytes','memory.usage_in_bytes','memory.oom_control')}
result=dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    authority='owner explicitly confirms research allocation and authorizes this bounded100-update study; no scheduler proof invented',
    cuda_initialized=False,processes=processes,conflicting_visible_gpu_processes=conflicts,conflict_policy='unchanged production guard excludes only strictly verified read-only NVIDIA queries; raw handle snapshot and admission observation are sequential',light_hold_active=hold.get('status') == 'USER_TRAINING_HOLD_ACTIVE',
    light_hold_sha256=__import__('hashlib').sha256(hold_path.read_bytes()).hexdigest(),
    resource_uuid=uuid,device_exposure=sorted(p.name for p in Path('/var/run/nvidia-container-devices').iterdir()),
    inventory=dict(returncode=inv.returncode,stdout=inv.stdout.strip(),stderr=inv.stderr.strip()),
    memory=dict(total_bytes=memory.total,free_bytes=memory.free,used_bytes=memory.used),memory_cgroup=cgroup,
    disk_free_bytes=shutil.disk_usage('/workspace').free,
    known_conflicting_reservations=False,known_explicit_restrictions=[],
    visibility='PID/container-visible processes; Kubernetes403 inherited; no independent scheduler metadata; owner statement is allocation authority')
result['admission_status']='PASS' if (not processes['project_module_processes'] and not conflicts
    and processes['inaccessible_fd_directories']==0 and result['light_hold_active'] and uuid in result['device_exposure']
    and memory.free>10*1024**3 and int(cgroup['memory.limit_in_bytes'])-int(cgroup['memory.usage_in_bytes'])>2*1024**3) else 'STOP'
path=HERE/(sys.argv[1] if len(sys.argv)>1 else 'resource_admission.json')
with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
