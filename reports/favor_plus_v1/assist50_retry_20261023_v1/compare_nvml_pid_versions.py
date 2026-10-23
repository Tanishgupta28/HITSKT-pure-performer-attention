"""One authorized, bounded CUDA context; compare NVML interfaces, zero Adam."""
import ctypes as c
import json
import os
from pathlib import Path

from ktbench.favor_plus_assist50 import admit, original
from ktbench.favor_plus_resource_telemetry import nvml_snapshot
from ktbench.gpu_process_diagnostics import process_evidence

HERE = Path(__file__).resolve().parent
record = json.loads((HERE/'allocation_record.json').read_text())
uuid, before = admit(record)
memory = nvml_snapshot(uuid)
assert memory.get('compute_processes') == [] and memory['free_bytes'] >= 10*1024**3
host = original.check_host_resources()

import torch
capacity = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(8*1024**3/capacity, 0)
token = torch.empty(1, device='cuda:0')

class P1(c.Structure):
    _fields_ = [('pid', c.c_uint), ('usedGpuMemory', c.c_ulonglong)]

class P2(c.Structure):
    _fields_ = P1._fields_ + [('gpuInstanceId', c.c_uint), ('computeInstanceId', c.c_uint)]

lib = c.CDLL('libnvidia-ml.so.1')
assert lib.nvmlInit_v2() == 0
rows = {}
try:
    handle = c.c_void_p()
    assert lib.nvmlDeviceGetHandleByUUID(uuid.encode(), c.byref(handle)) == 0
    for name, typ in [('nvmlDeviceGetComputeRunningProcesses', P1),
                      ('nvmlDeviceGetComputeRunningProcesses_v2', P2),
                      ('nvmlDeviceGetComputeRunningProcesses_v3', P2)]:
        fn = getattr(lib, name, None)
        if fn is None:
            rows[name] = dict(status='SYMBOL_UNAVAILABLE')
            continue
        count = c.c_uint()
        code = fn(handle, c.byref(count), None)
        row = dict(initial_status=code, count=count.value)
        if code in (0, 7) and 0 < count.value <= 64:
            data = (typ*(count.value+8))()
            count.value = len(data)
            code = fn(handle, c.byref(count), data)
            row['read_status'] = code
            if code == 0:
                row['processes'] = [dict(pid=p.pid, used_bytes=p.usedGpuMemory) for p in data[:count.value]]
        elif code == 0 and count.value == 0:
            row['processes'] = []
        rows[name] = row
finally:
    assert lib.nvmlShutdown() == 0
pids = {os.getpid(), *[p['pid'] for row in rows.values() for p in row.get('processes', [])]}
result = dict(resource_uuid=uuid, own_pid=os.getpid(), source='live CUDA context with native NVML metadata',
    nvml_interfaces=rows, identity=[process_evidence(p) for p in sorted(pids)],
    before_snapshot=before, before_memory=memory, host_before=host,
    cuda_context_created=True, requested_tensor_bytes=4, optimizer_calls=0,
    training_forward_backward_calls=0, guard_modified=False, identity_exception_installed=False)
with (HERE/'NVML_INTERFACE_COMPARISON.json').open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps({k:result[k] for k in ('own_pid','nvml_interfaces','optimizer_calls')}, indent=2))
# No post-conflict CUDA query, training, checkpoint or explicit retry.
