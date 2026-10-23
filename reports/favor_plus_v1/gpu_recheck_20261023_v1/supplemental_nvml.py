"""One read-only management snapshot, with no torch or CUDA context."""
import ctypes as c
import datetime as dt
import json
from pathlib import Path
import sys


class Process(c.Structure):
    _fields_ = [('pid', c.c_uint), ('usedGpuMemory', c.c_ulonglong),
                ('gpuInstanceId', c.c_uint), ('computeInstanceId', c.c_uint)]


class Utilization(c.Structure):
    _fields_ = [('gpu', c.c_uint), ('memory', c.c_uint)]


lib = c.CDLL('libnvidia-ml.so.1')
assert lib.nvmlInit_v2() == 0
result = {'observed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
          'observations': [], 'cuda_initialized': False}
try:
    for uuid in ('MIG-d2fcfd79-720e-5218-82c9-0a132be68920',
                 'GPU-5a568e3c-e96c-169f-b063-f4478340ff51'):
        handle = c.c_void_p()
        code = lib.nvmlDeviceGetHandleByUUID(uuid.encode(), c.byref(handle))
        row = {'uuid': uuid, 'handle_status': code, 'queries': {}}
        result['observations'].append(row)
        if code:
            continue
        for name in ('nvmlDeviceGetComputeRunningProcesses_v2',
                     'nvmlDeviceGetGraphicsRunningProcesses_v2',
                     'nvmlDeviceGetMPSComputeRunningProcesses_v2'):
            count = c.c_uint()
            fn = getattr(lib, name)
            code = fn(handle, c.byref(count), None)
            observation = {'initial_status': code, 'initial_count': count.value}
            row['queries'][name] = observation
            if code not in (0, 7) or count.value > 128:
                continue
            if count.value:
                data = (Process * (count.value + 8))()
                count.value = len(data)
                code = fn(handle, c.byref(count), data)
                observation['read_status'] = code
                if code:
                    continue
                observation['processes'] = [dict(pid=p.pid, used_bytes=p.usedGpuMemory,
                    gpu_instance=p.gpuInstanceId, compute_instance=p.computeInstanceId)
                    for p in data[:count.value]]
            else:
                observation['processes'] = []
        utilization = Utilization()
        code = lib.nvmlDeviceGetUtilizationRates(handle, c.byref(utilization))
        row['utilization'] = {'status': code, 'gpu_percent': utilization.gpu if code == 0 else None,
                              'memory_percent': utilization.memory if code == 0 else None}
finally:
    assert lib.nvmlShutdown() == 0
result['no_torch_import'] = 'torch' not in sys.modules
result['no_cuda_driver_import'] = not any('libcuda.so' in line for line in Path('/proc/self/maps').read_text().splitlines())
with Path(__file__).with_name('SUPPLEMENTAL_NVML.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result, indent=2))
