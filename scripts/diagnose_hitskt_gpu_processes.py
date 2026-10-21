#!/usr/bin/env python3
"""One finite read-only NVML/proc snapshot; never CUDA, training or admission."""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ktbench.gpu_process_diagnostics import (cgroup_summary,diagnostic_decision,namespace_parent,process_evidence)
from ktbench.favor_plus_resource_telemetry import DEVICE_EXPOSURE,LIGHT_HOLD,device_snapshot,nvml_snapshot

UUID='MIG-d2fcfd79-720e-5218-82c9-0a132be68920'


def nvidia_metadata(uuid,pids):
    lib=ctypes.CDLL('libnvidia-ml.so.1')
    def check(code):
        if code:raise RuntimeError('NVML status'+str(code))
    check(lib.nvmlInit_v2());result={}
    try:
        handle=ctypes.c_void_p();check(lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(handle)))
        for name,fn in (('driver_version','nvmlSystemGetDriverVersion'),('nvml_version','nvmlSystemGetNVMLVersion')):
            data=ctypes.create_string_buffer(128);code=getattr(lib,fn)(data,len(data))
            result[name]=dict(status=code,value=data.value.decode(errors='replace') if code==0 else None)
        for name,fn in (('gpu_instance','nvmlDeviceGetGpuInstanceId'),('compute_instance','nvmlDeviceGetComputeInstanceId')):
            value=ctypes.c_uint();code=getattr(lib,fn)(handle,ctypes.byref(value));result[name]=dict(status=code,value=value.value if code==0 else None)
        parent=ctypes.c_void_p();code=lib.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle,ctypes.byref(parent))
        if code==0:
            data=ctypes.create_string_buffer(128);code=lib.nvmlDeviceGetUUID(parent,data,len(data))
            result['parent_uuid']=dict(status=code,value=data.value.decode() if code==0 else None)
        result['process_names']=[]
        for pid in sorted(set(pids)):
            data=ctypes.create_string_buffer(1024);code=lib.nvmlSystemGetProcessName(ctypes.c_uint(pid),data,len(data))
            # Names are evidence only; never sufficient to associate two PIDs.
            name=data.value.decode(errors='replace') if code==0 else None
            result['process_names'].append(dict(pid=pid,status=code,basename=Path(name).name if name else None))
        class Process(ctypes.Structure):
            _fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]
        fn=getattr(lib,'nvmlDeviceGetMPSComputeRunningProcesses_v2',None)
        if fn is None:result['mps_compute']={'status':'SYMBOL_UNAVAILABLE'}
        else:
            count=ctypes.c_uint();code=fn(handle,ctypes.byref(count),None)
            rows=[]
            if code in (0,7) and count.value<=128:
                if count.value:
                    data=(Process*count.value)();code=fn(handle,ctypes.byref(count),data)
                    if code==0:rows=[dict(pid=p.pid,gpu_instance=p.gpuInstanceId,compute_instance=p.computeInstanceId) for p in data[:count.value]]
                result['mps_compute']=dict(status=code,processes=rows)
            else:result['mps_compute']=dict(status=code,count=count.value)
        maps=[line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if 'libnvidia-ml.so' in line and line.split()[-1].startswith('/')]
        result['loaded_nvml_libraries']=sorted(set(maps))
    finally:check(lib.nvmlShutdown())
    return result


def collect():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('diagnostics require CUDA hidden')
    own=os.getpid();snapshot=device_snapshot();memory=nvml_snapshot(UUID)
    candidates={67123,2847964,own,1,*[r['pid'] for r in snapshot['holders']],*[r['pid'] for r in memory.get('compute_processes',[])]}
    if len(candidates)>128:raise RuntimeError('unexpected process count; diagnostic bounded to128 identities')
    gpu_meta=nvidia_metadata(UUID,candidates)
    smi={}
    for name,args in (('inventory',['-L']),('compute',['--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader,nounits'])):
        completed=subprocess.run(['/usr/bin/nvidia-smi',*args],text=True,capture_output=True,timeout=20)
        # No arbitrary process argv: NVIDIA process-name query contains executable names only.
        smi[name]=dict(returncode=completed.returncode,stdout=completed.stdout.strip(),stderr=completed.stderr.strip())
    mounts=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);fields=left.split();kind=right.split()[0]
        if kind=='proc' or fields[4]=='/sys/fs/cgroup/memory':
            mounts.append(dict(filesystem=kind,mountpoint=fields[4],root_sha256=hashlib.sha256(fields[3].encode()).hexdigest(),
                mount_options=fields[5],root_is_slash=fields[3]=='/'))
    base=Path('/sys/fs/cgroup/memory')
    result=dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),resource_uuid=UUID,
        own_pid=own,own_proc_self=os.readlink('/proc/self'),process_evidence=[process_evidence(pid) for pid in sorted(candidates)],
        namespace_parent=namespace_parent(),mounts=mounts,snapshot=snapshot,nvml=memory,nvidia_metadata=gpu_meta,nvidia_smi=smi,
        exposed_devices=sorted(p.name for p in DEVICE_EXPOSURE.iterdir()),
        host_paths={p:Path(p).exists() for p in ('/host/proc','/host','/run/containerd/containerd.sock','/var/run/docker.sock','/proc/driver/nvidia/clients')},
        mps_paths={p:Path(p).exists() for p in ('/tmp/nvidia-mps','/run/nvidia-mps','/var/run/nvidia-mps')},
        selected_environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')},
        host_cgroup={name:(base/name).read_text().strip() for name in ('memory.limit_in_bytes','memory.usage_in_bytes','memory.oom_control')},
        light_hold_status=json.loads(LIGHT_HOLD.read_text())['status'],light_hold_sha256=hashlib.sha256(LIGHT_HOLD.read_bytes()).hexdigest(),
        no_torch_import='torch' not in sys.modules,no_cuda_driver_import=not any('libcuda.so' in line for line in Path('/proc/self/maps').read_text().splitlines()),
        additional_optimizer_calls=0,signals_sent=0,allocation_or_job_changes=0,diagnostic_decision=diagnostic_decision(snapshot,memory))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=collect()
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('observed_utc','resource_uuid','nvml','namespace_parent','diagnostic_decision','no_torch_import','no_cuda_driver_import')},indent=2))
