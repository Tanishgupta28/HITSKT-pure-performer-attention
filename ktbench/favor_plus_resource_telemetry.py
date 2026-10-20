"""Identity-first synchronous GPU admission. No torch, watcher, signals or CUDA.

A file-descriptor owner remains a conflict unless its actual root-owned binary
and complete argument structure establish a read-only management query. Capture
identity BEFORE descriptors so exiting queries do not lose their provenance.
"""
from __future__ import annotations
import ctypes
import datetime as dt
import hashlib
import os
from pathlib import Path
import re
import stat

from ktbench.favor_plus_gpu_access import (CUDA_VALIDATION, DEVICE_EXPOSURE, LIGHT_HOLD,
    require_gpu_authorization)

PROTECTED_NODES=frozenset(('/dev/nvidia0','/dev/nvidia-caps/nvidia-cap21','/dev/nvidia-caps/nvidia-cap22'))
MONITOR_PATHS=('/usr/bin/nvidia-smi','/usr/local/bin/nvidia-smi')
PILOT_ACTION='favor_plus_assist2017_recoverable_twenty_update_pilot'
RECOVERY_ACTION='favor_plus_assist2017_checkpoint_recovery_zero_updates'


def trusted_binaries(paths=None):
    result={}
    for name in MONITOR_PATHS if paths is None else paths:
        try:
            p=Path(name).resolve(strict=True);s=p.stat()
            if s.st_uid==0 and not s.st_mode & 0o022 and stat.S_ISREG(s.st_mode):
                result[str(p)]=dict(device=s.st_dev,inode=s.st_ino,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        except OSError:continue
    return result


def read_only_options(argv):
    """Parse every option, never accept a command by basename or partial flags."""
    flags=[];i=1
    while i<len(argv):
        arg=argv[i]
        if arg in ('-L','--list-gpus','-q','--query','-x','--xml-format'):
            flags.append(arg);i+=1;continue
        key,sep,value=arg.partition('=')
        if key not in ('--query-gpu','--query-compute-apps','--format','--id','-i'):
            return None
        if not sep:
            i+=1
            if i>=len(argv):return None
            value=argv[i]
        # A value cannot introduce another option; log only option names/hash.
        if not value or len(value)>2048 or value.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.,:/%-]+',value):
            return None
        flags.append(key);i+=1
    return flags


def start_ticks(proc):
    text=(proc/'stat').read_text()
    return int(text[text.rfind(')')+2:].split()[19])  # /proc stat field22


def process_identity(proc, trusted):
    row=dict(pid=int(proc.name),identity_complete=False,verified_read_only=False)
    try:
        first=start_ticks(proc)
        status=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
        row.update(start_time_ticks=first,uid=[int(x) for x in status['Uid'].split()],
            parent_pid=int(status['PPid']),state=status['State'].strip(),
            comm=(proc/'comm').read_text().strip()[:64])
        raw=(proc/'cmdline').read_bytes()
        argv=[part.decode('utf-8',errors='replace') for part in raw.split(b'\0') if part]
        exe=os.readlink(proc/'exe');exe_stat=(proc/'exe').stat()
        second=start_ticks(proc)
        if first!=second:
            row['identity_error']='PID reused during identity read';return row
        row.update(start_time_ticks=first,uid=[int(x) for x in status['Uid'].split()],
            parent_pid=int(status['PPid']),state=status['State'].strip(),
            comm=(proc/'comm').read_text().strip()[:64],executable=exe,
            executable_device=exe_stat.st_dev,executable_inode=exe_stat.st_ino,
            argv_sha256=hashlib.sha256(raw).hexdigest(),identity_complete=True)
        module=None
        if '-m' in argv:
            j=argv.index('-m')+1
            if j<len(argv) and re.fullmatch(r'(scripts|ktbench|torch|pytest|ipykernel)\.[A-Za-z0-9_.]{1,140}',argv[j]):module=argv[j]
        row['python_module']=module
        try:
            cwd=os.readlink(proc/'cwd')
            row['cwd_category']='HiTSKT' if cwd=='/workspace/capstone' else 'LightSAKT2' if cwd=='/workspace/LightSAKT2' else 'other'
        except OSError:row['cwd_category']='unavailable'
        binary=trusted.get(exe);options=read_only_options(argv) if argv else None
        if (binary is not None and binary['device']==exe_stat.st_dev and binary['inode']==exe_stat.st_ino
                and options is not None):
            row.update(verified_read_only=True,approved_option_names=options,
                trusted_binary_sha256=binary['sha256'])
    except (OSError,ValueError,KeyError) as exc:
        row['identity_error']=type(exc).__name__
    return row


def device_snapshot(proc_root=Path('/proc'),*,own_pid=None,trusted=None,after_identity=None):
    """Finite single scan; after_identity is a CPU-test race-injection callback."""
    own_pid=os.getpid() if own_pid is None else own_pid
    trusted=trusted_binaries() if trusted is None else trusted
    holders=[];light=[];inaccessible=[]
    for proc in sorted(proc_root.iterdir(),key=lambda p:int(p.name) if p.name.isdigit() else -1):
        if not proc.name.isdigit() or int(proc.name)==own_pid:continue
        identity=process_identity(proc,trusted)
        if (identity.get('cwd_category')=='LightSAKT2' and identity.get('comm','').startswith('python')
                and not identity.get('state','').startswith('Z')):light.append(identity)
        if after_identity is not None:after_identity(proc)
        try:fds=list((proc/'fd').iterdir())
        except FileNotFoundError:continue
        except PermissionError:
            inaccessible.append(int(proc.name));continue
        nodes=set()
        for fd in fds:
            try:
                target=os.readlink(fd)
                if target.startswith('/dev/nvidia'):nodes.add(target)
            except FileNotFoundError:continue
            except PermissionError:inaccessible.append(int(proc.name))
        if not nodes.intersection(PROTECTED_NODES):continue
        try:
            if start_ticks(proc)!=identity.get('start_time_ticks'):
                identity.update(verified_read_only=False,identity_complete=False,
                    identity_error='PID changed before descriptor observation')
            current_exe=(proc/'exe').stat()
            if (current_exe.st_dev!=identity.get('executable_device')
                    or current_exe.st_ino!=identity.get('executable_inode')):
                identity.update(verified_read_only=False,identity_complete=False,
                    identity_error='executable changed before descriptor observation')
        except FileNotFoundError:identity['exited_after_identity']=True
        except OSError:identity.update(verified_read_only=False,identity_error='process identity inaccessible after descriptors')
        except ValueError:identity.update(verified_read_only=False,identity_error='invalid process start time')
        classification='trusted_read_only_monitor' if identity['verified_read_only'] else 'unknown_or_compute_conflict'
        holders.append({**identity,'nvidia_handles':sorted(nodes),'classification':classification})
    conflicts=[r for r in holders if not r['verified_read_only']]
    return dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),own_pid=own_pid,
        holders=holders,conflicts=conflicts,light_processes=light,inaccessible_fd_pids=sorted(set(inaccessible)),
        visibility='visible PID namespace only; unknown/inaccessible owners fail closed; no arbitrary argv or environment stored')


class ResourceConflict(RuntimeError):
    def __init__(self,message,snapshot):
        super().__init__(message);self.snapshot=snapshot


def admit(record,action,*,snapshot_fn=device_snapshot):
    if action not in (PILOT_ACTION,RECOVERY_ACTION) or action not in record.get('user_approval',{}).get('actions',[]):
        raise ValueError('specific twenty-update/recovery user authorization required')
    resource=require_gpu_authorization(record,os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_VALIDATION)
    if not (DEVICE_EXPOSURE/resource).exists() or not LIGHT_HOLD.is_file():
        raise RuntimeError('exact transferred device exposure or LightSAKT2 hold missing')
    import json
    if json.loads(LIGHT_HOLD.read_text()).get('status')!='USER_TRAINING_HOLD_ACTIVE':
        raise RuntimeError('LightSAKT2 hold is not active')
    snapshot=snapshot_fn()
    if snapshot['conflicts'] or snapshot['inaccessible_fd_pids'] or snapshot['light_processes']:
        raise ResourceConflict('active unknown/compute GPU owner, inaccessible process or Light trainer',snapshot)
    return resource,snapshot


def nvml_snapshot(uuid):
    """MIG memory/compute-process observation; NVML never creates CUDA context."""
    lib=ctypes.CDLL('libnvidia-ml.so.1')
    def check(code):
        if code:raise RuntimeError(f'NVML status{code}')
    class Memory(ctypes.Structure):
        _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    class Process(ctypes.Structure):
        _fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),
                  ('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]
    check(lib.nvmlInit_v2())
    try:
        handle=ctypes.c_void_p();check(lib.nvmlDeviceGetHandleByUUID(uuid.encode(),ctypes.byref(handle)))
        memory=Memory();check(lib.nvmlDeviceGetMemoryInfo(handle,ctypes.byref(memory)))
        result=dict(total_bytes=memory.total,free_bytes=memory.free,used_bytes=memory.used)
        count=ctypes.c_uint(0)
        fn=getattr(lib,'nvmlDeviceGetComputeRunningProcesses_v2',None)
        if fn is None:return {**result,'compute_visibility':'API unavailable'}
        code=fn(handle,ctypes.byref(count),None)
        if code not in (0,7):return {**result,'compute_visibility':f'NVML status{code}'}
        if count.value>4096:raise RuntimeError('unexpected NVML process count')
        if count.value:
            values=(Process*count.value)();check(fn(handle,ctypes.byref(count),values))
            result['compute_processes']=[dict(pid=p.pid,used_gpu_memory_bytes=p.usedGpuMemory,
                gpu_instance=p.gpuInstanceId,compute_instance=p.computeInstanceId) for p in values[:count.value]]
        else:result['compute_processes']=[]
        return result
    finally:check(lib.nvmlShutdown())
