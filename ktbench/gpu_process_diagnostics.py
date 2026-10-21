"""Read-only PID evidence collection, never GPU admission or alias acceptance.

No torch, CUDA driver, namespace entry, ptrace, process signals, arbitrary argv
or environment dumps. Incomplete/changed identity remains explicitly unproven.
"""
from __future__ import annotations
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import re

NS_GET_PARENT=0xb702
INTEGER_FIELDS=('Pid','Tgid','PPid','TracerPid')
VECTOR_FIELDS=('NSpid','NStgid','Uid')


def parse_status(text):
    values=dict(line.split(':',1) for line in text.splitlines() if ':' in line)
    result={}
    for key in INTEGER_FIELDS:
        if key in values:
            v=values[key].strip()
            if not re.fullmatch(r'[0-9]+',v):raise ValueError('invalid integer PID metadata')
            result[key]=int(v)
            if key in ('Pid','Tgid') and result[key]<=0:raise ValueError('invalid process PID')
    for key in VECTOR_FIELDS:
        if key in values:
            tokens=values[key].split()
            if not tokens or any(not re.fullmatch(r'[0-9]+',v) for v in tokens):
                raise ValueError('invalid namespace/UID vector')
            result[key]=[int(v) for v in tokens]
            if key!='Uid' and any(v<=0 for v in result[key]):raise ValueError('invalid namespace PID')
    if 'CapEff' in values:
        raw=values['CapEff'].strip()
        if not re.fullmatch(r'[0-9a-fA-F]+',raw):raise ValueError('invalid capability mask')
        capabilities=int(raw,16)
        result.update(cap_sys_admin=bool(capabilities & (1<<21)),cap_sys_ptrace=bool(capabilities & (1<<19)))
    return result


def cgroup_summary(text):
    """Record safe hierarchy identifiers/fingerprints, not unrelated file contents."""
    rows=[]
    for line in text.splitlines():
        fields=line.split(':',2)
        if len(fields)!=3:raise ValueError('invalid cgroup metadata')
        number,controllers,path=fields
        if not number.isdigit() or not path.startswith('/'):raise ValueError('invalid cgroup hierarchy')
        pod=re.search(r'pod([0-9a-f_\-]{36})(?:\.slice|/)',path)
        container=re.search(r'cri-containerd-([0-9a-f]{64})\.scope',path)
        rows.append(dict(hierarchy=int(number),controllers=controllers.split(',') if controllers else [],
            root_only=path=='/',path_sha256=hashlib.sha256(path.encode()).hexdigest(),
            kubepods=path.startswith('/kubepods'),pod_uid=pod.group(1).replace('_','-') if pod else None,
            container_id_sha256=hashlib.sha256(container.group(1).encode()).hexdigest() if container else None))
    return dict(sha256=hashlib.sha256(text.encode()).hexdigest(),hierarchies=rows)


def process_evidence(pid,proc_root=Path('/proc'),*,after_read=None):
    from ktbench.favor_plus_resource_telemetry import process_identity,start_ticks,trusted_binaries
    if type(pid) is not int or pid<=0:raise ValueError('positive integer PID required')
    proc=Path(proc_root)/str(pid)
    if not proc.exists():return dict(pid=pid,status='NOT_VISIBLE',mapping_proven=False)
    result=dict(pid=pid,status='UNPROVEN',mapping_proven=False)
    try:
        start=start_ticks(proc)
        result['identity']=process_identity(proc,trusted_binaries())
        result['selected_status']=parse_status((proc/'status').read_text())
        if (result['selected_status'].get('Pid')!=pid or not result['selected_status'].get('NSpid')
                or not result['identity'].get('identity_complete')):raise ValueError('incomplete process identity')
        result['cgroup']=cgroup_summary((proc/'cgroup').read_text())
        result['namespace_links']={name:os.readlink(proc/'ns'/name) for name in ('pid','pid_for_children','mnt','cgroup','user')}
        result['namespace_inodes']={name:(proc/'ns'/name).stat().st_ino for name in ('pid','mnt','cgroup','user')}
        if after_read is not None:after_read(proc)
        end_status=parse_status((proc/'status').read_text());exe=(proc/'exe').stat()
        if (start_ticks(proc)!=start or exe.st_dev!=result['identity']['executable_device']
                or exe.st_ino!=result['identity']['executable_inode']
                or (proc/'ns/pid').stat().st_ino!=result['namespace_inodes']['pid']
                or cgroup_summary((proc/'cgroup').read_text())!=result['cgroup']
                or any(end_status.get(k)!=result['selected_status'].get(k) for k in ('Pid','Tgid','NSpid','Uid'))):
            result['status']='CHANGED_OR_REUSED';return result
        result.update(status='VISIBLE_IDENTITY_SNAPSHOT',start_time_ticks=start)
    except (OSError,ValueError) as exc:
        result.update(status='INCOMPLETE_OR_EXITED',error_type=type(exc).__name__,errno=getattr(exc,'errno',None))
    return result


def namespace_parent(path=Path('/proc/self/ns/pid')):
    """NS_GET_PARENT only reads a relationship; never setns/unshare/mount."""
    fd=None;parent=None
    try:
        fd=os.open(path,os.O_RDONLY);parent=fcntl.ioctl(fd,NS_GET_PARENT)
        return dict(status='PARENT_FD_VISIBLE',link=os.readlink('/proc/self/fd/'+str(parent)),mapping_proven=False)
    except OSError as exc:
        return dict(status='UNAVAILABLE',errno=exc.errno,error=errno.errorcode.get(exc.errno,'UNKNOWN'),mapping_proven=False)
    finally:
        if parent is not None:os.close(parent)
        if fd is not None:os.close(fd)


def diagnostic_decision(snapshot,nvml):
    """No observations from this diagnostic can authorize GPU execution."""
    reasons=[]
    if snapshot.get('conflicts'):reasons.append('visible_unknown_or_compute_device_holder')
    if snapshot.get('inaccessible_fd_pids'):reasons.append('inaccessible_process_owner')
    if snapshot.get('light_processes'):reasons.append('LightSAKT2_process_present')
    if 'compute_processes' not in nvml:reasons.append('NVML_compute_visibility_unavailable')
    elif nvml['compute_processes']:reasons.append('active_NVML_compute_ownership_unproven')
    return dict(status='BLOCKED_ACTIVE_OR_UNATTRIBUTED_GPU_USE' if reasons else 'NO_COMPUTE_OBSERVED_NOT_ALLOCATION_PROOF',
        reasons=reasons,gpu_execution_allowed=False,pid_mapping_proven=False,
        policy='Evidence only; no numeric alias whitelist, process-name exemption or grant from free VRAM')
