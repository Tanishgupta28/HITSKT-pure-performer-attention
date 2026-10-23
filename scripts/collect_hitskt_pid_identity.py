#!/usr/bin/env python3
"""Standalone CPU/procfs identity witness. Never authorizes GPU execution.

An administrator must attest the source/host scope. --view is only a label.
No NVIDIA API, torch, subprocess, process signals, namespace entry or polling.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

NAMESPACES = ('pid', 'pid_for_children', 'mnt', 'user', 'cgroup', 'time', 'time_for_children')


def read_bounded(path):
    with path.open('rb') as stream:
        value = stream.read(65537)
    if len(value) > 65536:
        raise ValueError('oversize proc metadata')
    return value.decode('utf-8', errors='strict')


def selected_status(text):
    wanted = ('Pid', 'Tgid', 'PPid', 'TracerPid', 'NSpid', 'NStgid', 'Uid')
    fields = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
    result = {}
    for key in wanted:
        parts = fields[key].split()
        if not parts or any(not re.fullmatch(r'[0-9]+', x) for x in parts):
            raise ValueError('invalid selected status')
        values = [int(x) for x in parts]
        if (key in ('Pid', 'Tgid', 'NSpid', 'NStgid') and min(values) <= 0
                or key == 'Uid' and len(values) != 4
                or key in ('Pid', 'Tgid', 'PPid', 'TracerPid') and len(values) != 1):
            raise ValueError('invalid selected status shape')
        result[key] = values if key in ('Uid', 'NSpid', 'NStgid') else values[0]
    return result


def namespace_links(proc):
    result = {}
    for name in NAMESPACES:
        try:
            link = os.readlink(proc/'ns'/name)
            if not re.fullmatch(r'[a-z_]+:\[[0-9]+\]', link):
                raise ValueError('invalid namespace link')
            result[name] = link
        except FileNotFoundError:
            if name not in ('time', 'time_for_children'):
                raise
            result[name] = None
    return result


def snapshot_pid(pid, proc_root=Path('/proc'), *, after_read=None):
    if type(pid) is not int or pid <= 0:
        raise ValueError('positive integer PID required')
    proc = proc_root/str(pid)

    def identity():
        stat = read_bounded(proc/'stat')
        # Field2 (comm) may contain spaces and closing parentheses.
        tail = stat[stat.rfind(')')+2:].split()
        start = int(tail[19])  # field22, not ps's rounded display time
        status = selected_status(read_bounded(proc/'status'))
        if status['Pid'] != pid or int(stat.split(' ', 1)[0]) != pid or start < 0:
            raise ValueError('PID/stat/status mismatch')
        cgroup = read_bounded(proc/'cgroup')
        exe = (proc/'exe').stat()
        optional = {}
        for name in ('uid_map', 'gid_map', 'timens_offsets'):
            try:
                optional[name] = read_bounded(proc/name).strip()
            except FileNotFoundError:
                optional[name] = None
        return dict(selected_status=status, start_time_ticks=start,
            namespaces=namespace_links(proc), cgroup_sha256=hashlib.sha256(cgroup.encode()).hexdigest(),
            container_ids=sorted(set(re.findall(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', cgroup))),
            pod_uids=sorted(set(x.replace('_', '-') for x in re.findall(r'pod([0-9a-f_-]{36})(?:/|\.slice)', cgroup))),
            executable_device=exe.st_dev, executable_inode=exe.st_ino, **optional)

    try:
        first = identity()
        if after_read is not None:
            after_read(proc)
        second = identity()
        return dict(pid=pid, status='STABLE_SNAPSHOT' if first == second else 'CHANGED_OR_REUSED',
                    identity=first, mapping_proven=False)
    except (OSError, ValueError, KeyError, IndexError, UnicodeError) as exc:
        return dict(pid=pid, status='INCOMPLETE_OR_EXITED', error_type=type(exc).__name__,
                    errno=getattr(exc, 'errno', None), mapping_proven=False)


def collect(pids, nonce, view, proc_root=Path('/proc')):
    if not re.fullmatch(r'[0-9a-f]{32,64}', nonce):
        raise ValueError('nonce must be32–64 lower-case hex characters')
    if view not in ('host', 'container') or not 1 <= len(pids) <= 64:
        raise ValueError('invalid view or bounded PID count')
    boot = read_bounded(proc_root/'sys/kernel/random/boot_id').strip()
    if not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', boot):
        raise ValueError('invalid boot identity')
    rows = [snapshot_pid(pid, proc_root) for pid in sorted(set(pids))]
    return dict(schema='hitskt_cpu_pid_witness_v1', observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        nonce=nonce, claimed_view=view, source_requires_administrator_attestation=True,
        boot_id=boot, clock_ticks_per_second=os.sysconf('SC_CLK_TCK'), reader_pid=os.getpid(),
        reader_namespaces=namespace_links(proc_root/'self'), proc1_namespaces=namespace_links(proc_root/'1'),
        processes=rows, mapping_proven=False, gpu_execution_allowed=False,
        no_torch_import='torch' not in sys.modules,
        no_cuda_driver_import=not any('libcuda.so' in line for line in read_bounded(proc_root/'self/maps').splitlines()),
        optimizer_calls=0, signals_sent=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', required=True, choices=('host', 'container'))
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--pid', action='append', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--hold-seconds', type=int, default=0,
                        help='optional foreground CPU witness lifetime,0–45s; no polling')
    args = parser.parse_args(argv)
    if not 0 <= args.hold_seconds <= 45 or args.hold_seconds and args.pid:
        parser.error('hold is bounded to0–45s and only for the collector itself')
    result = collect(args.pid or [os.getpid()], args.nonce, args.view)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)
    if args.hold_seconds:
        time.sleep(args.hold_seconds)  # one foreground pause, no watcher or GPU work
    return 0 if all(row['status'] == 'STABLE_SNAPSHOT' for row in result['processes']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
