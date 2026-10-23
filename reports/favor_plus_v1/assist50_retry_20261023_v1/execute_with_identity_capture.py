"""Single unchanged stage; on conflict collect CPU metadata, then re-raise."""
import json
import os
from pathlib import Path
import sys
from ktbench.favor_plus_assist50 import execute
from ktbench.favor_plus_resource_telemetry import ResourceConflict
from ktbench.gpu_process_diagnostics import process_evidence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261023_v2/prepared_plan.json'
record = json.loads((HERE/'allocation_record.json').read_text())
try:
    result = execute(PLAN, record, 30)
except ResourceConflict as exc:
    try:
        pids = {os.getpid(), *[r['pid'] for r in exc.snapshot.get('nvml', {}).get('compute_processes', [])]}
        libraries = sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                               if any(x in line for x in ('libcuda.', 'libnvidia-', 'libcudart.'))))
        fdinfo = []
        for fd in Path('/proc/self/fd').iterdir():
            try:
                target = os.readlink(fd)
                if target.startswith('/dev/nvidia'):
                    text = (Path('/proc/self/fdinfo')/fd.name).read_text()
                    fdinfo.append(dict(fd=int(fd.name), target=target, metadata=text[:8192]))
            except OSError:
                continue
        evidence = dict(own_pid=os.getpid(), process_evidence=[process_evidence(p) for p in sorted(pids)],
            scheduler_header=Path('/proc/self/sched').read_text().splitlines()[0],
            library_paths=libraries, own_nvidia_fdinfo=fdinfo, conflict_snapshot=exc.snapshot,
            capture_is_cpu_only=True, cuda_calls_by_capture=0, additional_optimizer_calls=0,
            guard_exception_suppressed=False)
        with (HERE/'LIVE_FAILURE_IDENTITY.json').open('x') as f:
            json.dump(evidence, f, indent=2)
            f.write('\n')
    except Exception as capture_error:
        print('CPU identity capture failed: '+type(capture_error).__name__, file=sys.stderr)
    raise
print(json.dumps(result, indent=2))
