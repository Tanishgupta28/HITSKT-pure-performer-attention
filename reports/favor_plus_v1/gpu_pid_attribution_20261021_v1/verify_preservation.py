"""CPU streaming-hash/evidence accounting; no torch or NVIDIA API calls."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

manifests=[]
for name in ('assist20_recovery_20261020_v1/ARTIFACT_INVENTORY.json','assist50_continuation_20261020_v1/EXPERIMENT_ARTIFACT_INVENTORY.json'):
    p=HERE.parent/name;old=json.loads(p.read_text())
    for path,expected in old['files'].items():
        value=ROOT/path
        assert value.is_file() and value.stat().st_size==expected['bytes'] and sha(value)==expected['sha256'],path
    manifests.append(dict(manifest=str(p.relative_to(ROOT)),files_verified=len(old['files']),unchanged=True))
plan_path=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261020_v2/prepared_plan.json'
plan=json.loads(plan_path.read_text())
sources=plan['contract_base']['source_sha256']
for path,expected in sources.items():assert sha(ROOT/path)==expected,path
assert sha(ROOT/'README.md')=='e5754c28ae89a628d46e0269523115404be13aba55d252a92c90ece2bf8ac6b8'
assert sha(ROOT/'experiments/sakt/ednet_kt1/full/independent_evaluation.console.log')=='0f5546314e0911001954ca04e2fccdad7be7e09dce685fb8d61f5fe7a7af76ff'
hold=Path('/workspace/LightSAKT2/experiments/USER_TRAINING_HOLD.json')
assert sha(hold)=='65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234'
evidence=json.loads((HERE/'RESOURCE_AND_NAMESPACE_EVIDENCE.json').read_text())
assert evidence['no_torch_import'] and evidence['no_cuda_driver_import']
assert evidence['additional_optimizer_calls']==evidence['signals_sent']==evidence['allocation_or_job_changes']==0
assert evidence['diagnostic_decision']['gpu_execution_allowed'] is False
cases=ET.parse(HERE/'targeted_cpu.xml').getroot().findall('.//testcase')
assert len(cases)==31 and all(not any(t.find(x) is not None for x in ('failure','error','skipped')) for t in cases)
result=dict(status='PASS_PRESERVATION_AND_CPU_CHECKS;GPU_IDENTITY_UNRESOLVED',observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    base_commit='4927490049e0c5f26833d05e9219c42ddcfa6db0',manifests=manifests,existing_source_files_unchanged=len(sources),
    checkpoints=[dict(update=n,sha256=sha(ROOT/f'experiments/hitskt_favor_plus_v1/assist2017/assist20_recovery_20261020_v1/checkpoints/checkpoint_{n:03d}.pt')) for n in (5,10,20)],
    light_hold_sha256=sha(hold),unrelated_readme_sakt_unchanged=True,production_guard_modified=False,
    historical67123_equals2847964='NOT VERIFIED',historical_competing_job='NOT VERIFIED',
    assigned_resource_uuid=evidence['resource_uuid'],observed_gpu_instance=2,observed_compute_instance=0,
    active_nvml_pid_in_formal_snapshot=[r['pid'] for r in evidence['nvml']['compute_processes']],
    own_collector_pid=evidence['own_pid'],namespace_parent=evidence['namespace_parent'],
    gpu_ownership_fully_resolved=False,independent_current_availability_verified=False,
    actual_cuda_execution=False,additional_optimizer_calls=0,total_training_updates=20,
    cpu_tests=dict(passed=31,new=29,existing_guard_cases=2,failures=0,skipped=0),
    source_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'ktbench/gpu_process_diagnostics.py',ROOT/'scripts/diagnose_hitskt_gpu_processes.py',ROOT/'tests/test_gpu_process_diagnostics.py')},
    nvml_header_sha256=sha(Path('/usr/local/cuda/include/nvml.h')),
    nvml_library_sha256=sha(Path('/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.161.08')),
    conclusion='NO-GO: host PID binding absent and active unattributed GPU use observed; retain existing owner permission, wait for host evidence and next instruction')
with (HERE/'DIAGNOSTIC_SUMMARY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','checkpoints')},indent=2))
