"""CPU accounting of the resource-blocked attempt; never starts CUDA/training."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from ktbench.favor_plus_pipeline import ROOT,sha256,write_json
from ktbench.favor_plus_assist20 import source_identity

HERE=Path(__file__).resolve().parent
BASE=ROOT/'experiments/hitskt_favor_plus_v1/assist2017'
V1=BASE/'assist50_continuation_20261020_v1'
V2=BASE/'assist50_continuation_20261020_v2'
PARENT=BASE/'assist20_recovery_20261020_v1'
failure=json.loads((V2/'study_failure.json').read_text())
proof=json.loads((V2/'stage_030/incoming_recovery_and_migration.json').read_text())
assert failure['optimizer_updates']==failure['optimizer_step_calls']==20
assert proof['status']=='PASS' and proof['optimizer_calls_during_recovery']==0
assert proof['incoming_checkpoint_sha256']==sha256(PARENT/'checkpoints/checkpoint_020.pt')
assert not list(V2.glob('stage_*/update_*.json')) and not list((V2/'checkpoints').iterdir())
assert not (V2/'stage_030/parent20_validation_replay.json').exists()
assert not (V2/'study_result.json').exists()
assert not list((V1/'checkpoints').iterdir())
plan=json.loads((V2/'prepared_plan.json').read_text())
assert all(plan['contract_base']['source_sha256'][k]==v for k,v in source_identity()['source_sha256'].items())
inventory=json.loads((HERE.parent/'assist20_recovery_20261020_v1/ARTIFACT_INVENTORY.json').read_text())
for name,v in inventory['files'].items():
 p=ROOT/name
 assert p.is_file() and p.stat().st_size==v['bytes'] and sha256(p)==v['sha256'],name
hold=Path('/workspace/LightSAKT2/experiments/USER_TRAINING_HOLD.json')
assert sha256(hold)=='65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234'
assert sha256(ROOT/'README.md')=='e5754c28ae89a628d46e0269523115404be13aba55d252a92c90ece2bf8ac6b8'
assert sha256(ROOT/'experiments/sakt/ednet_kt1/full/independent_evaluation.console.log')=='0f5546314e0911001954ca04e2fccdad7be7e09dce685fb8d61f5fe7a7af76ff'
tests=ET.parse(HERE/'targeted_cpu_final.xml').getroot().findall('.//testcase')
assert len(tests)==103 and all(t.find('failure') is None and t.find('error') is None and t.find('skipped') is None for t in tests)
post=json.loads((HERE/'NVML_PID_CONFLICT_INVESTIGATION.json').read_text())
assert post['nvml']['compute_processes']==[] and not post['snapshot']['conflicts'] and not post['snapshot']['light_processes']
prior=json.loads((HERE.parent/'assist20_recovery_20261020_v1/VERIFICATION_SUMMARY.json').read_text())
v20=next(v for v in prior['validation'] if v['update']==20)
result=dict(status='RESOURCE_IDENTITY_BLOCKED_BEFORE_UPDATE21;50_NOT_REACHED',
 source_commit_executed='b255713',optimizer_updates_total=20,additional_optimizer_calls=0,
 training_milestones={str(i):('HISTORICAL20_UNCHANGED' if i==20 else 'NOT RUN') for i in (20,30,40,50)},
 parent_checkpoint_sha256=proof['incoming_checkpoint_sha256'],parent_cuda_recovery=proof,
 frozen_validation20_metrics=v20['reported_metrics'],validation20_status='HISTORICAL ONLY; fresh full-sample replay stopped before evaluation',
 new_training_losses=[],new_training_gradient_norms=[],new_training_throughput='NOT MEASURED:0 updates',
 new_torch_gpu_peak_bytes='NOT VERIFIED: failure occurred before next cached CUDA memory boundary',
 failed_process_execution_seconds=failure['process_execution_seconds'],
 failure_scope=dict(phase=failure['phase'],visible_process_pid=failure['conflict_snapshot']['own_pid'],
   nvml_compute=failure['conflict_snapshot']['nvml']['compute_processes'],
   observed_total_mig_used_bytes=failure['conflict_snapshot']['nvml']['used_bytes'],
   interpretation='Unknown PID attribution; neither another workload nor own PID alias proved; do not bypass guard'),
 first_attempt=dict(status='ABORTED_BEFORE_CUDA',additional_optimizer_calls=0,cause='Adam betas tuple/list metadata representation; fixed without changing values/thresholds; preserved'),
 cpu_tests=dict(current_distinct_passed=103,failures=0,new_execution_cases=32,previous_controls=71,
   initial100_missed_json_boundary=True,corrected101_then_two_explicit_nvml_guard_cases=True),
 preservation=dict(original20_files_verified=len(inventory['files']),all_original20_files_unchanged=True,
   light_hold_sha256=sha256(hold),unrelated_readme_sakt_unchanged=True,signals_sent=0,light_dkt_modifications=0),
 post_failure_observation=post,
 next_requirement='Authoritative host/container PID mapping or visible allocation-level process identity for this MIG; alternatively an assigned MIG with verifiable ownership. Existing owner permission is not missing. No new CUDA to infer alias.',
 full_training_ready=False,new_test_evaluations=0,
 decision='NO-GO for more GPU execution until PID identity is established; no50-update training-readiness conclusion')
write_json(HERE/'VERIFICATION_SUMMARY.json',result)
files={}
for directory in (V1,V2):
 for p in sorted(directory.rglob('*')):
  if p.is_file():files[str(p.relative_to(ROOT))]=dict(bytes=p.stat().st_size,sha256=sha256(p))
write_json(HERE/'EXPERIMENT_ARTIFACT_INVENTORY.json',dict(files=files,count=len(files),additional_optimizer_calls=0,new_checkpoint_files=0))
print(json.dumps({k:v for k,v in result.items() if k not in ('parent_cuda_recovery','post_failure_observation','frozen_validation20_metrics')},indent=2))
