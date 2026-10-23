"""CPU streaming verification of preservation and the stopped attempt."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / 'experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261023_v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


preserved = []
for name in ('assist20_recovery_20261020_v1/ARTIFACT_INVENTORY.json',
             'assist50_continuation_20261020_v1/EXPERIMENT_ARTIFACT_INVENTORY.json'):
    manifest = HERE.parent/name
    files = json.loads(manifest.read_text())['files']
    for path, expected in files.items():
        value = ROOT/path
        assert value.stat().st_size == expected['bytes'] and sha(value) == expected['sha256'], path
    preserved.append(dict(manifest=str(manifest.relative_to(ROOT)), files=len(files), unchanged=True))
old = json.loads((ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261020_v2/prepared_plan.json').read_text())
plan = json.loads((RUN/'prepared_plan.json').read_text())
sources = old['contract_base']['source_sha256']
assert plan['contract_base']['source_sha256'] == sources
for path, expected in sources.items():
    assert sha(ROOT/path) == expected, path
for field in ('config', 'configuration_sha256', 'parent_contract', 'parent_checkpoint_sha256',
              'parent_component_sha256', 'parent_projection_sha256', 'dataset_identity',
              'exact_training_and_probe_samples', 'frozen_validation_samples', 'execution_limits'):
    assert plan['contract_base'][field] == old['contract_base'][field], field
assert sha(ROOT/'README.md') == 'e5754c28ae89a628d46e0269523115404be13aba55d252a92c90ece2bf8ac6b8'
assert sha(ROOT/'experiments/sakt/ednet_kt1/full/independent_evaluation.console.log') == '0f5546314e0911001954ca04e2fccdad7be7e09dce685fb8d61f5fe7a7af76ff'
hold = Path('/workspace/LightSAKT2/experiments/USER_TRAINING_HOLD.json')
assert sha(hold) == '65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234'
failure = json.loads((RUN/'study_failure.json').read_text())
proof = json.loads((RUN/'stage_030/incoming_recovery_and_migration.json').read_text())
assert failure['optimizer_updates'] == failure['optimizer_step_calls'] == 20
assert failure['failure_type'] == 'ResourceConflict' and failure['phase'] == 'diagnostics_20'
assert proof['status'] == 'PASS' and proof['eval_prediction_max_abs_difference'] == 0
assert proof['optimizer_calls_during_recovery'] == 0
assert proof['next_train_forward_loss_all_gradient_digest'] == 'EXACT'
assert proof['unchanged_model_adam_rng_projections'] is True
assert not list((RUN/'checkpoints').iterdir())
assert not list(RUN.glob('stage_*/update_*.json'))
assert not list(RUN.glob('stage_*/validation_predictions_*.npz'))
assert not (RUN/'study_result.json').exists()
release = json.loads((HERE/'resource_release.json').read_text())
assert release['status'] == 'PASS' and release['nvml']['compute_processes'] == []
assert release['nvml']['used_bytes'] == 39780352
cases = ET.parse(HERE/'targeted_cpu.xml').getroot().findall('.//testcase')
assert len(cases) == 32 and all(not any(t.find(x) is not None for x in ('failure', 'error', 'skipped')) for t in cases)
files = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(RUN.rglob('*')) if p.is_file()}
result = dict(status='PASS_PRESERVATION_AND_RECOVERY;CONTINUATION_RESOURCE_BLOCKED',
    verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(), base_commit='bd076beeb4b198c7e51363793aa5e6056276ca01',
    run_id=RUN.name, prepared_plan_sha256=sha(RUN/'prepared_plan.json'), manifests=preserved,
    existing_source_files_unchanged=len(sources), scientific_configuration_unchanged=True,
    historical20_checkpoint_sha256=proof['parent_checkpoint_sha256'], checkpoint20_fresh_cuda_recovery='PASS',
    additional_real_optimizer_calls=0, total_training_updates=20, new_checkpoints=0,
    checkpoints30_40_50='NOT REACHED', new_full_frozen_validation_replay='NOT RUN',
    nvml_pid=488211, own_process_pid=3843909, pid_relationship='UNKNOWN; no trusted host/runtime binding',
    parth_job_stopped='No workers observed before/after; individual job identity not verified',
    cuda_execution_occurred=True, production_guards_changed=False, gpu_identity_fully_resolved=False,
    cpu_tests=dict(passed=32, failures=0, skipped=0, new_tests=0),
    light_hold_sha256=sha(hold), unrelated_readme_sakt_unchanged=True,
    signals_sent=0, job_or_allocation_changes=0, automatic_retries=0,
    artifact_inventory=files, final_mig_compute_processes=release['nvml']['compute_processes'])
with (HERE/'VERIFICATION_SUMMARY.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps({k:v for k,v in result.items() if k != 'artifact_inventory'}, indent=2))
