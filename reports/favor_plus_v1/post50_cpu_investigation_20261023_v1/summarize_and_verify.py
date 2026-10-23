"""Saved-evidence verification/aggregation only; no model, backward or GPU call."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/post50_cpu_diagnostics_20261023_v1'
PREVIOUS=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1'


def load(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def stats(a):
    return dict(count=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(max(a)),above_point20=sum(x>.2 for x in a))


plan=load(RUN/'prepared_plan.json');design=load(HERE/'PREDECLARED_DESIGN.json')
assert plan['design_sha256']==sha(HERE/'PREDECLARED_DESIGN.json')
assert plan['checkpoint50_sha256']==sha(PREVIOUS/'checkpoints/checkpoint_050.pt')
snapshot=load(HERE/'EXECUTED_SOURCE_SNAPSHOT.json')
for path,expected in plan['source_sha256'].items():
    actual=sha(ROOT/snapshot['snapshot_path']) if path==snapshot['original_path'] else sha(ROOT/path)
    assert actual==expected,path
att=load(RUN/'ATTENTION_RESULTS.json');rows=att['results'];approx={}
for name in sorted({x['stage'] for x in rows}):
    group=[x for x in rows if x['stage']==name]
    lengths={}
    for length in (1,8,32,128,256,512,939):
        selected=[y for x in group for y in x['production_lengths'] if y['length']==length]
        if selected:lengths[str(length)]=stats([x['relative_output_l2'] for x in selected])
    capped=[next(y for y in x['production_lengths'] if y['length']==min(128,x['actual_positions'])) for x in group]
    configs={str(m):stats([y['relative_output_l2'] for x in group for y in x['feature_comparison'] if y['features']==m]) for m in (128,256,512)}
    approx[name]=dict(production_lengths=lengths,deployed_projection_cap128=stats([x['relative_output_l2'] for x in capped]),diagnostic_projection_feature_grid=configs,
        capped_concentration={key:float(np.median([x[key]['median'] for x in capped])) for key in ('pair_feature_ess','exact_attention_max_weight','exact_attention_effective_keys','query_norm','key_norm','iid_log10_relative_variance_proxy')},
        max_production_feature_kernel_abs_error=max(y['production_vs_feature_max_abs'] for x in group for y in x['production_lengths']))
    assert all(y['feature_kernel_agreement']=='PASS' for x in group for y in x['production_lengths'])
assert len(rows)==28 and len(plan['cohort'])==14
assert all(x['selection']['split']=='train' for x in rows)
annex=load(RUN/'MAX_HISTORY_ANNEX_RESULTS.json');assert annex['actual_positions']==939 and annex['optimizer_calls']==0
assert all(x['feature_kernel_agreement']=='PASS' for x in annex['results'])
metrics=load(RUN/'PREDICTIVE_METRIC_ANALYSIS.json');assert metrics['labels_identical_all_milestones'] and metrics['no_threshold_selected']
assert metrics['optimizer_calls']==0 and metrics['test_evaluations']==0
profiles=load(RUN/'CPU_PROFILE_RESULTS.json')
assert len(profiles['results'])==2
for i,x in enumerate(profiles['results']):
    assert x['gradients_finite'] and x['optimizer_calls']==0 and not x['cuda_initialized']
    assert x['host']['peak_process_rss_bytes']<4*1024**3 and x['remaining_saved_live_bytes']==0
    assert 'cudaGetDeviceCount' in (RUN/f'cpu_profile_{i}.log').read_text()
isolation=load(HERE/'CPU_PROFILING_ENVIRONMENT_WARNING.json')
assert isolation['predeclared_no_gpu_probes_strictly_satisfied'] is False
test_counts={}
for file in ('cpu_tests.xml','cpu_isolation_guard_tests.xml'):
    cases=ET.parse(HERE/file).getroot().findall('.//testcase')
    assert cases and not any(any(x.find(tag) is not None for tag in ('failure','error','skipped')) for x in cases)
    test_counts[file]=dict(passed=len(cases),failures=0,skips=0)
preservation=[]
manifests=[('reports/favor_plus_v1/shared50_from25_20261023_v1/ARTIFACT_INVENTORY.json','files'),
 ('reports/favor_plus_v1/shared25_pilot_20261023_v1/ARTIFACT_INVENTORY.json','files'),
 ('reports/favor_plus_v1/assist20_recovery_20261020_v1/ARTIFACT_INVENTORY.json','files'),
 ('reports/favor_plus_v1/assist50_continuation_20261020_v1/EXPERIMENT_ARTIFACT_INVENTORY.json','files'),
 ('reports/favor_plus_v1/assist50_retry_20261023_v1/VERIFICATION_SUMMARY.json','experiment_files')]
for name,key in manifests:
    content=load(ROOT/name)[key]
    for filename,identity in content.items():
        path=Path(filename);path=path if path.is_absolute() else ROOT/path
        assert sha(path)==identity['sha256'],str(path)
    preservation.append(dict(manifest=name,files=len(content),unchanged=True))
protected=load(ROOT/'reports/favor_plus_v1/shared50_from25_20261023_v1/PRESERVATION_BEFORE.json')['protected_files']
for path,expected in protected.items():assert sha(Path(path))==expected
baseline=load(PREVIOUS/'prepared_plan.json')['contract_base']
for name,expected in baseline['source_sha256'].items():assert sha(ROOT/name)==expected
store=ROOT/'data/processed/assist2017/full/session_store'
for name,expected in baseline['dataset_identity']['files'].items():assert sha(store/name)==expected
shapes=load(RUN/'TRAIN_SHAPE_ACCOUNTING.json')
result=dict(status='CPU_FINDINGS_VERIFIED;APPROXIMATION_REVIEW;CPU_ISOLATION_WARNING_RETAINED',
    checkpoint50_sha256=plan['checkpoint50_sha256'],original_total_optimizer_updates=50,new_optimizer_updates=0,
    model_cuda_operations_requested=0,implicit_device_count_warning='observed in2 CPU-profile logs; strict no-probe intent not satisfied; current runner refuses CUDA-linked CPU backward',
    fresh_validation_inference=0,test_evaluations=0,attention=approx,max_history_action=annex,
    tests=test_counts,unique_cases_across_these_suites=84,
    memory=dict(train_examples=shapes['train_examples'],train_labels=shapes['train_target_labels'],original_batches=shapes['original_batch_count'],
        original_component_sum_GiB_median=shapes['component_sum_bytes']['median']/1024**3,original_component_sum_GiB_max=shapes['component_sum_bytes']['maximum']/1024**3,
        original_batches_component_sum_above1GiB=sum(x['component_sum_bytes']>1024**3 for x in shapes['full_batches']),
        max_one_weighted_state_trajectory_GiB=shapes['workspace_component_bytes']['maximum']/1024**3,
        max_combined_accounted_components_GiB=max(x['component_sum_bytes']+x['max_one_weighted_state_per_recomputed_token_bytes'] for x in shapes['full_batches'])/1024**3,
        caveat='Partial component accounting only, not measured VRAM or safe allocation recommendation'),
    preserved=preservation,protected_unrelated_files_unchanged=True,all35_prior_sources_unchanged=True,all11_processed_hashes_unchanged=True,
    full_training='NO-GO',longer_bounded_training='CONDITIONAL GO with explicit approval and persistent approximation REVIEW; no execution now')
with (HERE/'VERIFICATION_SUMMARY.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
with (RUN/'INVESTIGATION_RESULT.json').open('x') as f:json.dump(dict(status=result['status'],optimizer_updates=50,new_optimizer_updates=0,
    checkpoint50_sha256=result['checkpoint50_sha256'],test_evaluations=0,no_automatic_continuation=True,
    cuda_profile_environment_warning=True,full_training_ready=False),f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('attention','max_history_action')},indent=2))
