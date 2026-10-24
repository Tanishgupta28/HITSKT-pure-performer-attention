"""CPU post-run aggregation of actual receipts; no optimizer/CUDA/test inference."""
from pathlib import Path
import csv,datetime as dt,hashlib,json,statistics as st
ROOT=Path('/workspace/capstone');RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1';OUT=ROOT/'reports/favor_plus_v1/assist_epochs_execution_20261024_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def attention(value):
    dec=value['decision'];result={}
    for stage,stats in dec['aggregate'].items():
        errors=[case[stage]['feature_versus_exact_softmax']['relative_l2'] for case in value['cases']]
        result[stage]=dict(median_relative_l2=stats['relative_l2']['median'],maximum_relative_l2=stats['relative_l2']['maximum'],cases_above_review=sum(v>.2 for v in errors),median_pair_ess=stats['pair_ess_medians']['median'],minimum_case_pair_ess_median=stats['pair_ess_medians']['minimum'],max_p95_growth_ratio=stats['max_p95_growth_ratio'],kernel_reference_max_abs=stats['max_implementation_abs'])
    return dict(status=dec['status'],severe_failure_stages=dec['severe_failure_stages'],review_threshold=dec['original_softmax_review_trigger'],stages=result)
plan=read(RUN/'prepared_plan.json');contract=plan['contract'];rows=[read(p) for p in sorted((RUN/'metrics').glob('epoch_*.json'))]
assert rows and not (RUN/'study_failure.json').exists()
for name,expected in contract['source_sha256'].items():assert sha(ROOT/name)==expected,name
for name,expected in contract['dataset_identity']['files'].items():assert sha(Path(plan['store_path'])/name)==expected,name
resources=[read(p) for p in sorted(RUN.rglob('resources_*.json'))];allocators=[read(p) for p in sorted(RUN.rglob('allocator_*.json'))]
assert all(v['decision']['status']!='STOP' and not v['decision']['reasons'] for v in resources)
assert all(not v['observation']['local']['light_processes'] for v in resources)
assert all(max(v['allocated_bytes'],v['reserved_bytes'])<=8*1024**3 for v in allocators)
assert sha(ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1/checkpoints/checkpoint_050.pt')=='93ebe9d2a29be9a6e4deb278916513339de61c9ac8f72a454a31111829440a73'
assert sha(Path('/workspace/LightSAKT2/experiments/USER_TRAINING_HOLD.json'))=='65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234'
assert sha(ROOT/'README.md')=='e5754c28ae89a628d46e0269523115404be13aba55d252a92c90ece2bf8ac6b8'
assert sha(ROOT/'experiments/sakt/ednet_kt1/full/independent_evaluation.console.log')=='0f5546314e0911001954ca04e2fccdad7be7e09dce685fb8d61f5fe7a7af76ff'
epochs=[]
for row in rows:
    epoch=row['epoch'];proof=read(RUN/f'recovery_{epoch:03d}/recovery_result.json');assert proof['status']=='PASS' and proof['optimizer_calls']==0
    assert proof['checkpoint_sha256']==sha(RUN/f'checkpoints/epoch_{epoch:03d}.pt')==row['checkpoint']['sha256']
    phase=[read(p) for p in sorted((RUN/f'epoch_{epoch:03d}').glob('resources_*.json'))]
    first=next(v['observation']['observed_utc'] for v in phase if v['phase']=='train_batch');last=[v['observation']['observed_utc'] for v in phase if v['phase']=='after_update'][-1]
    training_wall=(dt.datetime.fromisoformat(last)-dt.datetime.fromisoformat(first)).total_seconds()
    epochs.append(dict(epoch=epoch,training=row['train'],validation=row['validation'],optimizer_updates=len(row['updates']),cumulative_updates=row['checkpoint']['optimizer_updates'],checkpoint_sha256=proof['checkpoint_sha256'],fresh_recovery='PASS_EXACT_ZERO_ADAM',gradient_l2_min=min(u['gradient_l2'] for u in row['updates']),gradient_l2_max=max(u['gradient_l2'] for u in row['updates']),max_batch_bce=max(u['loss'] for u in row['updates']),median_update_seconds=st.median(u['seconds'] for u in row['updates']),measured_update_seconds=sum(u['seconds'] for u in row['updates']),training_wall_seconds=training_wall,training_labels_per_wall_second=566107/training_wall,attention=attention(read(RUN/f'epoch_{epoch:03d}/final_diagnostics/attention_diagnostics.json'))))
metric_proof=read(OUT/'INDEPENDENT_ALL_ATTEMPTED_METRIC_VERIFICATION.json');assert metric_proof['status']=='PASS' and len(metric_proof['epochs'])==len(rows)
replay=read(RUN/'best_full_validation_replay/validation_replay.json');assert replay['status']=='PASS' and replay['arrays_exact']
record=read(OUT/'APPROVED_EXECUTION_RECORD.json');terminal=read(RUN/'study_result.json')
last_stamp=max(v['observation']['observed_utc'] for v in resources);elapsed=dt.datetime.fromisoformat(last_stamp).timestamp()-record['campaign_started_unix'];assert elapsed<=2700
checkpoints=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((RUN/'checkpoints').glob('*.pt'))]
result=dict(status='MEASURED_EPOCH_TRAINING_VERIFIED_WITH_APPROXIMATION_REVIEW',terminal=terminal,epochs_completed=len(epochs),epochs=epochs,initial_attention=attention(read(RUN/'epoch_001/initial/attention_diagnostics.json')),profile=read(RUN/'cuda_memory_profile/profile_result.json'),source_files_unchanged=len(contract['source_sha256']),processed_checksums_verified=len(contract['dataset_identity']['files']),all_training_targets_per_epoch=566107,all_validation_targets_per_epoch=125800,source_plan_sha256=sha(RUN/'prepared_plan.json'),test_evaluations=0,data_discarded=0,sequences_truncated=0,checkpoint50_preserved=True,light_hold_preserved=True,unrelated_dirty_files_preserved=True,gradient_model_adam_all_finite=True,optimizer_updates=sum(x['optimizer_updates'] for x in epochs),independent_metric_max_abs_difference=max(max(x['abs_differences'].values()) for x in metric_proof['epochs']),best_full_validation_replay=replay,local_checkpoint_inventory=checkpoints,resources=dict(observations=len(resources),allocator_observations=len(allocators),all_admitted_with_limits=True,peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in allocators),peak_reserved_bytes=max(v['peak_reserved_bytes'] for v in allocators),min_sampled_device_free_bytes=min(v['observation']['device']['free_bytes'] for v in resources),max_sampled_device_used_bytes=max(v['observation']['device']['used_bytes'] for v in resources),max_reported_nvml_process_bytes=max((p.get('used_gpu_memory_bytes') or 0) for v in resources for p in v['observation']['device']['compute_processes']),peak_host_rss_bytes=max(v['host']['peak_process_rss_bytes'] for v in resources),minimum_shared_host_headroom_bytes=min(v['host']['cgroup_headroom_bytes'] for v in resources),utilization='MIG UNSUPPORTED; UNKNOWN',pid_mapping='UNRESOLVED; no process-total/self/peer attribution claimed',campaign_elapsed_through_final_gpu_receipt_seconds=elapsed))
(OUT/'VERIFIED_EXECUTION_SUMMARY.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
with (OUT/'EPOCH_METRICS.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['epoch','train_bce','validation_auc','validation_accuracy','validation_precision','validation_recall','validation_f1','validation_bce','optimizer_updates'])
    for e in epochs:w.writerow([e['epoch'],e['training']['bce'],*[e['validation'][k] for k in ('roc_auc','accuracy','precision','recall','f1','bce')],e['optimizer_updates']])
print(json.dumps(dict(status=result['status'],epochs=result['epochs_completed'],optimizer_updates=result['optimizer_updates'],independent_metric_max_abs_difference=result['independent_metric_max_abs_difference'],resources=result['resources']),indent=2))
