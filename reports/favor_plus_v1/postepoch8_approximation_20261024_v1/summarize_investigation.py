"""CPU aggregation and preservation proof; plots only measured frozen diagnostics."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/workspace/capstone');HERE=ROOT/'reports/favor_plus_v1/postepoch8_approximation_20261024_v1';RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/postepoch8_frozen_20261024_v1';PARENT=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(v):
    a=np.asarray(v,float);assert len(a) and np.isfinite(a).all()
    return dict(n=len(a),median=float(np.median(a)),mean=float(a.mean()),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),minimum=float(a.min()),maximum=float(a.max()))
cap=read(RUN/'CAPTURE_RESULTS.json');grid=read(RUN/'FEATURE_GRID_RESULTS.json');controls=read(RUN/'CONTROL_RESULTS.json');gap=read(RUN/'INITIAL_DRAW_GAP_RESULTS.json');proto=read(RUN/'ANTITHETIC_PROTOTYPE_REPLAY.json');long=read(RUN/'LONG_PROTOTYPE_REPLAY.json');end=read(RUN/'END_TO_END_SENSITIVITY.json');design=read(HERE/'PREDECLARED_DESIGN.json');stages=design['stages']
all_phases=[cap,grid,controls,gap,proto,long,end]
assert all(x['optimizer_calls']==0 and x['test_evaluations']==0 for x in all_phases)
capture_summary=[]
for state in ('initial','7','8'):
 for stage in stages:
  rows=[x for x in cap['records'] if x['state']==state and x['stage']==stage]
  for prefix in ('32','128','full'):
   chosen=[x['prefixes'][-1] if prefix=='full' else min(x['prefixes'],key=lambda v:abs(v['length']-min(int(prefix),x['length']))) for x in rows]
   ms=[x['metrics'] for x in chosen]
   capture_summary.append(dict(state=state,stage=stage,prefix=prefix,output_error=stats([x['relative_output_l2'] for x in ms]),mean_weight_tv=stats([x['attention_total_variation']['mean'] for x in ms]),exact_effective_keys=stats([x['exact_attention_effective_keys']['mean'] for x in ms]),pair_ess=stats([x['pair_feature_ess']['median'] for x in ms]),qnorm=stats([x['query_norm']['median'] for x in ms]),knorm=stats([x['key_norm']['median'] for x in ms]),pair_cosine=stats([x['geometry']['cosine']['median'] for x in chosen]),scaled_pair_sum_norm_squared=stats([x['geometry']['scaled_pair_sum_norm_squared']['median'] for x in chosen]),value_rms=stats([x['value_rms'] for x in ms]),exact_output_rms=stats([x['exact_output_rms'] for x in ms])))
grids=[];saved=[]
for stage in stages:
 rows=[x for x in grid['records'] if x['stage']==stage]
 base=np.array([x['saved128']['relative_output_l2'] for x in rows]);center=np.array([x['saved128_centered']['relative_output_l2'] for x in rows])
 saved.append(dict(stage=stage,saved128=stats(base),saved128_first_key_centered=stats(center),centering_improved_cases=int((center<base).sum()),cases=len(rows)))
 for arm in design['feature_sweep']['arms']:
  for m in (128,256,512):
   vals=[v['metrics']['relative_output_l2'] for row in rows for v in row['grid'] if v['arm']==arm and v['features']==m]
   grids.append(dict(stage=stage,arm=arm,features=m,output_error=stats(vals),cases_at_or_below_review=sum(x<=.2 for x in vals),note='112 correlated case/draw contrasts, not independent training seeds'))
draw_summary=[]
for state in ('initial','7','8'):
 for stage in stages:
  rows=[x for x in gap['records'] if x['state']==state and x['stage']==stage]
  draw_summary.append(dict(state=state,stage=stage,saved=stats([x['saved']['relative_output_l2'] for x in rows]),fresh=stats([y['metrics']['relative_output_l2'] for x in rows for y in x['fresh']]),saved_pair_ess=stats([x['saved']['pair_feature_ess']['median'] for x in rows]),fresh_pair_ess=stats([y['metrics']['pair_feature_ess']['median'] for x in rows for y in x['fresh']]),pair_max_feature_mass_mean=stats([x['original_pair_max_feature_mass']['mean'] for x in rows]),maximum_pair_projection_cosine=stats([x['original_max_projection_cosine']['mean'] for x in rows])))
control_summary=[]
for stage in stages:
 rows=[x for x in controls['records'] if x['stage']==stage]
 control_summary.append(dict(stage=stage,swaps={name:stats([x['swaps'][name]['relative_output_l2'] for x in rows]) for name in rows[0]['swaps']},scales={name:stats([x['scales'][name]['relative_output_l2'] for x in rows]) for name in rows[0]['scales']},max_float64_feature_abs=max(x['float64']['production_vs_feature_max_abs'] for x in rows),max_value_offset_absolute_error_change=max(x['value_offset']['absolute_error_invariance_max_abs'] for x in rows)))
prototype_summary=[dict(stage=stage,features=m,error=stats([r['metrics']['relative_output_l2'] for r in proto['records'] if r['stage']==stage and r['features_total']==m])) for stage in stages for m in (128,256,512)]
sensitivity=[]
for condition in ('exact_CorrectPadding_only','exact_all_four'):
 rows=[x for x in end['records'] if x['condition']==condition]
 sensitivity.append(dict(condition=condition,case_mean_absolute_probability_difference=stats([x['mean_abs_probability_difference'] for x in rows]),maximum_probability_difference=max(x['max_abs_probability_difference'] for x in rows),changed_classifications=sum(x['changed_classifications'] for x in rows),target_presentations=sum(x['targets'] for x in rows),not_a_predictive_metric=True))
tests={}
for name in ('oracle_tests.xml','antithetic_regression_tests.xml','exact_reference_tests.xml'):
 tree=ET.parse(HERE/name);assert not tree.findall('.//failure') and not tree.findall('.//error')
 for c in tree.findall('.//testcase'):tests[(c.attrib['classname'],c.attrib['name'])]=True
assert len(tests)==125
new_tests=sum(a.rsplit('.',1)[-1] in ('test_favor_plus_frozen_investigation','test_favor_plus_antithetic_experimental') for a,b in tests)
parent=read(PARENT/'prepared_plan.json')['contract'];preserved=[]
for name,expected in parent['source_sha256'].items():assert sha(ROOT/name)==expected,name
for name,expected in parent['dataset_identity']['files'].items():assert sha(Path(read(PARENT/'prepared_plan.json')['store_path'])/name)==expected,name
for epoch,expected in design['checkpoints'].items():assert sha(PARENT/f'checkpoints/epoch_{int(epoch):03d}.pt')==expected
inventory=read(ROOT/'reports/favor_plus_v1/assist_epochs_execution_20261024_v1/ARTIFACT_INVENTORY.json')
for item in inventory['files']:
 if item['path'].startswith('project-memory/'):continue
 assert sha(ROOT/item['path'])==item['sha256'],item['path'];preserved.append(item['path'])
for path,expected in [('README.md','e5754c28ae89a628d46e0269523115404be13aba55d252a92c90ece2bf8ac6b8'),('experiments/sakt/ednet_kt1/full/independent_evaluation.console.log','0f5546314e0911001954ca04e2fccdad7be7e09dce685fb8d61f5fe7a7af76ff'),('/workspace/LightSAKT2/experiments/USER_TRAINING_HOLD.json','65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234')]:assert sha(Path(path))==expected
result=dict(status='INVESTIGATION_COMPLETE; NO_ORIGINAL_BUG_FOUND; APPROXIMATION_REVIEW; ANTITHETIC_PROTOTYPE_ONLY',source_commit='24206be171bbf3d973ad388e484d1828ecd32492',training_examples=14,states=['reconstructed_initial','7','8'],capture_records=168,grid_comparisons=sum(len(x['grid']) for x in grid['records']),review_threshold=.2,review_threshold_changed=False,optimizer_calls=0,cuda_execution=False,test_evaluations=0,new_validation_inference=0,capture=capture_summary,saved_projection_comparison=saved,feature_grid=grids,draw_gap=draw_summary,controls=control_summary,prototype_default_policy=prototype_summary,prototype_cpu_benchmark=proto['benchmark'],long_prototype=long['records'],end_to_end=sensitivity,projection_checks=cap['projection_checks'],historical_prefix_sanity=read(HERE/'HISTORICAL_PREFIX_SANITY.json')['max_absolute_relative_error_difference'],production_feature_max_abs=max(p['metrics']['production_vs_feature_max_abs'] for x in cap['records'] for p in x['prefixes']),prototype_feature_max_abs=max(x['metrics']['production_vs_feature_max_abs'] for x in proto['records']+long['records']),tests=dict(unique_passed=len(tests),new_passed=new_tests,existing_passed=len(tests)-new_tests,final_failures=0,initial_prototype_failure_preserved=True),resources=dict(max_peak_rss_bytes=max(x['host']['peak_process_rss_bytes'] for x in all_phases),minimum_shared_host_headroom_bytes=min(x['host']['cgroup_headroom_bytes'] for x in all_phases),phase_seconds={name:x['seconds'] for name,x in zip(['capture','grid','controls','draw_gap','prototype','long_prototype','end_to_end'],all_phases)},cpu_environment=read(HERE/'CPU_ENVIRONMENT.json')),preservation=dict(source_files_unchanged=len(parent['source_sha256']),processed_checksums_unchanged=len(parent['dataset_identity']['files']),prior_published_nonmemory_files_unchanged=len(preserved),protected_checkpoint7_sha256=design['checkpoints']['7'],protected_checkpoint8_sha256=design['checkpoints']['8'],original_results_changed=False,other_workloads_modified=False))
(HERE/'INVESTIGATION_SUMMARY.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8));labels={'original_ORF':'Independent ORF','first_valid_key_centering_ORF':'First-key centering','antithetic_ORF':'Antithetic ORF'}
for ax,stage in zip(axes.flat,stages):
 for arm in design['feature_sweep']['arms']:
  rows=[x for x in grids if x['stage']==stage and x['arm']==arm];x=[v['features'] for v in rows]
  line=ax.plot(x,[v['output_error']['median'] for v in rows],marker='o',label=labels[arm])[0]
  ax.fill_between(x,[v['output_error']['p10'] for v in rows],[v['output_error']['p90'] for v in rows],color=line.get_color(),alpha=.10)
 original=next(x for x in saved if x['stage']==stage)['saved128']['median'];ax.axhline(original,color='black',ls=':',label='Saved128 draw')
 ax.axhline(.2,color='red',ls='--',label='Unchanged review0.20');ax.set_title(stage.split('.')[0]);ax.set_xticks([128,256,512]);ax.set_xlabel('Total positive features');ax.set_ylabel('Raw attention relative L2');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('Frozen trained representations7/8,14 TRAIN cases,≤128 positions\nShading: descriptive10–90% across correlated case/draw contrasts; not confidence intervals',fontsize=11)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(HERE/'FROZEN_FEATURE_COMPARISON.png',dpi=160);plt.close(fig)
cpe='correct_padding_encoder.layers.0.attention';rows=[x for x in draw_summary if x['stage']==cpe];fig,axes=plt.subplots(1,3,figsize=(12,3.8));x=np.arange(3)
for ax,keya,keyb,title in [(axes[0],'saved','fresh','Relative attention-output L2'),(axes[1],'saved_pair_ess','fresh_pair_ess','Pair-feature ESS')]:
 ax.plot(x,[v[keya]['median'] for v in rows],marker='o',label='Saved128 draw');ax.plot(x,[v[keyb]['median'] for v in rows],marker='o',label='Independent128 draws');ax.set_ylabel(title);ax.legend(fontsize=8)
axes[2].plot(x,[v['pair_max_feature_mass_mean']['mean'] for v in rows],marker='o');axes[2].set_ylabel('Mean maximum feature mass (saved draw)')
for ax in axes:ax.set_xticks(x,['Initial','Epoch7','Epoch8']);ax.grid(alpha=.2)
fig.suptitle('CorrectPaddingEncoder: projection-specific alignment after learning');fig.tight_layout();fig.savefig(HERE/'PROJECTION_ALIGNMENT_EVIDENCE.png',dpi=160);plt.close(fig)
print(json.dumps(dict(status=result['status'],tests=result['tests'],peak_rss_gib=result['resources']['max_peak_rss_bytes']/1024**3,preservation=result['preservation']),indent=2))
