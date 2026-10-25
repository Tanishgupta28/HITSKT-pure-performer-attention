"""CPU-only research planning. No Torch import, CUDA entry point or optimizer.

Recipes are declarations, not a new training implementation or authorization.
Historical configurations/checkpoints and already-frozen pilot plans are inputs.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1/prepared_plan.json')
EPOCH_CSV=Path('reports/favor_plus_v1/assist_epochs_execution_20261024_v1/EPOCH_METRICS.csv')
COMPARISON=Path('reports/favor_plus_v1/four_arm_cuda_comparison_20261024_v1/fp64_v2/COMPARISON_SUMMARY.json')
OUT=Path('reports/favor_plus_v1/performance_study_20261025_v1')
RUN=Path('experiments/hitskt_favor_plus_v1/assist2017/performance_plan_20261025_v1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validation_record(rows,selected_epoch):
    """Filter validation BEFORE reading metrics; never return test/train rows."""
    val=[r for r in rows if r['split']=='validation']
    selected=[r for r in val if int(r['epoch'])==selected_epoch]
    if len(selected)!=1:raise ValueError('one selected full-validation row required')
    row=selected[0]
    if float(row['roc_auc'])!=max(float(r['roc_auc']) for r in val):raise ValueError('selected epoch is not validation AUC maximum')
    return dict(epoch=selected_epoch,targets=int(row['targets']),roc_auc=float(row['roc_auc']),
        bce=float(row['loss']),f1=float(row['f1']),accuracy=float(row['accuracy']),
        precision=float(row['precision']),recall=float(row['recall']),threshold=float(row['threshold']))


def recipes(model_config,schedule_sha):
    common=dict(model_config=copy.deepcopy(model_config),seed=42,projection_seed=42,
        projection_policy='shared_nested512_pool_all_four_arms_v1',accumulation_dtype='float64',chunk_size=32,
        schedule_sha256=schedule_sha,optimizer=dict(name='Adam',lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=0,gradient_clip=None),
        protocol='HiTSKT_V1_chronological_within_student_retrospective_teacher_forced',review_threshold=.20,
        execution_approved=False,optimizer_updates_executed=0,initialization='fresh; never reuse historical learned weights',
        pilot_updates=20,full_comparison_epochs_proposed=8,full_comparison_early_stopping=False,
        selection='strict full-validation AUC maximum within common eight-epoch ceiling; threshold0.5',
        truncate_actions=False,drop_targets=False,modification='none')
    result=[]
    for arm,features,mapping,priority in [('orf128',128,'orf',1),('orf512',512,'orf',1),('anti512',512,'antithetic',2)]:
        item=copy.deepcopy(common);item.update(arm=arm,features=features,feature_mapping=mapping,priority=priority,
            checkpoint_identity='matched_study_format4',implementation_status='CUDA verified only for explicit FP64 attention; training unrun',
            namespace=f'experiments/hitskt_favor_plus_v1/assist2017/performance_20261025_{arm}_seed42_NEW_RUN_ID')
        result.append(item)
    for suffix,mod in [('gated_fusion','question_conditioned_history_response_gate'),('session_retrieval','question_conditioned_exact_attention_over_past_session_states')]:
        item=copy.deepcopy(result[1]);item.update(arm='orf512_'+suffix,modification=mod,priority=3,
            checkpoint_identity='NEW separate identity required before implementation',implementation_status='DESIGN ONLY; not implemented or CPU/CUDA verified',
            namespace=f'experiments/hitskt_favor_plus_v1/assist2017/performance_20261025_{suffix}_seed42_NEW_RUN_ID')
        result.append(item)
    return result


def validate_plan(plan):
    if plan['execution_approved'] or plan['optimizer_updates_executed']!=0:raise ValueError('preparation cannot authorize or report training')
    arms={x['arm']:x for x in plan['recipes']}
    a,b=arms['orf128'],arms['orf512']
    ignored={'arm','features','namespace'}
    if {k:v for k,v in a.items() if k not in ignored}!={k:v for k,v in b.items() if k not in ignored}:raise ValueError('primary pair has a non-feature confound')
    for x in arms.values():
        if x['execution_approved'] or x['optimizer_updates_executed']!=0 or x['review_threshold']!=.20:raise ValueError('scope/review drift')
        if x['schedule_sha256']!=plan['schedule_sha256'] or x['seed']!=42 or x['drop_targets'] or x['truncate_actions']:raise ValueError('matched data/seed/coverage required')
        if x['accumulation_dtype']!='float64' or x['projection_seed']!=42:raise ValueError('explicit precision/draw required')
    if len({x['namespace'] for x in arms.values()})!=len(arms):raise ValueError('experiment directories must be separate')
    if plan['test_evaluations_authorized']!=0:raise ValueError('no test evaluation during planning/selection')
    for key,value in dict(allocator_gib=8,advisory_total_gib=10,admission_free_gib=16,runtime_free_gib=6,host_rss_gib=4,host_headroom_gib=2,pacing_seconds=.1).items():
        if plan['budget'][key]!=value:raise ValueError('unchanged shared resource safeguards required')
    if any(x['pilot_updates']!=20 or x['full_comparison_epochs_proposed']!=8 for x in arms.values()):raise ValueError('equal prospective update/epoch budgets required')
    return 'PASS'


def storage_components(target_sessions,offsets,numbers,features,parameter_count=6931457):
    """Analytical component accounting, not a GPU peak or an admission proof."""
    b=len(target_sessions);counts=[min(15,int(numbers[t])-1) for t in target_sessions]
    hist=[j for t,c in zip(target_sessions,counts) for j in range(t-c,t)]
    lh=max(int(offsets[j+1]-offsets[j])+1 for j in hist)
    lt=max(int(offsets[t+1]-offsets[t])+1 for t in target_sessions);ls=max(counts)+1
    states=0;log_features=0;ff=0
    for rows,n in [(len(hist),lh),(b,ls),(b,lt),(b,lt)]:
        states+=rows*4*((n+31)//32)*features*(64+2)*8
        log_features+=rows*4*n*(2*features+64)*8
        ff+=2*rows*n*2048*4
    persistent=parameter_count*4*4+4*4*features*64*4
    return dict(component_sum_bytes=states+log_features+ff+persistent,saved_boundary_bytes=states,
        saved_log_features_value_bytes=log_features,two_ff_activation_bytes=ff,persistent_weights_grad_Adam_projection_bytes=persistent,
        batch_size=b,max_action_length=lh,max_target_length=lt,
        naive_exact_full_target_score_bytes_per_stage=b*4*lt*lt*4,
        short_session_retrieval_score_bytes=b*4*lt*15*4)


def prepare(root=ROOT):
    root=Path(root);old=json.loads((root/PARENT).read_text());c=old['contract']
    identity={}
    for name,want in c['source_sha256'].items():
        got=sha(root/name)
        if got!=want:raise ValueError('historical source changed:'+name)
        identity[name]=got
    store=Path(old['store_path'])
    for name,want in c['dataset_identity']['files'].items():
        if sha(store/name)!=want:raise ValueError('dataset changed:'+name)
    offsets=np.load(store/'session_offsets.npy',mmap_mode='r');numbers=np.load(store/'session_number.npy',mmap_mode='r');split=np.load(store/'session_split.npy',mmap_mode='r')
    train=np.flatnonzero((split==0)&(numbers>1));val=np.flatnonzero((split==1)&(numbers>1))
    baseline_val=np.flatnonzero(split==1)
    if not np.array_equal(val,baseline_val):raise ValueError('baseline/HiTSKT validation population differs')
    val_events=np.concatenate([np.arange(offsets[t],offsets[t+1],dtype='<i8') for t in val]);labels=np.load(store/'correct.npy',mmap_mode='r')[val_events]
    val_identity=dict(targets=len(val_events),positive=int(labels.sum()),negative=int(len(labels)-labels.sum()),
        ordered_event_ids_sha256=hashlib.sha256(val_events.tobytes()).hexdigest(),ordered_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        adapters='current RollingSessionDataset and RollingTargetDataset select same validation sessions; not a historical checkpoint replay')
    if len(val_events)!=125800 or len(train)!=7060:raise ValueError('full protocol counts changed')
    schedules={};accounting={128:[],512:[]}
    for epoch,rows in old['schedules']['train'].items():
        if int(epoch)>8:continue
        sessions=[t for r in rows for t in r['target_sessions']]
        if sorted(sessions)!=train.tolist():raise ValueError('training omission/duplicate at epoch'+epoch)
        if sum(r['supervised_targets'] for r in rows)!=566107:raise ValueError('training label count changed')
        schedules[epoch]=rows
        for row in rows:
            for m in (128,512):accounting[m].append(dict(epoch=int(epoch),batch=len(accounting[m]),**storage_components(row['target_sessions'],offsets,numbers,m)))
    if len(schedules)!=8:raise ValueError('eight complete common schedules required')
    matrix=[];sources={str(PARENT):sha(root/PARENT),str(EPOCH_CSV):sha(root/EPOCH_CSV),str(COMPARISON):sha(root/COMPARISON)}
    for model in ('hitskt','dkt','dkvmn','sakt','rkt'):
        directory=root/'experiments'/model/'assist2017/full';conf=json.loads((directory/'config.json').read_text());final=json.loads((directory/'final_results.json').read_text())
        with (directory/'metrics.csv').open() as f:selected=validation_record(list(csv.DictReader(f)),int(final['best_epoch']))
        if selected['targets']!=len(val_events) or selected['roc_auc']!=conf['best_validation_auc']:raise ValueError('stored full-validation evidence mismatch')
        files={str((directory/n).relative_to(root)):sha(directory/n) for n in ('config.json','metrics.csv','final_results.json')};sources.update(files)
        matrix.append(dict(model=model,metric_provenance='historical stored full-validation row; fresh validation replay NOT VERIFIED',epochs_completed=final['epochs_completed'],
            parameter_count=final['total_parameters'],history_sessions=conf.get('history_sessions'),history_interactions=conf.get('history_length'),seed=conf['seed'],selected_validation=selected,sources=files))
    with (root/EPOCH_CSV).open() as f:epoch_rows=list(csv.DictReader(f))
    selected=max(epoch_rows,key=lambda x:float(x['validation_auc']));matrix.append(dict(model='FAVOR+128 historical native FP32',metric_provenance='eight-epoch study; full selected-checkpoint validation replay PASS',epochs_completed=8,
        parameter_count=6931457,seed=42,history_sessions=15,selected_validation=dict(epoch=int(selected['epoch']),targets=125800,roc_auc=float(selected['validation_auc']),f1=float(selected['validation_f1']),bce=float(selected['validation_bce']))))
    for name in ('ktbench/favor_plus_performance_study.py','scripts/prepare_hitskt_performance_study.py'):
        sources[name]=sha(root/name)
    schedule_sha=canonical_sha(schedules)
    plan=dict(kind='assist2017_performance_research_plan_v1',execution_approved=False,optimizer_updates_executed=0,test_evaluations_authorized=0,
        dataset_identity=c['dataset_identity'],validation_identity=val_identity,schedule_sha256=schedule_sha,schedule_reference=str(PARENT),
        schedules='exact first8 train schedules at reference; use a COMMON new sealed schedule if profiling requires batch-size reduction',
        recipes=recipes(c['model_config'],schedule_sha),baseline_validation_context=matrix,source_sha256={**identity,**sources},
        budget=dict(allocator_gib=8,advisory_total_gib=10,admission_free_gib=16,runtime_free_gib=6,host_rss_gib=4,host_headroom_gib=2,pacing_seconds=.1,
            pilot_max_updates_per_arm=20,primary_pilot_total_updates=40,pilot_campaign_ceiling_seconds_proposed=900,full_shape_profile_seconds_proposed=600,
            full_study_reservation_seconds_proposed_not_runtime_prediction=7200,full_512_fp64_runtime='NOT ESTIMABLE before representative profiling; no automatic budget increase'),
        runner_status='DECLARATIVE PLAN ONLY. Existing20update runner binds four arms; two-arm/full-epoch execution needs separate reviewed scope/variant wiring; modifications are design-only.',
        seeds='42 for discovery; 43/44 confirmation proposed only after locking method/hyperparameters; no seed sweep approved',
        approval='New GPU profile/training instruction required; old campaign approvals/false templates do not authorize this plan.')
    validate_plan(plan)
    memory={str(m):dict(max_accounted=max(rows,key=lambda x:x['component_sum_bytes']),
        warning='Analytical retained-component sum, NOT measured GPU peak, complete accounting, memory forecast or proof of fit; FP64 attention/FP32 FF and parameters') for m,rows in accounting.items()}
    directory=root/OUT;directory.mkdir(exist_ok=False);run=root/RUN;run.mkdir(exist_ok=False)
    def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    write(directory/'EXPERIMENT_PLAN.json',plan);write(run/'prepared_plan.json',plan);write(directory/'MEMORY_COMPONENT_ACCOUNTING.json',memory)
    write(directory/'VALIDATION_CONTEXT.json',matrix);write(directory/'COMMON_TRAIN_SCHEDULES.json',schedules)
    with (directory/'VALIDATION_CONTEXT.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['model','selected_epoch','full_validation_auc','full_validation_f1','full_validation_bce','targets','epochs_completed','provenance'])
        for row in matrix:
            v=row['selected_validation'];writer.writerow([row['model'],v['epoch'],v['roc_auc'],v['f1'],v['bce'],v['targets'],row['epochs_completed'],row['metric_provenance']])
    return dict(status='PREPARED_NOT_AUTHORIZED',plan_sha256=sha(directory/'EXPERIMENT_PLAN.json'),recipes=len(plan['recipes']),validation=val_identity,
        optimizer_updates=0,CUDA_initialized=False,retained_component_max_GiB={m:v['max_accounted']['component_sum_bytes']/1024**3 for m,v in memory.items()})
