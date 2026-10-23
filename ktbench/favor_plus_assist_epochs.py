"""Separate full-coverage ASSIST run: <=10 epochs, full validation, patience3.

Historical pilot/model/data/shared policies stay immutable. Resource-budget
increases require an actual explicit response; preparation never grants one.
No automatic test inference, failed-step retry or data/sequence omission.
"""
from __future__ import annotations
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np
import torch
from ktbench import favor_plus_shared_gpu as shared
from ktbench.batching import LengthBucketTokenBatchSampler,collate_hitskt,padded_token_cost
from ktbench.config import seed_everything
from ktbench.data.session_store import SessionStore,RollingSessionDataset
from ktbench.favor_plus_bounded_assist import severe_check
from ktbench.favor_plus_pipeline import ROOT,PROTOCOL,fresh_run,model_config,profile_for,sha256,write_json,runtime,target_count
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree,finite_tree,fixed_example_diagnostics
from ktbench.favor_plus_pilot_recovery import digest
from ktbench.favor_plus_continuation_recovery import optimizer_groups_equal
from ktbench.favor_plus_real_pilot import host_memory
from ktbench.metrics import binary_metrics,masked_bce_loss
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.training import capture_rng_state,restore_rng_state

KIND='hitskt_favor_plus_assist_epochs_v1'
ACTION='favor_plus_assist2017_complete_train_up_to10_epochs_patience3'
HERE=ROOT/'reports/favor_plus_v1/assist_epochs_v1_20261023'
DESIGN=HERE/'PREDECLARED_DESIGN.json'
AUTH=HERE/'USER_TRAINING_AUTHORIZATION.md'
PRIOR=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_from25_20261023_v1'
STORE=ROOT/'data/processed/assist2017/full/session_store'
CHECKPOINT50=PRIOR/'checkpoints/checkpoint_050.pt'
SHA50='93ebe9d2a29be9a6e4deb278916513339de61c9ac8f72a454a31111829440a73'
GIB=1024**3
BUCKETS=dict(token_budget=32768,max_batch_size=64,bucket_size=512,seed=42)
COMPONENTS=('state_dict','optimizer_state_dict','rng_state','progress','contract','reference_eval_logits','reference_probe')


class BudgetBlocked(RuntimeError):pass


def source_identity():
    old=json.loads((PRIOR/'prepared_plan.json').read_text())['contract_base']['source_sha256']
    for name,expected in old.items():
        if sha256(ROOT/name)!=expected:raise ValueError('historical source changed: '+name)
    return {**old,**{name:sha256(ROOT/name) for name in ('ktbench/favor_plus_assist_epochs.py','scripts/train_assist_favor_plus_epochs.py')}}


def budget(record):
    name=record.get('budget_profile')
    if name=='existing_1gib':
        return dict(allocator_bytes=GIB,advisory_total_bytes=2*GIB,admission_free_bytes=8*GIB,
            runtime_free_bytes=6*GIB,max_padded_tokens=2048,max_batch_size=1)
    if (name=='requested_8gib' and record.get('budget_increase_approved') is True
            and isinstance(record.get('actual_budget_approval_evidence'),str)
            and record['actual_budget_approval_evidence'].strip() and record.get('approval_is_proposal') is False):
        return dict(allocator_bytes=8*GIB,advisory_total_bytes=10*GIB,admission_free_bytes=16*GIB,
            runtime_free_bytes=6*GIB,max_padded_tokens=32768,max_batch_size=64)
    raise BudgetBlocked('8GiB resource proposal is not approval; existing1GiB/cost2048 remain in force')


def authorize(record,plan_path,*,now=None):
    if (record.get('action')!=ACTION or record.get('shared_permission_confirmed') is not True
            or record.get('full_training_approved') is not True or record.get('max_epochs')!=10
            or record.get('patience')!=3 or record.get('resource_uuid')!=shared.TRANSFERRED_MIG
            or record.get('prepared_plan_sha256')!=sha256(plan_path)
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=shared.TRANSFERRED_MIG
            or record.get('known_explicit_restriction') is not False
            or record.get('known_incompatible_reservation') is not False):raise ValueError('actual full-run shared scope/UUID/plan required')
    clock=now or dt.datetime.now(dt.timezone.utc)
    start=dt.datetime.fromisoformat(record['starts_utc']);end=dt.datetime.fromisoformat(record['expires_utc'])
    if start.tzinfo is None or end.tzinfo is None or not start<=clock<end or not 0<(end-start).total_seconds()<=7200:raise ValueError('fresh <=2h operator record required')
    return budget(record)


def schedule(dataset,epoch,training):
    sampler=LengthBucketTokenBatchSampler([dataset.shape(i) for i in range(len(dataset))],shuffle=training,**BUCKETS)
    sampler.set_epoch(epoch if training else 0);rows=list(sampler)
    if sorted(i for row in rows for i in row)!=list(range(len(dataset))):raise ValueError('missing/duplicate supervised sessions')
    return rows


def audit_dataset(store):
    reference=json.loads((PRIOR/'prepared_plan.json').read_text())['contract_base']['dataset_identity']
    if store.metadata!=reference['metadata']:raise ValueError('original verified metadata mismatch')
    for name,expected in reference['files'].items():
        if sha256(store.root/name)!=expected:raise ValueError('verified processed checksum mismatch: '+name)
    if np.any(np.diff(store.session_offsets)<=0):raise ValueError('empty/misaligned sessions')
    for start,stop in zip(store.student_offsets[:-1],store.student_offsets[1:]):
        start,stop=int(start),int(stop)
        if (np.any(np.diff(store.session_split[start:stop].astype(int))<0)
                or not np.array_equal(store.session_number[start:stop],np.arange(1,stop-start+1))
                or not np.all(store.session_student[start:stop]==store.session_student[start])):raise ValueError('student/split chronology mismatch')
    splits={name:RollingSessionDataset(store,name,history_sessions=15) for name in ('train','validation')}
    for name,ds in splits.items():
        for raw in ds.targets:
            target=int(raw);start,stop=ds._history_bounds(target)
            if (stop!=target or start>=target or not np.all(store.session_student[start:target+1]==store.session_student[target])
                    or np.any(store.session_split[start:target+1]>ds.split)):raise ValueError('history student/future-split leakage')
    counts={k:dict(sessions=len(v),labels=target_count(v)) for k,v in splits.items()}
    if counts!={'train':{'sessions':7060,'labels':566107},'validation':{'sessions':1977,'labels':125800}}:raise ValueError('published complete target counts changed')
    return reference,splits,counts


def shape_manifest(ds,rows):
    return [dict(indices=row,target_sessions=[int(ds.targets[i]) for i in row],
        batch_size=len(row),padded_tokens=padded_token_cost([ds.shape(i) for i in row]),
        supervised_targets=sum(ds.shape(i).target_length for i in row)) for row in rows]


def shape_admission(plan,limits):
    rows=[(split,e,i,row) for split,epochs in plan['schedules'].items() for e,batches in epochs.items() for i,row in enumerate(batches)]
    bad=[dict(split=s,epoch=e,batch=i,batch_size=r['batch_size'],padded_tokens=r['padded_tokens']) for s,e,i,r in rows
         if r['padded_tokens']>limits['max_padded_tokens'] or r['batch_size']>limits['max_batch_size']]
    return dict(status='BLOCKED_RESOURCE_BUDGET' if bad else 'PASS_ALL_SHAPES',incompatible_batches=len(bad),
        first_incompatible=bad[0] if bad else None,limits=limits,data_skipped=0,actions_truncated=0,cuda_initialized=False)


def prepare(run_id):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU preparation requires CUDA hidden')
    sources=source_identity()
    if sha256(CHECKPOINT50)!=SHA50:raise ValueError('preserved diagnostic50 changed')
    store=SessionStore(STORE);identity,splits,counts=audit_dataset(store)
    schedules=dict(train={str(e):shape_manifest(splits['train'],schedule(splits['train'],e,True)) for e in range(1,11)},
        validation={'0':shape_manifest(splits['validation'],schedule(splits['validation'],0,False))})
    prior=json.loads((PRIOR/'prepared_plan.json').read_text())['contract_base']
    checkpoints=torch.load(CHECKPOINT50,map_location='cpu',weights_only=True)
    contract=dict(run_id=run_id,model_config=asdict(model_config('assist2017',store.metadata['num_questions'],store.metadata['num_skills'])),
        favor_config=asdict(FavorPlusConfig()),profile=profile_for('assist2017'),seed=42,protocol=PROTOCOL,
        max_epochs=10,patience=3,selection='strict full_validation_roc_auc improvement;min_delta0',threshold=.5,
        initialization='FRESH42; no diagnostic weights/Adam imported',dataset_identity=identity,target_counts=counts,
        source_sha256=sources,design_sha256=sha256(DESIGN),user_authorization_sha256=sha256(AUTH),
        expected_projection_sha256=checkpoints['projection_sha256'],parameter_tensor_count=len(checkpoints['optimizer_state_dict']['state']),
        expected_optimizer_groups=checkpoints['optimizer_state_dict']['param_groups'],
        sampler_sha256=digest(schedules),sampler_schedule={split:{e:[row['target_sessions'] for row in batches] for e,batches in epochs.items()} for split,epochs in schedules.items()},
        diagnostic_samples=prior['diagnostic_samples'],review_and_severe_stops=prior['config']['review_and_severe_stops'])
    del checkpoints
    plan=dict(kind=KIND,contract=contract,schedules=schedules,sampler_sha256=digest(schedules),
        store_path=str(store.root),preserved_diagnostic50_sha256=SHA50,test_evaluations=0)
    output=fresh_run('assist2017',run_id);(output/'checkpoints').mkdir();(output/'metrics').mkdir()
    write_json(output/'prepared_plan.json',plan)
    blocked=shape_admission(plan,budget({'budget_profile':'existing_1gib'}));write_json(output/'CURRENT_BUDGET_ADMISSION.json',blocked)
    return dict(output=str(output),prepared_plan_sha256=sha256(output/'prepared_plan.json'),counts=counts,
        training_batches_per_epoch={e:len(v) for e,v in schedules['train'].items()},admission=blocked,new_optimizer_updates=0)


def load_plan(path):
    plan=json.loads(path.read_text());c=plan['contract']
    if (plan.get('kind')!=KIND or c['source_sha256']!=source_identity() or c['design_sha256']!=sha256(DESIGN)
            or c['user_authorization_sha256']!=sha256(AUTH) or plan['sampler_sha256']!=digest(plan['schedules'])
            or c['max_epochs']!=10 or c['patience']!=3):raise ValueError('source/data/config/sampler/task contract changed')
    identity,splits,counts=audit_dataset(SessionStore(plan['store_path']))
    if identity!=c['dataset_identity'] or counts!=c['target_counts']:raise ValueError('verified data changed')
    if c['sampler_sha256']!=plan['sampler_sha256'] or c['sampler_schedule']!={s:{e:[row['target_sessions'] for row in batches] for e,batches in epochs.items()} for s,epochs in plan['schedules'].items()}:raise ValueError('checkpoint sampler identity mismatch')
    expected_model=asdict(model_config('assist2017',identity['metadata']['num_questions'],identity['metadata']['num_skills']))
    if (c['model_config']!=expected_model or c['favor_config']!=asdict(FavorPlusConfig())
            or c['profile']!=profile_for('assist2017') or c['seed']!=42 or c['protocol']!=PROTOCOL
            or c['threshold']!=.5):raise ValueError('verified architecture/attention/optimizer/protocol changed')
    return plan,splits


def selection(progress,epoch,validation):
    auc=float(validation['roc_auc']);f1=float(validation['f1'])
    if type(epoch) is not int or not 1<=epoch<=10 or not math.isfinite(auc) or not math.isfinite(f1):raise ValueError('finite full validation at epoch1..10 required')
    improved=progress.get('best_auc') is None or auc>progress['best_auc']
    result={**progress,'bad_epochs':0 if improved else progress.get('bad_epochs',0)+1}
    if improved:result.update(best_auc=auc,best_epoch=epoch)
    if progress.get('best_f1') is None or f1>progress['best_f1']:result.update(best_f1=f1,best_f1_epoch=epoch)
    result['stop']=result['bad_epochs']>=3 or epoch==10
    return result,improved


def checkpoint_validate(payload,contract):
    if (payload.get('kind')!=KIND or payload.get('contract')!=contract or payload.get('variant')!='hitskt_favor_plus_v1'
            or payload.get('format_version')!=2 or payload.get('config')!=contract['model_config']
            or payload.get('favor_config')!=contract['favor_config']):raise ValueError('epoch checkpoint contract mismatch')
    if set(payload.get('component_sha256',{}))!=set(COMPONENTS):raise ValueError('missing checkpoint hashes')
    for key in COMPONENTS:
        if digest(payload[key])!=payload['component_sha256'][key]:raise ValueError('checkpoint component mismatch: '+key)
    projection={k:digest(v) for k,v in payload['state_dict'].items() if k.endswith('.projection')}
    if projection!=contract['expected_projection_sha256']:raise ValueError('FAVOR+ projections changed')
    if not finite_tree(payload):raise FloatingPointError('nonfinite checkpoint')
    p=payload['progress'];steps=p['optimizer_updates']
    if type(steps) is not int or steps<0 or not 1<=p['epoch']<=10 or p['next_batch']<0:raise ValueError('invalid epoch/sampler cursor')
    schedule=contract['sampler_schedule']['train'];current=schedule[str(p['epoch'])]
    expected=sum(len(rows) for e,rows in schedule.items() if int(e)<p['epoch'])+p['next_batch']
    if p['next_batch']>len(current) or steps!=expected:raise ValueError('optimizer calls disagree with exact epoch/sampler cursor')
    if 'train_probabilities' in p:
        if p['train_probabilities'].ndim!=1 or p['train_labels'].shape!=p['train_probabilities'].shape or p['train_targets_done']!=p['train_labels'].numel():raise ValueError('resumable metric-prefix shape mismatch')
    opt=payload['optimizer_state_dict']
    if not optimizer_groups_equal(opt['param_groups'],contract['expected_optimizer_groups']):raise ValueError('original Adam settings/parameter groups changed')
    if steps:
        if len(opt['state'])!=contract['parameter_tensor_count']:raise ValueError('incomplete Adam coverage')
        if any(float(s['step'])!=steps or not bool((s['exp_avg_sq']>=0).all()) for s in opt['state'].values()):raise ValueError('Adam counter/second moments mismatch')
    for group in opt['param_groups']:
        if (group['lr']!=5e-5 or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8 or group['weight_decay']!=0.):raise ValueError('original Adam settings changed')
    return dict(status='PASS',epoch=p['epoch'],optimizer_updates=steps,projection_sha256=projection)


def atomic_save(path,payload,contract):
    path=Path(path);checkpoint_validate(payload,contract)
    if path.exists():raise FileExistsError(path)
    fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.stem,suffix='.partial.pt')
    with os.fdopen(fd,'wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    loaded=torch.load(temp,map_location='cpu',weights_only=True);checkpoint_validate(loaded,contract)
    if not equal_tree(loaded,payload):raise RuntimeError('checkpoint serialization mismatch')
    os.link(temp,path);os.unlink(temp)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return dict(path=str(path),sha256=sha256(path),atomic_no_overwrite=True,**checkpoint_validate(loaded,contract))


def make_batch(ds,row):
    b=collate_hitskt([ds[i] for i in row['indices']],num_questions=ds.store.metadata['num_questions'],num_skills=ds.store.metadata['num_skills'])
    if b.padded_tokens!=row['padded_tokens'] or int(b.target_metric_mask.sum())!=row['supervised_targets']:raise ValueError('collation shape/label alignment changed')
    return b


def fixed_logits(net,b):
    was=net.training;net.eval()
    try:
        with torch.no_grad():return net(b).detach().cpu()
    finally:net.train(was)


def probe(net,opt,b):
    rng=capture_rng_state();was=net.training;net.train();opt.zero_grad(set_to_none=True)
    try:
        out=net(b);loss=masked_bce_loss(out,b.target_labels,b.target_metric_mask);loss.backward()
        value=digest(cpu_tree(dict(logits=out,loss=loss,gradients={n:p.grad for n,p in net.named_parameters()})))
        return dict(digest=value,loss=float(loss),optimizer_calls=0)
    finally:opt.zero_grad(set_to_none=True);restore_rng_state(rng);net.train(was)


def save_checkpoint(net,opt,contract,progress,b,path):
    payload=cpu_tree(net.checkpoint());payload.update(kind=KIND,contract=contract,optimizer_state_dict=cpu_tree(opt.state_dict()),
        progress=cpu_tree(progress),reference_eval_logits=fixed_logits(net,b),reference_probe=probe(net,opt,b),rng_state=cpu_tree(capture_rng_state()))
    payload['component_sha256']={key:digest(payload[key]) for key in COMPONENTS}
    return atomic_save(path,payload,contract)


def restore_checkpoint(payload,net,opt,contract,b):
    checkpoint_validate(payload,contract);net.load_state_dict(payload['state_dict'],strict=True);opt.load_state_dict(payload['optimizer_state_dict'])
    restore_rng_state(payload['rng_state']);before=(digest(cpu_tree(net.state_dict())),digest(cpu_tree(opt.state_dict())),digest(cpu_tree(capture_rng_state())))
    old=opt.step;calls=0
    def forbidden(*args,**kwargs):
        nonlocal calls
        calls+=1;raise RuntimeError('no Adam call allowed during recovery')
    opt.step=forbidden
    try:
        if not torch.equal(fixed_logits(net,b),payload['reference_eval_logits']) or probe(net,opt,b)!=payload['reference_probe']:raise RuntimeError('prediction/RNG-gradient recovery mismatch')
        after=(digest(cpu_tree(net.state_dict())),digest(cpu_tree(opt.state_dict())),digest(cpu_tree(capture_rng_state())))
        if before!=after or calls:raise RuntimeError('recovery changed model/Adam/RNG')
        return dict(status='PASS',optimizer_calls=0,fixed_logit_max_abs_difference=0.,next_probe='EXACT',epoch=payload['progress']['epoch'])
    finally:opt.step=old


class Boundaries:
    def __init__(self,plan_path,record,output,started):
        self.path=plan_path;self.record=record;self.output=output;self.started=started;self.n=0;self.last=None;self.previous=set();self.allocator_cached=None
    def check(self,phase,*,cuda=True,admission=False):
        limits=authorize(self.record,self.path);shared.check_exposure_and_hold(shared.TRANSFERRED_MIG)
        host=host_memory();obs=shared.observe(shared.TRANSFERRED_MIG);decision=shared.assess(obs,host,admission=admission)
        if obs['device'].get('free_bytes',0)<limits['admission_free_bytes' if admission else 'runtime_free_bytes']:decision['reasons'].append('epoch budget headroom floor');decision['status']='STOP'
        pids={x['pid'] for x in decision['reported_compute_processes']}
        self.last=dict(phase=phase,observation=obs,host=host,decision=decision,pids_added=sorted(pids-self.previous),pids_departed=sorted(self.previous-pids),pid_mapping_proven=False,cached_allocator=self.allocator_cached)
        self.n+=1;write_json(self.output/f'resources_{self.n:05d}.json',self.last);self.previous=pids
        if decision['status']=='STOP':raise shared.SharedResourceStop('; '.join(decision['reasons']),self.last)
        if time.time()-self.started>2700:raise TimeoutError('unchanged45min execution ceiling')
        if cuda:
            own=dict(allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            self.allocator_cached={**own,'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            self.last.update(own);write_json(self.output/f'allocator_{self.n:05d}.json',self.last)
            if max(own['allocated_bytes'],own['reserved_bytes'])>limits['allocator_bytes']:raise MemoryError('explicit epoch allocator/reserved cap')
        return self.last


def context(path,record,phase,*,allow_completed=False):
    limits=authorize(record,path);plan,splits=load_plan(path);root=path.parent
    if root.resolve()!=(ROOT/'experiments/hitskt_favor_plus_v1/assist2017'/plan['contract']['run_id']).resolve():raise ValueError('epoch namespace mismatch')
    if (root/'study_failure.json').exists() or ((root/'study_result.json').exists() and not allow_completed):raise RuntimeError('failed/completed study cannot restart')
    admission=shape_admission(plan,limits)
    if admission['status']!='PASS_ALL_SHAPES':raise BudgetBlocked(json.dumps(admission))
    output=root/phase;output.mkdir(exist_ok=False);started=record.get('campaign_started_unix',time.time())
    b=Boundaries(path,record,output,started)
    for i in range(2):
        b.check('admission',cuda=False,admission=True)
        if i==0:time.sleep(.2)
    return plan,splits,output,b,limits


def initialize(plan,limits):
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    seed_everything(42);torch.use_deterministic_algorithms(True)
    total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(limits['allocator_bytes']/total,0)
    c=plan['contract'];net=HiTSKTFavorPlus(model_config('assist2017',c['model_config']['num_questions'],c['model_config']['num_skills']),FavorPlusConfig(**c['favor_config'])).to('cuda:0')
    opt=torch.optim.Adam(net.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    if {k:digest(v.cpu()) for k,v in net.state_dict().items() if k.endswith('.projection')}!=c['expected_projection_sha256']:raise ValueError('fresh projection policy mismatch')
    return net,opt


def record_failure(root,out,exc,b,phase,progress):
    value=dict(status='FAIL',phase=phase,type=type(exc).__name__,message=str(exc),progress=cpu_tree(progress),
        last_cached_resources=b.last,no_cuda_calls_by_handler=True,no_automatic_retry=True,
        valid_checkpoints=[dict(path=str(x),sha256=sha256(x)) for x in sorted((root/'checkpoints').glob('*.pt'))])
    # Metric prefix tensors belong to checkpoints; keep failure JSON primitive.
    value['progress']={k:v for k,v in value['progress'].items() if not isinstance(v,torch.Tensor)}
    write_json(out/'failure.json',value)
    if not (root/'study_failure.json').exists():write_json(root/'study_failure.json',value)


def profile(path,record):
    plan,splits,out,b,limits=context(path,record,'cuda_memory_profile')
    rows=plan['schedules']['train']['1'];small={**rows[0],'indices':rows[0]['indices'][:1]}
    small=shape_manifest(splits['train'],[small['indices']])[0]
    ordered=sorted(range(len(rows)),key=lambda i:rows[i]['padded_tokens'])
    train=splits['train']
    workspace_index=max(range(len(rows)),key=lambda i:sum(len(train.shape(j).history_lengths) for j in rows[i]['indices'])*min(32,max(max(train.shape(j).history_lengths)+1 for j in rows[i]['indices'])))
    chosen=[('train',small),*[('train',rows[i]) for i in dict.fromkeys([ordered[0],ordered[len(ordered)//2],ordered[-1],workspace_index])]]
    chosen.append(('validation',max(plan['schedules']['validation']['0'],key=lambda r:r['padded_tokens'])))
    progress=dict(optimizer_updates=0);results=[]
    reserve=[]
    try:
        net,opt=initialize(plan,limits);b.check('initialized');snap=digest(cpu_tree(net.state_dict()));rng=capture_rng_state()
        # Account for both Adam moments without taking a fake optimizer step.
        reserve=[torch.zeros_like(p) for p in net.parameters() for _ in range(2)];b.check('adam_size_buffers_reserved')
        for i,(split,row) in enumerate(chosen):
            b.check('before_profile');net.train(split=='train');opt.zero_grad(set_to_none=True);batch=make_batch(splits[split],row).to('cuda:0')
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.set_grad_enabled(split=='train'):
                loss=masked_bce_loss(net(batch),batch.target_labels,batch.target_metric_mask)
                if split=='train':loss.backward()
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            if not math.isfinite(float(loss)) or float(loss)>20:raise FloatingPointError('profile loss failure')
            norm=None
            if split=='train':
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):raise FloatingPointError('profile gradient failure')
                norm=math.hypot(*(float(p.grad.double().norm()) for p in net.parameters()))
                if norm>100 or not math.isfinite(norm):raise FloatingPointError('profile gradient severe gate')
            if seconds>30:raise TimeoutError('unchanged30s profile-step ceiling')
            results.append(dict(case=i,split=split,shape=row,seconds=seconds,loss=float(loss),gradient_l2=norm,resources=b.check('after_profile'),optimizer_calls=0))
            del loss,batch;opt.zero_grad(set_to_none=True);restore_rng_state(rng)
        if digest(cpu_tree(net.state_dict()))!=snap:raise RuntimeError('zero-update profile changed weights/projections')
        result=dict(status='PASS',cases=results,optimizer_calls=0,adam_moment_bytes_reserved=sum(x.numel()*x.element_size() for x in reserve),runtime=runtime(torch.device('cuda:0')),plan_sha256=sha256(path),budget_profile=record['budget_profile'])
        write_json(out/'profile_result.json',result);return result
    except BaseException as exc:record_failure(path.parent,out,exc,b,'profile',progress);raise


def epoch_data(net,opt,ds,rows,device,boundary,progress,*,training,checkpoint_callback=None,checkpoint_interval=25,logits_output=None):
    net.train(training);count=target_count(ds);probabilities=np.empty(count,np.float32);labels=np.empty(count,np.uint8)
    cursor=0;total=0.;seen=[];updates=[];start_batch=0
    if training and progress.get('next_batch',0):
        start_batch=progress['next_batch'];cursor=progress['train_targets_done'];total=progress['train_loss_sum']
        probabilities[:cursor]=progress['train_probabilities'].numpy();labels[:cursor]=progress['train_labels'].numpy()
        seen=[i for row in rows[:start_batch] for i in row['indices']]
        if cursor!=sum(row['supervised_targets'] for row in rows[:start_batch]):raise ValueError('resumed metric-prefix cursor mismatch')
    for index in range(start_batch,len(rows)):
        row=rows[index]
        boundary.check('train_batch' if training else 'validation_batch');batch=make_batch(ds,row).to(device)
        with torch.set_grad_enabled(training):
            if training:opt.zero_grad(set_to_none=True)
            if training:torch.cuda.synchronize() if device.type=='cuda' else None
            start=time.perf_counter();logits=net(batch);loss=masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask)
            if not finite_tree(logits) or not math.isfinite(float(loss)) or float(loss)>20:raise FloatingPointError('nonfinite/excessive loss/logits')
            if training:
                loss.backward();grads=[p.grad for p in net.parameters()]
                if any(g is None or not torch.isfinite(g).all() for g in grads):raise FloatingPointError('nonfinite/missing gradient')
                norm=math.hypot(*(float(g.double().norm()) for g in grads))
                if not math.isfinite(norm) or norm>100:raise FloatingPointError('gradient severe gate')
                boundary.check('pre_adam')
                if time.perf_counter()-start>30:raise TimeoutError('30s pre-Adam gate')
                progress['optimizer_step_calls']+=1;opt.step();progress['optimizer_updates']+=1
                if device.type=='cuda':torch.cuda.synchronize()
                if not finite_tree(net.state_dict()) or not finite_tree(opt.state_dict()):raise FloatingPointError('model/Adam nonfinite')
                seconds=time.perf_counter()-start
                if seconds>30:raise TimeoutError('30s optimizer-step gate')
                updates.append(dict(batch=index,loss=float(loss),gradient_l2=norm,seconds=seconds,optimizer_updates=progress['optimizer_updates']))
        valid=batch.target_metric_mask;n=int(valid.sum());probabilities[cursor:cursor+n]=logits[valid].detach().sigmoid().cpu().numpy();labels[cursor:cursor+n]=batch.target_labels[valid].cpu().numpy()
        if logits_output is not None:logits_output[cursor:cursor+n]=logits[valid].detach().cpu().numpy()
        total+=float(loss.detach())*n;cursor+=n;seen.extend(row['indices'])
        if training:
            progress.update(next_batch=index+1,train_loss_sum=total,train_targets_done=cursor,
                train_probabilities=torch.from_numpy(probabilities[:cursor].copy()),train_labels=torch.from_numpy(labels[:cursor].copy()))
            boundary.check('after_update');time.sleep(.1)
            if checkpoint_callback is not None and ((index+1)%checkpoint_interval==0 or index+1==len(rows)):checkpoint_callback(index+1)
    if cursor!=count or sorted(seen)!=list(range(len(ds))):raise ValueError('full-epoch missing/duplicate labels/sessions')
    metrics=binary_metrics(torch.from_numpy(probabilities),torch.from_numpy(labels),loss=total/count);metrics['bce']=metrics['loss']
    if not finite_tree(metrics):raise FloatingPointError('nonfinite metrics')
    return metrics,probabilities,labels,updates


def diagnostics(net,train,contract,boundary,out,baseline=None):
    results=[]
    for i,item in enumerate(contract['diagnostic_samples']):
        boundary.check('fixed_training_diagnostic');batch=make_batch(train,shape_manifest(train,[[item['dataset_index']]])[0]).to('cuda:0')
        results.append(fixed_example_diagnostics(net,batch,baseline=None if baseline is None else baseline[i]))
    rules=contract['review_and_severe_stops']['severe_stops'];decision=severe_check(results,{'prospective_severe_stops':rules})
    write_json(out/'attention_diagnostics.json',dict(cases=results,decision=decision,review_threshold=.20))
    if decision['status']=='STOP':raise FloatingPointError('unchanged severe attention gate')
    return results,decision


def execute_epoch(path,record,epoch):
    if type(epoch) is not int or not 1<=epoch<=10:raise ValueError('only1..10 epochs')
    plan,splits,out,b,limits=context(path,record,f'epoch_{epoch:03d}');root=path.parent;c=plan['contract']
    prof=json.loads((root/'cuda_memory_profile/profile_result.json').read_text())
    if prof['status']!='PASS' or prof['plan_sha256']!=sha256(path) or prof['budget_profile']!=record['budget_profile']:raise RuntimeError('matching zero-update memory profile prerequisite')
    progress=dict(epoch=epoch,next_batch=0,optimizer_updates=0,optimizer_step_calls=0,best_auc=None,best_f1=None,bad_epochs=0,phase='TRAIN')
    try:
        net,opt=initialize(plan,limits);b.check('initialized');write_json(out/'runtime.json',runtime(torch.device('cuda:0')))
        proof_batch=make_batch(splits['train'],shape_manifest(splits['train'],[[c['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0')
        if epoch>1:
            previous=root/'checkpoints'/f'epoch_{epoch-1:03d}.pt';receipt=json.loads((root/f'recovery_{epoch-1:03d}/recovery_result.json').read_text())
            if receipt['status']!='PASS' or receipt['checkpoint_sha256']!=sha256(previous):raise ValueError('fresh previous-epoch recovery required')
            payload=torch.load(previous,map_location='cpu',weights_only=True);restore_checkpoint(payload,net,opt,c,proof_batch);progress=payload['progress'];del payload
            if progress.get('stop'):raise RuntimeError('patience3/max10 stop already reached')
            if progress['epoch']!=epoch-1 or progress.get('phase')!='EPOCH_COMPLETE':raise ValueError('previous completed epoch identity required')
            progress.update(epoch=epoch,next_batch=0,phase='TRAIN',train_targets_done=0,train_loss_sum=0.,train_probabilities=torch.empty(0),train_labels=torch.empty(0,dtype=torch.uint8))
        baseline_path=root/'epoch_001'/'initial_attention_diagnostics.json'
        if epoch==1:
            initial_dir=out/'initial';initial_dir.mkdir();initial,_=diagnostics(net,splits['train'],c,b,initial_dir)
            write_json(baseline_path,initial)
        baseline=json.loads(baseline_path.read_text())
        def checkpoint_callback(cursor):
            b.check('checkpoint');receipt=save_checkpoint(net,opt,c,progress,proof_batch,root/'checkpoints'/f'epoch_{epoch:03d}_batch_{cursor:04d}.pt')
            write_json(out/f'checkpoint_batch_{cursor:04d}.json',receipt)
        tm,_,_,updates=epoch_data(net,opt,splits['train'],plan['schedules']['train'][str(epoch)],torch.device('cuda:0'),b,progress,training=True,checkpoint_callback=checkpoint_callback)
        write_json(out/'training_epoch_metrics.json',dict(metrics=tm,updates=updates,full_targets_verified=True))
        logits_saved=np.empty(c['target_counts']['validation']['labels'],np.float32)
        vm,prob,y,_=epoch_data(net,opt,splits['validation'],plan['schedules']['validation']['0'],torch.device('cuda:0'),b,progress,training=False,logits_output=logits_saved)
        write_json(out/'full_validation_metrics.json',dict(metrics=vm,full_targets_verified=True,cleared_for_next_epoch=False))
        with (out/'full_validation_predictions.npz').open('xb') as f:np.savez_compressed(f,probability=prob,label=y,logits=logits_saved)
        final_dir=out/'final_diagnostics';final_dir.mkdir();_,decision=diagnostics(net,splits['train'],c,b,final_dir,baseline)
        progress,improved=selection(progress,epoch,vm);progress.update(phase='EPOCH_COMPLETE',train_metrics=tm,validation_metrics=vm)
        receipt=save_checkpoint(net,opt,c,progress,proof_batch,root/'checkpoints'/f'epoch_{epoch:03d}.pt')
        if improved:write_json(root/'metrics'/f'best_selection_{epoch:03d}.json',dict(epoch=epoch,checkpoint=receipt,validation=vm))
        result=dict(status='PASS_FULL_EPOCH',epoch=epoch,train=tm,validation=vm,improved=improved,selection={k:v for k,v in progress.items() if k in ('best_auc','best_epoch','best_f1','best_f1_epoch','bad_epochs','stop')},
            updates=updates,checkpoint=receipt,attention_status=decision['status'],resources=b.check('epoch_complete'),test_evaluations=0)
        write_json(root/'metrics'/f'epoch_{epoch:03d}.json',result);write_json(out/'epoch_result.json',result);curves(root)
        return result
    except BaseException as exc:record_failure(root,out,exc,b,'epoch',progress);raise


def verify_epoch(path,record,epoch):
    if type(epoch) is not int or not 1<=epoch<=10:raise ValueError('epoch1..10 required')
    plan,splits,out,b,limits=context(path,record,f'recovery_{epoch:03d}');root=path.parent;c=plan['contract'];progress={}
    try:
        checkpoint=root/'checkpoints'/f'epoch_{epoch:03d}.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=True);progress=payload['progress']
        if progress['epoch']!=epoch or progress.get('phase')!='EPOCH_COMPLETE':raise ValueError('fresh recovery requires matching complete epoch')
        net,opt=initialize(plan,limits);batch=make_batch(splits['train'],shape_manifest(splits['train'],[[c['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0')
        result=restore_checkpoint(payload,net,opt,c,batch);result.update(checkpoint_sha256=sha256(checkpoint),resources=b.check('recovery_complete'))
        write_json(out/'recovery_result.json',result)
        if progress['stop']:
            write_json(root/'study_result.json',dict(status='EPOCH_TRAINING_COMPLETE',epochs_completed=epoch,best_epoch=progress['best_epoch'],best_validation_auc=progress['best_auc'],best_validation_f1=progress['best_f1'],
                best_validation_f1_epoch=progress['best_f1_epoch'],best_checkpoint_path=str(root/'checkpoints'/f'epoch_{progress["best_epoch"]:03d}.pt'),best_checkpoint_sha256=sha256(root/'checkpoints'/f'epoch_{progress["best_epoch"]:03d}.pt'),
                optimizer_updates=progress['optimizer_updates'],test_evaluations=0,no_automatic_more_training=True))
        return result
    except BaseException as exc:record_failure(root,out,exc,b,'recovery',progress);raise


def curves(root):
    rows=[json.loads(p.read_text()) for p in sorted((root/'metrics').glob('epoch_*.json'))]
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5));x=[r['epoch'] for r in rows]
    axes[0].plot(x,[r['train']['loss'] for r in rows],label='training BCE');axes[0].plot(x,[r['validation']['loss'] for r in rows],label='full validation BCE');axes[0].legend()
    for ax,key in zip(axes[1:],('roc_auc','f1')):ax.plot(x,[r['validation'][key] for r in rows]);ax.set_ylabel('full validation '+key)
    for ax in axes:ax.set_xlabel('completed epoch');ax.set_xticks(x)
    fig.tight_layout();fig.savefig(root/f'curves_through_epoch_{x[-1]:03d}.png',dpi=160);plt.close(fig)


def validate_best(path,record):
    """Fresh full-validation replay only. Never constructs/evaluates test data."""
    if not (path.parent/'study_result.json').exists():raise ValueError('final selection must exist before best replay')
    selected=json.loads((path.parent/'study_result.json').read_text());epoch=selected['best_epoch']
    plan,splits,out,b,limits=context(path,record,'best_full_validation_replay',allow_completed=True)
    net,opt=initialize(plan,limits);checkpoint=path.parent/'checkpoints'/f'epoch_{epoch:03d}.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    proof=make_batch(splits['train'],shape_manifest(splits['train'],[[plan['contract']['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0')
    recovered=restore_checkpoint(payload,net,opt,plan['contract'],proof)
    logits=np.empty(plan['contract']['target_counts']['validation']['labels'],np.float32)
    metrics,prob,y,_=epoch_data(net,None,splits['validation'],plan['schedules']['validation']['0'],torch.device('cuda:0'),b,{},training=False,logits_output=logits)
    expected=json.loads((path.parent/'metrics'/f'epoch_{epoch:03d}.json').read_text())['validation']
    comparison={key:abs(float(metrics[key])-float(expected[key])) for key in ('roc_auc','accuracy','precision','recall','f1','mse','bce')}
    with np.load(path.parent/f'epoch_{epoch:03d}'/'full_validation_predictions.npz') as z:
        exact=all(np.array_equal(z[key],value) for key,value in (('probability',prob),('label',y),('logits',logits)))
    result=dict(status='PASS' if exact and max(comparison.values())<=1e-6 else 'FAIL',epoch=epoch,validation=metrics,
        comparison_abs=comparison,unchanged_metric_tolerance=1e-6,arrays_exact=exact,checkpoint_sha256=sha256(checkpoint),recovery=recovered,
        optimizer_calls=0,test_evaluations=0,resources=b.check('best_validation_replay_complete'))
    write_json(out/'validation_replay.json',result)
    if result['status']!='PASS':raise RuntimeError('fresh best full-validation reproduction failed')
    return result
