"""Separate matched ORF128/512 epoch controller; no historical module mutation.

Profiling is zero-update and independently scoped. Real training requires an
actual new pair-runtime approval record. No test dataset is ever constructed.
"""
from __future__ import annotations
from dataclasses import asdict
import datetime as dt
import hashlib,json,math,os,random,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
from ktbench import favor_plus_shared_gpu as shared
from ktbench.favor_plus_pipeline import ROOT,sha256,write_json,runtime,target_count,fresh_run
from ktbench.favor_plus_matched_comparison import MatchedArmConfig,MatchedComparisonModel,learned_digest,arm_projection
from ktbench.models.favor_plus import FavorPlusAttention,causal_favor_attention
from ktbench.models.hitskt import HiTSKTConfig
from ktbench.favor_plus_assist_epochs import (audit_dataset,shape_manifest,make_batch,epoch_data,
    fixed_logits,probe,curves)
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree,finite_tree,fixed_example_diagnostics
from ktbench.favor_plus_pilot_recovery import digest
from ktbench.favor_plus_continuation_recovery import optimizer_groups_equal
from ktbench.favor_plus_bounded_assist import severe_check
from ktbench.favor_plus_real_pilot import host_memory
from ktbench.favor_plus_performance_study import canonical_sha,validate_plan,storage_components
from ktbench.favor_plus_post50_diagnostics import attention_diagnostics
from ktbench.data.session_store import SessionStore
from ktbench.metrics import masked_bce_loss
from ktbench.training import capture_rng_state,restore_rng_state

ARMS=('orf128','orf512');KIND='hitskt_favor_plus_matched_pair_epochs_v1';FORMAT=5
PROFILE_ACTION='favor_plus_pair_full_shape_profile_only_v1'
TRAIN_ACTION='favor_plus_pair_eight_epochs_v1'
GIB=1024**3
LIMITS=dict(allocator_bytes=8*GIB,advisory_total_bytes=10*GIB,admission_free_bytes=16*GIB,
    runtime_free_bytes=6*GIB,host_rss_bytes=4*GIB,host_headroom_bytes=2*GIB,
    max_step_seconds=30,pause_seconds=.1)
BLUEPRINT=ROOT/'reports/favor_plus_v1/performance_study_20261025_v1'
PARENT=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1/prepared_plan.json'
SOURCES=('ktbench/favor_plus_paired_epochs.py','scripts/run_favor_plus_paired_epochs.py',
    'ktbench/favor_plus_matched_comparison.py','ktbench/favor_plus_performance_study.py',
    'ktbench/favor_plus_post50_diagnostics.py')
COMPONENTS=('state_dict','optimizer_state_dict','rng_state','progress','contract','reference_eval_logits','reference_probe')


class PairedEpochModel(MatchedComparisonModel):
    def checkpoint(self):
        value=super().checkpoint();value.update(format_version=FORMAT,variant=KIND)
        return value

    @classmethod
    def from_checkpoint(cls,path,*,map_location='cpu'):
        v=torch.load(path,map_location=map_location,weights_only=True)
        if v.get('format_version')!=FORMAT or v.get('variant')!=KIND:raise ValueError('not a paired epoch checkpoint')
        net=cls(HiTSKTConfig(**v['config']),MatchedArmConfig(**v['comparison_config']))
        net.load_state_dict(v['state_dict'],strict=True);return net


def fresh_model(config,arm):
    if arm not in ARMS:raise ValueError('primary ORF pair only')
    torch.random.default_generator.manual_seed(42)
    return PairedEpochModel(config,MatchedArmConfig(arm,accumulation_dtype='float64'))


def selection(progress,epoch,metrics):
    if type(epoch) is not int or not 1<=epoch<=8 or not finite_tree(metrics):raise ValueError('finite completed epoch1..8 required')
    if progress.get('last_selected_epoch',0)!=epoch-1:raise ValueError('selection must follow consecutive complete epochs')
    auc=float(metrics['roc_auc']);f1=float(metrics['f1']);improved=progress.get('best_auc') is None or auc>progress['best_auc']
    p={**progress,'last_selected_epoch':epoch,'stop':epoch==8}
    if improved:p.update(best_auc=auc,best_epoch=epoch)
    if p.get('best_f1') is None or f1>p['best_f1']:p.update(best_f1=f1,best_f1_epoch=epoch)
    return p,improved


def prepare(run_id):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU preparation requires CUDA hidden')
    if not run_id or run_id in ('.','..') or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in run_id):raise ValueError('simple fresh run ID required')
    blueprint=json.loads((BLUEPRINT/'EXPERIMENT_PLAN.json').read_text());validate_plan(blueprint)
    old=json.loads(PARENT.read_text());store=SessionStore(old['store_path']);identity,splits,counts=audit_dataset(store)
    schedules=dict(train=json.loads((BLUEPRINT/'COMMON_TRAIN_SCHEDULES.json').read_text()),validation=old['schedules']['validation'])
    if canonical_sha(schedules['train'])!=blueprint['schedule_sha256']:raise ValueError('matched epoch schedule changed')
    for split,epochs in schedules.items():
        for epoch,rows in epochs.items():
            if sorted(i for row in rows for i in row['indices'])!=list(range(len(splits[split]))):raise ValueError('coverage changed')
            if rows!=shape_manifest(splits[split],[row['indices'] for row in rows]):raise ValueError('actual complete shapes changed')
    sources={**old['contract']['source_sha256'],**blueprint['source_sha256'],
        **{str(p.relative_to(ROOT)):sha256(p) for p in (PARENT,BLUEPRINT/'EXPERIMENT_PLAN.json',BLUEPRINT/'COMMON_TRAIN_SCHEDULES.json')},
        **{name:sha256(ROOT/name) for name in SOURCES}}
    for name,want in sources.items():
        if sha256(ROOT/name)!=want:raise ValueError('source drift:'+name)
    ids=validation_event_ids(schedules,splits);truth=np.asarray(splits['validation'].store.correct[ids],dtype=np.uint8)
    contracts={};config=HiTSKTConfig(**old['contract']['model_config']);fresh=[]
    for arm in ARMS:
        net=fresh_model(config,arm);opt=torch.optim.Adam(net.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        c=dict(arm=arm,model_config=asdict(config),comparison_config=asdict(net.comparison),seed=42,max_epochs=8,
            threshold=.5,selection='strict full-validation AUC; equal8epochs; no performance early stopping',
            expected_projection_sha256={n:digest(v) for n,v in net.named_buffers() if n.endswith('.projection')},
            fresh_learned_sha256=learned_digest(net),fresh_cpu_rng_sha256=digest(torch.random.get_rng_state()),
            parameter_tensor_count=len(list(net.parameters())),expected_optimizer_groups=cpu_tree(opt.state_dict()['param_groups']),
            sampler_sha256=digest(schedules),sampler_schedule={s:{e:[r['target_sessions'] for r in rr] for e,rr in ee.items()} for s,ee in schedules.items()},
            diagnostic_samples=old['contract']['diagnostic_samples'],review_and_severe_stops=old['contract']['review_and_severe_stops'],
            dataset_identity=identity,target_counts=counts,source_sha256=sources,protocol=old['contract']['protocol'],
            validation_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),validation_labels_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
            diagnostic_prefixes=[32,128],audit_projection_seeds=[17,101])
        contracts[arm]=c;fresh.append((c['fresh_learned_sha256'],c['fresh_cpu_rng_sha256']));del net,opt
    if len(set(fresh))!=1:raise ValueError('initialization/RNG is not matched')
    root=fresh_run('assist2017',run_id)
    for arm in ARMS:
        for sub in ('checkpoints','metrics'):(root/arm/sub).mkdir(parents=True,exist_ok=False)
    plan=dict(kind=KIND,run_id=run_id,contracts=contracts,schedules=schedules,store_path=str(store.root),
        source_sha256=sources,limits=LIMITS,test_evaluations=0,training_approved=False,
        profile_max_seconds=600,proposed_pair_training_seconds=7200,proposed_arm_training_seconds=3600,
        notice='Profile10min zeroAdam separately authorized; proposed60min/arm replaces old45min ONLY after actual new explicit runtime approval.')
    write_json(root/'prepared_plan.json',plan)
    for action,name in ((PROFILE_ACTION,'PROFILE_RECORD_TEMPLATE.json'),(TRAIN_ACTION,'TRAINING_RECORD_TEMPLATE.json')):
        write_json(root/name,dict(action=action,prepared_plan_sha256=sha256(root/'prepared_plan.json'),resource_uuid=shared.TRANSFERRED_MIG,
            limits=LIMITS,shared_permission_confirmed=True,known_explicit_restriction=False,known_incompatible_reservation=False,
            approval_is_proposal=True,actual_approval_evidence='',profile_approved=False,training_approved=False,
            max_optimizer_updates=0 if action==PROFILE_ACTION else None,max_epochs_per_arm=8,
            total_seconds=600 if action==PROFILE_ACTION else 7200,per_arm_seconds=600 if action==PROFILE_ACTION else 3600,
            starts_utc='SET_AFTER_ACTUAL_APPROVAL',expires_utc='SET_AFTER_ACTUAL_APPROVAL',campaign_started_unix=None,arm_started_unix={}))
    return dict(status='PREPARED_NOT_TRAINING_AUTHORIZED',plan=str(root/'prepared_plan.json'),plan_sha256=sha256(root/'prepared_plan.json'),fresh_parameters_and_cpu_rng_matched=True,
        train_updates_per_arm=sum(map(len,schedules['train'].values())),counts=counts)


def load_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan['kind']!=KIND or tuple(plan['contracts'])!=ARMS or plan['limits']!=LIMITS:raise ValueError('paired contract identity mismatch')
    if Path(path).parent.resolve()!=(ROOT/'experiments/hitskt_favor_plus_v1/assist2017'/plan['run_id']).resolve():raise ValueError('separate namespace required')
    for name,want in plan['source_sha256'].items():
        if sha256(ROOT/name)!=want:raise ValueError('paired source drift:'+name)
    identity,splits,counts=audit_dataset(SessionStore(plan['store_path']))
    for arm,c in plan['contracts'].items():
        if c['dataset_identity']!=identity or c['target_counts']!=counts or c['max_epochs']!=8 or c['seed']!=42 or c['threshold']!=.5:raise ValueError('data/protocol changed')
        if c['sampler_sha256']!=digest(plan['schedules']):raise ValueError('schedule changed')
        if c['comparison_config']!=asdict(MatchedArmConfig(arm,accumulation_dtype='float64')):raise ValueError('attention policy changed')
    return plan,splits


def authorize(record,path,action,arm=None,*,now=None):
    if (record.get('action')!=action or record.get('prepared_plan_sha256')!=sha256(path) or record.get('resource_uuid')!=shared.TRANSFERRED_MIG
        or os.environ.get('CUDA_VISIBLE_DEVICES')!=shared.TRANSFERRED_MIG or record.get('limits')!=LIMITS
        or record.get('shared_permission_confirmed') is not True or record.get('known_explicit_restriction') is not False
        or record.get('known_incompatible_reservation') is not False or record.get('approval_is_proposal') is not False
        or not isinstance(record.get('actual_approval_evidence'),str) or not record['actual_approval_evidence'].strip()):raise PermissionError('actual scope-bound approval required')
    current=now or dt.datetime.now(dt.timezone.utc);start=dt.datetime.fromisoformat(record['starts_utc']);end=dt.datetime.fromisoformat(record['expires_utc'])
    if start.tzinfo is None or end.tzinfo is None or not start<=current<end or not 0<(end-start).total_seconds()<=7200:raise PermissionError('fresh <=2h scope required')
    if action==PROFILE_ACTION:
        if record.get('profile_approved') is not True or record.get('max_optimizer_updates')!=0 or record.get('total_seconds')!=600:raise PermissionError('profile is10min zeroAdam only')
    elif action==TRAIN_ACTION:
        if (record.get('training_approved') is not True or record.get('eight_epoch_pair_runtime_approved') is not True
            or record.get('total_seconds')!=7200 or record.get('per_arm_seconds')!=3600 or record.get('max_epochs_per_arm')!=8
            or not record.get('actual_new_time_budget_evidence')):raise PermissionError('NEW explicit eight-epoch/2hpair/60minarm approval required')
        if arm not in ARMS or arm not in record.get('arm_started_unix',{}):raise PermissionError('explicit arm clock required')
    else:raise PermissionError('unregistered action')
    stamp=current.timestamp();origin=record.get('campaign_started_unix')
    if not isinstance(origin,(float,int)) or not start.timestamp()<=origin<=stamp or stamp-origin>record['total_seconds']:raise TimeoutError('global scope clock expired or invalid')
    if action==TRAIN_ACTION:
        arm_start=record['arm_started_unix'][arm]
        if not isinstance(arm_start,(float,int)) or not origin<=arm_start<=stamp or stamp-arm_start>3600:raise TimeoutError('arm60min ceiling')


class Boundaries:
    def __init__(self,path,record,out,action,arm=None):
        self.path,self.record,self.output,self.action,self.arm=path,record,out,action,arm;self.n=0;self.last=None;self.cached=None
        authorize(record,path,action,arm)
    def check(self,phase,*,cuda=True,admission=False):
        authorize(self.record,self.path,self.action,self.arm);shared.check_exposure_and_hold(shared.TRANSFERRED_MIG)
        obs=shared.observe(shared.TRANSFERRED_MIG);host=host_memory();decision=shared.assess(obs,host,admission=admission)
        if obs['device'].get('free_bytes',0)<LIMITS['admission_free_bytes' if admission else 'runtime_free_bytes']:
            decision['status']='STOP';decision['reasons'].append('paired approved headroom floor')
        self.last=dict(phase=phase,observation=obs,host=host,decision=decision,cached_own_allocator=self.cached);self.n+=1
        write_json(self.output/f'resources_{self.n:05d}.json',self.last)
        if decision['status']=='STOP':raise shared.SharedResourceStop('; '.join(decision['reasons']),self.last)
        if cuda:
            self.cached=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            self.last['own_allocator']=self.cached;write_json(self.output/f'allocator_{self.n:05d}.json',self.last)
            if max(self.cached.values())>LIMITS['allocator_bytes']:raise MemoryError('unchanged8GiB allocated/reserved/peak cap')
        return self.last
    def admit(self):
        self.check('admission1',cuda=False,admission=True);time.sleep(.2);self.check('admission2',cuda=False,admission=True)


def initialize(contract,boundary):
    boundary.admit();torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(8*GIB/total,0);torch.cuda.manual_seed(42);random.seed(42);np.random.seed(42)
    net=fresh_model(HiTSKTConfig(**contract['model_config']),contract['arm'])
    if learned_digest(net)!=contract['fresh_learned_sha256'] or digest(torch.random.get_rng_state())!=contract['fresh_cpu_rng_sha256']:raise ValueError('fresh initialization/RNG drift')
    if {n:digest(v) for n,v in net.named_buffers() if n.endswith('.projection')}!=contract['expected_projection_sha256']:raise ValueError('projection drift')
    net=net.cuda();boundary.check('initialized');return net


def context(path,record,action,phase,arm=None):
    path=Path(path);authorize(record,path,action,arm);plan,splits=load_plan(path)
    if (path.parent/'study_failure.json').exists():raise RuntimeError('failed study cannot retry automatically')
    out=path.parent/phase;out.mkdir(parents=True,exist_ok=False);b=Boundaries(path,record,out,action,arm)
    return plan,splits,out,b


def failure(root,out,b,exc,phase,progress):
    value=dict(status='STOP_NO_RETRY',phase=phase,type=type(exc).__name__,message=str(exc),
        progress={k:v for k,v in progress.items() if not isinstance(v,torch.Tensor)},last_cached_resources=b.last,
        checkpoints=[dict(path=str(x),sha256=sha256(x)) for x in sorted(root.glob('*/checkpoints/*.pt')) if not x.name.startswith('.')],no_cuda_calls_in_handler=True)
    write_json(out/'failure.json',value);write_json(root/'study_failure.json',value)


def profile_cases(plan,splits):
    rows=[r for batches in plan['schedules']['train'].values() for r in batches];ds=splits['train']
    def history_max(r):return max(max(ds.shape(i).history_lengths)+1 for i in r['indices'])
    def target_max(r):return max(ds.shape(i).target_length+1 for i in r['indices'])
    def workspace(r):return sum(len(ds.shape(i).history_lengths) for i in r['indices'])*min(32,history_max(r))
    order=sorted(range(len(rows)),key=lambda i:rows[i]['padded_tokens'])
    def component(r):return storage_components(r['target_sessions'],ds.store.session_offsets,ds.store.session_number,512)['component_sum_bytes']
    chosen=list(dict.fromkeys([order[0],order[len(order)//2],order[-1],max(range(len(rows)),key=lambda i:workspace(rows[i])),max(range(len(rows)),key=lambda i:history_max(rows[i])),max(range(len(rows)),key=lambda i:target_max(rows[i])),max(range(len(rows)),key=lambda i:component(rows[i]))]))
    cases=[('train',shape_manifest(ds,[[rows[0]['indices'][0]]])[0])]+[('train',rows[i]) for i in sorted(chosen,key=lambda i:rows[i]['padded_tokens'])]
    val=plan['schedules']['validation']['0'];cases.append(('validation',max(val,key=lambda r:r['padded_tokens'])))
    return cases


def validation_event_ids(schedules,splits):
    offsets=splits['validation'].store.session_offsets
    return np.concatenate([np.arange(offsets[t],offsets[t+1],dtype='<i8') for r in schedules['validation']['0'] for t in r['target_sessions']])


def profile(path,record,arm):
    if arm not in ARMS:raise ValueError('primary pair only')
    plan,splits,out,b=context(path,record,PROFILE_ACTION,f'{arm}/full_shape_profile',arm);progress=dict(optimizer_updates=0);c=plan['contracts'][arm]
    try:
        if arm=='orf512':
            prior=json.loads((Path(path).parent/'orf128/full_shape_profile/profile_result.json').read_text())
            if prior['status']!='PASS_ZERO_UPDATE_FULL_SHAPES' or prior['plan_sha256']!=sha256(path):raise ValueError('exact-plan128 profile before512')
        net=initialize(c,b);before=digest(cpu_tree(net.state_dict()));rng=capture_rng_state();reserve=[torch.zeros_like(p) for p in net.parameters() for _ in range(2)];b.check('moments_reserved');rows=[]
        for index,(split,row) in enumerate(profile_cases(plan,splits)):
            b.check('before_profile');net.train(split=='train');net.zero_grad(set_to_none=True);batch=make_batch(splits[split],row).to('cuda:0')
            torch.cuda.synchronize();stamp=time.perf_counter()
            with torch.set_grad_enabled(split=='train'):
                loss=masked_bce_loss(net(batch),batch.target_labels,batch.target_metric_mask)
                if split=='train':loss.backward()
            torch.cuda.synchronize();seconds=time.perf_counter()-stamp
            if not torch.isfinite(loss) or float(loss)>20:raise FloatingPointError('profile loss gate')
            grad=None
            if split=='train':
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):raise FloatingPointError('profile gradient gate')
                grad=math.hypot(*(float(p.grad.double().norm()) for p in net.parameters()))
                if not math.isfinite(grad) or grad>100:raise FloatingPointError('profile gradient norm severe gate')
            if seconds>30:raise TimeoutError('30s profile step ceiling')
            value=dict(index=index,split=split,shape=row,forward_backward_seconds=seconds,loss=float(loss),gradient_l2=grad,optimizer_calls=0,resources=b.check('after_profile'))
            rows.append(value);write_json(out/f'case_{index:02d}.json',value);del loss,batch;net.zero_grad(set_to_none=True);restore_rng_state(rng);time.sleep(.1)
        if digest(cpu_tree(net.state_dict()))!=before:raise ValueError('profile changed model/projections')
        result=dict(status='PASS_ZERO_UPDATE_FULL_SHAPES',arm=arm,plan_sha256=sha256(path),cases=rows,optimizer_calls=0,weights_projections_unchanged=True,
            moment_bytes_reserved=sum(x.numel()*x.element_size() for x in reserve),runtime=runtime(torch.device('cuda:0')),resources=b.check('profile_complete'))
        write_json(out/'profile_result.json',result);return result
    except BaseException as exc:failure(Path(path).parent,out,b,exc,'profile',progress);raise


def checkpoint_validate(v,c):
    if (v.get('kind')!=KIND or v.get('format_version')!=FORMAT or v.get('variant')!=KIND or v.get('contract')!=c
        or v.get('config')!=c['model_config'] or v.get('comparison_config')!=c['comparison_config']):raise ValueError('paired checkpoint identity mismatch')
    if set(v.get('component_sha256',{}))!=set(COMPONENTS):raise ValueError('checkpoint hashes missing')
    for name in COMPONENTS:
        if digest(v[name])!=v['component_sha256'][name]:raise ValueError('checkpoint corruption:'+name)
    if not finite_tree(v):raise FloatingPointError('nonfinite checkpoint')
    if {n:digest(x) for n,x in v['state_dict'].items() if n.endswith('.projection')}!=c['expected_projection_sha256']:raise ValueError('fixed projections changed')
    p=v['progress'];epoch=p['epoch'];step=p['optimizer_updates'];schedule=c['sampler_schedule']['train']
    if type(epoch) is not int or not 1<=epoch<=8 or type(step) is not int or not 0<=p['next_batch']<=len(schedule[str(epoch)]):raise ValueError('invalid epoch/cursor')
    expected=sum(len(rows) for e,rows in schedule.items() if int(e)<epoch)+p['next_batch']
    if step!=expected or p['optimizer_step_calls']!=step:raise ValueError('Adam/sampler counters mismatch')
    opt=v['optimizer_state_dict']
    if not optimizer_groups_equal(opt['param_groups'],c['expected_optimizer_groups']):raise ValueError('Adam settings changed')
    if len(opt['state'])!=(c['parameter_tensor_count'] if step else 0):raise ValueError('Adam state coverage')
    if step and any(float(x['step'])!=step or not (x['exp_avg_sq']>=0).all() for x in opt['state'].values()):raise ValueError('Adam counters/moments mismatch')
    if 'train_probabilities' in p and (p['train_probabilities'].shape!=p['train_labels'].shape or p['train_targets_done']!=len(p['train_labels'])):raise ValueError('metric prefix mismatch')
    return dict(status='PASS',epoch=epoch,optimizer_updates=step)


def save_checkpoint(net,opt,c,progress,proof,path):
    payload=cpu_tree(net.checkpoint());payload.update(kind=KIND,contract=c,progress=cpu_tree(progress),optimizer_state_dict=cpu_tree(opt.state_dict()),
        reference_eval_logits=fixed_logits(net,proof),reference_probe=probe(net,opt,proof),rng_state=cpu_tree(capture_rng_state()))
    payload['component_sha256']={k:digest(payload[k]) for k in COMPONENTS};checkpoint_validate(payload,c)
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.stem,suffix='.partial.pt')
    with os.fdopen(fd,'wb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    restored=torch.load(temp,map_location='cpu',weights_only=True);checkpoint_validate(restored,c)
    if not equal_tree(payload,restored):raise ValueError('serialization mismatch')
    os.link(temp,path);os.unlink(temp);directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return dict(path=str(path),sha256=sha256(path),atomic_no_overwrite=True,**checkpoint_validate(restored,c))


def restore_checkpoint(v,net,opt,c,proof):
    checkpoint_validate(v,c);net.load_state_dict(v['state_dict'],strict=True);opt.load_state_dict(v['optimizer_state_dict']);restore_rng_state(v['rng_state'])
    before=(digest(cpu_tree(net.state_dict())),digest(cpu_tree(opt.state_dict())),digest(cpu_tree(capture_rng_state())))
    original=opt.step
    def forbidden(*a,**kw):raise AssertionError('recovery must never take an optimizer step')
    opt.step=forbidden
    try:
        if not torch.equal(fixed_logits(net,proof),v['reference_eval_logits']) or probe(net,opt,proof)!=v['reference_probe']:raise ValueError('next-computation recovery mismatch')
        after=(digest(cpu_tree(net.state_dict())),digest(cpu_tree(opt.state_dict())),digest(cpu_tree(capture_rng_state())))
        if before!=after:raise ValueError('recovery changed model/Adam/RNG')
        return dict(status='PASS_EXACT',optimizer_calls=0,epoch=v['progress']['epoch'],next_dropout_loss_gradients='EXACT')
    finally:opt.step=original


def diagnostics(net,train,c,b,out,baseline=None):
    rows=[];extended=[];names=('action_encoder','session_encoder','correct_padding_encoder','decoder')
    for i,item in enumerate(c['diagnostic_samples']):
        b.check('fixed_train_diagnostic');batch=make_batch(train,shape_manifest(train,[[item['dataset_index']]])[0]).to(next(net.parameters()).device)
        def save(name,arrays):
            with (out/f'case{i}_{name.replace(".","_")}.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        handles=[]
        def collect(name):
            def hook(module,args,kwargs):
                with torch.no_grad():
                    q0,k0,v0=args;qm=kwargs['query_mask'];km=kwargs['key_mask'];q=module._heads(module.query(torch.where(qm[...,None],q0,0.)));k=module._heads(module.key(torch.where(km[...,None],k0,0.)));v=module._heads(module.value(torch.where(km[...,None],v0,0.)))
                    row=int(km.sum(-1).argmax());length=min(q.shape[2],128);q,k,v=[x[row:row+1,:,:length] for x in (q,k,v)];qm=qm[row:row+1,:length];km=km[row:row+1,:length]
                    actual=causal_favor_attention(q,k,v,module.projection,query_mask=qm,key_mask=km,chunk_size=32,accumulation_dtype='float64')
                    arrays={n:x.detach().cpu().numpy() for n,x in [('q',q),('k',k),('v',v),('query_mask',qm),('key_mask',km),('projection',module.projection),('actual',actual)]}
                    with (out/f'extended_case{i}_{name.replace(".","_")}.npz').open('xb') as f:np.savez_compressed(f,**arrays)
                    metrics=attention_diagnostics(*(arrays[x][0] for x in ('q','k','v')),arrays['projection'],qm.cpu().numpy()[0],km.cpu().numpy()[0],arrays['actual'][0])
                    if metrics['feature_kernel_agreement']!='PASS':raise FloatingPointError('extended independent kernel mismatch')
                    stage=next(j for j,n in enumerate(names) if name.startswith(n+'.'));audits=[]
                    for seed in c['audit_projection_seeds']:
                        projection=arm_projection(q.shape[1],q.shape[-1],MatchedArmConfig(c['arm'],projection_seed=seed,accumulation_dtype='float64'),stage).numpy()
                        audits.append(dict(seed=seed,metrics=attention_diagnostics(*(arrays[x][0] for x in ('q','k','v')),projection,arrays['query_mask'][0],arrays['key_mask'][0])))
                    extended.append(dict(case=i,stage=name,prefix=length,metrics=metrics,audit_draws=audits,review=metrics['relative_output_l2']>.20))
            return hook
        try:
            for name,module in net.core.named_modules():
                if isinstance(module,FavorPlusAttention):handles.append(module.register_forward_pre_hook(collect(name),with_kwargs=True))
            rows.append(fixed_example_diagnostics(net.core,batch,baseline=None if baseline is None else baseline[i],save_prefix=save))
        finally:
            for h in handles:h.remove()
    decision=severe_check(rows,{'prospective_severe_stops':c['review_and_severe_stops']['severe_stops']})
    write_json(out/'attention_diagnostics.json',dict(cases=rows,extended_prefixes=extended,decision=decision,review_threshold=.20,extended_scope='additional REVIEW; original severe approximation gates evaluated on original32-prefix grid'))
    if decision['status']=='STOP':raise FloatingPointError('unchanged severe attention gate')
    return rows,decision


def execute_epoch(path,record,arm,epoch):
    wall=time.perf_counter()
    if arm not in ARMS or type(epoch) is not int or not 1<=epoch<=8:raise ValueError('pair/epoch1..8 only')
    plan,splits,out,b=context(path,record,TRAIN_ACTION,f'{arm}/epoch_{epoch:03d}',arm);root=Path(path).parent;arm_root=root/arm;c=plan['contracts'][arm];progress={}
    try:
        for a in ARMS:
            receipt=json.loads((root/a/'full_shape_profile/profile_result.json').read_text())
            if receipt['status']!='PASS_ZERO_UPDATE_FULL_SHAPES' or receipt['plan_sha256']!=sha256(path):raise ValueError('both exact-plan full-shape profiles required')
        if arm=='orf512':
            prior=json.loads((root/'orf128/best_full_validation_replay/validation_replay.json').read_text())
            if prior['status']!='PASS_EXACT':raise ValueError('finish and verify128 before512')
        net=initialize(c,b);opt=torch.optim.Adam(net.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        proof=make_batch(splits['train'],shape_manifest(splits['train'],[[c['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0')
        progress=dict(epoch=epoch,next_batch=0,optimizer_updates=0,optimizer_step_calls=0,best_auc=None,best_f1=None,last_selected_epoch=0)
        if epoch>1:
            prev=arm_root/'checkpoints'/f'epoch_{epoch-1:03d}.pt';receipt=json.loads((arm_root/f'recovery_{epoch-1:03d}/recovery_result.json').read_text())
            if receipt['status']!='PASS_EXACT' or receipt['checkpoint_sha256']!=sha256(prev):raise ValueError('fresh prior recovery required')
            v=torch.load(prev,map_location='cpu',weights_only=True);restore_checkpoint(v,net,opt,c,proof);progress=v['progress'];del v
            if progress.get('stop') or progress['epoch']!=epoch-1 or progress.get('phase')!='EPOCH_COMPLETE':raise ValueError('prior complete epoch required')
            progress.update(epoch=epoch,next_batch=0,train_targets_done=0,train_loss_sum=0.,train_probabilities=torch.empty(0),train_labels=torch.empty(0,dtype=torch.uint8))
        write_json(out/'runtime.json',runtime(torch.device('cuda:0')))
        if epoch==1:
            init=out/'initial';init.mkdir();initial,_=diagnostics(net,splits['train'],c,b,init);write_json(arm_root/'initial_attention.json',initial)
        baseline=json.loads((arm_root/'initial_attention.json').read_text())
        def checkpoint_callback(cursor):
            b.check('checkpoint');write_json(out/f'checkpoint_{cursor:04d}.json',save_checkpoint(net,opt,c,progress,proof,arm_root/'checkpoints'/f'epoch_{epoch:03d}_batch_{cursor:04d}.pt'))
        progress['phase']='TRAIN';train,_,_,updates=epoch_data(net,opt,splits['train'],plan['schedules']['train'][str(epoch)],torch.device('cuda:0'),b,progress,training=True,checkpoint_callback=checkpoint_callback)
        write_json(out/'training_metrics.json',dict(metrics=train,updates=updates));logits=np.empty(c['target_counts']['validation']['labels'],np.float32)
        val,prob,y,_=epoch_data(net,None,splits['validation'],plan['schedules']['validation']['0'],torch.device('cuda:0'),b,{},training=False,logits_output=logits)
        ids=validation_event_ids(plan['schedules'],splits)
        if hashlib.sha256(ids.tobytes()).hexdigest()!=c['validation_ids_sha256'] or hashlib.sha256(y.tobytes()).hexdigest()!=c['validation_labels_sha256'] or not np.array_equal(y,splits['validation'].store.correct[ids]):raise ValueError('full validation ID/label alignment mismatch')
        with (out/'full_validation_predictions.npz').open('xb') as f:np.savez_compressed(f,probability=prob,label=y,logits=logits,event_id=ids)
        final=out/'final';final.mkdir();_,decision=diagnostics(net,splits['train'],c,b,final,baseline)
        progress,improved=selection(progress,epoch,val);progress.update(phase='EPOCH_COMPLETE',train_metrics=train,validation_metrics=val)
        receipt=save_checkpoint(net,opt,c,progress,proof,arm_root/'checkpoints'/f'epoch_{epoch:03d}.pt')
        result=dict(status='PASS_FULL_EPOCH',arm=arm,epoch=epoch,train=train,validation=val,improved=improved,
            checkpoint=receipt,updates=updates,attention_status=decision['status'],optimizer_updates=progress['optimizer_updates'],epoch_wall_seconds=time.perf_counter()-wall,test_evaluations=0,resources=b.check('epoch_complete'))
        write_json(arm_root/'metrics'/f'epoch_{epoch:03d}.json',result);write_json(out/'epoch_result.json',result);curves(arm_root);return result
    except BaseException as exc:failure(root,out,b,exc,'epoch',progress);raise


def verify_epoch(path,record,arm,epoch):
    if arm not in ARMS or type(epoch) is not int or not 1<=epoch<=8:raise ValueError('pair/epoch1..8 only')
    plan,splits,out,b=context(path,record,TRAIN_ACTION,f'{arm}/recovery_{epoch:03d}',arm);c=plan['contracts'][arm];root=Path(path).parent
    try:
        file=root/arm/'checkpoints'/f'epoch_{epoch:03d}.pt';v=torch.load(file,map_location='cpu',weights_only=True)
        if v['progress'].get('phase')!='EPOCH_COMPLETE' or v['progress']['epoch']!=epoch:raise ValueError('matching complete epoch required')
        net=initialize(c,b);opt=torch.optim.Adam(net.parameters(),lr=5e-5);proof=make_batch(splits['train'],shape_manifest(splits['train'],[[c['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0')
        result=restore_checkpoint(v,net,opt,c,proof);result.update(checkpoint_sha256=sha256(file),resources=b.check('fresh_recovery_complete'));write_json(out/'recovery_result.json',result)
        if epoch==8:write_json(root/arm/'study_result.json',dict(status='EIGHT_EPOCHS_COMPLETE',best_epoch=v['progress']['best_epoch'],best_auc=v['progress']['best_auc'],best_f1=v['progress']['best_f1'],best_f1_epoch=v['progress']['best_f1_epoch'],optimizer_updates=v['progress']['optimizer_updates'],test_evaluations=0))
        return result
    except BaseException as exc:failure(root,out,b,exc,'recovery',{});raise


def validate_best(path,record,arm):
    root=Path(path).parent;selected=json.loads((root/arm/'study_result.json').read_text());epoch=selected['best_epoch']
    plan,splits,out,b=context(path,record,TRAIN_ACTION,f'{arm}/best_full_validation_replay',arm);c=plan['contracts'][arm]
    try:
        net=initialize(c,b);opt=torch.optim.Adam(net.parameters(),lr=5e-5);file=root/arm/'checkpoints'/f'epoch_{epoch:03d}.pt';v=torch.load(file,map_location='cpu',weights_only=True)
        proof=make_batch(splits['train'],shape_manifest(splits['train'],[[c['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0');restore_checkpoint(v,net,opt,c,proof)
        logits=np.empty(c['target_counts']['validation']['labels'],np.float32);metrics,prob,y,_=epoch_data(net,None,splits['validation'],plan['schedules']['validation']['0'],torch.device('cuda:0'),b,{},training=False,logits_output=logits)
        expected=json.loads((root/arm/f'metrics/epoch_{epoch:03d}.json').read_text())['validation']
        with np.load(root/arm/f'epoch_{epoch:03d}/full_validation_predictions.npz') as z:exact=all(np.array_equal(z[k],x) for k,x in [('probability',prob),('label',y),('logits',logits),('event_id',validation_event_ids(plan['schedules'],splits))])
        differences={k:abs(float(metrics[k])-float(expected[k])) for k in ('roc_auc','f1','loss','accuracy','precision','recall','mse')}
        result=dict(status='PASS_EXACT' if exact and max(differences.values())<=1e-6 else 'FAIL',arm=arm,epoch=epoch,validation=metrics,arrays_exact=exact,metric_abs=differences,checkpoint_sha256=sha256(file),optimizer_calls=0,resources=b.check('best_replay_complete'))
        write_json(out/'validation_replay.json',result)
        if result['status']!='PASS_EXACT':raise ValueError('selected full validation replay mismatch')
        return result
    except BaseException as exc:failure(root,out,b,exc,'best_replay',{});raise


def compare_results(path):
    """CPU-only reporting after all sixteen epochs and both best replays exist."""
    plan,_=load_plan(path);root=Path(path).parent;summaries={};rows=[]
    if (root/'study_failure.json').exists():raise ValueError('failed comparison cannot be reported as complete')
    for arm in ARMS:
        result=json.loads((root/arm/'study_result.json').read_text());replay=json.loads((root/arm/'best_full_validation_replay/validation_replay.json').read_text())
        if result['status']!='EIGHT_EPOCHS_COMPLETE' or replay['status']!='PASS_EXACT':raise ValueError('complete8epochs/exactreplay required')
        for epoch in range(1,9):
            record=json.loads((root/arm/f'metrics/epoch_{epoch:03d}.json').read_text())
            recovery=json.loads((root/arm/f'recovery_{epoch:03d}/recovery_result.json').read_text())
            if record['status']!='PASS_FULL_EPOCH' or recovery['status']!='PASS_EXACT' or recovery['checkpoint_sha256']!=record['checkpoint']['sha256']:raise ValueError('epoch/recovery evidence incomplete')
            rows.append(dict(arm=arm,epoch=epoch,train_bce=record['train']['bce'],validation=record['validation'],epoch_wall_seconds=record['epoch_wall_seconds'],attention_status=record['attention_status']))
        summaries[arm]=dict(selected_epoch=replay['epoch'],selected_validation=replay['validation'],checkpoint_sha256=replay['checkpoint_sha256'],optimizer_updates=result['optimizer_updates'])
    gains={key:summaries['orf512']['selected_validation'][key]-summaries['orf128']['selected_validation'][key] for key in ('roc_auc','f1','loss')}
    value=dict(status='COMPLETE_VALIDATION_ONLY_COMPARISON',arms=summaries,epochs=rows,selected_validation_512_minus128=gains,
        significance='NOT ESTABLISHED: one training seed, validation selection',test_evaluations=0,
        next_hypothesis_if512_has_no_auc_gain='separately versioned question-conditioned fusion gate; no automatic training')
    write_json(root/'paired_comparison.json',value);return value


def campaign(path,record_path):
    """Finite foreground workflow. Actual runtime approval required before launch."""
    path,record_path=Path(path),Path(record_path);base=json.loads(record_path.read_text());root=path.parent
    initial={**base,'arm_started_unix':{'orf128':time.time()}}
    authorize(initial,path,TRAIN_ACTION,'orf128');load_plan(path)
    out=root/'approved_campaign';out.mkdir(exist_ok=False);starts={}
    try:
        for arm in ARMS:
            starts[arm]=time.time();record={**base,'arm_started_unix':dict(starts),'parent_record_sha256':sha256(record_path)}
            file=out/f'{arm}_execution_record.json';write_json(file,record)
            stages=[(a,e) for e in range(1,9) for a in ('train-epoch','verify-epoch')]+[('validate-best',None)]
            for action,epoch in stages:
                authorize(record,path,TRAIN_ACTION,arm)
                command=[sys.executable,'scripts/run_favor_plus_paired_epochs.py',action,'--plan',str(path),'--record',str(file),'--arm',arm,'--execute-cuda']
                if epoch is not None:command+=['--epoch',str(epoch)]
                with (out/f'{arm}_{action}_{epoch or 0:03d}.log').open('xb') as log:subprocess.run(command,cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,check=True)
                print(arm,action,epoch,'PASS',flush=True)
        return compare_results(path)
    except BaseException as exc:
        write_json(out/'CAMPAIGN_STOP.json',dict(status='STOP_NO_RETRY',type=type(exc).__name__,message=str(exc),campaign_started_unix=base['campaign_started_unix'],no_other_jobs_changed=True));raise
