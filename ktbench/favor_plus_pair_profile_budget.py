"""Zero-update profiling instrumentation; original paired/model sources untouched.

Timing proxies support resource planning, not optimizer correctness or convergence.
All CUDA stages share the operator's single600s scope. No Adam.step exists here.
"""
from __future__ import annotations
import json,math,os,time
from pathlib import Path
import numpy as np
import torch
from ktbench import favor_plus_paired_epochs as pair

SOURCES=('ktbench/favor_plus_pair_profile_budget.py','scripts/profile_favor_plus_pair_budget.py')


def stratify(rows,bins):
    """Equal-frequency padded-cost bins, deterministic midpoint representative."""
    if not rows or type(bins) is not int or bins<1:raise ValueError('nonempty positive-bin design required')
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['padded_tokens'],i));bins=min(bins,len(order));result=[]
    for chunk in np.array_split(np.asarray(order),bins):
        index=int(chunk[len(chunk)//2]);result.append(dict(weight=len(chunk),index=index,shape=rows[index]))
    if sum(x['weight'] for x in result)!=len(rows):raise ValueError('strata do not represent the whole schedule')
    return result


def prepare(path,out):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU preparation only')
    path,out=Path(path),Path(out);plan,splits=pair.load_plan(path)
    if (path.parent/'study_failure.json').exists():raise ValueError('never retry a failed campaign')
    training=[r for rows in plan['schedules']['train'].values() for r in rows];validation=plan['schedules']['validation']['0']
    design=dict(kind='fixed_B8_zero_update_runtime_profile_v1',plan_sha256=pair.sha256(path),training=stratify(training,32),validation=stratify(validation,8),
        repeats=2,warmups_per_arm=1,io_proxy_repeats=2,initial_diagnostic_passes_per_arm=1,
        checkpoint_count_per_arm=sum(math.ceil(len(rr)/25)+1 for rr in plan['schedules']['train'].values()),
        planned_train_batches=len(training),planned_validation_batches=len(validation),diagnostic_events_per_arm=9,validation_passes_per_arm=9,
        budget_screen_reserve_fraction=.25,profile_max_seconds=600,optimizer_updates=0,
        estimation='weighted median2 repeat observed body time; includes guards/pacing/state-finiteness transfers, excludes real Adam math.25percent prospective planning reserve is NOT an error tolerance or rigorous bound.',
        source_sha256={n:pair.sha256(pair.ROOT/n) for n in SOURCES})
    out.mkdir(exist_ok=False);pair.write_json(out/'RUNTIME_PROFILE_DESIGN.json',design)
    pair.write_json(out/'CORE_PROFILE_DESIGN.json',dict(plan_sha256=pair.sha256(path),cases=[dict(split=s,shape=r) for s,r in pair.profile_cases(plan,splits)],optimizer_updates=0))
    return dict(status='CPU_READY_NO_GPU_UPDATES',train_strata=len(design['training']),validation_strata=len(design['validation']),checkpoint_count_per_arm=design['checkpoint_count_per_arm'],planned_train_batches=len(training),planned_validation_batches=len(validation))


def load_design(path,record):
    design_path=Path(record['runtime_design_path']);design=json.loads(design_path.read_text())
    if pair.sha256(design_path)!=record['runtime_design_sha256'] or design['plan_sha256']!=pair.sha256(path):raise ValueError('profile design drift')
    for name,want in design['source_sha256'].items():
        if pair.sha256(pair.ROOT/name)!=want:raise ValueError('profiling instrumentation source drift:'+name)
    return design


class NoStepOptimizer:
    def __init__(self,net):self.net=net
    def zero_grad(self,*,set_to_none=True):self.net.zero_grad(set_to_none=set_to_none)
    def step(self,*a,**k):raise PermissionError('zero-update instrumentation forbids optimizer steps')


def observe_body(net,ds,row,b,reserve,training):
    stamp=time.perf_counter();b.check('runtime_train_batch' if training else 'runtime_validation_batch')
    net.train(training)
    if training:net.zero_grad(set_to_none=True)
    batch=pair.make_batch(ds,row).to('cuda:0');torch.cuda.synchronize();fb_start=time.perf_counter()
    with torch.set_grad_enabled(training):
        logits=net(batch);loss=pair.masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask)
        if not pair.finite_tree(logits) or not torch.isfinite(loss) or float(loss)>20:raise FloatingPointError('profile loss/logit gate')
        if training:loss.backward()
    torch.cuda.synchronize();fb_seconds=time.perf_counter()-fb_start;grad=None
    if training:
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):raise FloatingPointError('profile gradient gate')
        grad=math.hypot(*(float(p.grad.double().norm()) for p in net.parameters()))
        if not math.isfinite(grad) or grad>100:raise FloatingPointError('profile gradient norm gate')
        b.check('runtime_pre_Adam_NO_ADAM_CALL')
        if not pair.finite_tree(net.state_dict()) or not pair.finite_tree(reserve):raise FloatingPointError('model/moment-size-buffer finiteness')
    mask=batch.target_metric_mask;prob=logits[mask].detach().sigmoid().cpu().numpy();labels=batch.target_labels[mask].cpu().numpy()
    if len(prob)!=row['supervised_targets'] or len(labels)!=len(prob):raise ValueError('profile target alignment')
    if training:b.check('runtime_post_update_NO_UPDATE');time.sleep(.1)
    seconds=time.perf_counter()-stamp
    if fb_seconds>30 or seconds>30:raise TimeoutError('30s profile body ceiling')
    return dict(body_seconds=seconds,forward_backward_seconds=fb_seconds,loss=float(loss),gradient_l2=grad,targets=len(labels),optimizer_updates=0)


def io_proxy(net,reserve,plan,c,proof,out,b,index):
    stamp=time.perf_counter();b.check('checkpoint_IO_proxy');ref=pair.fixed_logits(net,proof);probe=pair.probe(net,NoStepOptimizer(net),proof)
    payload=dict(kind='UNTRAINED_IO_PROXY_NOT_A_CHECKPOINT',format_version=0,model=pair.cpu_tree(net.state_dict()),
        moment_sized_zero_buffers=pair.cpu_tree(reserve),contract=c,reference_logits=ref,zero_step_probe=probe,
        placeholder_probability=torch.zeros(c['target_counts']['train']['labels']),placeholder_labels=torch.zeros(c['target_counts']['train']['labels'],dtype=torch.uint8))
    target=out/f'UNTRAINED_IO_PROXY_{index}.bin';tmp=target.with_suffix('.partial.bin');write_start=time.perf_counter()
    with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.link(tmp,target);tmp.unlink();write_seconds=time.perf_counter()-write_start;read_start=time.perf_counter()
    restored=torch.load(target,map_location='cpu',weights_only=True)
    if not pair.equal_tree(payload,restored):raise ValueError('IO proxy roundtrip mismatch')
    read_seconds=time.perf_counter()-read_start;sha=pair.sha256(target);seconds=time.perf_counter()-stamp
    if seconds>30:raise TimeoutError('IO proxy30s ceiling')
    result=dict(path=str(target),sha256=sha,bytes=target.stat().st_size,total_seconds=seconds,write_seconds=write_seconds,read_verify_seconds=read_seconds,
        optimizer_updates=0,notice='initial model plus moment-sized ZERO buffers/placeholders; NOT actual Adam state or a resumable training checkpoint')
    pair.write_json(out/f'io_proxy_{index}.json',result);return result


def profile(path,record,arm):
    design=load_design(path,record);plan,splits,out,b=pair.context(path,record,pair.PROFILE_ACTION,f'{arm}/runtime_profile',arm);c=plan['contracts'][arm]
    try:
        for a in pair.ARMS:
            v=json.loads((Path(path).parent/a/'full_shape_profile/profile_result.json').read_text())
            if v['status']!='PASS_ZERO_UPDATE_FULL_SHAPES' or v['plan_sha256']!=pair.sha256(path):raise ValueError('both full-shape profiles must pass first')
        started=time.perf_counter();net=pair.initialize(c,b);startup_seconds=time.perf_counter()-started
        before=pair.digest(pair.cpu_tree(net.state_dict()));rng=pair.capture_rng_state();reserve=[torch.zeros_like(p) for p in net.parameters() for _ in range(2)];b.check('runtime_moment_buffers')
        warm=design['training'][len(design['training'])//2]['shape'];pair.restore_rng_state(rng);warmup=observe_body(net,splits['train'],warm,b,reserve,True)
        results={}
        for phase,training in [('training',True),('validation',False)]:
            rows=[]
            for i,item in enumerate(design[phase]):
                samples=[]
                for repeat in range(2):
                    pair.restore_rng_state(rng);samples.append(observe_body(net,splits['train' if training else 'validation'],item['shape'],b,reserve,training))
                row=dict(stratum=i,weight=item['weight'],shape=item['shape'],samples=samples,body_median_seconds=float(np.median([s['body_seconds'] for s in samples])))
                rows.append(row);pair.write_json(out/f'{phase}_{i:02d}.json',row)
            results[phase]=rows
        initial=out/'initial_diagnostics';initial.mkdir();diag_start=time.perf_counter();pair.diagnostics(net,splits['train'],c,b,initial);diag_seconds=time.perf_counter()-diag_start
        proof=pair.make_batch(splits['train'],pair.shape_manifest(splits['train'],[[c['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0')
        ios=[io_proxy(net,reserve,plan,c,proof,out,b,i) for i in range(2)]
        if pair.digest(pair.cpu_tree(net.state_dict()))!=before:raise ValueError('zero-update profiling changed model/projections')
        result=dict(status='PASS_ZERO_UPDATE_RUNTIME_PROFILE',arm=arm,plan_sha256=pair.sha256(path),initialization_seconds=startup_seconds,
            warmup=warmup,strata=results,initial_diagnostics_seconds=diag_seconds,io_proxies=ios,
            weights_projections_unchanged=True,optimizer_updates=0,resources=b.check('runtime_profile_complete'))
        pair.write_json(out/'runtime_profile_result.json',result);return result
    except BaseException as exc:pair.failure(Path(path).parent,out,b,exc,'runtime_profile',dict(optimizer_updates=0));raise


def estimate_arm(result,design,startup_proxy_seconds=0.):
    def total(phase):return sum(r['weight']*r['body_median_seconds'] for r in result['strata'][phase])
    train=total('training');validation=total('validation')*design['validation_passes_per_arm'];io=float(np.median([x['total_seconds'] for x in result['io_proxies']]))
    checkpoints=design['checkpoint_count_per_arm']*io;recoveries=8*io;diagnostics=design['diagnostic_events_per_arm']*result['initial_diagnostics_seconds'];startups=17*startup_proxy_seconds
    subtotal=train+validation+checkpoints+recoveries+diagnostics+startups
    return dict(training_body_proxy_seconds=train,validation_proxy_seconds=validation,checkpoint_proxy_seconds=checkpoints,recovery_proxy_seconds=recoveries,
        diagnostics_proxy_seconds=diagnostics,worker_startup_proxy_seconds=startups,subtotal_seconds=subtotal,
        with_planning_reserve_seconds=subtotal*(1+design['budget_screen_reserve_fraction']),real_Adam_step_cost_measured=False,
        caveat='stratified initial-model timing plus untrained IO proxies; not an upper bound, learned-state benchmark or completion guarantee')
