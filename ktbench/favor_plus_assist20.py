"""One recoverable ASSIST2017 campaign; planned5/10/20 stages, no retries.

Scientific model/attention/optimizer/data code is imported unchanged. GPU entry
points gate before any CUDA API and use identity-first synchronous telemetry.
"""
from __future__ import annotations
import datetime as dt
import json
import math
import os
from pathlib import Path
import signal
import time
from dataclasses import asdict

import numpy as np
import torch

from ktbench.favor_plus_bounded_assist import (CUDA_REPORT, capture, causal_check,
    continuation_probe, evaluate, select_samples, severe_check)
from ktbench.favor_plus_pipeline import ROOT, PROTOCOL, fresh_run, model_config, profile_for, runtime, sha256, write_json
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree,finite_tree
from ktbench.favor_plus_real_pilot import FAVOR,check_host_resources,collate_selected,host_memory,pilot_source_identity,verified_dataset_identity,require_cuda_verification
from ktbench.data.session_store import SessionStore
from ktbench.metrics import masked_bce_loss
from ktbench.models.favor_plus import FavorPlusConfig
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus
from ktbench.favor_plus_resource_telemetry import (PILOT_ACTION,RECOVERY_ACTION,
    ResourceConflict,admit,nvml_snapshot)
from ktbench.favor_plus_pilot_recovery import KIND,atomic_checkpoint,digest,no_step_recovery,seal,validate

DESIGN=ROOT/'reports/favor_plus_v1/assist20_recovery_20261020_v1/PREDECLARED_DESIGN.json'
FROZEN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist_bounded_review_20261020_v2/prepared_plan.json'


def utc():return dt.datetime.now(dt.timezone.utc)


def source_identity():
    result=pilot_source_identity()
    for name in ('ktbench/favor_plus_bounded_assist.py','ktbench/favor_plus_resource_telemetry.py',
                 'ktbench/favor_plus_pilot_recovery.py','ktbench/favor_plus_assist20.py','scripts/assist20_favor_plus.py'):
        result['source_sha256'][name]=sha256(ROOT/name)
    return result


def prepare(run_id):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):
        raise ValueError('preparation requires CUDA hidden')
    design=json.loads(DESIGN.read_text());frozen=json.loads(FROZEN.read_text())
    if design['frozen_original_plan_sha256']!=sha256(FROZEN):raise ValueError('frozen historical selection changed')
    store=SessionStore(frozen['store_path'])
    reference=json.loads((ROOT/'reports/favor_plus_v1/numerical_quality_v1/dataset_pipeline_plan.json').read_text())['datasets']['assist2017']
    dataset_id=verified_dataset_identity(store,reference)
    train,val,samples=select_samples(store)
    if samples!=frozen['samples']:raise ValueError('existing frozen train/validation/diagnostic identities changed')
    # Number of parameter tensors is inherited from the validated original pilot,
    # not guessed from new model initialization. Four projections are buffers.
    old=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/real_assist_five_updates_20261020_v1/pilot_model.pt'
    if sha256(old)!=frozen['prior_checkpoint_sha256']:raise ValueError('preserved five-step checkpoint hash mismatch')
    original=torch.load(old,map_location='cpu',weights_only=True)
    count=len(original['optimizer_state_dict']['state']);del original
    selected=samples['training'][:21]
    configuration=dict(model_config=frozen['model_config'],favor_config=asdict(FavorPlusConfig(**FAVOR)),
        profile=profile_for('assist2017'),seed=42,protocol=PROTOCOL,limits=design['limits'],
        review_and_severe_stops={k:design[k] for k in ('severe_stops','historical_review_triggers')})
    contract=dict(run_id=run_id,source_sha256=source_identity()['source_sha256'],dataset_identity=dataset_id,
        config=configuration,configuration_sha256=digest(configuration),design_sha256=sha256(DESIGN),
        exact_training_and_probe_samples=selected,diagnostic_samples=samples['diagnostics'],
        frozen_validation_samples=samples['validation'],validation_sample_sha256=digest(samples['validation']),
        parameter_tensor_count=count,epoch=1,sampler_seed=42)
    plan=dict(kind=KIND,created_utc=utc().isoformat(),design=design,source=source_identity(),
        contract_base=contract,samples=dict(training=selected[:20],next_step_probe_only=selected[20],
        diagnostics=samples['diagnostics'],validation=samples['validation'],coverage=samples['validation_coverage']),
        store_path=str(store.root),frozen_original_plan_sha256=sha256(FROZEN),
        baseline_cpu_diagnostics_sha256=sha256(FROZEN.parent/'initial_diagnostics.json'),
        optimizer_updates=0,cuda_status='NOT RUN',test_evaluations=0)
    check_host_resources()
    output=fresh_run('assist2017',run_id)
    (output/'checkpoints').mkdir()
    write_json(output/'prepared_plan.json',plan)
    return dict(output=str(output),prepared_plan_sha256=sha256(output/'prepared_plan.json'),
        training_updates_authorized=20,next_sample_probe_only=True,validation_coverage=samples['validation_coverage'],cuda_status='NOT RUN')


def load_plan(path):
    plan=json.loads(path.read_text());contract=plan['contract_base']
    if (plan['kind']!=KIND or plan['design']!=json.loads(DESIGN.read_text())
            or contract['source_sha256']!=source_identity()['source_sha256']
            or contract['configuration_sha256']!=digest(contract['config'])
            or contract['design_sha256']!=sha256(DESIGN) or plan['frozen_original_plan_sha256']!=sha256(FROZEN)):
        raise ValueError('study plan/source/configuration identity mismatch')
    store=SessionStore(plan['store_path'])
    reference=json.loads((ROOT/'reports/favor_plus_v1/numerical_quality_v1/dataset_pipeline_plan.json').read_text())['datasets']['assist2017']
    if verified_dataset_identity(store,reference)!=contract['dataset_identity']:raise ValueError('dataset identity mismatch')
    train,val,samples=select_samples(store)
    if (plan['samples']['training']!=samples['training'][:20] or plan['samples']['next_step_probe_only']!=samples['training'][20]
            or plan['samples']['validation']!=samples['validation'] or plan['samples']['diagnostics']!=samples['diagnostics']
            or contract['exact_training_and_probe_samples']!=samples['training'][:21]
            or contract['frozen_validation_samples']!=samples['validation']
            or contract['validation_sample_sha256']!=digest(samples['validation'])
            or contract['diagnostic_samples']!=samples['diagnostics']):
        raise ValueError('sampler/validation sample mismatch')
    config=contract['config']
    if (config['profile']!=profile_for('assist2017') or config['model_config']!=asdict(model_config('assist2017',store.metadata['num_questions'],store.metadata['num_skills']))
            or config['favor_config']!=asdict(FavorPlusConfig(**FAVOR)) or config['protocol']!=PROTOCOL
            or config['limits']!=plan['design']['limits']
            or config['review_and_severe_stops']!={k:plan['design'][k] for k in ('severe_stops','historical_review_triggers')}
            or plan['baseline_cpu_diagnostics_sha256']!=sha256(FROZEN.parent/'initial_diagnostics.json')):
        raise ValueError('original architecture/optimizer/protocol mismatch')
    return plan,{**contract,'prepared_plan_sha256':sha256(path)},train,val


def stage_interval(previous,to):
    if type(to) is not int or {5:0,10:5,20:10}.get(to)!=previous:
        raise ValueError('only0→5→10→20; no failed-step restart or50/100 stage')
    return range(previous+1,to+1)


def fixed_logits(model,batch):
    was=model.training;model.eval()
    try:
        with torch.no_grad():return model(batch).detach()
    finally:model.train(was)


def context(plan_path,allocation,action,phase,to):
    # All gates before data IO/model/seeding/CUDA. Also refuse failed campaigns.
    uuid,snapshot=admit(allocation,action)
    root=plan_path.parent
    if (root/'study_failure.json').exists():raise RuntimeError('campaign failed; no automatic GPU restart/recovery')
    plan,contract,train,val=load_plan(plan_path)
    require_cuda_verification(CUDA_REPORT,allocation)
    check_host_resources()
    memory=nvml_snapshot(uuid)
    if memory.get('compute_processes'):
        raise ResourceConflict('MIG has active NVML compute processes before CUDA',dict(**snapshot,nvml=memory))
    if memory['free_bytes']<10*1024**3:raise MemoryError('less than8GiB allocation plus2GiB overhead headroom free')
    output=root/f'{phase}_{to:03d}';output.mkdir(exist_ok=False)
    write_json(output/'admission.json',dict(resource_uuid=uuid,snapshot=snapshot,nvml=memory,
        allocation=allocation,host_memory=check_host_resources(),cuda_initialized=False,source=source_identity()))
    return output,plan,contract,train,val


def initialize(plan):
    from ktbench.config import seed_everything
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    seed_everything(42);torch.use_deterministic_algorithms(True,warn_only=False)
    capacity=torch.cuda.get_device_properties(0).total_memory
    if capacity<10*1024**3:raise MemoryError('insufficient device capacity')
    torch.cuda.set_per_process_memory_fraction(8*1024**3/capacity,0)
    config=plan['contract_base']['config']
    net=HiTSKTFavorPlus(model_config('assist2017',config['model_config']['num_questions'],
        config['model_config']['num_skills']),FavorPlusConfig(**FAVOR)).to('cuda:0')
    optimizer=torch.optim.Adam(net.parameters(),lr=config['profile']['learning_rate'],weight_decay=config['profile']['weight_decay'])
    if len(list(net.parameters()))!=plan['contract_base']['parameter_tensor_count']:raise ValueError('parameter-state coverage changed')
    return net,optimizer


class Boundaries:
    def __init__(self,allocation,action,started,limits,output):
        self.allocation=allocation;self.action=action;self.started=started;self.limits=limits;self.output=output
        self.last=None;self.monitor_events=[];self.calls=0
    def check(self,phase,*,cuda=True):
        _,snapshot=admit(self.allocation,self.action)  # no CUDA calls if conflict raised
        if snapshot['holders']:
            self.monitor_events.append(dict(phase=phase,snapshot=snapshot))
            write_json(self.output/f'monitor_event_{len(self.monitor_events):03d}.json',self.monitor_events[-1])
        host=check_host_resources()
        elapsed=(utc()-dt.datetime.fromisoformat(self.started)).total_seconds()
        if elapsed<0 or elapsed>self.limits['max_execution_seconds']:raise TimeoutError('single-campaign45min deadline')
        if self.calls>20:raise RuntimeError('20 optimizer-call limit')
        values=dict(observed_utc=utc().isoformat(),phase=phase,calls=self.calls,host=host,
            campaign_elapsed_seconds=elapsed,snapshot=snapshot)
        if cuda:
            values.update(allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            if values['allocated_bytes']>self.limits['max_cuda_allocated_bytes']:raise MemoryError('8GiB GPU allocation ceiling')
        self.last=values
        return values


def record_failure(root,output,exc,phase,calls,returned,boundaries,started):
    # Deliberately no CUDA query/serialization: use data already copied to CPU.
    existing=[]
    for path in sorted((root/'checkpoints').glob('checkpoint_*.pt')):
        existing.append(dict(path=str(path),sha256=sha256(path)))
    value=dict(status='FAIL',failure_type=type(exc).__name__,message=str(exc),phase=phase,
        optimizer_step_calls=calls,optimizer_updates=returned,observed_utc=utc().isoformat(),
        process_execution_seconds=time.perf_counter()-started,last_safe_resources=boundaries.last,
        conflict_snapshot=getattr(exc,'snapshot',None),valid_published_checkpoints=existing,
        no_cuda_calls_in_failure_handler=True,no_automatic_restart=True)
    write_json(output/'failure.json',value)
    if not (root/'study_failure.json').exists():write_json(root/'study_failure.json',value)


def execute(plan_path,allocation,to):
    previous={5:0,10:5,20:10}.get(to)
    if previous is None:raise ValueError('unapproved milestone')
    stage_interval(previous,to)
    output,plan,contract,train,val=context(plan_path,allocation,PILOT_ACTION,'stage',to)
    root=plan_path.parent
    started_utc=utc().isoformat()
    prior=None
    if previous:
        prior_path=root/'checkpoints'/f'checkpoint_{previous:03d}.pt'
        prior=torch.load(prior_path,map_location='cpu',weights_only=True);validate(prior,contract)
        receipt=json.loads((root/f'verify_{previous:03d}'/'recovery_result.json').read_text())
        if receipt.get('status')!='PASS' or receipt.get('checkpoint_sha256')!=sha256(prior_path):raise ValueError('fresh-process recovery PASS required')
        if not (root/f'stage_{previous:03d}'/'stage_result.json').exists():raise ValueError('previous stage did not finish safely')
        started_utc=prior['campaign_started_utc']
    b=Boundaries(allocation,PILOT_ACTION,started_utc,plan['design']['limits'],output)
    calls=returned=previous;phase='initialization';stamp=time.perf_counter()
    old_handler=signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(InterruptedError('termination; no automatic restart')))
    try:
        b.calls=calls;b.check(phase,cuda=False)
        net,optimizer=initialize(plan)
        from ktbench.training import capture_rng_state,restore_rng_state
        if prior is not None:
            net.load_state_dict(prior['state_dict'],strict=True);optimizer.load_state_dict(prior['optimizer_state_dict'])
            if not equal_tree(cpu_tree(optimizer.state_dict()),prior['optimizer_state_dict']):raise RuntimeError('continued Adam reload mismatch')
            restore_rng_state(prior['rng_state'])
        write_json(output/'runtime.json',runtime(torch.device('cuda:0')))
        torch.cuda.reset_peak_memory_stats(0)
        baseline=json.loads((FROZEN.parent/'initial_diagnostics.json').read_text())
        def diagnose(step,validation=False):
            nonlocal phase
            phase=f'diagnostics_{step}';b.check(phase)
            stages=capture(net,train,plan['samples']['diagnostics'],output,f'm{step:03d}',baseline)
            # Field name differs in the new design but rules are byte-identical.
            decision=severe_check(stages,{'prospective_severe_stops':plan['design']['severe_stops']})
            write_json(output/f'containment_{step:03d}.json',decision)
            if decision['status']=='STOP':raise FloatingPointError('unchanged severe approximation/norm/ESS/kernel gate failed')
            causal=causal_check(net,train,plan['samples']['diagnostics'][-1])
            result=dict(update=step,containment=decision,causal=causal,resources=b.check(phase))
            if validation:
                # Use same exact frozen targets, explicit eval, no RNG draws.
                tm,_,_=evaluate(net,train,plan['samples']['diagnostics'])
                vm,p,y=evaluate(net,val,plan['samples']['validation'])
                with (output/f'validation_predictions_{step:03d}.npz').open('xb') as f:np.savez_compressed(f,probability=p.numpy(),label=y.numpy())
                result.update(sampled_training_eval=tm,sampled_validation=vm,test_evaluations=0,
                    preliminary_sampled_results=True,resources=b.check(f'validation_{step}'))
            write_json(output/f'diagnostics_result_{step:03d}.json',result)
            return result
        if previous==0:diagnose(0,validation=True)
        rows=[]
        for step in stage_interval(previous,to):
            phase=f'optimizer_update_{step}';b.check(phase)
            item=plan['samples']['training'][step-1]
            batch=collate_selected(train,item).to('cuda:0');net.train();optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(0);begin=time.perf_counter()
            logits=net(batch);loss=masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask)
            gate=plan['design']['severe_stops']
            if not finite_tree(logits) or not finite_tree(loss) or float(loss)>gate['single_update_bce_max']:
                raise FloatingPointError('nonfinite/excessive loss before optimizer')
            loss.backward();grads=[p.grad for p in net.parameters()]
            if any(g is None or not finite_tree(g) for g in grads):raise FloatingPointError('missing/nonfinite gradient')
            norm=math.hypot(*(float(g.double().norm()) for g in grads))
            if not math.isfinite(norm) or norm>gate['gradient_l2_max']:raise FloatingPointError('excessive/nonfinite gradient norm')
            if prior is not None and step==previous+1:
                # Actual next training pass must match the saved no-step probe.
                import hashlib
                h=hashlib.sha256()
                for name,value in [('logits',logits),*[(n,p.grad) for n,p in net.named_parameters()]]:
                    h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
                observed=dict(sha256_logits_and_all_parameter_gradients=h.hexdigest(),loss=float(loss),
                    dataset_index=item['dataset_index'],optimizer_step_calls=0,rng_restored=True)
                if observed!=prior['next_update_probe']:raise RuntimeError('actual next training behavior differs from checkpoint proof')
                write_json(output/'actual_next_step_matches_checkpoint.json',dict(status='PASS',update=step,probe=observed))
                del prior;prior=None
            b.check('pre_adam')
            if time.perf_counter()-begin>120:raise TimeoutError('120s pre-optimizer limit')
            before=[p.detach().clone() for p in net.parameters()]
            if calls>=20:raise RuntimeError('20-call hard stop')
            calls+=1;b.calls=calls;optimizer.step();returned+=1
            torch.cuda.synchronize(0);seconds=time.perf_counter()-begin
            change=math.hypot(*(float((p.detach()-old).double().norm()) for p,old in zip(net.parameters(),before)));del before
            finite=finite_tree(net.state_dict()) and finite_tree(optimizer.state_dict())
            # Record memory before the next admission snapshot so conflict exits
            # retain the most recent post-Adam high-water without touching CUDA.
            current=dict(observed_utc=utc().isoformat(),phase='post_adam',calls=calls,host=host_memory(),
                allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            b.last=current
            row=dict(update=step,training_bce=float(loss.detach()),gradient_l2=norm,gradients_finite=True,
                adam_and_model_finite=finite,parameter_update_l2=change,optimizer_step_calls=calls,optimizer_updates=returned,
                supervised_targets=int(batch.target_metric_mask.sum()),padded_tokens=item['padded_tokens'],
                synchronized_training_seconds=seconds,resources=current)
            write_json(output/f'update_{step:03d}.json',row);rows.append(row)
            if not finite or not math.isfinite(change) or change==0:raise FloatingPointError('invalid model/Adam/parameter change after optimizer')
            if seconds>120:raise TimeoutError('120s post-optimizer limit')
            b.check('post_update')
        phase='checkpoint_preparation';b.check(phase)
        # Next training sample is used for proof only, never an Adam update.
        next_item=contract['exact_training_and_probe_samples'][to]
        probe=continuation_probe(net,optimizer,train,next_item)
        fixed=collate_selected(train,plan['samples']['diagnostics'][0]).to('cuda:0')
        prediction=fixed_logits(net,fixed).cpu()
        payload=cpu_tree(net.checkpoint())
        payload.update(kind=KIND,optimizer_updates=returned,optimizer_step_calls=calls,
            optimizer_state_dict=cpu_tree(optimizer.state_dict()),rng_state=cpu_tree(capture_rng_state()),
            sampler=dict(cursor=to,epoch=1,seed=42,exact_samples=contract['exact_training_and_probe_samples']),
            contract=contract,reference_eval_logits=prediction,next_update_probe=probe,campaign_started_utc=started_utc)
        seal(payload)
        phase='atomic_checkpoint_publication'
        receipt=atomic_checkpoint(root/'checkpoints'/f'checkpoint_{to:03d}.pt',payload,contract)
        write_json(output/'checkpoint_receipt.json',receipt)
        del payload
        # Once the CPU checkpoint is published, even a later conflict retains it.
        b.check('checkpoint_published')
        result=diagnose(to,validation=to==20)
        final=dict(status='PASS',optimizer_updates=returned,optimizer_step_calls=calls,checkpoint=receipt,
            stage_execution_seconds=time.perf_counter()-stamp,training_interval_seconds=sum(r['synchronized_training_seconds'] for r in rows),
            resources=b.check('stage_complete'),diagnostic_result=result,full_training_ready=False,test_evaluations=0)
        write_json(output/'stage_result.json',final)
        return final
    except BaseException as exc:
        record_failure(root,output,exc,phase,calls,returned,b,stamp);raise
    finally:signal.signal(signal.SIGTERM,old_handler)


def verify(plan_path,allocation,update):
    if type(update) is not int or update not in (5,10,20):raise ValueError('unapproved recovery update')
    output,plan,contract,train,val=context(plan_path,allocation,RECOVERY_ACTION,'verify',update)
    root=plan_path.parent;path=root/'checkpoints'/f'checkpoint_{update:03d}.pt'
    payload=torch.load(path,map_location='cpu',weights_only=True);validate(payload,contract)
    if payload['optimizer_updates']!=update:raise ValueError('wrong checkpoint update')
    b=Boundaries(allocation,RECOVERY_ACTION,payload['campaign_started_utc'],plan['design']['limits'],output)
    b.calls=update;phase='recovery_initialization';stamp=time.perf_counter()
    try:
        b.check(phase,cuda=False);net,optimizer=initialize(plan)
        write_json(output/'runtime.json',runtime(torch.device('cuda:0')))
        torch.cuda.reset_peak_memory_stats(0)
        fixed=collate_selected(train,plan['samples']['diagnostics'][0]).to('cuda:0')
        next_item=contract['exact_training_and_probe_samples'][update]
        b.check('before_recovery_proof')
        result=no_step_recovery(payload,net,optimizer,contract,
            evaluate_logits=lambda model:fixed_logits(model,fixed),
            probe=lambda model,opt:continuation_probe(model,opt,train,next_item))
        result.update(checkpoint_sha256=sha256(path),recovery_execution_seconds=time.perf_counter()-stamp,
            resources=b.check('after_recovery_proof'),source=source_identity(),test_evaluations=0)
        write_json(output/'recovery_result.json',result)
        if update==20:
            write_json(root/'study_result.json',dict(status='TWENTY_UPDATES_COMPLETED',optimizer_updates=20,
                checkpoint_sha256=sha256(path),fresh_process_recovery='PASS5/10/20; zero extra Adam calls',
                no_automatic_further_training=True,full_training_ready=False,test_evaluations=0))
        return result
    except BaseException as exc:
        record_failure(root,output,exc,phase,update,update,b,stamp);raise
