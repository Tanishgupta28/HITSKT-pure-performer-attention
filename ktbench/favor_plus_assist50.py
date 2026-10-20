"""Explicitly approved exact20→50 continuation; scientific code stays unchanged."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from ktbench import favor_plus_assist20 as original
from ktbench import favor_plus_pilot_recovery as parent_recovery
from ktbench import favor_plus_resource_telemetry as telemetry
from ktbench.favor_plus_continuation_contract import ACTION,PARENT_SHA256,require_explicit_approval,stage_interval
from ktbench.favor_plus_continuation_recovery import KIND,atomic_checkpoint,no_step_recovery,seal,validate
from ktbench.favor_plus_gpu_access import CUDA_VALIDATION,require_gpu_authorization
from ktbench.favor_plus_pipeline import ROOT,fresh_run,runtime,sha256,write_json
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree,finite_tree
from ktbench.metrics import masked_bce_loss

HERE=ROOT/'reports/favor_plus_v1/assist50_continuation_20261020_v1'
DESIGN=HERE/'PREDECLARED_DESIGN.json'
PROPOSAL=ROOT/'reports/favor_plus_v1/assist50_preparation_20261020_v1/PROPOSED_CONTINUATION_PLAN.json'
PARENT_PLAN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist20_recovery_20261020_v1/prepared_plan.json'
PARENT_CHECKPOINT=PARENT_PLAN.parent/'checkpoints/checkpoint_020.pt'


def source_identity():
    result=original.source_identity()
    for name in ('ktbench/favor_plus_continuation_contract.py','ktbench/favor_plus_continuation_recovery.py',
                 'ktbench/favor_plus_assist50.py','scripts/assist50_favor_plus.py'):
        result['source_sha256'][name]=sha256(ROOT/name)
    return result


def prepare(run_id):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU preparation requires CUDA hidden')
    design=json.loads(DESIGN.read_text());proposal=json.loads(PROPOSAL.read_text())
    if sha256(PROPOSAL)!=design['approved_proposal_sha256']:raise ValueError('approved proposal changed')
    old,pc,_,_=original.load_plan(PARENT_PLAN)
    if sha256(PARENT_CHECKPOINT)!=PARENT_SHA256:raise ValueError('exact original20 binary required')
    payload=torch.load(PARENT_CHECKPOINT,map_location='cpu',weights_only=True);parent_recovery.validate(payload,pc)
    frozen=json.loads(original.FROZEN.read_text())['samples']
    from ktbench.favor_plus_continuation_contract import proposed_handoff
    regenerated=proposed_handoff(payload,PARENT_SHA256,frozen)
    if any(proposal.get(k)!=v for k,v in regenerated.items()):raise ValueError('proposal differs from verified parent/frozen identities')
    approval=dict(approved=True,action=ACTION,prepared_plan_sha256=sha256(PROPOSAL),
        parent_checkpoint_sha256=PARENT_SHA256,additional_optimizer_calls=30,max_total_optimizer_updates=50,
        actual_user_approval_evidence=str(HERE/'OWNER_AND_CONTINUATION_AUTHORIZATION.md'))
    require_explicit_approval(approval,sha256(PROPOSAL))
    contract=dict(run_id=run_id,config=copy.deepcopy(pc['config']),configuration_sha256=pc['configuration_sha256'],
        parent_contract=pc,parent_checkpoint_sha256=PARENT_SHA256,parent_component_sha256=payload['component_sha256'],
        parent_projection_sha256=payload['projection_sha256'],parent_optimizer_param_groups=payload['optimizer_state_dict']['param_groups'],
        dataset_identity=pc['dataset_identity'],source_sha256=source_identity()['source_sha256'],
        exact_training_and_probe_samples=proposal['exact_training_and_probe_samples'],
        parameter_tensor_count=pc['parameter_tensor_count'],diagnostic_samples=pc['diagnostic_samples'],
        frozen_validation_samples=pc['frozen_validation_samples'],validation_sample_sha256=pc['validation_sample_sha256'],
        execution_limits=proposal['execution_limits'],design_sha256=sha256(DESIGN),proposal_sha256=sha256(PROPOSAL),
        user_approval=approval,authorization_evidence_sha256=sha256(Path(approval['actual_user_approval_evidence'])))
    plan=dict(kind=KIND,created_utc=original.utc().isoformat(),contract_base=contract,design=design,
        source=source_identity(),samples=dict(training=proposal['exact_training_and_probe_samples'][:50],
        next_probe_only=proposal['exact_training_and_probe_samples'][50],diagnostics=pc['diagnostic_samples'],
        validation=pc['frozen_validation_samples']),parent_plan_sha256=sha256(PARENT_PLAN),
        original_approval_scope='20→50 total;30 additional; no full training',cuda_status='NOT RUN',optimizer_updates=20)
    original.check_host_resources();output=fresh_run('assist2017',run_id);(output/'checkpoints').mkdir()
    write_json(output/'prepared_plan.json',plan)
    return dict(output=str(output),prepared_plan_sha256=sha256(output/'prepared_plan.json'),cuda_status='NOT RUN',optimizer_updates=20)


def load_plan(path):
    plan=json.loads(path.read_text());c=plan['contract_base']
    if (plan.get('kind')!=KIND or c['source_sha256']!=source_identity()['source_sha256']
            or c['design_sha256']!=sha256(DESIGN) or plan['design']!=json.loads(DESIGN.read_text())
            or c['proposal_sha256']!=sha256(PROPOSAL) or plan['parent_plan_sha256']!=sha256(PARENT_PLAN)
            or sha256(PARENT_CHECKPOINT)!=PARENT_SHA256):raise ValueError('continuation source/design/parent/proposal mismatch')
    old,pc,train,val=original.load_plan(PARENT_PLAN)
    proposal=json.loads(PROPOSAL.read_text())
    if (c['parent_contract']!=pc or c['config']!=pc['config'] or c['configuration_sha256']!=pc['configuration_sha256']
            or c['dataset_identity']!=pc['dataset_identity'] or c['parameter_tensor_count']!=pc['parameter_tensor_count']
            or c['parent_checkpoint_sha256']!=PARENT_SHA256
            or c['exact_training_and_probe_samples']!=proposal['exact_training_and_probe_samples']
            or c['diagnostic_samples']!=pc['diagnostic_samples'] or c['frozen_validation_samples']!=pc['frozen_validation_samples']
            or c['validation_sample_sha256']!=pc['validation_sample_sha256']
            or c['execution_limits']!=proposal['execution_limits']
            or c['execution_limits']!=plan['design']['execution_limits']
            or plan['samples']!=dict(training=c['exact_training_and_probe_samples'][:50],next_probe_only=c['exact_training_and_probe_samples'][50],
                diagnostics=pc['diagnostic_samples'],validation=pc['frozen_validation_samples'])):
        raise ValueError('unchanged scientific configuration/sampler/validation contract required')
    require_explicit_approval(c['user_approval'],sha256(PROPOSAL))
    if sha256(Path(c['user_approval']['actual_user_approval_evidence']))!=c['authorization_evidence_sha256']:
        raise ValueError('actual user approval evidence changed')
    return plan,{**c,'prepared_plan_sha256':sha256(path)},train,val


def admit(record,*,snapshot_fn=telemetry.device_snapshot):
    # Actual20→50 approval is separate from allocation ownership and old20 scopes.
    if ACTION not in record.get('user_approval',{}).get('actions',[]):raise ValueError('explicit20→50 GPU action required')
    require_explicit_approval(record.get('continuation_approval',{}),sha256(PROPOSAL))
    uuid=require_gpu_authorization(record,os.environ.get('CUDA_VISIBLE_DEVICES'),CUDA_VALIDATION)
    if (not (telemetry.DEVICE_EXPOSURE/uuid).exists() or not telemetry.LIGHT_HOLD.is_file()
            or json.loads(telemetry.LIGHT_HOLD.read_text()).get('status')!='USER_TRAINING_HOLD_ACTIVE'):
        raise RuntimeError('assigned device or LightSAKT2 restart hold unavailable')
    snap=snapshot_fn()
    if snap['conflicts'] or snap['inaccessible_fd_pids'] or snap['light_processes']:
        raise telemetry.ResourceConflict('unknown/compute owner, inaccessible process or Light trainer',snap)
    return uuid,snap


def context(plan_path,record,phase,update):
    uuid,snapshot=admit(record)  # no CUDA API or plan/model work before approval/allocation
    root=plan_path.parent
    if (root/'study_failure.json').exists() or (root/'study_result.json').exists():raise RuntimeError('failed/completed campaign cannot restart')
    plan,contract,train,val=load_plan(plan_path)
    if record['continuation_approval']!=contract['user_approval']:raise ValueError('allocation scope differs from approved plan')
    original.require_cuda_verification(original.CUDA_REPORT,record)
    host=original.check_host_resources();memory=telemetry.nvml_snapshot(uuid)
    if memory.get('compute_processes')!=[]:
        raise telemetry.ResourceConflict('NVML compute ownership not empty/verified before CUDA',{**snapshot,'nvml':memory})
    if memory['free_bytes']<10*1024**3:raise MemoryError('8GiB allocator plus2GiB free headroom required')
    output=root/f'{phase}_{update:03d}';output.mkdir(exist_ok=False)
    write_json(output/'admission.json',dict(resource_uuid=uuid,snapshot=snapshot,nvml=memory,
        allocation=record,host_memory=host,cuda_initialized=False,source=source_identity()))
    return output,plan,contract,train,val


class Boundaries:
    def __init__(self,record,started,limits,output):
        self.record=record;self.started=started;self.limits=limits;self.output=output
        self.calls=20;self.last=None;self.monitor_events=[];self.nvml_peak_observed_used=0
    def check(self,phase,*,cuda=True):
        uuid,snapshot=admit(self.record)
        host=original.check_host_resources();memory=telemetry.nvml_snapshot(uuid)
        other=[p for p in memory.get('compute_processes',[]) if p['pid']!=os.getpid()]
        if 'compute_processes' not in memory or other:
            raise telemetry.ResourceConflict('unverified/competing NVML compute process',{**snapshot,'nvml':memory})
        self.nvml_peak_observed_used=max(self.nvml_peak_observed_used,memory['used_bytes'])
        if memory['free_bytes']<2*1024**3:raise MemoryError('2GiB device headroom violated')
        if snapshot['holders']:
            self.monitor_events.append(dict(phase=phase,snapshot=snapshot))
            write_json(self.output/f'monitor_event_{len(self.monitor_events):03d}.json',self.monitor_events[-1])
        elapsed=(original.utc()-original.dt.datetime.fromisoformat(self.started)).total_seconds()
        if elapsed<0 or elapsed>self.limits['max_execution_seconds']:raise TimeoutError('45min continuation campaign limit')
        if not 20<=self.calls<=50:raise RuntimeError('20→50 TOTAL-call ceiling')
        value=dict(observed_utc=original.utc().isoformat(),phase=phase,calls=self.calls,additional_optimizer_calls=self.calls-20,
            campaign_elapsed_seconds=elapsed,host=host,snapshot=snapshot,nvml=memory,
            nvml_peak_observed_used_bytes=self.nvml_peak_observed_used)
        if cuda:
            value.update(allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            if value['allocated_bytes']>self.limits['max_cuda_allocated_bytes']:raise MemoryError('8GiB allocation ceiling')
        self.last=value;return value


def restore_proof(payload,model,optimizer,contract,train,diagnostics,*,parent=False):
    fixed=original.collate_selected(train,diagnostics[0]).to(next(model.parameters()).device)
    item=contract['exact_training_and_probe_samples'][payload['optimizer_updates']]
    fn=parent_recovery.no_step_recovery if parent else no_step_recovery
    return fn(payload,model,optimizer,contract,evaluate_logits=lambda m:original.fixed_logits(m,fixed),
        probe=lambda m,o:original.continuation_probe(m,o,train,item))


def checkpoint(net,optimizer,contract,step,started,root,output,train,diagnostics):
    from ktbench.training import capture_rng_state
    probe=original.continuation_probe(net,optimizer,train,contract['exact_training_and_probe_samples'][step])
    fixed=original.collate_selected(train,diagnostics[0]).to('cuda:0')
    prediction=original.fixed_logits(net,fixed).cpu()
    payload=cpu_tree(net.checkpoint())
    payload.update(kind=KIND,optimizer_updates=step,optimizer_step_calls=step,additional_optimizer_calls=step-20,
        optimizer_state_dict=cpu_tree(optimizer.state_dict()),rng_state=cpu_tree(capture_rng_state()),
        sampler=dict(cursor=step,epoch=1,seed=42,exact_samples=contract['exact_training_and_probe_samples']),
        contract=contract,reference_eval_logits=prediction,next_update_probe=probe,campaign_started_utc=started)
    seal(payload)
    receipt=atomic_checkpoint(root/'checkpoints'/f'checkpoint_{step:03d}.pt',payload,contract)
    write_json(output/f'checkpoint_receipt_{step:03d}.json',receipt)
    return receipt,probe


def execute(plan_path,record,to):
    previous={30:20,40:30,50:40}.get(to)
    if previous is None:raise ValueError('only30/40/50 TOTAL endpoints')
    stage_interval(previous,to)
    output,plan,c,train,val=context(plan_path,record,'stage',to);root=plan_path.parent
    started_utc=original.utc().isoformat();stamp=time.perf_counter()
    if previous==20:
        path=PARENT_CHECKPOINT;prior=torch.load(path,map_location='cpu',weights_only=True)
        parent_recovery.validate(prior,c['parent_contract'])
        if (prior['component_sha256']!=c['parent_component_sha256'] or prior['projection_sha256']!=c['parent_projection_sha256']
                or prior['optimizer_state_dict']['param_groups']!=c['parent_optimizer_param_groups']):raise ValueError('parent migration components mismatch')
    else:
        path=root/'checkpoints'/f'checkpoint_{previous:03d}.pt'
        prior=torch.load(path,map_location='cpu',weights_only=True);validate(prior,c)
        receipt=json.loads((root/f'verify_{previous:03d}'/'recovery_result.json').read_text())
        if receipt['status']!='PASS' or receipt['checkpoint_sha256']!=sha256(path):raise ValueError('fresh recovery PASS prerequisite missing')
        if not (root/f'stage_{previous:03d}'/'stage_result.json').exists():raise ValueError('previous stage incomplete')
        started_utc=prior['campaign_started_utc']
    b=Boundaries(record,started_utc,c['execution_limits'],output);calls=returned=previous;b.calls=calls;phase='initialization'
    handler=signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(InterruptedError('stop without automatic restart')))
    try:
        b.check(phase,cuda=False);net,opt=original.initialize(plan)
        write_json(output/'runtime.json',runtime(torch.device('cuda:0')));torch.cuda.reset_peak_memory_stats(0)
        proof=restore_proof(prior,net,opt,c['parent_contract'] if previous==20 else c,train,c['diagnostic_samples'],parent=previous==20)
        write_json(output/'incoming_recovery_and_migration.json',dict(**proof,parent_checkpoint_sha256=PARENT_SHA256,
            incoming_checkpoint_sha256=sha256(path),incoming_component_sha256=prior['component_sha256'],
            source_contract_sha256=parent_recovery.digest(prior['contract']),destination_contract_sha256=parent_recovery.digest(c),
            unchanged_model_adam_rng_projections=True,source_checkpoint_unchanged=True))
        expected_probe=prior['next_update_probe'];del prior
        baseline=json.loads((original.FROZEN.parent/'initial_diagnostics.json').read_text())
        def diagnose(step):
            nonlocal phase
            phase=f'diagnostics_{step}';b.check(phase)
            stages=original.capture(net,train,c['diagnostic_samples'],output,f'm{step:03d}',baseline)
            rules=c['config']['review_and_severe_stops']['severe_stops']
            decision=original.severe_check(stages,{'prospective_severe_stops':rules})
            write_json(output/f'containment_{step:03d}.json',decision)
            if decision['status']=='STOP':raise FloatingPointError('unchanged severe approximation/kernel/norm/ESS gate')
            causal=original.causal_check(net,train,c['diagnostic_samples'][-1])
            tm,_,_=original.evaluate(net,train,c['diagnostic_samples'])
            vm,p,y=original.evaluate(net,val,c['frozen_validation_samples'])
            with (output/f'validation_predictions_{step:03d}.npz').open('xb') as f:np.savez_compressed(f,probability=p.numpy(),label=y.numpy())
            if step==20:
                with np.load(PARENT_PLAN.parent/'stage_020/validation_predictions_020.npz') as z:
                    if not np.array_equal(z['probability'],p.numpy()) or not np.array_equal(z['label'],y.numpy()):
                        raise RuntimeError('checkpoint20 frozen-validation replay is not EXACT')
                old=json.loads((PARENT_PLAN.parent/'stage_020/diagnostics_result_020.json').read_text())
                if vm!=old['sampled_validation'] or tm!=old['sampled_training_eval']:
                    raise RuntimeError('checkpoint20 metric replay changed')
                write_json(output/'parent20_validation_replay.json',dict(status='PASS',probabilities_labels_and_metrics='EXACT',optimizer_calls=0))
            result=dict(update=step,containment=decision,causal=causal,sampled_training_eval=tm,sampled_validation=vm,
                preliminary_sampled_results=True,test_evaluations=0,resources=b.check(f'diagnostics_complete_{step}'))
            write_json(output/f'diagnostics_result_{step:03d}.json',result);return result
        if previous==20:diagnose(20)
        rows=[];receipts=[]
        for step in stage_interval(previous,to):
            phase=f'optimizer_update_{step}';b.check(phase)
            item=c['exact_training_and_probe_samples'][step-1]
            batch=original.collate_selected(train,item).to('cuda:0');net.train();opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(0);begin=time.perf_counter()
            logits=net(batch);loss=masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask)
            gates=c['config']['review_and_severe_stops']['severe_stops']
            if not finite_tree(logits) or not finite_tree(loss) or float(loss)>gates['single_update_bce_max']:raise FloatingPointError('invalid/excessive loss')
            loss.backward();grads=[p.grad for p in net.parameters()]
            if any(g is None or not finite_tree(g) for g in grads):raise FloatingPointError('missing/nonfinite gradient')
            norm=math.hypot(*(float(g.double().norm()) for g in grads))
            if not math.isfinite(norm) or norm>gates['gradient_l2_max']:raise FloatingPointError('excessive gradient norm')
            if expected_probe is not None:
                h=hashlib.sha256()
                for name,value in [('logits',logits),*[(n,p.grad) for n,p in net.named_parameters()]]:
                    h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
                observed=dict(sha256_logits_and_all_parameter_gradients=h.hexdigest(),loss=float(loss),
                    dataset_index=item['dataset_index'],optimizer_step_calls=0,rng_restored=True)
                if observed!=expected_probe:raise RuntimeError('actual next logits/loss/gradients differ from recovery probe')
                write_json(output/f'actual_next_step_matches_checkpoint_{step:03d}.json',dict(status='PASS',update=step,probe=observed));expected_probe=None
            b.check('pre_adam')
            if time.perf_counter()-begin>c['execution_limits']['max_step_seconds']:raise TimeoutError('120s pre-Adam')
            before=[p.detach().clone() for p in net.parameters()]
            if calls>=50 or calls-20>=30:raise RuntimeError('30 additional/50 TOTAL hard stop')
            calls+=1;b.calls=calls;opt.step();returned+=1
            torch.cuda.synchronize(0);seconds=time.perf_counter()-begin
            change=math.hypot(*(float((p.detach()-v).double().norm()) for p,v in zip(net.parameters(),before)));del before
            finite=finite_tree(net.state_dict()) and finite_tree(opt.state_dict())
            b.last=dict(observed_utc=original.utc().isoformat(),phase='post_adam_cached',calls=calls,
                host=original.host_memory(),allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            row=dict(update=step,training_bce=float(loss.detach()),gradient_l2=norm,gradients_finite=True,
                adam_and_model_finite=finite,parameter_update_l2=change,optimizer_step_calls=calls,
                optimizer_updates=returned,additional_optimizer_calls=calls-20,supervised_targets=int(batch.target_metric_mask.sum()),
                padded_tokens=item['padded_tokens'],sample=item,synchronized_training_seconds=seconds,resources=b.last)
            write_json(output/f'update_{step:03d}.json',row);rows.append(row)
            if not finite or not math.isfinite(change) or change==0:raise FloatingPointError('invalid Adam/model/parameter delta')
            if seconds>c['execution_limits']['max_step_seconds']:raise TimeoutError('120s post-Adam')
            b.check('post_update')
            if step%5==0:
                phase=f'checkpoint_{step}';b.check(phase)
                receipt,expected_probe=checkpoint(net,opt,c,step,started_utc,root,output,train,c['diagnostic_samples'])
                receipts.append(receipt);b.check('checkpoint_published')
        final_diag=diagnose(to)
        result=dict(status='PASS',optimizer_updates=returned,optimizer_step_calls=calls,additional_optimizer_calls=calls-20,
            checkpoints=receipts,stage_execution_seconds=time.perf_counter()-stamp,
            training_interval_seconds=sum(r['synchronized_training_seconds'] for r in rows),
            resources=b.check('stage_complete'),diagnostic_result=final_diag,full_training_ready=False,test_evaluations=0)
        write_json(output/'stage_result.json',result);return result
    except BaseException as exc:
        original.record_failure(root,output,exc,phase,calls,returned,b,stamp);raise
    finally:signal.signal(signal.SIGTERM,handler)


def verify(plan_path,record,update):
    if type(update) is not int or update not in (30,40,50):raise ValueError('only30/40/50 fresh CUDA verification')
    output,plan,c,train,val=context(plan_path,record,'verify',update);root=plan_path.parent
    path=root/'checkpoints'/f'checkpoint_{update:03d}.pt';p=torch.load(path,map_location='cpu',weights_only=True);validate(p,c)
    if p['optimizer_updates']!=update:raise ValueError('wrong update checkpoint')
    b=Boundaries(record,p['campaign_started_utc'],c['execution_limits'],output);b.calls=update
    stamp=time.perf_counter();phase='recovery_initialization'
    try:
        b.check(phase,cuda=False);net,opt=original.initialize(plan)
        write_json(output/'runtime.json',runtime(torch.device('cuda:0')));torch.cuda.reset_peak_memory_stats(0)
        result=restore_proof(p,net,opt,c,train,c['diagnostic_samples'])
        result.update(checkpoint_sha256=sha256(path),recovery_execution_seconds=time.perf_counter()-stamp,
            resources=b.check('after_recovery_proof'),source=source_identity(),test_evaluations=0)
        write_json(output/'recovery_result.json',result)
        if update==50:
            write_json(root/'study_result.json',dict(status='FIFTY_TOTAL_UPDATES_COMPLETED',optimizer_updates=50,
                additional_optimizer_calls=30,parent_checkpoint_sha256=PARENT_SHA256,checkpoint_sha256=sha256(path),
                fresh_process_recovery='PASS30/40/50; zero extra Adam calls',no_automatic_further_training=True,
                full_training_ready=False,test_evaluations=0))
        return result
    except BaseException as exc:
        original.record_failure(root,output,exc,phase,update,update,b,stamp);raise
