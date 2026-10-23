"""Separate shared-GPU20→50 controller; legacy exclusive sources stay frozen.

Reuses unchanged scientific diagnostics, Adam, checkpoint format and exact
recovery proofs. Shared mode is sealed into the new plan/checkpoint contract.
"""
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
from ktbench import favor_plus_assist50 as exclusive
from ktbench import favor_plus_assist20 as original
from ktbench import favor_plus_pilot_recovery as parent_recovery
from ktbench import favor_plus_shared_gpu as shared
from ktbench.favor_plus_continuation_contract import PARENT_SHA256, stage_interval
from ktbench.favor_plus_continuation_recovery import KIND, no_step_recovery, optimizer_groups_equal, validate
from ktbench.favor_plus_pipeline import ROOT, fresh_run, runtime, sha256, write_json
from ktbench.favor_plus_pilot_diagnostics import finite_tree
from ktbench.metrics import masked_bce_loss

PLAN_KIND = 'assist2017_favor_plus_shared20_to50_v1'
HERE = ROOT/'reports/favor_plus_v1/shared_gpu_v1'
POLICY_FILE = HERE/'PREDECLARED_POLICY.json'
AUTHORIZATION = HERE/'USER_SHARED_AUTHORIZATION.md'
BASE_PLAN = ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261020_v2/prepared_plan.json'
PARENT_PLAN = exclusive.PARENT_PLAN
PARENT_CHECKPOINT = exclusive.PARENT_CHECKPOINT


def source_identity():
    result = exclusive.source_identity()
    for p in ('ktbench/favor_plus_shared_gpu.py', 'ktbench/favor_plus_shared_assist50.py',
              'scripts/assist50_favor_plus_shared.py'):
        result['source_sha256'][p] = sha256(ROOT/p)
    return result


def make_contract(run_id, base):
    policy = json.loads(POLICY_FILE.read_text())
    shared.validate_policy(policy)
    c = copy.deepcopy(base)
    c.pop('prepared_plan_sha256', None)
    c.update(run_id=run_id, execution_mode=shared.MODE, source_sha256=source_identity()['source_sha256'],
        shared_resource_policy=policy, shared_policy_sha256=sha256(POLICY_FILE),
        shared_authorization_sha256=sha256(AUTHORIZATION), baseline_exclusive_plan_sha256=sha256(BASE_PLAN))
    c['execution_limits'] = dict(max_execution_seconds=policy['max_execution_seconds'],
        max_step_seconds=policy['max_step_seconds'], max_cuda_allocated_bytes=policy['allocator_cap_bytes'],
        max_cuda_reserved_bytes=policy['reserved_cap_bytes'], max_process_rss_bytes=policy['host_process_rss_bytes'],
        minimum_cgroup_headroom_bytes=policy['host_headroom_bytes'], batch_size=1, max_padded_tokens=2048)
    return c


def prepare(run_id):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('CPU preparation requires CUDA hidden')
    old, base, _, _ = exclusive.load_plan(BASE_PLAN)
    payload = torch.load(PARENT_CHECKPOINT, map_location='cpu', weights_only=True)
    parent_recovery.validate(payload, base['parent_contract'])
    c = make_contract(run_id, base)
    plan = dict(kind=PLAN_KIND, execution_mode=shared.MODE, created_utc=original.utc().isoformat(),
        contract_base=c, source=source_identity(), samples=old['samples'], optimizer_updates=20,
        cuda_status='NOT RUN', readiness='CPU preparation; shared execution record required', test_evaluations=0)
    original.check_host_resources()
    output = fresh_run('assist2017', run_id)
    (output/'checkpoints').mkdir()
    write_json(output/'prepared_plan.json', plan)
    return dict(output=str(output), prepared_plan_sha256=sha256(output/'prepared_plan.json'),
                execution_mode=shared.MODE, optimizer_updates=20, cuda_status='NOT RUN')


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan.get('kind') != PLAN_KIND or plan.get('execution_mode') != shared.MODE:
        raise ValueError('explicit shared plan required; legacy mode is separate')
    old, base, train, val = exclusive.load_plan(BASE_PLAN)
    c = plan['contract_base']
    if (c != make_contract(c['run_id'], base) or plan['samples'] != old['samples']
            or plan['source']['source_sha256'] != source_identity()['source_sha256']):
        raise ValueError('shared science/source/data/policy/sample contract mismatch')
    return plan, {**c, 'prepared_plan_sha256':sha256(path)}, train, val


def authorize(record, plan_sha):
    uuid = shared.authorize(record, os.environ.get('CUDA_VISIBLE_DEVICES'),
        plan_sha256=plan_sha, policy_sha256=sha256(POLICY_FILE))
    shared.check_exposure_and_hold(uuid)
    return uuid


def context(plan_path, record, phase, update):
    # Permission/policy/scope are checked before plan/model/CUDA work.
    uuid = authorize(record, sha256(plan_path))
    root = plan_path.parent
    if (root/'study_failure.json').exists() or (root/'study_result.json').exists():
        raise RuntimeError('failed/completed shared campaign cannot restart')
    plan, c, train, val = load_plan(plan_path)
    expected = ROOT/'experiments/hitskt_favor_plus_v1/assist2017'/c['run_id']/'prepared_plan.json'
    if plan_path.resolve() != expected.resolve():
        raise ValueError('shared execution must remain in its fresh FAVOR+ experiment namespace')
    original.require_cuda_verification(original.CUDA_REPORT, record)
    observations = []
    for i in range(shared.POLICY['admission_samples']):
        host = original.check_host_resources()
        observed = shared.observe(uuid)
        decision = shared.check_observation(observed, host, admission=True)
        observations.append(dict(observation=observed, decision=decision, host=host))
        if i+1 < shared.POLICY['admission_samples']:
            time.sleep(shared.POLICY['admission_sample_spacing_seconds'])
    output = root/f'{phase}_{update:03d}'
    output.mkdir(exist_ok=False)
    write_json(output/'admission.json', dict(mode=shared.MODE, allocation=record,
        observations=observations, cuda_initialized=False, pid_mapping_proven=False, source=source_identity()))
    return output, plan, c, train, val


def initialize(plan):
    # Same seed/precision/constructor/Adam as original.initialize; only the
    # explicitly versioned allocator cap changes, BEFORE model allocation.
    from ktbench.config import seed_everything
    torch.set_num_threads(1); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    seed_everything(42); torch.use_deterministic_algorithms(True, warn_only=False)
    capacity = torch.cuda.get_device_properties(0).total_memory
    if capacity < shared.POLICY['admission_free_bytes']:
        raise MemoryError('insufficient shared device capacity')
    torch.cuda.set_per_process_memory_fraction(shared.POLICY['allocator_cap_bytes']/capacity, 0)
    config = plan['contract_base']['config']
    net = original.HiTSKTFavorPlus(original.model_config('assist2017', config['model_config']['num_questions'],
        config['model_config']['num_skills']), original.FavorPlusConfig(**original.FAVOR)).to('cuda:0')
    optimizer = torch.optim.Adam(net.parameters(), lr=config['profile']['learning_rate'],
                                 weight_decay=config['profile']['weight_decay'])
    if len(list(net.parameters())) != plan['contract_base']['parameter_tensor_count']:
        raise ValueError('parameter-state coverage changed')
    return net, optimizer


class Boundaries:
    def __init__(self, record, started, limits, output):
        self.record=record; self.started=started; self.limits=limits; self.output=output
        self.calls=20; self.last=None; self.sequence=0; self.previous_pids=set()
        self.expected_plan_sha256=record['prepared_plan_sha256']

    def check(self, phase, *, cuda=True):
        uuid = authorize(self.record, self.expected_plan_sha256)
        host = original.check_host_resources()
        observed = shared.observe(uuid)
        # Save failing telemetry BEFORE any CUDA allocator query.
        decision = shared.assess(observed, host)
        pids = {p['pid'] for p in decision['reported_compute_processes']}
        value = dict(observed_utc=original.utc().isoformat(), phase=phase, calls=self.calls,
            additional_optimizer_calls=self.calls-20, observation=observed, decision=decision, host=host,
            reported_pids_added=sorted(pids-self.previous_pids), reported_pids_departed=sorted(self.previous_pids-pids),
            pid_mapping_proven=False)
        self.last = {**(self.last or {}), **value}
        self.sequence += 1
        write_json(self.output/f'resources_{self.sequence:04d}.json', value)
        if decision['status'] == 'STOP':
            raise shared.SharedResourceStop('; '.join(decision['reasons']), value)
        elapsed = (original.utc()-original.dt.datetime.fromisoformat(self.started)).total_seconds()
        if elapsed < 0 or elapsed > self.limits['max_execution_seconds']:
            raise TimeoutError('shared45min campaign budget')
        if not 20 <= self.calls <= 50:
            raise RuntimeError('30 additional/50 TOTAL hard stop')
        if cuda:
            own = dict(allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            self.last.update(own)
            shared.check_observation(observed, host, own_cuda=own)
        self.previous_pids=pids
        return self.last


restore_proof = exclusive.restore_proof
checkpoint = exclusive.checkpoint


def capture_checked(net, train, selected, output, tag, baseline, boundaries):
    results=[]
    for i, item in enumerate(selected):
        boundaries.check(f'{tag}_prefix_{i}')
        results.extend(original.capture(net, train, [item], output, f'{tag}_sample{i}', [baseline[i]]))
    write_json(output/f'{tag}_diagnostics.json', results)
    return results


def evaluate_checked(net, dataset, selected, boundaries, label):
    # Original evaluate, with a boundary before each sample. Iterator wrapping
    # adds no RNG/metric operation and preserves exact accumulation/order.
    def samples():
        for i, item in enumerate(selected):
            boundaries.check(f'{label}_sample_{i}')
            yield item
    return original.evaluate(net, dataset, samples())

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
                or not optimizer_groups_equal(prior['optimizer_state_dict']['param_groups'],c['parent_optimizer_param_groups'])):raise ValueError('parent migration components mismatch')
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
        b.check(phase,cuda=False);net,opt=initialize(plan)
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
            stages=capture_checked(net,train,c['diagnostic_samples'],output,f'm{step:03d}',baseline,b)
            rules=c['config']['review_and_severe_stops']['severe_stops']
            decision=original.severe_check(stages,{'prospective_severe_stops':rules})
            write_json(output/f'containment_{step:03d}.json',decision)
            if decision['status']=='STOP':raise FloatingPointError('unchanged severe approximation/kernel/norm/ESS gate')
            causal=original.causal_check(net,train,c['diagnostic_samples'][-1])
            tm,_,_=evaluate_checked(net,train,c['diagnostic_samples'],b,'training_eval')
            vm,p,y=evaluate_checked(net,val,c['frozen_validation_samples'],b,'validation')
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
            if time.perf_counter()-begin>c['execution_limits']['max_step_seconds']:raise TimeoutError('shared30s pre-Adam')
            before=[p.detach().clone() for p in net.parameters()]
            if calls>=50 or calls-20>=30:raise RuntimeError('30 additional/50 TOTAL hard stop')
            calls+=1;b.calls=calls;opt.step();returned+=1
            torch.cuda.synchronize(0);seconds=time.perf_counter()-begin
            change=math.hypot(*(float((p.detach()-v).double().norm()) for p,v in zip(net.parameters(),before)));del before
            finite=finite_tree(net.state_dict()) and finite_tree(opt.state_dict())
            b.last=dict(**{k:v for k,v in (b.last or {}).items() if k not in ('observed_utc','phase','calls','host','allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')},observed_utc=original.utc().isoformat(),phase='post_adam_cached',calls=calls,
                host=original.host_memory(),allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            row=dict(update=step,training_bce=float(loss.detach()),gradient_l2=norm,gradients_finite=True,
                adam_and_model_finite=finite,parameter_update_l2=change,optimizer_step_calls=calls,
                optimizer_updates=returned,additional_optimizer_calls=calls-20,supervised_targets=int(batch.target_metric_mask.sum()),
                padded_tokens=item['padded_tokens'],sample=item,synchronized_training_seconds=seconds,resources=b.last)
            write_json(output/f'update_{step:03d}.json',row);rows.append(row)
            if not finite or not math.isfinite(change) or change==0:raise FloatingPointError('invalid Adam/model/parameter delta')
            if seconds>c['execution_limits']['max_step_seconds']:raise TimeoutError('shared30s post-Adam')
            b.check('post_update')
            time.sleep(shared.POLICY['inter_update_pause_seconds'])
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
        b.check(phase,cuda=False);net,opt=initialize(plan)
        write_json(output/'runtime.json',runtime(torch.device('cuda:0')));torch.cuda.reset_peak_memory_stats(0)
        result=restore_proof(p,net,opt,c,train,c['diagnostic_samples'])
        result.update(checkpoint_sha256=sha256(path),recovery_execution_seconds=time.perf_counter()-stamp,
            resources=b.check('after_recovery_proof'),source=source_identity(),test_evaluations=0)
        write_json(output/'recovery_result.json',result)
        if update==50:
            write_json(root/'study_result.json',dict(status='FIFTY_TOTAL_UPDATES_COMPLETED_SHARED_GPU',optimizer_updates=50,
                additional_optimizer_calls=30,parent_checkpoint_sha256=PARENT_SHA256,checkpoint_sha256=sha256(path),
                fresh_process_recovery='PASS30/40/50; zero extra Adam calls',no_automatic_further_training=True,
                full_training_ready=False,test_evaluations=0))
        return result
    except BaseException as exc:
        original.record_failure(root,output,exc,phase,update,update,b,stamp);raise
