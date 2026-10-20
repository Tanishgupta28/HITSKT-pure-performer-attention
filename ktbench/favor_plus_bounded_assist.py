"""Prospectively bounded ASSIST study. Separate from five-step/full pipelines.

CPU review freezes inputs and validation sample. GPU stages cannot exceed a
single 100-call/45-minute campaign; continuation is explicit at20 and50 only.
No test data selection, projection redraw during training, retries or HPO.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ktbench.batching import collate_hitskt, padded_token_cost
from ktbench.data.session_store import RollingSessionDataset, SessionStore, make_session_dataloader
from ktbench.favor_plus_gpu_access import CUDA_VALIDATION, LIGHT_HOLD, require_execution_access
from ktbench.favor_plus_pipeline import ROOT, PROTOCOL, fresh_run, model_config, profile_for, runtime, sha256, write_json
from ktbench.favor_plus_pilot_diagnostics import cpu_tree, equal_tree, finite_tree, fixed_example_diagnostics
from ktbench.favor_plus_real_pilot import FAVOR, check_host_resources, collate_selected, pilot_source_identity, require_cuda_verification, verified_dataset_identity
from ktbench.metrics import binary_metrics, masked_bce_loss
from ktbench.models.favor_plus import FavorPlusConfig, orthogonal_gaussian_projection
from ktbench.models.hitskt_favor_plus import HiTSKTFavorPlus, VARIANT
from scripts.favor_plus_quality_common import describe, independent_attention, output_errors

DESIGN = ROOT/'reports/favor_plus_v1/assist_bounded_v1/PREDECLARED_DESIGN.json'
OLD_PILOT = ROOT/'experiments/hitskt_favor_plus_v1/assist2017/real_assist_five_updates_20261020_v1'
CUDA_REPORT = ROOT/'experiments/hitskt_favor_plus_v1/assist2017/cuda_assist_owner_20261020_v2/gpu_validation.json'
ACTION = 'favor_plus_assist2017_bounded_100_update_validation_pilot'
KIND = 'assist2017_favor_plus_bounded_training_validation_v1'


def now():
    return dt.datetime.now(dt.timezone.utc)


def identity():
    result = pilot_source_identity()
    for name in ('ktbench/favor_plus_bounded_assist.py', 'scripts/bounded_assist_favor_plus.py'):
        result['source_sha256'][name] = sha256(ROOT/name)
    return result


def descriptor(dataset, index):
    target = int(dataset.targets[index])
    start, end = dataset._history_bounds(target)
    shape = dataset.shape(index)
    split = int(dataset.store.session_split[target])
    if (end != target or start >= target or split != dataset.split
            or not np.all(dataset.store.session_student[start:target+1] == dataset.store.session_student[target])
            or not np.all(dataset.store.session_split[start:target+1] <= split)):
        raise ValueError('nonchronological/cross-student or future-split visibility')
    return dict(dataset_index=int(index), target_session=target, history_start=start,
        history_end_exclusive=end, student=int(dataset.store.session_student[target]),
        split='train' if split == 0 else 'validation', history_lengths=list(shape.history_lengths),
        target_length=shape.target_length, padded_tokens=padded_token_cost([shape]))


def stratum(item):
    length = item['target_length']
    return (0 if length <= 32 else 1 if length <= 128 else 2)*2 + (len(item['history_lengths']) > 4)


def select_samples(store):
    train = RollingSessionDataset(store, 'train', history_sessions=15)
    validation = RollingSessionDataset(store, 'validation', history_sessions=15)
    buckets = [[] for _ in range(6)]
    for i in range(len(train)):
        item = descriptor(train, i)
        if item['padded_tokens'] <= 2048:
            buckets[stratum(item)].append(item)
    diagnostics, missing, students = [], [], set()
    for b, items in enumerate(buckets):
        if not items:
            missing.append(b)
            continue
        chosen = next((a for a in items if a['student'] not in students), items[0])
        students.add(chosen['student'])
        diagnostics.append({**chosen, 'stratum': b})
    _, sampler = make_session_dataloader(train, token_budget=2048, maximum_batch_size=1,
        bucket_size=512, shuffle=True, seed=42, workers=0, pin_memory=False)
    sampler.set_epoch(1)
    selected, skipped = [], 0
    for batch in sampler:
        if len(batch) != 1:
            raise ValueError('singleton batches required')
        item = descriptor(train, batch[0])
        if item['padded_tokens'] > 2048:
            skipped += 1
            continue
        selected.append(item)
        if len(selected) == 100:
            break
    if len(selected) != 100 or len({a['target_session'] for a in selected}) != 100:
        raise ValueError('100 distinct whole training targets required')
    val_buckets = [[] for _ in range(6)]
    for i in range(len(validation)):
        item = descriptor(validation, i)
        if item['padded_tokens'] <= 2048:
            val_buckets[stratum(item)].append(item)
    frozen = []
    for offset in range(32):
        for items in val_buckets:
            if offset < len(items) and len(frozen) < 32:
                frozen.append(items[offset])
    if len(frozen) != 32:
        raise ValueError('32 eligible validation targets required')
    labels = torch.cat([collate_selected(validation, a).target_labels.masked_select(
        collate_selected(validation, a).target_metric_mask) for a in frozen]).long()
    coverage = dict(targets=labels.numel(), positive=int(labels.sum()), negative=int((labels == 0).sum()))
    if coverage['targets'] < 500 or min(coverage['positive'], coverage['negative']) < 25:
        raise ValueError('predeclared validation coverage unmet; do not train or adapt sample to scores')
    return train, validation, dict(training=selected, diagnostics=diagnostics,
        missing_diagnostic_strata=missing, validation=frozen, validation_coverage=coverage,
        skipped_oversize_training_singletons=skipped)


def severe_check(stage_sets, design):
    """Return containment failures. The old0.20 REVIEW is deliberately separate."""
    gate = design['prospective_severe_stops']
    reasons, aggregate = [], {}
    names = sorted({name for stages in stage_sets for name in stages})
    for name in names:
        rows = [s[name] for s in stage_sets]
        errors = [r['feature_versus_exact_softmax']['relative_l2'] for r in rows]
        ess = [r['kernel_statistics']['feature_ess']['median'] for r in rows]
        growth = [x for r in rows for x in r.get('p95_ratio_to_initial', {}).values()]
        if not all(math.isfinite(x) for x in errors+ess+growth):
            reasons.append(name)
            aggregate[name] = {'nonfinite_diagnostic': True}
            continue
        aggregate[name] = dict(relative_l2=describe(errors), pair_ess_medians=describe(ess),
            max_implementation_abs=max(r['implementation_error']['max_abs'] for r in rows),
            max_p95_growth_ratio=max(growth, default=1.))
        a = aggregate[name]
        if (not finite_tree(a) or a['max_implementation_abs'] > gate['feature_kernel_max_abs']
                or np.median(errors) > gate['stage_median_relative_l2']
                or max(errors) > gate['any_prefix_relative_l2']
                or np.median(ess) < gate['stage_median_pair_ess_min']
                or a['max_p95_growth_ratio'] > gate['qk_p95_growth_ratio']):
            reasons.append(name)
    if not names:
        reasons.append('no stages observed')
    return dict(status='STOP' if reasons else 'CONTINUE_BOUNDED_DIAGNOSTIC_ONLY',
        severe_failure_stages=reasons, aggregate=aggregate,
        original_softmax_review_trigger=.20, full_training_ready=False)


def capture(model, dataset, selected, output, tag, baseline=None):
    results = []
    for i, item in enumerate(selected):
        batch = collate_selected(dataset, item).to(next(model.parameters()).device)
        def save(name, arrays):
            with (output/f'{tag}_e{i}_{name.replace(".", "_")}.npz').open('xb') as f:
                np.savez_compressed(f, **arrays)
        stages = fixed_example_diagnostics(model, batch,
            baseline=None if baseline is None else baseline[i], save_prefix=save)
        results.append(stages)
    write_json(output/f'{tag}_diagnostics.json', results)
    return results


def query_errors(feature, exact, mask):
    # Per valid query/head: distinct from the complete-output relative norm.
    denominator = np.linalg.norm(exact, axis=-1)
    error = np.linalg.norm(feature-exact, axis=-1)/np.maximum(denominator, 1e-30)
    return describe(error[np.broadcast_to(mask[:, None, :], error.shape)])


def prepare_review(run_id):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('CPU review requires CUDA_VISIBLE_DEVICES empty or -1')
    torch.set_num_threads(1)
    design = json.loads(DESIGN.read_text())
    memory = check_host_resources()
    ref = ROOT/'reports/favor_plus_v1/numerical_quality_v1/dataset_pipeline_plan.json'
    store = SessionStore(ROOT/'data/processed/assist2017/full/session_store')
    dataset_identity = verified_dataset_identity(store, json.loads(ref.read_text())['datasets']['assist2017'])
    train, validation, selected = select_samples(store)
    output = fresh_run('assist2017', run_id)
    plan = dict(kind=KIND, design_sha256=sha256(DESIGN), design=design, source=identity(),
        dataset_identity=dataset_identity, model_config=asdict(model_config('assist2017',
        store.metadata['num_questions'], store.metadata['num_skills'])), profile=profile_for('assist2017'),
        protocol=PROTOCOL, samples=selected, store_path=str(store.root), created_utc=now().isoformat(),
        prior_checkpoint_sha256=sha256(OLD_PILOT/'pilot_model.pt'), test_evaluations=0,
        validation_note='sampled preliminary metrics only; frozen before any new training/validation scores')
    write_json(output/'prepared_plan.json', plan)
    from ktbench.config import seed_everything
    seed_everything(42)
    initial = HiTSKTFavorPlus(model_config('assist2017', store.metadata['num_questions'],
        store.metadata['num_skills']), FavorPlusConfig(**FAVOR)).eval()
    first = capture(initial, train, selected['diagnostics'], output, 'initial')
    del initial
    trained = HiTSKTFavorPlus.from_checkpoint(OLD_PILOT/'pilot_model.pt').eval()
    after = capture(trained, train, selected['diagnostics'], output, 'after_five', baseline=first)
    del trained
    # The same Q/K/V are used for every count/seed. No whole-model count substitution.
    comparisons = []
    for path in sorted(output.glob('*_e*_*.npz')):
        check_host_resources()
        with np.load(path) as z:
            q,k,v,qm,km = [z[key] for key in ('q','k','v','query_mask','key_mask')]
        for m in design['frozen_review']['features']:
            for seed in design['frozen_review']['projection_seeds']:
                omega = orthogonal_gaussian_projection(q.shape[1], m, q.shape[-1], seed).numpy()
                feature, exact, stats = independent_attention(q,k,v,omega,qm,km)
                comparisons.append(dict(prefix=path.name, features=m, projection_seed=seed,
                    error=output_errors(feature, exact, qm), per_query_relative_l2=query_errors(feature,exact,qm),
                    kernel_statistics=stats))
    write_json(output/'frozen_feature_comparisons.json', comparisons)
    # Isolate changing CPE Q/K/V using the preserved original single-example arrays.
    old_name = 'correct_padding_encoder_layers_0_attention'
    with np.load(OLD_PILOT/f'prefix_00_{old_name}.npz') as z:
        a = {k:z[k] for k in z.files}
    with np.load(OLD_PILOT/f'prefix_05_{old_name}.npz') as z:
        b = {k:z[k] for k in z.files}
    counterfactual = []
    for qstep in (0,5):
        for kstep in (0,5):
            for vstep in (0,5):
                q,k,v = [(a if step == 0 else b)[key] for key,step in (('q',qstep),('k',kstep),('v',vstep))]
                feature, exact, stats = independent_attention(q,k,v,a['projection'],a['query_mask'],a['key_mask'])
                counterfactual.append(dict(q_step=qstep,k_step=kstep,v_step=vstep,
                    error=output_errors(feature,exact,a['query_mask']), kernel_statistics=stats))
    write_json(output/'saved_cpe_qkv_decomposition.json', counterfactual)
    decisions = dict(initial=severe_check(first, design), after_five=severe_check(after, design))
    result = dict(phase1_status='SAFE_FOR_BOUNDED_DIAGNOSTIC' if all(
        r['status'] != 'STOP' for r in decisions.values()) else 'STOP', decisions=decisions,
        original_outcome='REVIEW_REQUIRED;0.20 unchanged', comparisons=len(comparisons),
        host_memory=check_host_resources(), initial_host_memory=memory,
        prepared_plan_sha256=sha256(output/'prepared_plan.json'), optimizer_updates=0,
        cuda_initialized=False, test_evaluations=0)
    write_json(output/'review_decision.json', result)
    return dict(output=str(output), **result)


def authorize_bounded(allocation):
    # Compose the unchanged common allocation/CUDA gate with a NEW study-specific
    # approval. Five-step approval alone never permits100 updates. No old gate edited.
    if ACTION not in allocation.get('user_approval', {}).get('actions', []):
        raise ValueError('explicit bounded100-update/validation user approval required')
    if json.loads(LIGHT_HOLD.read_text()).get('status') != 'USER_TRAINING_HOLD_ACTIVE':
        raise RuntimeError('LightSAKT2 restart hold is not active')
    return require_execution_access(allocation, os.environ.get('CUDA_VISIBLE_DEVICES'), CUDA_VALIDATION)


def stage_range(previous, milestone):
    expected = {20:0, 50:20, 100:50}
    if type(milestone) is not int or milestone not in expected or previous != expected[milestone]:
        raise ValueError('only explicit0→20→50→100 continuation is permitted')
    return range(previous+1, milestone+1)


def budget_check(design, started, calls, allocated, *, clock=None):
    elapsed = (now() if clock is None else clock)-dt.datetime.fromisoformat(started)
    limits = design['limits']
    if elapsed.total_seconds() < 0 or elapsed.total_seconds() > limits['max_execution_seconds']:
        raise TimeoutError('single-campaign45-minute wall deadline exceeded')
    if calls > limits['max_optimizer_updates']:
        raise RuntimeError('100-call campaign limit exceeded')
    if allocated > limits['max_cuda_allocated_bytes']:
        raise MemoryError('8GiB CUDA allocation ceiling exceeded')


def validate_plan(path):
    plan = json.loads(path.read_text())
    if (plan['kind'] != KIND or plan['design_sha256'] != sha256(DESIGN)
            or plan['design'] != json.loads(DESIGN.read_text())
            or plan['source']['source_sha256'] != identity()['source_sha256']):
        raise ValueError('plan/design/source mismatch; no execution')
    review = json.loads((path.parent/'review_decision.json').read_text())
    if review.get('phase1_status') != 'SAFE_FOR_BOUNDED_DIAGNOSTIC' or review['prepared_plan_sha256'] != sha256(path):
        raise ValueError('matching phase1 containment PASS required')
    store = SessionStore(plan['store_path'])
    ref = json.loads((ROOT/'reports/favor_plus_v1/numerical_quality_v1/dataset_pipeline_plan.json').read_text())
    if verified_dataset_identity(store, ref['datasets']['assist2017']) != plan['dataset_identity']:
        raise ValueError('dataset identity changed')
    train, validation, samples = select_samples(store)
    if samples != plan['samples'] or plan['protocol'] != PROTOCOL or plan['profile'] != profile_for('assist2017'):
        raise ValueError('selection/protocol/config mismatch')
    return plan, train, validation


def evaluate(model, dataset, items):
    was_training = model.training
    model.eval()
    probs, labels, losses = [], [], []
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for item in items:
                check_host_resources()
                batch = collate_selected(dataset, item).to(device)
                logits = model(batch)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('nonfinite validation/training-eval logits')
                mask = batch.target_metric_mask
                loss = masked_bce_loss(logits, batch.target_labels, mask)
                probs.append(logits[mask].sigmoid().cpu())
                labels.append(batch.target_labels[mask].cpu())
                losses.append(float(loss)*int(mask.sum()))
        probability, label = torch.cat(probs), torch.cat(labels)
        result = binary_metrics(probability, label, loss=sum(losses)/len(label))
        if not math.isfinite(result['roc_auc']):
            result['roc_auc'] = None
        return result, probability, label
    finally:
        model.train(was_training)


def causal_check(model, dataset, item):
    """Re-collate raw labels, including own, last/future, rather than edit shifted inputs."""
    example = dataset[item['dataset_index']]
    device = next(model.parameters()).device
    def collate(ex):
        return collate_hitskt([ex], num_questions=model.config.num_questions, num_skills=model.config.num_skills).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            before = model(collate(example))
            for position in sorted({0, len(example.target)//2, len(example.target)-1}):
                responses = list(example.target.correct)
                responses[position] = 1-responses[position]
                modified = replace(example, target=replace(example.target, correct=responses))
                after = model(collate(modified))
                if not torch.equal(before[:, :position+1], after[:, :position+1]):
                    raise RuntimeError('raw own/future response changed own or earlier predictions')
        return dict(status='PASS', changed_raw_positions=sorted({0,len(example.target)//2,len(example.target)-1}),
            requirement='own/earlier exact equality after raw-response re-collation')
    finally:
        model.train(was_training)


def continuation_probe(model, optimizer, dataset, item):
    """Train-mode forward/backward without Adam; RNG restored, no update counted."""
    from ktbench.training import capture_rng_state, restore_rng_state
    state = cpu_tree(capture_rng_state())
    mode = model.training
    optimizer.zero_grad(set_to_none=True)
    try:
        model.train()
        batch = collate_selected(dataset, item).to(next(model.parameters()).device)
        logits = model(batch)
        loss = masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask)
        loss.backward()
        if not finite_tree(logits) or any(p.grad is None or not finite_tree(p.grad) for p in model.parameters()):
            raise FloatingPointError('nonfinite continuation probe')
        digest = hashlib.sha256()
        for name, value in [('logits',logits), *[(n,p.grad) for n,p in model.named_parameters()]]:
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return dict(sha256_logits_and_all_parameter_gradients=digest.hexdigest(), loss=float(loss),
            dataset_index=item['dataset_index'], optimizer_step_calls=0, rng_restored=True)
    finally:
        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(state)
        model.train(mode)


def execute_stage(plan_path, allocation, run_id, milestone, previous_checkpoint=None):
    authorize_bounded(allocation)  # no CUDA API or output creation before approval/conflict checks
    plan, train, validation = validate_plan(plan_path)
    require_cuda_verification(CUDA_REPORT, allocation)  # all18 original source hashes remain unchanged
    design = plan['design']
    prior = None
    if previous_checkpoint is not None:
        prior = torch.load(previous_checkpoint,map_location='cpu',weights_only=True)
        status = json.loads((previous_checkpoint.parent/'stage_result.json').read_text())
        if (prior.get('kind') != KIND or prior.get('prepared_plan_sha256') != sha256(plan_path)
                or status.get('execution_status') != 'PASS' or status.get('containment_status') != 'CONTINUE_BOUNDED_DIAGNOSTIC_ONLY'
                or sha256(previous_checkpoint) != status.get('checkpoint_sha256')
                or prior.get('optimizer_step_calls') != prior.get('optimizer_updates')):
            raise ValueError('valid reviewed predecessor checkpoint required; failed steps never restart')
    previous = 0 if prior is None else prior['optimizer_updates']
    steps = stage_range(previous, milestone)
    started = now().isoformat() if prior is None else prior['campaign_started_utc']
    budget_check(design,started,previous,0)
    output = fresh_run('assist2017',run_id)
    write_json(output/'execution_config.json',dict(plan_path=str(plan_path),prepared_plan_sha256=sha256(plan_path),
        source=identity(), allocation=allocation, milestone=milestone, previous=previous,
        previous_checkpoint=None if prior is None else str(previous_checkpoint),
        previous_checkpoint_sha256=None if prior is None else sha256(previous_checkpoint),
        campaign_started_utc=started, test_evaluations=0))
    calls, returned, phase = previous, previous, 'initialization'
    stage_started = time.perf_counter()
    def boundary():
        authorize_bounded(allocation)
        memory = check_host_resources()
        budget_check(design,started,calls,torch.cuda.memory_allocated(0))
        return memory
    def interrupted(signum,frame):
        raise InterruptedError('termination received; no step restart')
    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        from ktbench.config import seed_everything
        from ktbench.training import capture_rng_state, restore_rng_state
        seed_everything(42)
        torch.use_deterministic_algorithms(True,warn_only=False)
        capacity = torch.cuda.get_device_properties(0).total_memory
        if capacity < 10*1024**3:
            raise MemoryError('less than8GiB quota plus2GiB runtime headroom')
        torch.cuda.set_per_process_memory_fraction(8*1024**3/capacity,0)
        device = torch.device('cuda:0')
        net = HiTSKTFavorPlus(model_config('assist2017',train.store.metadata['num_questions'],
            train.store.metadata['num_skills']),FavorPlusConfig(**FAVOR)).to(device)
        optimizer = torch.optim.Adam(net.parameters(),lr=plan['profile']['learning_rate'],
            weight_decay=plan['profile']['weight_decay'])
        if prior is not None:
            net.load_state_dict(prior['state_dict'],strict=True)
            optimizer.load_state_dict(prior['optimizer_state_dict'])
            if not equal_tree(prior['optimizer_state_dict'],cpu_tree(optimizer.state_dict())) or not finite_tree(optimizer.state_dict()):
                raise RuntimeError('fresh-process optimizer reload mismatch')
            restore_rng_state(prior['rng_state'])
            probe = continuation_probe(net,optimizer,train,plan['samples']['training'][previous])
            if probe != prior['next_update_probe']:
                raise RuntimeError('fresh-process train-mode logits/gradient/RNG continuation mismatch')
            write_json(output/'fresh_process_continuation.json',dict(status='PASS',probe=probe,
                previous_checkpoint_sha256=sha256(previous_checkpoint), additional_optimizer_updates=0))
        del prior
        write_json(output/'runtime.json',runtime(device))
        torch.cuda.reset_peak_memory_stats(0)
        baseline = json.loads((plan_path.parent/'initial_diagnostics.json').read_text())
        def diagnose(step):
            nonlocal phase
            phase = f'milestone_{step}_diagnostics'
            boundary()
            stages = capture(net,train,plan['samples']['diagnostics'],output,f'm{step:03d}',baseline)
            containment = severe_check(stages,design)
            write_json(output/f'm{step:03d}_containment.json',containment)
            if containment['status'] == 'STOP':
                raise FloatingPointError('predeclared severe approximation/concentration/norm/kernel criterion failed')
            causal = causal_check(net,train,plan['samples']['diagnostics'][-1])
            train_metrics,_,_ = evaluate(net,train,plan['samples']['diagnostics'])
            val_metrics,prob,labels = evaluate(net,validation,plan['samples']['validation'])
            with (output/f'm{step:03d}_validation_predictions.npz').open('xb') as f:
                np.savez_compressed(f,probability=prob.numpy(),label=labels.numpy())
            row = dict(optimizer_updates=step, sampled_training_eval=train_metrics,
                sampled_validation=val_metrics,validation_is_preliminary=True,test_evaluations=0,
                causal=causal, containment=containment,host_memory=boundary(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0))
            write_json(output/f'milestone_{step:03d}.json',row)
            return row
        if previous == 0:
            diagnose(0)
        rows = []
        for step in steps:
            phase = f'optimizer_update_{step}'
            boundary()
            batch = collate_selected(train,plan['samples']['training'][step-1]).to(device)
            net.train(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(0); stamp = time.perf_counter()
            logits = net(batch)
            loss = masked_bce_loss(logits,batch.target_labels,batch.target_metric_mask)
            if not finite_tree(logits) or not finite_tree(loss) or float(loss) > design['prospective_severe_stops']['single_update_bce_max']:
                raise FloatingPointError('nonfinite/excessive training BCE; not stepped')
            loss.backward()
            grads = [p.grad for p in net.parameters()]
            if any(g is None or not finite_tree(g) for g in grads):
                raise FloatingPointError('nonfinite/missing gradients; not stepped')
            gradnorm = math.hypot(*(float(g.double().norm()) for g in grads))
            if not math.isfinite(gradnorm) or gradnorm > design['prospective_severe_stops']['gradient_l2_max']:
                raise FloatingPointError('nonfinite/excessive global gradient norm; not stepped')
            boundary()
            if time.perf_counter()-stamp > design['limits']['max_step_seconds']:
                raise TimeoutError('pre-step120s deadline; no optimizer call')
            before = [p.detach().clone() for p in net.parameters()]
            if calls >= 100:
                raise RuntimeError('100-call hard limit')
            calls += 1  # even a partially raising Adam call consumes the budget
            optimizer.step(); returned += 1
            torch.cuda.synchronize(0); seconds = time.perf_counter()-stamp
            delta = math.hypot(*(float((p.detach()-old).double().norm()) for p,old in zip(net.parameters(),before)))
            del before
            finite = finite_tree(net.state_dict()) and finite_tree(optimizer.state_dict())
            row = dict(update=step,training_bce=float(loss.detach()),gradient_l2=gradnorm,
                gradients_finite=True,model_adam_finite=finite,parameter_delta_l2=delta,
                optimizer_step_calls=calls,optimizer_updates=returned,training_seconds=seconds,
                supervised_targets=int(batch.target_metric_mask.sum()),
                padded_tokens=batch.padded_tokens, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0))
            write_json(output/f'update_{step:03d}.json',row); rows.append(row)
            if not finite or not math.isfinite(delta) or delta == 0:
                raise FloatingPointError('nonfinite/unchanged state after Adam; no retry')
            if seconds > design['limits']['max_step_seconds']:
                raise TimeoutError('120s update deadline exceeded')
            boundary()
        # Preserve the reached state even if the milestone review then stops the
        # study. This snapshot is deliberately NOT an eligible continuation.
        phase = 'pre_review_preservation'
        boundary()
        snapshot = cpu_tree(net.checkpoint())
        snapshot.update(kind=KIND+'_unreviewed_snapshot', optimizer_updates=returned,
            optimizer_step_calls=calls, optimizer_state_dict=cpu_tree(optimizer.state_dict()),
            rng_state=cpu_tree(capture_rng_state()), prepared_plan_sha256=sha256(plan_path))
        snapshot_path = output/f'unreviewed_{milestone:03d}.pt'
        with snapshot_path.open('xb') as f:
            torch.save(snapshot, f)
        del snapshot
        write_json(output/'unreviewed_snapshot.json', dict(path=snapshot_path.name,
            sha256=sha256(snapshot_path), continuation_eligible=False))
        summary = diagnose(milestone)
        phase = 'checkpoint_and_reload'
        boundary()
        next_probe = None if milestone == 100 else continuation_probe(net,optimizer,train,plan['samples']['training'][milestone])
        net.eval()
        fixed = collate_selected(train,plan['samples']['diagnostics'][0]).to(device)
        with torch.no_grad():
            before_logits = net(fixed).detach()
        payload = cpu_tree(net.checkpoint())
        payload.update(kind=KIND,optimizer_updates=returned,optimizer_step_calls=calls,
            optimizer_state_dict=cpu_tree(optimizer.state_dict()),rng_state=cpu_tree(capture_rng_state()),
            prepared_plan_sha256=sha256(plan_path),campaign_started_utc=started,next_update_probe=next_probe)
        checkpoint = output/f'checkpoint_{milestone:03d}.pt'
        with checkpoint.open('xb') as f:
            torch.save(payload,f)
        loaded = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not equal_tree(payload,loaded):
            raise RuntimeError('model/Adam/RNG/cursor checkpoint serialization mismatch')
        restored = HiTSKTFavorPlus.from_checkpoint(checkpoint).to(device).eval()
        with torch.no_grad():
            after_logits = restored(fixed)
        if not torch.equal(before_logits,after_logits):
            raise RuntimeError('exact checkpoint CUDA logits mismatch')
        del restored,loaded,payload
        boundary()
        result = dict(execution_status='PASS',containment_status=summary['containment']['status'],
            optimizer_step_calls=calls,optimizer_updates=returned,milestone=milestone,
            checkpoint_sha256=sha256(checkpoint),checkpoint_serialization='exact model/Adam/RNG/cursor',
            reload_max_abs_logit_difference=0., training_interval_seconds=sum(r['training_seconds'] for r in rows),
            stage_execution_seconds=time.perf_counter()-stage_started,
            campaign_elapsed_seconds=(now()-dt.datetime.fromisoformat(started)).total_seconds(),
            stage_training_bce_target_weighted=sum(r['training_bce']*r['supervised_targets'] for r in rows)/sum(r['supervised_targets'] for r in rows),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
            host_memory=check_host_resources(),full_training_ready=False,test_evaluations=0)
        write_json(output/'stage_result.json',result)
        (output/'_BOUNDED_STAGE_COMPLETE').write_text('diagnostic study only; not benchmark success\n')
        return dict(output=str(output),**result)
    except BaseException as exc:
        write_json(output/'stage_failure.json',dict(execution_status='FAIL',phase=phase,
            failure_type=type(exc).__name__,message=str(exc),optimizer_step_calls=calls,
            optimizer_updates=returned,stage_execution_seconds=time.perf_counter()-stage_started,
            note='no automatic retry or optimizer reset; inspect all applied-step records'))
        raise
    finally:
        signal.signal(signal.SIGTERM,old_handler)
