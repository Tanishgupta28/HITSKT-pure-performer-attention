"""Preparation and separately authorized real-Adam memory profiling.

No CUDA is touched by preparation. The old B8 plan/profiles remain immutable.
Profile weights are disposable sandbox weights, never training initializations.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from ktbench import favor_plus_paired_epochs as pair
from ktbench import favor_plus_microbatch as micro
from ktbench.favor_plus_pair_profile_budget import stratify

KIND = 'favor_plus_logical8_microbatch_v1'
PROFILE = KIND + '_real_Adam_profile'
TRAIN = KIND + '_matched8epochs'
PHYSICAL_BATCHES = (2, 4)
PROFILE_SECONDS = 900
PROFILE_UPDATE_CEILING = 160
HEADROOM_TARGET = 6 * pair.GIB
SOURCES = ('ktbench/favor_plus_microbatch.py', 'ktbench/favor_plus_micro_profile.py',
           'ktbench/favor_plus_micro_epochs.py', 'scripts/run_favor_plus_microbatch.py')


def cpu_only():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1') or torch.cuda.is_initialized():
        raise PermissionError('CPU preparation requires CUDA hidden/uninitialized')


def accounting(dataset, row, physical_batch, features):
    parts = micro.partition(dataset, row, physical_batch)
    sizes = [pair.storage_components(r['target_sessions'], dataset.store.session_offsets,
                                    dataset.store.session_number, features) for r in parts]
    return dict(physical_batches=len(parts), padded_tokens=sum(r['padded_tokens'] for r in parts),
                largest_component_bytes=max(r['component_sum_bytes'] for r in sizes),
                component_notice='retained-component accounting only, NOT measured peak or proof of fit')


def augment_extremes(rows, summary):
    """Splitting changes which logical group contains the largest microbatch."""
    result = list(rows)
    for physical in PHYSICAL_BATCHES:
        for item in summary[str(physical)]['max_components'].values():
            row = item['row']
            if row['indices'] not in [r['indices'] for r in result]:
                result.append(row)
    return result


def prepare(parent_path, run_id):
    cpu_only()
    parent_path = Path(parent_path).resolve()
    original, splits = pair.load_plan(parent_path)
    if original.get('common_batch_policy', {}).get('max_batch_size') != 8:
        raise ValueError('exact common-B8 parent required')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('fresh simple run ID required')
    # Source binding includes this adapter, not any edits to the old engine.
    sources = {**original['source_sha256'], str(parent_path.relative_to(pair.ROOT)): pair.sha256(parent_path),
               **{n: pair.sha256(pair.ROOT/n) for n in SOURCES}}
    for arm in pair.ARMS:
        for suffix in ('full_shape_profile/profile_result.json', 'runtime_profile/runtime_profile_result.json'):
            file = parent_path.parent/arm/suffix
            sources[str(file.relative_to(pair.ROOT))] = pair.sha256(file)
    old_summary = pair.ROOT/'reports/favor_plus_v1/b8_profile_20261025_v1/MEASURED_BUDGET_SUMMARY.json'
    sources[str(old_summary.relative_to(pair.ROOT))] = pair.sha256(old_summary)
    rows = [r for rr in original['schedules']['train'].values() for r in rr]
    extremes = []
    for split, row in pair.profile_cases(original, splits):
        if split == 'train' and row['indices'] not in [r['indices'] for r in extremes]:
            extremes.append(row)
    design = dict(worst_training=extremes, training_strata=stratify(rows, 15), repeats=2,
                  worst_validation=max(original['schedules']['validation']['0'], key=lambda r:r['padded_tokens']),
                  validation_strata=stratify(original['schedules']['validation']['0'], 8),
                  max_optimizer_steps_per_worker=len(extremes)+30,
                  max_optimizer_steps_total=2*len(PHYSICAL_BATCHES)*(len(extremes)+30),
                  max_seconds=PROFILE_SECONDS, memory_acceptance_bytes=HEADROOM_TARGET,
                  actual_Adam_required=True, real_dataset_training_evidence=False)
    if design['max_optimizer_steps_total'] > PROFILE_UPDATE_CEILING:
        raise ValueError('profile design exceeds prospective160-step ceiling')
    summary = {}
    for physical in (1, 2, 4):
        per_epoch = {}
        for epoch, rr in original['schedules']['train'].items():
            physical_rows = [p for r in rr for p in micro.partition(splits['train'], r, physical)]
            if [i for r in physical_rows for i in r['indices']] != [i for r in rr for i in r['indices']]:
                raise ValueError('microbatch sample order changed')
            if sorted(i for r in physical_rows for i in r['indices']) != list(range(len(splits['train']))):
                raise ValueError('missing or duplicated interaction group')
            per_epoch[epoch] = dict(logical_updates=len(rr), physical_forwards=len(physical_rows),
                supervised_targets=sum(r['supervised_targets'] for r in physical_rows),
                padded_tokens=sum(r['padded_tokens'] for r in physical_rows),
                max_padded_cost=max(r['padded_tokens'] for r in physical_rows))
        summary[str(physical)] = dict(epochs=per_epoch, logical_updates=sum(len(rr) for rr in original['schedules']['train'].values()),
            physical_forwards=sum(r['physical_forwards'] for r in per_epoch.values()),
            max_components={str(m): max((dict(row=r, **accounting(splits['train'], r, physical, m)) for r in rows),
                                      key=lambda v:v['largest_component_bytes']) for m in (128,512)})
    design['worst_training'] = augment_extremes(extremes, summary)
    design['max_optimizer_steps_per_worker'] = len(design['worst_training'])+30
    design['max_optimizer_steps_total'] = 2*len(PHYSICAL_BATCHES)*design['max_optimizer_steps_per_worker']
    if design['max_optimizer_steps_total'] > PROFILE_UPDATE_CEILING:
        raise ValueError('augmented worst-case design exceeds prospective160-step ceiling')
    val_extremes = [design['worst_validation']]
    for physical in PHYSICAL_BATCHES:
        for features in (128,512):
            row = max(original['schedules']['validation']['0'], key=lambda r:accounting(splits['validation'],r,physical,features)['largest_component_bytes'])
            if row['indices'] not in [r['indices'] for r in val_extremes]:
                val_extremes.append(row)
    design['worst_validation_cases'] = val_extremes
    # A diagnostic-only length sort INSIDE logical batches. Never used by execution.
    length_sort = {}
    for physical in PHYSICAL_BATCHES:
        ordered_cost = sorted_cost = 0
        for r in rows:
            ordered_cost += accounting(splits['train'], r, physical, 512)['padded_tokens']
            order = sorted(r['indices'], key=lambda i:(max(splits['train'].shape(i).history_lengths),
                len(splits['train'].shape(i).history_lengths), splits['train'].shape(i).target_length, i))
            sorted_row = pair.shape_manifest(splits['train'], [order])[0]
            sorted_cost += accounting(splits['train'], sorted_row, physical, 512)['padded_tokens']
        length_sort[str(physical)] = dict(ordered_padded_tokens=ordered_cost, sorted_padded_tokens=sorted_cost,
            executed=False, choice='NOT SELECTED; within-group reorder changes dropout assignment and metric order; no cross-group reordering allowed')
    plan = dict(kind=KIND, run_id=run_id, parent_plan=str(parent_path), parent_sha256=pair.sha256(parent_path),
        source_sha256=sources, limits=pair.LIMITS, logical_schedules_sha256=pair.digest(original['schedules']),
        physical_choices=list(PHYSICAL_BATCHES), profile_design=design, accounting=summary, length_sort=length_sort,
        optimizer='original Adam lr5e-5 betas.9/.999 eps1e-8 wd0 foreach=None fused=None; one step per original logical group',
        loss='sum_micro(masked_mean_BCE * n_supervised_micro/n_supervised_logical)',
        stochastic_difference='dropout kernel calls and floating-point reduction differ from physicalB8; same policy for both new arms',
        profile_approved=False, training_approved=False, test_evaluations=0)
    out = pair.fresh_run('assist2017', run_id)
    pair.write_json(out/'prepared_plan.json', plan)
    template = dict(action=PROFILE, prepared_plan_sha256=pair.sha256(out/'prepared_plan.json'), limits=pair.LIMITS,
        resource_uuid=pair.shared.TRANSFERRED_MIG, shared_permission_confirmed=True,
        known_explicit_restriction=False, known_incompatible_reservation=False,
        profile_approved=False, training_approved=False, approval_is_proposal=True, actual_approval_evidence='',
        total_seconds=PROFILE_SECONDS, max_optimizer_updates=design['max_optimizer_steps_total'],
        starts_utc='SET_ONLY_AFTER_ACTUAL_APPROVAL', expires_utc='SET_ONLY_AFTER_ACTUAL_APPROVAL', campaign_started_unix=None)
    pair.write_json(out/'PROFILE_SCOPE_TEMPLATE.json', template)
    load_plan(out/'prepared_plan.json')
    return dict(status='CPU_PREPARED_NOT_AUTHORIZED', plan=str(out/'prepared_plan.json'),
                source_bound=True, cuda_initialized=False, optimizer_steps=0,
                profile_proposed_max_optimizer_updates=design['max_optimizer_steps_total'], accounting=summary,
                length_sort=length_sort)


def load_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    if (plan['kind'] != KIND or plan['limits'] != pair.LIMITS or plan['physical_choices'] != list(PHYSICAL_BATCHES)
            or path.parent.resolve() != (pair.ROOT/'experiments/hitskt_favor_plus_v1/assist2017'/plan['run_id']).resolve()):
        raise ValueError('separate microbatch plan identity/limits required')
    if pair.sha256(Path(plan['parent_plan'])) != plan['parent_sha256']:
        raise ValueError('original B8 parent changed')
    for n, want in plan['source_sha256'].items():
        if pair.sha256(pair.ROOT/n) != want:
            raise ValueError('microbatch source drift: '+n)
    original, splits = pair.load_plan(Path(plan['parent_plan']))
    if pair.digest(original['schedules']) != plan['logical_schedules_sha256']:
        raise ValueError('logical schedule changed')
    if plan['profile_design']['max_optimizer_steps_total'] > PROFILE_UPDATE_CEILING:
        raise ValueError('profile step ceiling changed')
    return plan, original, splits


def authorize(record, path, action, *, now=None):
    if (record.get('action') != action or record.get('prepared_plan_sha256') != pair.sha256(path)
            or record.get('limits') != pair.LIMITS or record.get('resource_uuid') != pair.shared.TRANSFERRED_MIG
            or os.environ.get('CUDA_VISIBLE_DEVICES') != pair.shared.TRANSFERRED_MIG
            or record.get('shared_permission_confirmed') is not True
            or record.get('known_explicit_restriction') is not False or record.get('known_incompatible_reservation') is not False
            or record.get('approval_is_proposal') is not False
            or not isinstance(record.get('actual_approval_evidence'),str) or not record['actual_approval_evidence'].strip()):
        raise PermissionError('actual separate action/plan/resource-bound approval required')
    if action == PROFILE:
        if (record.get('profile_approved') is not True or record.get('training_approved') is not False
                or record.get('total_seconds') != PROFILE_SECONDS or type(record.get('max_optimizer_updates')) is not int
                or not 0 < record['max_optimizer_updates'] <= PROFILE_UPDATE_CEILING):
            raise PermissionError('new <=15min/160-real-Adam profile approval required')
    elif action == TRAIN:
        if (record.get('training_approved') is not True or record.get('new_runtime_budget_approved') is not True
                or record.get('profile_approved') is not False or record.get('max_epochs_per_arm') != 8
                or record.get('physical_batch') not in PHYSICAL_BATCHES
                or type(record.get('total_seconds')) is not int or not 0 < record['total_seconds'] <= 12*3600
                or type(record.get('per_arm_seconds')) is not int or not 0 < record['per_arm_seconds'] <= record['total_seconds']):
            raise PermissionError('separate matched8epochs training/runtime approval required')
    else:
        raise PermissionError('unregistered execution action')
    current = now or dt.datetime.now(dt.timezone.utc)
    start = dt.datetime.fromisoformat(record['starts_utc']); end = dt.datetime.fromisoformat(record['expires_utc'])
    origin = record.get('campaign_started_unix')
    if (start.tzinfo is None or end.tzinfo is None or not start <= current < end
            or not 0 < (end-start).total_seconds() <= record['total_seconds']
            or not isinstance(origin, (float, int)) or not start.timestamp() <= origin <= current.timestamp()
            or current.timestamp()-origin >= record['total_seconds']):
        raise PermissionError('absolute execution window/clock invalid or expired')
    return min(end.timestamp(), origin+record['total_seconds'])


class Boundaries:
    """Same shared policy/limits as the immutable B8 worker, new approval scope."""
    def __init__(self, path, record, out, action, clock=None):
        self.path, self.record, self.output, self.action = Path(path), record, Path(out), action
        self.clock = clock; self.n = 0; self.last = None; self.cached = None
        authorize(record, path, action)

    def check(self, phase, *, cuda=True, admission=False):
        deadline = authorize(self.record, self.path, self.action)
        if self.clock is not None:
            if (self.clock['parent_sha256'] != pair.sha256(Path(self.clock['parent_path']))
                    or self.clock['campaign_started_unix'] != self.record['campaign_started_unix']
                    or not self.record['campaign_started_unix'] <= self.clock['arm_started_unix'] <= time.time()
                    or time.time() >= min(deadline, self.clock['arm_started_unix']+self.record['per_arm_seconds'])):
                raise TimeoutError('immutable parent/arm clock expired or changed')
        pair.shared.check_exposure_and_hold(pair.shared.TRANSFERRED_MIG)
        obs = pair.shared.observe(pair.shared.TRANSFERRED_MIG); host = pair.host_memory()
        decision = pair.shared.assess(obs, host, admission=admission)
        if obs['device'].get('free_bytes', 0) < pair.LIMITS['admission_free_bytes' if admission else 'runtime_free_bytes']:
            decision['status'] = 'STOP'; decision['reasons'].append('unchanged paired headroom floor')
        self.last = dict(phase=phase, observation=obs, host=host, decision=decision, cached_own_allocator=self.cached)
        self.n += 1; pair.write_json(self.output/f'resources_{self.n:05d}.json', self.last)
        if decision['status'] == 'STOP':
            raise pair.shared.SharedResourceStop('; '.join(decision['reasons']), self.last)
        if cuda:
            self.cached = dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            self.last['own_allocator'] = self.cached
            pair.write_json(self.output/f'allocator_{self.n:05d}.json', self.last)
            if max(self.cached.values()) > pair.LIMITS['allocator_bytes']:
                raise MemoryError('unchanged8GiB allocated/reserved/peak cap')
        return self.last

    def admit(self):
        self.check('admission1', cuda=False, admission=True)
        time.sleep(.2)
        self.check('admission2', cuda=False, admission=True)


def save_profile_checkpoint(net, optimizer, proof, progress, path, plan_sha):
    payload = dict(kind=KIND+'_PROFILE_ONLY_NOT_TRAINING', format_version=0,
        model=pair.cpu_tree(net.state_dict()), optimizer=pair.cpu_tree(optimizer.state_dict()),
        rng=pair.cpu_tree(pair.capture_rng_state()), progress=progress.copy(), plan_sha256=plan_sha,
        logits=pair.fixed_logits(net, proof), next_probe=pair.probe(net, optimizer, proof))
    if not pair.finite_tree(payload):
        raise FloatingPointError('nonfinite profiling checkpoint')
    path = Path(path); tmp = path.with_suffix('.partial.pt')
    with tmp.open('xb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    restored = torch.load(tmp, map_location='cpu', weights_only=True)
    if not pair.equal_tree(payload, restored):
        raise ValueError('profile checkpoint roundtrip mismatch')
    os.link(tmp, path); tmp.unlink()
    return dict(path=str(path), sha256=pair.sha256(path), actual_Adam_state=True, training_checkpoint=False)


def profile(path, scope, arm, physical):
    path = Path(path); authorize(scope, path, PROFILE)
    plan, original, splits = load_plan(path)
    if arm not in pair.ARMS or physical not in PHYSICAL_BATCHES:
        raise ValueError('only fixed primary arms/candidates')
    if scope['max_optimizer_updates'] != plan['profile_design']['max_optimizer_steps_total']:
        raise PermissionError('exact design step budget required')
    if (path.parent/'STOP_NO_RETRY.json').exists():
        raise RuntimeError('failed study must not retry')
    out = path.parent/f'profile_b{physical}'/arm; out.mkdir(parents=True, exist_ok=False)
    b = Boundaries(path, scope, out, PROFILE)
    progress = dict(optimizer_updates=0, optimizer_step_calls=0)
    design = plan['profile_design']; c = original['contracts'][arm]
    try:
        net = pair.initialize(c, b)
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-5, betas=(.9,.999), eps=1e-8, weight_decay=0.)
        initial_weights = pair.learned_digest(net)
        (out/'initial').mkdir(); initial, _ = pair.diagnostics(net, splits['train'], c, b, out/'initial')
        jobs = [('worst', i, 0, row, 0) for i, row in enumerate(design['worst_training'])]
        jobs += [('stratum', i, rep, item['shape'], item['weight'])
                 for i, item in enumerate(design['training_strata']) for rep in range(design['repeats'])]
        training = []
        for kind, index, rep, row, weight in jobs:
            if progress['optimizer_updates'] >= design['max_optimizer_steps_per_worker']:
                raise RuntimeError('profile optimizer budget exhausted')
            record, *_ = micro.logical_batch(net, optimizer, splits['train'], row, physical, 'cuda:0', b,
                                              training=True, progress=progress)
            item = dict(kind=kind, index=index, repeat=rep, weight=weight, shape=row, **record,
                        optimizer_updates=progress['optimizer_updates'], resources=b.check('profile_logical_update'))
            training.append(item); pair.write_json(out/f'train_{len(training):03d}.json', item)
        validation = []
        val_jobs = [('worst', -1-i, dict(shape=row, weight=0)) for i,row in enumerate(design['worst_validation_cases'])]
        val_jobs += [('stratum', i, item) for i, item in enumerate(design['validation_strata'])]
        for kind, index, item in val_jobs:
            record, *_ = micro.logical_batch(net, None, splits['validation'], item['shape'], physical, 'cuda:0', b,
                                              training=False)
            validation.append(dict(kind=kind, index=index, weight=item['weight'], **record))
        (out/'final').mkdir(); pair.diagnostics(net, splits['train'], c, b, out/'final', initial)
        if pair.learned_digest(net) == initial_weights:
            raise ValueError('real-Adam profile did not change parameters')
        if {n:pair.digest(v) for n,v in net.named_buffers() if n.endswith('.projection')} != c['expected_projection_sha256']:
            raise ValueError('profile changed fixed projections')
        proof = pair.make_batch(splits['train'], pair.shape_manifest(splits['train'], [[c['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0')
        checkpoint = save_profile_checkpoint(net, optimizer, proof, progress, out/'PROFILE_ONLY.pt', pair.sha256(path))
        resource = b.check('profile_complete')
        peak = max(resource['own_allocator'].values())
        result = dict(status='PASS_REAL_ADAM_HEADROOM' if peak <= HEADROOM_TARGET else 'REVIEW_INSUFFICIENT_HEADROOM',
            arm=arm, physical_batch=physical, memory_acceptance_bytes=HEADROOM_TARGET,
            plan_sha256=pair.sha256(path), training=training, validation=validation,
            actual_Adam_steps=progress['optimizer_updates'], optimizer_state_finite=True,
            weights_updated=True, projections_unchanged=True, checkpoint=checkpoint, resources=resource,
            runtime=pair.runtime(torch.device('cuda:0')), test_evaluations=0, trained_AUC_evidence=False)
        pair.write_json(out/'profile_result.json', result)
        # A below-cap but above-target result is valid negative evidence for this
        # candidate, not an unsafe automatic resize. The frozen study still
        # measures the other arm; training will reject this candidate.
        return result
    except BaseException as exc:
        pair.write_json(out/'failure.json', dict(type=type(exc).__name__, message=str(exc), progress=progress,
            last_cached_resources=b.last, no_retry=True, other_jobs_changed=False))
        pair.write_json(path.parent/'STOP_NO_RETRY.json', dict(phase='real_Adam_profile', arm=arm, physical_batch=physical,
            type=type(exc).__name__, message=str(exc), progress=progress, no_cuda_calls_in_handler=True))
        raise


def recover_profile(path, scope, arm, physical):
    path = Path(path); authorize(scope, path, PROFILE)
    plan, original, splits = load_plan(path)
    source = path.parent/f'profile_b{physical}'/arm
    receipt = json.loads((source/'profile_result.json').read_text())
    file = source/'PROFILE_ONLY.pt'
    if receipt['status'] not in ('PASS_REAL_ADAM_HEADROOM','REVIEW_INSUFFICIENT_HEADROOM') or receipt['checkpoint']['sha256'] != pair.sha256(file):
        raise ValueError('passed exact profile required')
    value = torch.load(file, map_location='cpu', weights_only=True)
    if value['kind'] != KIND+'_PROFILE_ONLY_NOT_TRAINING' or value['plan_sha256'] != pair.sha256(path):
        raise ValueError('not this profiling checkpoint')
    out = source/'fresh_recovery'; out.mkdir(exist_ok=False)
    b = Boundaries(path, scope, out, PROFILE)
    net = pair.initialize(original['contracts'][arm], b)
    opt = torch.optim.Adam(net.parameters(), lr=5e-5, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    net.load_state_dict(value['model']); opt.load_state_dict(value['optimizer']); pair.restore_rng_state(value['rng'])
    sample = original['contracts'][arm]['diagnostic_samples'][0]['dataset_index']
    proof = pair.make_batch(splits['train'], pair.shape_manifest(splits['train'], [[sample]])[0]).to('cuda:0')
    before = pair.digest(pair.cpu_tree(dict(model=net.state_dict(), opt=opt.state_dict(), rng=pair.capture_rng_state())))
    opt.step = lambda *a, **k: (_ for _ in ()).throw(AssertionError('no optimizer steps during recovery'))
    if not torch.equal(pair.fixed_logits(net, proof), value['logits']) or pair.probe(net, opt, proof) != value['next_probe']:
        raise ValueError('profile fresh recovery prediction/gradient mismatch')
    after = pair.digest(pair.cpu_tree(dict(model=net.state_dict(), opt=opt.state_dict(), rng=pair.capture_rng_state())))
    if before != after:
        raise ValueError('recovery changed model/Adam/RNG')
    result = dict(status='PASS_EXACT', optimizer_steps=0, profile_sha256=pair.sha256(source/'profile_result.json'),
                  checkpoint_sha256=pair.sha256(file), resources=b.check('fresh_recovery_complete'))
    pair.write_json(out/'recovery_result.json', result)
    return result


def profile_campaign(path, scope_path):
    path, scope_path = Path(path), Path(scope_path)
    record = json.loads(scope_path.read_text()); authorize(record, path, PROFILE)
    load_plan(path)
    out = path.parent/'approved_profile_campaign'; out.mkdir(exist_ok=False)
    identity = pair.sha256(scope_path)
    try:
        for physical in PHYSICAL_BATCHES:
            for arm in pair.ARMS:
                for action in ('profile-worker', 'recover-profile'):
                    deadline = authorize(record, path, PROFILE)
                    if pair.sha256(scope_path) != identity or deadline-time.time() < 30:
                        raise PermissionError('no new stage without unchanged scope/window headroom')
                    command = [sys.executable, 'scripts/run_favor_plus_microbatch.py', action, '--plan', str(path),
                               '--scope', str(scope_path), '--arm', arm, '--physical-batch', str(physical), '--execute-cuda']
                    with (out/f'{action}_{physical}_{arm}.log').open('xb') as stream:
                        subprocess.run(command, cwd=pair.ROOT, env=os.environ, stdout=stream, stderr=subprocess.STDOUT,
                                       check=True, timeout=deadline-time.time()-15)
        return estimate(path)
    except BaseException as exc:
        if not (path.parent/'STOP_NO_RETRY.json').exists():
            pair.write_json(path.parent/'STOP_NO_RETRY.json', dict(type=type(exc).__name__, message=str(exc), no_retry=True))
        raise


def estimate(path):
    plan, original, _ = load_plan(path)
    result = {}
    for physical in PHYSICAL_BATCHES:
        arms = {}
        for arm in pair.ARMS:
            folder = Path(path).parent/f'profile_b{physical}'/arm
            data = json.loads((folder/'profile_result.json').read_text())
            recovery = json.loads((folder/'fresh_recovery/recovery_result.json').read_text())
            if (data['status'] not in ('PASS_REAL_ADAM_HEADROOM','REVIEW_INSUFFICIENT_HEADROOM') or data['plan_sha256'] != pair.sha256(path)
                    or recovery['status'] != 'PASS_EXACT' or recovery['profile_sha256'] != pair.sha256(folder/'profile_result.json')):
                raise ValueError('successful source-bound real-Adam profiles/recoveries required')
            train = 0.
            for i, item in enumerate(plan['profile_design']['training_strata']):
                measured = [r['seconds'] for r in data['training'] if r['kind']=='stratum' and r['index']==i]
                if len(measured) != 2:
                    raise ValueError('two matched timing repeats required')
                train += item['weight'] * float(np.median(measured))
            val = 9 * sum(r['weight']*r['seconds'] for r in data['validation'] if r['kind']=='stratum')
            # Prior measured I/O/diagnostic/worker costs are clearly separate proxies.
            previous = json.loads((Path(plan['parent_plan']).parent/arm/'runtime_profile/runtime_profile_result.json').read_text())
            checkpoints = sum(math.ceil(len(rr)/25)+1 for rr in original['schedules']['train'].values())
            io = float(np.median([x['total_seconds'] for x in previous['io_proxies']]))
            auxiliary = (checkpoints+8)*io + 9*previous['initial_diagnostics_seconds']
            old_summary = json.loads((pair.ROOT/'reports/favor_plus_v1/b8_profile_20261025_v1/MEASURED_BUDGET_SUMMARY.json').read_text())
            auxiliary += old_summary['arms'][arm]['runtime_estimate']['worker_startup_proxy_seconds']
            arms[arm] = dict(actual_Adam_training_body_seconds=train, validation_seconds=val,
                prior_IO_diagnostics_and_startup_proxy_seconds=auxiliary,
                with25percent_reserve_seconds=1.25*(train+val+auxiliary),
                memory_status=data['status'],
                caveat='not a wall-time guarantee; worker startup, full learned epochs and future shared load remain unmeasured')
        result[str(physical)] = dict(arms=arms, pair_with25percent_reserve_seconds=sum(x['with25percent_reserve_seconds'] for x in arms.values()))
    return dict(status='MEASURED_PROFILE_ESTIMATE_NOT_TRAINING_APPROVAL', configurations=result)


def training_template(path, output):
    """Only AFTER real-Adam profiles: choose common headroom-qualified fastest policy."""
    path, output = Path(path), Path(output)
    estimates = estimate(path)
    eligible = [(int(key), value) for key, value in estimates['configurations'].items()
                if all(v['memory_status']=='PASS_REAL_ADAM_HEADROOM' for v in value['arms'].values())]
    if not eligible:
        raise PermissionError('no matched configuration meets prospective memory headroom')
    physical, value = min(eligible, key=lambda x:x[1]['pair_with25percent_reserve_seconds'])
    pair_seconds = math.ceil(value['pair_with25percent_reserve_seconds']/3600)*3600
    arm_seconds = math.ceil(max(v['with25percent_reserve_seconds'] for v in value['arms'].values())/3600)*3600
    if pair_seconds > 12*3600:
        raise PermissionError('measured estimate exceeds this controller12h preparation bound; new study decision needed')
    record = dict(action=TRAIN, prepared_plan_sha256=pair.sha256(path), limits=pair.LIMITS,
        resource_uuid=pair.shared.TRANSFERRED_MIG, shared_permission_confirmed=True,
        known_explicit_restriction=False, known_incompatible_reservation=False,
        profile_approved=False, training_approved=False, new_runtime_budget_approved=False,
        approval_is_proposal=True, actual_approval_evidence='', physical_batch=physical, max_epochs_per_arm=8,
        total_seconds=pair_seconds, per_arm_seconds=min(arm_seconds,pair_seconds),
        starts_utc='SET_ONLY_AFTER_ACTUAL_APPROVAL', expires_utc='SET_ONLY_AFTER_ACTUAL_APPROVAL', campaign_started_unix=None,
        profile_sha256={arm:pair.sha256(path.parent/f'profile_b{physical}'/arm/'profile_result.json') for arm in pair.ARMS},
        selection='memory/runtime only, no validation/test tuning; actual new training/runtime approval still required')
    if output.exists():
        raise FileExistsError(output)
    pair.write_json(output, record)
    return dict(status='UNAPPROVED_TRAINING_TEMPLATE', selected_physical_batch=physical, proposed_seconds=pair_seconds)
