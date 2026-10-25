"""Separate format-6 epoch execution for logical-B8 gradient accumulation.

No training is authorized by the source or by a profiling record. Successful
real-Adam profiles with >=2GiB reserved/allocated headroom, fresh recovery and
separate training/runtime approval are prerequisites for every epoch worker.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from ktbench import favor_plus_paired_epochs as pair
from ktbench import favor_plus_microbatch as micro
from ktbench import favor_plus_micro_profile as policy

FORMAT = 6
KIND = 'hitskt_favor_plus_logical8_microbatch_epochs_v1'


def contract(plan, original, arm, physical):
    value = copy.deepcopy(original['contracts'][arm])
    value.update(source_sha256=plan['source_sha256'], microbatch_policy=dict(
        physical_batch=physical, logical_membership='original frozen B8 groups',
        accumulation='supervised-target weighted; one original Adam step per group',
        order='contiguous original order; no length sorting; no gradient clipping'),
        micro_plan_sha256=pair.digest(plan))
    return value


def _base_view(value):
    return {**value, 'kind':pair.KIND, 'variant':pair.KIND, 'format_version':pair.FORMAT}


def validate_checkpoint(value, c):
    if value.get('kind') != KIND or value.get('variant') != KIND or value.get('format_version') != FORMAT:
        raise ValueError('separate format6 microbatch checkpoint required')
    return pair.checkpoint_validate(_base_view(value), c)


def save_checkpoint(net, optimizer, c, progress, proof, path):
    value = pair.cpu_tree(net.checkpoint())
    value.update(kind=KIND, variant=KIND, format_version=FORMAT, contract=c,
        progress=pair.cpu_tree(progress), optimizer_state_dict=pair.cpu_tree(optimizer.state_dict()),
        reference_eval_logits=pair.fixed_logits(net, proof), reference_probe=pair.probe(net, optimizer, proof),
        rng_state=pair.cpu_tree(pair.capture_rng_state()))
    value['component_sha256'] = {n:pair.digest(value[n]) for n in pair.COMPONENTS}
    validate_checkpoint(value, c)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.'+path.stem, suffix='.partial.pt')
    with os.fdopen(fd, 'wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    restored = torch.load(temp, map_location='cpu', weights_only=True)
    validate_checkpoint(restored, c)
    if not pair.equal_tree(value, restored):
        raise ValueError('microbatch checkpoint roundtrip mismatch')
    os.link(temp, path); os.unlink(temp)
    directory = os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(path=str(path), sha256=pair.sha256(path), atomic_no_overwrite=True, **validate_checkpoint(value, c))


def restore(value, net, optimizer, c, proof):
    validate_checkpoint(value, c)
    return pair.restore_checkpoint(_base_view(value), net, optimizer, c, proof)


def require_profiles(path, record):
    physical = record['physical_batch']
    if set(record.get('profile_sha256', {})) != set(pair.ARMS):
        raise PermissionError('both chosen matched profile identities required')
    for arm in pair.ARMS:
        folder = Path(path).parent/f'profile_b{physical}'/arm
        file = folder/'profile_result.json'
        profile = json.loads(file.read_text()); recovery = json.loads((folder/'fresh_recovery/recovery_result.json').read_text())
        if (record['profile_sha256'][arm] != pair.sha256(file) or profile['plan_sha256'] != pair.sha256(path)
                or profile['status'] != 'PASS_REAL_ADAM_HEADROOM' or profile['physical_batch'] != physical
                or max(profile['resources']['own_allocator'].values()) > policy.HEADROOM_TARGET
                or recovery['status'] != 'PASS_EXACT' or recovery['profile_sha256'] != pair.sha256(file)
                or recovery['checkpoint_sha256'] != pair.sha256(folder/'PROFILE_ONLY.pt')):
            raise PermissionError('both real-Adam >=2GiB-headroom profiles and fresh recoveries must pass')


def worker(path, record, arm, epoch, action, clock):
    path = Path(path); policy.authorize(record, path, policy.TRAIN)
    if arm not in pair.ARMS or type(epoch) is not int or not 1 <= epoch <= 8 or action not in ('train-epoch', 'verify-epoch', 'validate-best'):
        raise ValueError('registered arm, stage and epoch1..8 required')
    plan, original, splits = policy.load_plan(path); require_profiles(path, record)
    if (path.parent/'STOP_NO_RETRY.json').exists():
        raise RuntimeError('failed study cannot retry')
    arm_root = path.parent/f'training_b{record["physical_batch"]}'/arm
    phase = f'epoch_{epoch:03d}' if action=='train-epoch' else f'recovery_{epoch:03d}' if action=='verify-epoch' else 'best_full_validation_replay'
    out = arm_root/phase; out.mkdir(parents=True, exist_ok=False)
    boundary = policy.Boundaries(path, record, out, policy.TRAIN, clock)
    c = contract(plan, original, arm, record['physical_batch']); progress = {}
    try:
        net = pair.initialize(c, boundary)
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-5, betas=(.9,.999), eps=1e-8, weight_decay=0.)
        proof = pair.make_batch(splits['train'], pair.shape_manifest(splits['train'], [[c['diagnostic_samples'][0]['dataset_index']]])[0]).to('cuda:0')
        pair.write_json(out/'runtime.json', pair.runtime(torch.device('cuda:0')))
        progress = dict(epoch=epoch, next_batch=0, optimizer_updates=0, optimizer_step_calls=0,
                        best_auc=None, best_f1=None, last_selected_epoch=0)
        if action=='train-epoch':
            if epoch > 1:
                file = arm_root/'checkpoints'/f'epoch_{epoch-1:03d}.pt'
                prior = json.loads((arm_root/f'recovery_{epoch-1:03d}/recovery_result.json').read_text())
                if prior['status'] != 'PASS_EXACT' or prior['checkpoint_sha256'] != pair.sha256(file):
                    raise ValueError('fresh prior-epoch recovery required')
                value = torch.load(file, map_location='cpu', weights_only=True)
                restore(value, net, optimizer, c, proof); progress = value['progress']; del value
                if progress.get('stop') or progress['phase'] != 'EPOCH_COMPLETE':
                    raise ValueError('prior completed nonterminal epoch required')
                progress.update(epoch=epoch, next_batch=0, train_targets_done=0, train_loss_sum=0.,
                    train_probabilities=torch.empty(0), train_labels=torch.empty(0,dtype=torch.uint8))
            else:
                (out/'initial').mkdir()
                initial, _ = pair.diagnostics(net, splits['train'], c, boundary, out/'initial')
                pair.write_json(arm_root/'initial_attention.json', initial)
            baseline = json.loads((arm_root/'initial_attention.json').read_text())
            def checkpoint_callback(cursor):
                boundary.check('checkpoint')
                receipt = save_checkpoint(net, optimizer, c, progress, proof, arm_root/'checkpoints'/f'epoch_{epoch:03d}_batch_{cursor:04d}.pt')
                pair.write_json(out/f'checkpoint_{cursor:04d}.json', receipt)
            progress['phase'] = 'TRAIN'
            train, _, _, updates = micro.epoch_data(net, optimizer, splits['train'], original['schedules']['train'][str(epoch)],
                'cuda:0', boundary, progress, training=True, physical_batch=record['physical_batch'], checkpoint_callback=checkpoint_callback)
            pair.write_json(out/'training_metrics.json', dict(metrics=train, updates=updates))
            values = np.empty(c['target_counts']['validation']['labels'], np.float32)
            val, prob, y, _ = micro.epoch_data(net, None, splits['validation'], original['schedules']['validation']['0'],
                'cuda:0', boundary, {}, training=False, physical_batch=record['physical_batch'], logits_output=values)
            ids = pair.validation_event_ids(original['schedules'], splits)
            if (pair.hashlib.sha256(ids.tobytes()).hexdigest() != c['validation_ids_sha256']
                    or pair.hashlib.sha256(y.tobytes()).hexdigest() != c['validation_labels_sha256']
                    or not np.array_equal(y, splits['validation'].store.correct[ids])):
                raise ValueError('full validation IDs/labels changed')
            with (out/'full_validation_predictions.npz').open('xb') as f:
                np.savez_compressed(f, probability=prob, label=y, logits=values, event_id=ids)
            (out/'final').mkdir(); _, decision = pair.diagnostics(net, splits['train'], c, boundary, out/'final', baseline)
            progress, improved = pair.selection(progress, epoch, val)
            progress.update(phase='EPOCH_COMPLETE', train_metrics=train, validation_metrics=val)
            checkpoint = save_checkpoint(net, optimizer, c, progress, proof, arm_root/'checkpoints'/f'epoch_{epoch:03d}.pt')
            result = dict(status='PASS_FULL_EPOCH', arm=arm, epoch=epoch, train=train, validation=val, improved=improved,
                checkpoint=checkpoint, updates=updates, attention_status=decision['status'], optimizer_updates=progress['optimizer_updates'],
                resources=boundary.check('epoch_complete'), test_evaluations=0)
            (arm_root/'metrics').mkdir(exist_ok=True)
            pair.write_json(arm_root/'metrics'/f'epoch_{epoch:03d}.json', result); pair.curves(arm_root)
        else:
            selected = epoch
            if action=='validate-best':
                metrics = [json.loads((arm_root/'metrics'/f'epoch_{e:03d}.json').read_text()) for e in range(1,9)]
                selected = max(metrics, key=lambda v:v['validation']['roc_auc'])['epoch']
            file = arm_root/'checkpoints'/f'epoch_{selected:03d}.pt'
            value = torch.load(file, map_location='cpu', weights_only=True)
            result = restore(value, net, optimizer, c, proof)
            if action=='validate-best':
                replay, prob, y, _ = micro.epoch_data(net, None, splits['validation'], original['schedules']['validation']['0'],
                    'cuda:0', boundary, {}, training=False, physical_batch=record['physical_batch'])
                old = np.load(arm_root/f'epoch_{selected:03d}'/'full_validation_predictions.npz')
                if replay != value['progress']['validation_metrics'] or not np.array_equal(prob,old['probability']) or not np.array_equal(y,old['label']):
                    raise ValueError('best full-validation replay must be exact')
                result.update(selected_epoch=selected, metrics=replay)
            result.update(checkpoint_sha256=pair.sha256(file), resources=boundary.check('recovery_complete'))
        pair.write_json(out/('epoch_result.json' if action=='train-epoch' else 'recovery_result.json'), result)
        return result
    except BaseException as exc:
        pair.failure(path.parent, out, boundary, exc, action, progress)
        if not (path.parent/'STOP_NO_RETRY.json').exists():
            pair.write_json(path.parent/'STOP_NO_RETRY.json', dict(type=type(exc).__name__, message=str(exc), no_retry=True))
        raise


def campaign(path, scope_path):
    path, scope_path = Path(path), Path(scope_path)
    record = json.loads(scope_path.read_text()); policy.authorize(record, path, policy.TRAIN)
    policy.load_plan(path); require_profiles(path, record)
    out = path.parent/'approved_training_campaign'; out.mkdir(exist_ok=False)
    parent_sha = pair.sha256(scope_path); stages = []
    for arm in pair.ARMS:
        clock = dict(parent_path=str(scope_path), parent_sha256=parent_sha, campaign_started_unix=record['campaign_started_unix'], arm_started_unix=time.time())
        clock_path = out/f'CLOCK_{arm}.json'; pair.write_json(clock_path, clock)
        for epoch in range(1,9):
            for action in ('train-epoch','verify-epoch'):
                stages.append((arm, epoch, action, clock_path))
        stages.append((arm,8,'validate-best',clock_path))
        # Arm starts only when its actual stages start, not when planning the other arm.
        for _, epoch, action, cp in [s for s in stages if s[0]==arm]:
            policy.authorize(record, path, policy.TRAIN)
            if pair.sha256(scope_path) != parent_sha:
                raise PermissionError('parent record changed')
            remaining = min(record['campaign_started_unix']+record['total_seconds'], clock['arm_started_unix']+record['per_arm_seconds'],
                            dt_expiry(record))-time.time()
            if remaining < 120:
                raise TimeoutError('no new epoch/recovery without remaining budget headroom')
            command = [sys.executable, 'scripts/run_favor_plus_microbatch.py', action, '--plan', str(path), '--scope', str(scope_path),
                       '--arm', arm, '--epoch', str(epoch), '--clock', str(cp), '--execute-training']
            with (out/f'{arm}_{action}_{epoch:03d}.log').open('xb') as stream:
                subprocess.run(command, cwd=pair.ROOT, env=os.environ, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return dict(status='BOTH_EIGHT_EPOCH_ARMS_COMPLETED', test_evaluations=0)


def dt_expiry(record):
    return policy.dt.datetime.fromisoformat(record['expires_utc']).timestamp()
