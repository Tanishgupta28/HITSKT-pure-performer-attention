"""Matched logical batches with bounded physical batches.

No attention/model/data changes. A logical batch retains the original target
membership and order and takes exactly one Adam step. Its masked mean BCE is
the sum of microbatch BCE sums divided by the logical supervised-target count.
Dropout streams and floating-point reduction order need not match a physical B8
run; both new feature-count arms use the same explicit microbatch policy.
"""
from __future__ import annotations

import math
import time
import numpy as np
import torch

from ktbench import favor_plus_paired_epochs as pair
from ktbench.metrics import binary_metrics


def partition(dataset, row, physical_batch):
    if type(physical_batch) is not int or physical_batch not in (1, 2, 4):
        raise ValueError("physical batch must be 1, 2 or 4")
    if not row['indices'] or len(row['indices']) > 8:
        raise ValueError("original logical batch of at most 8 required")
    parts = pair.shape_manifest(dataset, [row['indices'][i:i+physical_batch]
                                        for i in range(0, len(row['indices']), physical_batch)])
    if ([i for p in parts for i in p['indices']] != row['indices']
            or [i for p in parts for i in p['target_sessions']] != row['target_sessions']
            or sum(p['supervised_targets'] for p in parts) != row['supervised_targets']):
        raise ValueError("microbatch membership, order or target count changed")
    return parts


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def logical_batch(net, optimizer, dataset, row, physical_batch, device, boundary,
                  *, training, progress=None):
    """One complete logical batch; never retry a partially accumulated step.

    Metric arrays detach before the next physical forward; no graphs or GPU
    batches are retained across microbatches. Adam is called only after all
    microbatches/guards/finite-gradient checks succeed.
    """
    parts = partition(dataset, row, physical_batch)
    progress = {} if progress is None else progress
    net.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    _sync(device)
    started = time.perf_counter()
    probabilities, labels, logit_values = [], [], []
    weighted_loss = 0.
    fb_seconds = 0.
    for index, part in enumerate(parts):
        boundary.check('micro_train' if training else 'micro_validation')
        batch = pair.make_batch(dataset, part).to(device)
        _sync(device)
        before = time.perf_counter()
        with torch.set_grad_enabled(training):
            logits = net(batch)
            loss = pair.masked_bce_loss(logits, batch.target_labels, batch.target_metric_mask)
            if not pair.finite_tree(logits) or not math.isfinite(float(loss.detach())) or float(loss.detach()) > 20:
                raise FloatingPointError('microbatch nonfinite/excessive loss or logits')
            weight = part['supervised_targets'] / row['supervised_targets']
            if training:
                (loss * weight).backward()
        _sync(device)
        fb_seconds += time.perf_counter() - before
        mask = batch.target_metric_mask
        probabilities.append(logits[mask].detach().sigmoid().cpu().numpy())
        labels.append(batch.target_labels[mask].detach().cpu().numpy())
        logit_values.append(logits[mask].detach().cpu().numpy())
        weighted_loss += float(loss.detach()) * weight
        del logits, loss, batch
        boundary.check('micro_backward_complete' if training else 'micro_forward_complete')
        if time.perf_counter() - started > pair.LIMITS['max_step_seconds']:
            raise TimeoutError('unchanged 30s logical-step gate before Adam')
        if training:
            time.sleep(pair.LIMITS['pause_seconds'])
    gradient = None
    adam_seconds = 0.
    if training:
        gradients = [p.grad for p in net.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('missing/nonfinite accumulated gradient')
        gradient = math.hypot(*(float(g.double().norm()) for g in gradients))
        if not math.isfinite(gradient) or gradient > 100:
            raise FloatingPointError('unchanged accumulated-gradient severe gate')
        boundary.check('pre_real_adam')
        if time.perf_counter() - started > pair.LIMITS['max_step_seconds']:
            raise TimeoutError('unchanged 30s pre-Adam gate')
        progress['optimizer_step_calls'] = progress.get('optimizer_step_calls', 0) + 1
        _sync(device)
        before = time.perf_counter()
        optimizer.step()
        _sync(device)
        adam_seconds = time.perf_counter() - before
        progress['optimizer_updates'] = progress.get('optimizer_updates', 0) + 1
        if not pair.finite_tree(net.state_dict()) or not pair.finite_tree(optimizer.state_dict()):
            raise FloatingPointError('nonfinite parameters or real Adam state')
        state = optimizer.state_dict()['state']
        if (len(state) != len(list(net.parameters())) or any(
                float(s['step']) != progress['optimizer_updates'] or not (s['exp_avg_sq'] >= 0).all()
                for s in state.values())):
            raise ValueError('Adam coverage/counter/second-moment mismatch')
        boundary.check('real_adam_complete')
    seconds = time.perf_counter() - started
    if seconds > pair.LIMITS['max_step_seconds']:
        raise TimeoutError('unchanged 30s logical-step ceiling')
    result = dict(loss=weighted_loss, gradient_l2=gradient, seconds=seconds,
                  forward_backward_seconds=fb_seconds, adam_seconds=adam_seconds,
                  physical_batches=len(parts), targets=row['supervised_targets'],
                  all_gradients_finite=True if training else None,
                  adam_state_finite=True if training else None)
    values = tuple(np.concatenate(x) for x in (probabilities, labels, logit_values))
    if len(values[0]) != row['supervised_targets']:
        raise ValueError('logical metric alignment changed')
    return result, *values


def epoch_data(net, optimizer, dataset, rows, device, boundary, progress, *, training,
               physical_batch, checkpoint_callback=None, checkpoint_interval=25,
               logits_output=None):
    """Resumable only at complete logical-update boundaries, with full metrics."""
    count = pair.target_count(dataset)
    probabilities = np.empty(count, np.float32)
    labels = np.empty(count, np.uint8)
    start = progress.get('next_batch', 0) if training else 0
    cursor = progress.get('train_targets_done', 0) if training else 0
    total = progress.get('train_loss_sum', 0.) if training else 0.
    seen = [i for row in rows[:start] for i in row['indices']]
    updates = []
    if cursor != sum(r['supervised_targets'] for r in rows[:start]):
        raise ValueError('logical resume target cursor mismatch')
    if start:
        probabilities[:cursor] = progress['train_probabilities'].numpy()
        labels[:cursor] = progress['train_labels'].numpy()
    for index in range(start, len(rows)):
        row = rows[index]
        record, prob, y, logits = logical_batch(net, optimizer, dataset, row, physical_batch,
                                               device, boundary, training=training, progress=progress)
        size = len(y)
        probabilities[cursor:cursor+size] = prob
        labels[cursor:cursor+size] = y
        if logits_output is not None:
            logits_output[cursor:cursor+size] = logits
        total += record['loss'] * size
        cursor += size
        seen.extend(row['indices'])
        if training:
            updates.append(dict(batch=index, optimizer_updates=progress['optimizer_updates'], **record))
            progress.update(next_batch=index+1, train_loss_sum=total, train_targets_done=cursor,
                            train_probabilities=torch.from_numpy(probabilities[:cursor].copy()),
                            train_labels=torch.from_numpy(labels[:cursor].copy()))
            if checkpoint_callback and ((index+1) % checkpoint_interval == 0 or index+1 == len(rows)):
                checkpoint_callback(index+1)
    if cursor != count or sorted(seen) != list(range(len(dataset))):
        raise ValueError('full epoch missing/duplicate sessions or targets')
    metrics = binary_metrics(torch.from_numpy(probabilities), torch.from_numpy(labels), loss=total/count)
    metrics['bce'] = metrics['loss']
    if not pair.finite_tree(metrics):
        raise FloatingPointError('nonfinite epoch metrics')
    return metrics, probabilities, labels, updates
