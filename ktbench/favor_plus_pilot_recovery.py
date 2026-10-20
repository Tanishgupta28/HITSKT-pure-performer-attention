"""CPU serialization/integrity and no-step recovery shared by pilot and tests."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch

from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree,finite_tree

KIND='assist2017_favor_plus_recoverable20_v1'


def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            if v.device.type!='cpu':raise ValueError('digest requires already-CPU tensors')
            h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
            h.update(v.contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for key in sorted(v,key=repr):visit(key);visit(v[key])
        elif isinstance(v,(tuple,list)):
            h.update(type(v).__name__.encode())
            for item in v:visit(item)
        else:h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
        h.update(b'\0')
    visit(value);return h.hexdigest()


def seal(payload):
    keys=('state_dict','optimizer_state_dict','rng_state','sampler','contract','reference_eval_logits','next_update_probe')
    payload['component_sha256']={k:digest(payload[k]) for k in keys}
    payload['projection_sha256']={k:digest(v) for k,v in payload['state_dict'].items() if k.endswith('.projection')}
    return payload


def validate(payload,contract):
    if payload.get('kind')!=KIND or payload.get('contract')!=contract:
        raise ValueError('checkpoint study/source/data/config contract mismatch')
    config=contract['config']
    if (payload.get('variant')!='hitskt_favor_plus_v1' or payload.get('format_version')!=2
            or payload.get('config')!=config['model_config']
            or payload.get('favor_config')!=config['favor_config']):
        raise ValueError('checkpoint architecture/feature configuration mismatch')
    step=payload.get('optimizer_updates')
    if type(step) is not int or step not in (5,10,20) or payload.get('optimizer_step_calls')!=step:
        raise ValueError('checkpoint must contain exactly5,10 or20 returned optimizer calls')
    sampler=payload.get('sampler',{})
    if (sampler.get('cursor')!=step or sampler.get('epoch')!=1 or sampler.get('seed')!=42
            or sampler.get('exact_samples')!=contract['exact_training_and_probe_samples']):
        raise ValueError('checkpoint sampler cursor/identities changed')
    for key,expected in payload.get('component_sha256',{}).items():
        if digest(payload[key])!=expected:raise ValueError(f'checkpoint component hash mismatch:{key}')
    if set(payload.get('component_sha256',{}))!={'state_dict','optimizer_state_dict','rng_state',
            'sampler','contract','reference_eval_logits','next_update_probe'}:
        raise ValueError('missing component hashes')
    projections={k:digest(v) for k,v in payload['state_dict'].items() if k.endswith('.projection')}
    if not projections or projections!=payload.get('projection_sha256'):
        raise ValueError('FAVOR+ projections missing or changed')
    if not finite_tree(payload):raise FloatingPointError('nonfinite checkpoint component')
    optimizer=payload['optimizer_state_dict'];count=contract['parameter_tensor_count']
    if len(optimizer['state'])!=count:
        raise ValueError('Adam parameter-state coverage mismatch')
    if any(float(s.get('step',-1))!=step or 'exp_avg' not in s or 'exp_avg_sq' not in s for s in optimizer['state'].values()):
        raise ValueError('Adam counters/buffers inconsistent with returned updates')
    return dict(status='PASS',updates=step,component_sha256=payload['component_sha256'],
        projection_sha256=projections,sampler_cursor=step,adam_parameter_states=count)


def atomic_checkpoint(path,payload,contract):
    """Validate before publication; hardlink is atomic and cannot replace a file.

    Interrupted .partial.pt files remain evidence but are never eligible. The
    final .pt alone contains all hashes/contract; receipt is a convenience.
    """
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    validate(payload,contract)
    fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.stem+'.',suffix='.partial.pt')
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(payload,f);f.flush();os.fsync(f.fileno())
        loaded=torch.load(temp,map_location='cpu',weights_only=True)
        result=validate(loaded,contract)
        if not equal_tree(payload,loaded):raise RuntimeError('checkpoint serialization changed exact tensors/RNG')
        os.link(temp,path)  # EEXIST refuses overwrite even if another creator raced
        os.unlink(temp)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        result.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),atomic_no_overwrite=True)
        return result
    except BaseException:
        # Retain partial evidence, never publish it as valid/recover it implicitly.
        raise


def no_step_recovery(payload,model,optimizer,contract,*,evaluate_logits,probe):
    """Constructors/optimizer loading precede RNG restoration; forbid Adam.step."""
    from ktbench.training import capture_rng_state,restore_rng_state
    validate(payload,contract)
    model.load_state_dict(payload['state_dict'],strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    if not equal_tree(cpu_tree(optimizer.state_dict()),payload['optimizer_state_dict']):
        raise RuntimeError('reloaded Adam states differ')
    restore_rng_state(payload['rng_state'])
    rng_before=cpu_tree(capture_rng_state())
    state_before=cpu_tree(model.state_dict());optimizer_before=cpu_tree(optimizer.state_dict())
    calls=0
    old_step=optimizer.step
    def forbidden_step(*args,**kwargs):
        nonlocal calls
        calls+=1
        raise RuntimeError('recovery verification must never call optimizer.step')
    optimizer.step=forbidden_step
    try:
        logits=evaluate_logits(model).detach().cpu()
        if not torch.equal(logits,payload['reference_eval_logits']):
            raise RuntimeError('fresh-process checkpoint prediction mismatch')
        observed=probe(model,optimizer)
        if observed!=payload['next_update_probe']:
            raise RuntimeError('restored-RNG next train-mode logits/loss/gradients mismatch')
        if (not equal_tree(state_before,cpu_tree(model.state_dict()))
                or not equal_tree(optimizer_before,cpu_tree(optimizer.state_dict()))
                or not equal_tree(rng_before,cpu_tree(capture_rng_state())) or calls):
            raise RuntimeError('no-step verification mutated model/Adam/RNG')
        return dict(status='PASS',optimizer_calls_during_recovery=0,eval_prediction_max_abs_difference=0.,
            next_train_forward_loss_all_gradient_digest='EXACT',model_adam_rng_projection_integrity='PASS',
            optimizer_updates=payload['optimizer_updates'])
    finally:optimizer.step=old_step
