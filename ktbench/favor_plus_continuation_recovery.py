"""Separate20→50 checkpoint contract; the original20 files remain immutable."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import tempfile
import torch
from ktbench.favor_plus_pilot_diagnostics import cpu_tree,equal_tree,finite_tree
from ktbench.favor_plus_pilot_recovery import digest,seal

KIND='assist2017_favor_plus_continuation20_to50_v1'
CHECKPOINT_UPDATES=(25,30,35,40,45,50)
COMPONENTS={'state_dict','optimizer_state_dict','rng_state','sampler','contract',
            'reference_eval_logits','next_update_probe'}


def validate(payload,contract):
    if payload.get('kind')!=KIND or payload.get('contract')!=contract:
        raise ValueError('continuation checkpoint contract mismatch')
    config=contract['config'];step=payload.get('optimizer_updates')
    if (type(step) is not int or step not in CHECKPOINT_UPDATES
            or payload.get('optimizer_step_calls')!=step
            or payload.get('additional_optimizer_calls')!=step-20):
        raise ValueError('continuation counters must be25/30/35/40/45/50 TOTAL,30 additional maximum')
    if (payload.get('variant')!='hitskt_favor_plus_v1' or payload.get('format_version')!=2
            or payload.get('config')!=config['model_config']
            or payload.get('favor_config')!=config['favor_config']):
        raise ValueError('unchanged attention/architecture configuration required')
    sampler=payload.get('sampler',{})
    if (sampler.get('cursor')!=step or sampler.get('epoch')!=1 or sampler.get('seed')!=42
            or sampler.get('exact_samples')!=contract['exact_training_and_probe_samples']):
        raise ValueError('continuation sampler cursor/identities mismatch')
    if set(payload.get('component_sha256',{}))!=COMPONENTS:
        raise ValueError('missing checkpoint component hashes')
    if any(digest(payload[k])!=v for k,v in payload['component_sha256'].items()):
        raise ValueError('checkpoint component digest mismatch')
    projections={k:digest(v) for k,v in payload['state_dict'].items() if k.endswith('.projection')}
    if (not projections or projections!=payload.get('projection_sha256')
            or projections!=contract['parent_projection_sha256']):
        raise ValueError('fixed parent FAVOR+ projection buffers changed')
    if not finite_tree(payload):raise FloatingPointError('nonfinite checkpoint/model/Adam/RNG')
    opt=payload['optimizer_state_dict'];count=contract['parameter_tensor_count']
    ids=[i for g in opt['param_groups'] for i in g['params']]
    if (opt['param_groups']!=contract['parent_optimizer_param_groups']
            or len(opt['state'])!=count or len(ids)!=count or len(set(ids))!=count
            or set(ids)!=set(opt['state'])):
        raise ValueError('original Adam hyperparameters/parameter coverage changed')
    for state in opt['state'].values():
        if (float(state.get('step',-1))!=step or 'exp_avg' not in state
                or 'exp_avg_sq' not in state or not bool((state['exp_avg_sq']>=0).all())):
            raise ValueError('Adam counters/moments inconsistent with total updates')
    return dict(status='PASS',updates=step,additional_optimizer_calls=step-20,
        component_sha256=payload['component_sha256'],projection_sha256=projections,
        sampler_cursor=step,adam_parameter_states=count,
        parent_checkpoint_sha256=contract['parent_checkpoint_sha256'])


def atomic_checkpoint(path,payload,contract):
    """Same fsync/roundtrip/no-overwrite protocol, with the NEW validator.

    Deliberately separate instead of widening the source-bound original20 gate.
    Partial writes are retained but never eligible for resumption.
    """
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    validate(payload,contract)
    fd,temp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.stem+'.',suffix='.partial.pt')
    with os.fdopen(fd,'wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    loaded=torch.load(temp,map_location='cpu',weights_only=True)
    result=validate(loaded,contract)
    if not equal_tree(payload,loaded):raise RuntimeError('checkpoint serialization mismatch')
    os.link(temp,path);os.unlink(temp)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    result.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),atomic_no_overwrite=True)
    return result


def no_step_recovery(payload,model,optimizer,contract,*,evaluate_logits,probe):
    from ktbench.training import capture_rng_state,restore_rng_state
    validate(payload,contract)
    model.load_state_dict(payload['state_dict'],strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    if not equal_tree(cpu_tree(optimizer.state_dict()),payload['optimizer_state_dict']):
        raise RuntimeError('continued Adam reload mismatch')
    # Initialization and loading precede RNG restoration; no reseeding after this.
    restore_rng_state(payload['rng_state'])
    rng=cpu_tree(capture_rng_state());weights=cpu_tree(model.state_dict())
    adam=cpu_tree(optimizer.state_dict());calls=0;old=optimizer.step
    def forbidden(*args,**kwargs):
        nonlocal calls
        calls+=1
        raise RuntimeError('recovery must not take an optimizer step')
    optimizer.step=forbidden
    try:
        logits=evaluate_logits(model).detach().cpu()
        if not torch.equal(logits,payload['reference_eval_logits']):raise RuntimeError('checkpoint predictions differ')
        if probe(model,optimizer)!=payload['next_update_probe']:raise RuntimeError('checkpoint next forward/loss/gradients differ')
        if (calls or not equal_tree(weights,cpu_tree(model.state_dict()))
                or not equal_tree(adam,cpu_tree(optimizer.state_dict()))
                or not equal_tree(rng,cpu_tree(capture_rng_state()))):
            raise RuntimeError('recovery modified weights/Adam/RNG')
        return dict(status='PASS',optimizer_calls_during_recovery=0,
            eval_prediction_max_abs_difference=0.,next_train_forward_loss_all_gradient_digest='EXACT',
            model_adam_rng_projection_integrity='PASS',optimizer_updates=payload['optimizer_updates'])
    finally:optimizer.step=old
