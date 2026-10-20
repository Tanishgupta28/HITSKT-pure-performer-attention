"""CPU-only contract preparation; contains no CUDA or training entry point.

The original recoverable20 validator/controller remain untouched. A later
versioned controller must bind this explicit parent handoff and obtain new
user approval; this file does not extend the old20 execution permission.
"""
from __future__ import annotations
import copy

PARENT_SHA256 = '0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4'
ACTION = 'favor_plus_assist2017_resume20_to50_thirty_additional_updates'


def stage_interval(previous, to):
    if type(previous) is not int or type(to) is not int or {30:20,40:30,50:40}.get(to) != previous:
        raise ValueError('continuation requires20→30→40→50, never restart/reset or exceed50')
    return range(previous+1, to+1)


def proposed_handoff(parent, parent_sha256, frozen_samples):
    """Caller first validates the sealed CPU payload with its original contract."""
    if (parent_sha256 != PARENT_SHA256 or parent.get('optimizer_updates') != 20
            or parent.get('optimizer_step_calls') != 20 or parent['sampler'].get('cursor') != 20):
        raise ValueError('exact validated20-update parent required')
    contract = parent['contract']
    ordered = frozen_samples['training']
    if (len(ordered) < 51 or ordered[:21] != contract['exact_training_and_probe_samples']
            or parent['sampler']['exact_samples'] != ordered[:21]
            or frozen_samples['validation'] != contract['frozen_validation_samples']
            or frozen_samples['diagnostics'] != contract['diagnostic_samples']):
        raise ValueError('original sampler/diagnostic/validation identity mismatch')
    if len({item['target_session'] for item in ordered[:51]}) != 51:
        raise ValueError('duplicate continuation target identity')
    return copy.deepcopy(dict(
        kind='assist2017_favor_plus_continuation20_to50_proposal_v1',
        parent_checkpoint_sha256=parent_sha256,
        parent_component_sha256=parent['component_sha256'],
        parent_projection_sha256=parent['projection_sha256'],
        original_contract=contract,
        scientific_config=contract['config'],
        exact_training_and_probe_samples=ordered[:51],
        start_sampler_cursor=20, next_training_sample=ordered[20],
        next_probe_only_after50=ordered[50], total_completed_updates=20,
        requested_additional_optimizer_calls=30, max_total_optimizer_updates=50,
        stage_endpoints=[30,40,50], checkpoint_updates=[25,30,35,40,45,50],
        diagnostic_and_validation_updates=[20,30,40,50],
        execution_limits=dict(max_execution_seconds=2700,max_step_seconds=120,
            max_cuda_allocated_bytes=8*1024**3,max_process_rss_bytes=4*1024**3,
            minimum_cgroup_headroom_bytes=2*1024**3,batch_size=1,max_padded_tokens=2048),
        execution_action=ACTION, explicit_continuation_approval_received=False,
        cuda_execution='NOT RUN',additional_optimizer_calls=0,
        controller_status='VERSIONED_EXECUTION_CONTROLLER_REQUIRED; ORIGINAL20_CONTROLLER_REFUSES50',
        restore_policy='strict model/Adam/projections load, then restore saved RNG; cursor20; prove next-step digest before update21; no reseeding after restoration',
        failure_policy='stop without retry/reset on unknown resource owner, nonfinite state, causal/reload failure or unchanged severe/resource/time gate; preserve last atomic checkpoint',
        thresholds='original .20 REVIEW and severe stops unchanged; continued REVIEW is not softmax quality PASS',
        test_evaluations=0,full_training_authorized=False))


def require_explicit_approval(record, plan_sha256):
    """Scope checker for a future controller, not allocation or execution approval."""
    if (record.get('approved') is not True or record.get('action') != ACTION
            or record.get('prepared_plan_sha256') != plan_sha256
            or record.get('parent_checkpoint_sha256') != PARENT_SHA256
            or record.get('additional_optimizer_calls') != 30
            or record.get('max_total_optimizer_updates') != 50
            or not record.get('actual_user_approval_evidence')):
        raise ValueError('new explicit user approval bound to exact20→50 plan is required')
