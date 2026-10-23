"""Explicit shared-resource policy; identity observations never become aliases.

Stdlib/NVML only. This module never imports torch, signals peers, initializes
CUDA, changes allocation/MPS/clock settings or grants permission from free VRAM.
"""
from __future__ import annotations
import ctypes
import datetime as dt
import json
import re

from ktbench.favor_plus_gpu_access import DEVICE_EXPOSURE, LIGHT_HOLD, TRANSFERRED_MIG
from ktbench.favor_plus_resource_telemetry import ResourceConflict, device_snapshot, nvml_snapshot

MODE = 'authorized_shared_gpu_v1'
ACTION = 'favor_plus_assist2017_shared_resume20_to50'
GIB = 1024**3
POLICY = dict(mode=MODE, allocator_cap_bytes=GIB, reserved_cap_bytes=GIB,
    advisory_total_vram_budget_bytes=2*GIB, driver_overhead_allowance_bytes=GIB,
    admission_free_bytes=8*GIB, runtime_free_bytes=6*GIB,
    host_process_rss_bytes=4*GIB, host_headroom_bytes=2*GIB,
    max_admission_utilization_pct=75, allow_unsupported_mig_utilization=True,
    admission_samples=2, admission_sample_spacing_seconds=.2,
    inter_update_pause_seconds=.1, max_step_seconds=30, max_execution_seconds=2700,
    max_telemetry_age_seconds=5,
    max_total_optimizer_updates=50, max_additional_optimizer_calls=30,
    batch_size=1, max_padded_tokens=2048)


class SharedResourceStop(ResourceConflict):
    """Unsafe resources or missing telemetry, not an ownership accusation."""


def validate_policy(policy):
    # Versioned, prospective policy: no ad-hoc relaxation from CLI or a record.
    if not isinstance(policy, dict) or json.dumps(policy, sort_keys=True, allow_nan=False) != json.dumps(POLICY, sort_keys=True):
        raise ValueError('exact predeclared shared_gpu_v1 resource policy required')


def authorize(record, visible, *, plan_sha256, policy_sha256, now=None):
    """Metadata only. A record documents actual permission, never creates it."""
    if any(not isinstance(x, str) or not re.fullmatch('[0-9a-f]{64}', x) for x in (plan_sha256, policy_sha256)):
        raise ValueError('actual plan/policy SHA256 values required')
    if not isinstance(record, dict) or record.get('mode') != MODE or record.get('exclusive') is not False:
        raise ValueError('explicit nonexclusive shared-GPU mode required; no implicit fallback')
    for flag in ('shared_permission_confirmed', 'execution_approved', 'budget_and_window_acknowledged',
                 'limited_pid_visibility_acknowledged'):
        if record.get(flag) is not True:
            raise ValueError('shared GPU record requires '+flag)
    for key in ('permission_evidence', 'execution_evidence', 'coordination_evidence', 'operator'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError('missing actual shared-use evidence: '+key)
    if (record.get('resource_uuid') != TRANSFERRED_MIG or visible != TRANSFERRED_MIG
            or record.get('action') != ACTION or record.get('prepared_plan_sha256') != plan_sha256
            or record.get('policy_sha256') != policy_sha256
            or record.get('max_total_optimizer_updates') != 50
            or record.get('max_additional_optimizer_calls') != 30):
        raise ValueError('exact assigned UUID/shared plan/policy/20→50 scope required')
    if record.get('known_explicit_restriction') is not False or record.get('known_incompatible_reservation') is not False:
        raise ValueError('shared-use restrictions/reservations must be checked')
    clock = now or dt.datetime.now(dt.timezone.utc)
    try:
        start = dt.datetime.fromisoformat(record['starts_utc'])
        end = dt.datetime.fromisoformat(record['expires_utc'])
        if (start.tzinfo is None or end.tzinfo is None or not start <= clock < end
                or not 0 < (end-start).total_seconds() <= 7200):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise ValueError('fresh shared operator window, at most2h, required') from None
    return TRANSFERRED_MIG


def utilization_snapshot(uuid):
    """One MIG query. Unsupported is explicit, never interpreted as zero load."""
    lib = ctypes.CDLL('libnvidia-ml.so.1')
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError('NVML initialization unavailable')
    try:
        handle = ctypes.c_void_p()
        code = lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle))
        if code:
            raise RuntimeError('NVML device handle unavailable: '+str(code))
        class U(ctypes.Structure):
            _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]
        value = U()
        code = lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(value))
        return dict(status='ok' if code == 0 else 'unsupported' if code == 3 else 'error',
                    nvml_status=code, gpu_percent=value.gpu if code == 0 else None,
                    memory_percent=value.memory if code == 0 else None)
    finally:
        lib.nvmlShutdown()


def observe(uuid):
    return dict(observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), resource_uuid=uuid,
                device=nvml_snapshot(uuid), utilization=utilization_snapshot(uuid),
                local=device_snapshot(), pid_attribution='UNRESOLVED; self/peer not inferred')


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def assess(observation, host, *, admission=False, own_cuda=None, policy=POLICY, now=None):
    """Pure CPU policy. Peers are tracked; unsafe pressure/visibility stops us."""
    validate_policy(policy)
    reasons, warnings = [], ['PID namespace identity remains unresolved']
    try:
        stamp = dt.datetime.fromisoformat(observation['observed_utc'])
        age = ((now or dt.datetime.now(dt.timezone.utc))-stamp).total_seconds()
        if stamp.tzinfo is None or not 0 <= age <= policy['max_telemetry_age_seconds']:
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        reasons.append('missing/stale observation timestamp')
    device = observation.get('device', {})
    if not isinstance(device, dict):
        reasons.append('invalid global memory visibility')
        device = {}
    if observation.get('resource_uuid') != TRANSFERRED_MIG:
        reasons.append('wrong resource UUID')
    if not all(_integer(device.get(k)) for k in ('total_bytes', 'free_bytes', 'used_bytes')):
        reasons.append('missing/invalid global memory visibility')
    elif not device['total_bytes'] or device['free_bytes']+device['used_bytes'] > device['total_bytes']:
        reasons.append('inconsistent device memory telemetry')
    elif device['free_bytes'] < policy['admission_free_bytes' if admission else 'runtime_free_bytes']:
        reasons.append('device memory headroom below shared floor')
    processes = device.get('compute_processes')
    if not isinstance(processes, list) or any(not isinstance(p, dict) or not _integer(p.get('pid'), minimum=1) for p in processes):
        reasons.append('missing/invalid NVML compute visibility')
        processes = []
    # Neither equal nor unequal PID integers establish self/peer across domains.
    local = observation.get('local')
    if not isinstance(local, dict) or not all(isinstance(local.get(k), list) for k in ('holders', 'inaccessible_fd_pids', 'light_processes')):
        reasons.append('missing local process telemetry')
        local = {}
    if local.get('light_processes'):
        reasons.append('protected LightSAKT2 activity: HiTSKT yields without signaling it')
    if local.get('inaccessible_fd_pids'):
        warnings.append('local descriptor visibility incomplete; not proof of peer ownership')
    if local.get('holders'):
        warnings.append('visible GPU holders recorded as shared/unattributed activity')
    utilization = observation.get('utilization', {})
    if utilization.get('status') == 'ok':
        values = [utilization.get('gpu_percent'), utilization.get('memory_percent')]
        if any(not _integer(v) or v > 100 for v in values):
            reasons.append('invalid utilization telemetry')
        elif admission and values[0] > policy['max_admission_utilization_pct']:
            reasons.append('GPU busy above shared admission threshold; defer once, no polling')
    elif (utilization.get('status') == 'unsupported' and utilization.get('nvml_status') == 3
          and policy['allow_unsupported_mig_utilization']):
        warnings.append('MIG utilization unsupported: cap/headroom/pacing/runtime fallback; load UNKNOWN')
    else:
        reasons.append('utilization query error (not an unsupported-MIG result)')
    if not all(_integer(host.get(k)) for k in ('peak_process_rss_bytes', 'cgroup_headroom_bytes')):
        reasons.append('missing host memory visibility')
    elif (host['peak_process_rss_bytes'] > policy['host_process_rss_bytes']
          or host['cgroup_headroom_bytes'] < policy['host_headroom_bytes']):
        reasons.append('host RSS/headroom limit')
    if own_cuda is not None:
        if not all(_integer(own_cuda.get(k)) for k in ('allocated_bytes', 'reserved_bytes')):
            reasons.append('missing own allocator telemetry')
        elif (own_cuda['allocated_bytes'] > policy['allocator_cap_bytes']
              or own_cuda['reserved_bytes'] > policy['reserved_cap_bytes']):
            reasons.append('own PyTorch allocated/reserved cap exceeded')
    return dict(status='STOP' if reasons else 'ADMIT_SHARED_WITH_LIMITED_ATTRIBUTION',
                reasons=reasons, warnings=warnings, reported_compute_processes=processes,
                local_holders=local.get('holders', []), local_visibility_gaps=local.get('inaccessible_fd_pids', []),
                pid_mapping_proven=False, exclusive_ownership_claimed=False)


def check_observation(observation, host, **kwargs):
    result = assess(observation, host, **kwargs)
    if result['status'] == 'STOP':
        raise SharedResourceStop('; '.join(result['reasons']), dict(observation=observation, decision=result, host=host))
    return result


def check_exposure_and_hold(uuid):
    if not (DEVICE_EXPOSURE/uuid).exists() or not LIGHT_HOLD.is_file():
        raise RuntimeError('exact shared device exposure or protected LightSAKT2 hold missing')
    if json.loads(LIGHT_HOLD.read_text()).get('status') != 'USER_TRAINING_HOLD_ACTIVE':
        raise RuntimeError('LightSAKT2 hold not active; no changes permitted')
