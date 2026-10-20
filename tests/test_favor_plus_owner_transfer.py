"""CPU-only checks of the narrowly owner-authorized historical-MIG transfer."""
import copy
import datetime as dt
from pathlib import Path

import pytest

from ktbench import favor_plus_gpu_access as access

NOW = dt.datetime(2026, 10, 20, 10, tzinfo=dt.timezone.utc)


def record():
    # Unit fixture only, never allocation evidence.
    return dict(allocation_kind=access.OWNER_TRANSFER, resource_uuid=access.TRANSFERRED_MIG,
        exclusive=True, owner='UNIT FIXTURE', job_id='UNIT FIXTURE',
        owner_confirmation_evidence='UNIT FIXTURE ONLY', owner_confirms_existing_allocation=True,
        resource_transfer=dict(from_project='LightSAKT2',to_model='hitskt_favor_plus_v1',authorized=True),
        starts_utc='2026-10-20T09:30:00Z',expires_utc='2026-10-20T11:30:00Z',
        user_approval=dict(approved=True,evidence='UNIT FIXTURE ONLY',
                           actions=[access.CUDA_VALIDATION,access.REAL_PILOT]))


def test_owner_confirmation_accepts_exact_mig_without_fabricated_external_record():
    r=record()
    assert 'external_allocation_evidence' not in r and 'independent' not in r
    for action in (access.CUDA_VALIDATION,access.REAL_PILOT):
        assert access.require_gpu_authorization(r,access.TRANSFERRED_MIG,action,now=NOW)==access.TRANSFERRED_MIG


@pytest.mark.parametrize('key,value',[
    ('owner_confirms_existing_allocation',False),('owner_confirmation_evidence',''),
    ('exclusive',False),('owner',''),('job_id',''),('resource_transfer',{}),
    ('resource_transfer',dict(from_project='other',to_model='hitskt_favor_plus_v1',authorized=True)),
    ('resource_transfer',dict(from_project='LightSAKT2',to_model='hitskt_favor_plus_v1',authorized=False)),
    ('resource_uuid','GPU-5a568e3c-e96c-169f-b063-f4478340ff51'),
    ('resource_uuid','MIG-11111111-2222-3333-4444-555555555555'),
    ('known_conflicting_reservation',True),('known_explicit_restriction',True),
    ('expires_utc','2026-10-20T11:30:01Z'),('expires_utc','2026-10-20T09:31:00Z'),
    ('user_approval',dict(approved=True,evidence='fixture',actions=['full_training'])),
])
def test_owner_record_cannot_expand_resource_authorization_or_ignore_restrictions(key,value):
    r=record();r[key]=value
    with pytest.raises(ValueError):
        access.require_gpu_authorization(r,r['resource_uuid'],access.REAL_PILOT,now=NOW)


def test_conflict_snapshot_ignores_self_but_refuses_another_gpu_owner(tmp_path):
    for pid,node in [(10,'/dev/nvidia0'),(20,'/dev/nvidia-caps/nvidia-cap22'),(30,'/dev/nvidia1')]:
        fds=tmp_path/str(pid)/'fd';fds.mkdir(parents=True);(fds/'3').symlink_to(node)
    assert access.conflicting_device_processes(tmp_path,own_pid=10)==[20]


@pytest.mark.parametrize('executable,args,blocked',[
    ('/usr/bin/nvidia-smi',[b'--query-gpu=memory.used',b'--format=csv,noheader'],False),
    ('/usr/bin/nvidia-smi',[],False),
    ('/usr/bin/nvidia-smi',[b'--gpu-reset'],True),
    ('/tmp/nvidia-smi',[b'--query-gpu=memory.used'],True),
    ('/usr/bin/python',[b'train.py'],True),
])
def test_only_trusted_read_only_management_queries_are_not_compute_conflicts(tmp_path,executable,args,blocked):
    proc=tmp_path/'20';(proc/'fd').mkdir(parents=True)
    (proc/'fd'/'3').symlink_to('/dev/nvidia0');(proc/'exe').symlink_to(executable)
    (proc/'cmdline').write_bytes(b'\0'.join([executable.encode(),*args,b'']))
    assert access.conflicting_device_processes(tmp_path,own_pid=10)==([20] if blocked else [])


@pytest.mark.parametrize('missing',['hold','exposure','conflict'])
def test_execution_gate_checks_actual_release_and_exposure_before_cuda(tmp_path,monkeypatch,missing):
    r=record();r['starts_utc']='2026-01-01T00:00:00Z';r['expires_utc']='2026-01-01T02:00:00Z'
    hold=tmp_path/'hold.json';hold.write_text('{}');devices=tmp_path/'devices';devices.mkdir()
    (devices/access.TRANSFERRED_MIG).touch()
    monkeypatch.setattr(access,'LIGHT_HOLD',hold);monkeypatch.setattr(access,'DEVICE_EXPOSURE',devices)
    monkeypatch.setattr(access,'require_gpu_authorization',lambda *a,**k:access.TRANSFERRED_MIG)
    monkeypatch.setattr(access,'conflicting_device_processes',lambda:[20] if missing=='conflict' else [])
    if missing=='hold':hold.unlink()
    elif missing=='exposure':(devices/access.TRANSFERRED_MIG).unlink()
    with pytest.raises(RuntimeError):
        access.require_execution_access(r,access.TRANSFERRED_MIG,access.REAL_PILOT)
