"""Golden inputs and bounded-resource checks for the independent audit tools."""
import json
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest


def test_independent_session_oracle_exact_ten_hour_boundary_and_student_reset():
    from scripts.audit_hitskt_data import session_fields
    students=np.array([1]*6+[2]*5)
    times=np.array([0,35999,71999,107999,143999,179999,0,36000,72000,108000,144000])
    number,count,split,position=session_fields(students,times,36000)
    assert number.tolist()==[1,1,2,3,4,5,1,2,3,4,5]
    assert count.tolist()==[5]*11
    assert split.tolist()==[0,0,0,0,1,2,0,0,0,1,2]
    assert position.tolist()==[1,2,1,1,1,1,1,1,1,1,1]


def test_rkt_chunk_oracle_uses_exact_sparse_key_positions():
    from scripts.audit_rkt_cache_algebra import subset_for_chunk
    keys=np.array([10,30,50,90])
    counts=np.arange(16).reshape(4,4)
    positions,values=subset_for_chunk((keys,counts),np.array([20,30,40,50,60]))
    assert positions.tolist()==[1,3]
    assert np.array_equal(values,counts[1:3])


def tiny_batch(slots=1):
    tensor=SimpleNamespace(numel=lambda:slots)
    return SimpleNamespace(history_question=tensor,target_question=tensor)


def test_memory_admission_serializes_requests_that_cannot_fit_together(tmp_path,monkeypatch):
    from scripts import audit_memory_guard as guard
    monkeypatch.setattr(guard,'STATE',tmp_path/'budget.json')
    monkeypatch.setattr(guard,'LIMIT',int(1.5*1024**3))
    active=maximum=0
    mutex=threading.Lock()
    errors=[]
    def request():
        nonlocal active,maximum
        try:
            with guard.admit(tiny_batch(),SimpleNamespace(width=8,heads=2)):
                with mutex:active+=1;maximum=max(active,maximum)
                time.sleep(.04)
                with mutex:active-=1
        except Exception as error:errors.append(error)
    threads=[threading.Thread(target=request) for _ in range(3)]
    for thread in threads:thread.start()
    for thread in threads:thread.join(timeout=10)
    assert not errors and not any(t.is_alive() for t in threads)
    assert maximum==1
    state=json.loads(guard.STATE.read_text())
    assert state['holders']=={} and state['maximum_estimated_bytes']<=guard.LIMIT


def test_memory_admission_rejects_oversized_batch_without_allocation(tmp_path,monkeypatch):
    from scripts import audit_memory_guard as guard
    monkeypatch.setattr(guard,'STATE',tmp_path/'budget.json')
    with pytest.raises(MemoryError):
        with guard.admit(tiny_batch(10**12),SimpleNamespace(width=128,heads=2)):
            pytest.fail('Oversized batch admitted')
    assert not guard.STATE.exists()


def test_memory_admission_removes_dead_process_reservation(tmp_path,monkeypatch):
    from scripts import audit_memory_guard as guard
    monkeypatch.setattr(guard,'STATE',tmp_path/'budget.json')
    guard.STATE.write_text(json.dumps({'holders':{'999999999':{'start':'0','bytes':guard.LIMIT}},'maximum_estimated_bytes':guard.LIMIT}))
    with guard.admit(tiny_batch(),SimpleNamespace(width=8,heads=2)):
        state=json.loads(guard.STATE.read_text())
        assert '999999999' not in state['holders']
    assert json.loads(guard.STATE.read_text())['holders']=={}
