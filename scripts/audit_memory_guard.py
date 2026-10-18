"""Shared RAM admission for audit-owned CPU inference only."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

LIMIT=24*1024**3
STATE=Path('.inspection/independent_correctness_audit/inference_memory_budget.json')


def identity(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:return None


@contextmanager
def admit(batch, config):
    # The unchanged operator creates an outer product and its cumulative copy.
    # Include another full copy, model/runtime and feed-forward headroom. Long
    # singleton sessions can exceed the normal token budget, so use real shapes.
    head_width=config.width//config.heads
    slots=max(batch.history_question.numel(),batch.target_question.numel())
    estimate=3*slots*config.heads*head_width**2*4+1024**3
    if estimate>LIMIT:raise MemoryError(f'Historical batch estimate {estimate} exceeds audit budget {LIMIT}')
    pid=str(os.getpid());start=identity(pid)
    acquired=False
    STATE.parent.mkdir(parents=True,exist_ok=True)
    while not acquired:
        with STATE.open('a+') as file:
            fcntl.flock(file,fcntl.LOCK_EX)
            file.seek(0);raw=file.read()
            state=json.loads(raw) if raw else {'holders':{},'maximum_estimated_bytes':0}
            holders={p:v for p,v in state['holders'].items() if identity(p)==v['start']}
            used=sum(v['bytes'] for v in holders.values())
            if used+estimate<=LIMIT:
                assert pid not in holders
                holders[pid]={'start':start,'bytes':estimate}
                state['maximum_estimated_bytes']=max(state['maximum_estimated_bytes'],used+estimate)
                acquired=True
            state.update(holders=holders,limit_bytes=LIMIT)
            file.seek(0);file.truncate();json.dump(state,file);file.flush()
        if not acquired:time.sleep(.25)
    try:yield
    finally:
        with STATE.open('r+') as file:
            fcntl.flock(file,fcntl.LOCK_EX)
            state=json.load(file)
            state['holders'].pop(pid,None)
            file.seek(0);file.truncate();json.dump(state,file);file.flush()
