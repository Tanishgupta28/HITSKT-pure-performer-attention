"""Foreground, bounded orchestration of the frozen source-bound epoch CLI.
No optimizer implementation, resource change, retry, polling watcher or test action.
"""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path('/workspace/capstone')
EVIDENCE=ROOT/'reports/favor_plus_v1/assist_epochs_execution_20261024_v1'
RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1'
PLAN=RUN/'prepared_plan.json';RECORD=EVIDENCE/'APPROVED_EXECUTION_RECORD.json'
start=json.loads(RECORD.read_text())['campaign_started_unix'];events=[]
assert json.loads((RUN/'recovery_001/recovery_result.json').read_text())['status']=='PASS'
for epoch in range(2,11):
    previous=json.loads((RUN/f'metrics/epoch_{epoch-1:03d}.json').read_text())
    if previous['selection']['stop']:break
    # Reserve measured previous phase duration plus60s for final evaluation.
    previous_seconds=events[-1]['seconds'] if events else sum(x['seconds'] for x in previous['updates'])*2+60
    if 2700-(time.time()-start)<previous_seconds+60:
        (EVIDENCE/'CONTROLLER_TIME_STOP.json').write_text(json.dumps(dict(status='STOP_BEFORE_NEXT_EPOCH',epochs_completed=epoch-1,reason='unchanged45min campaign headroom; no partial extra epoch started',elapsed_seconds=time.time()-start,required_seconds=previous_seconds+60),indent=2)+'\n')
        break
    begin=time.time()
    for action,prefix in (('train-epoch','epoch'),('verify-epoch','recovery')):
        argv=[sys.executable,'scripts/train_assist_favor_plus_epochs.py',action,'--plan',str(PLAN),'--record',str(RECORD),'--epoch',str(epoch),'--execute-cuda']
        with (EVIDENCE/f'{prefix}_{epoch:03d}.console.log').open('xb') as stream:
            code=subprocess.run(argv,cwd=ROOT,env=os.environ,stdout=stream,stderr=subprocess.STDOUT).returncode
        if code:
            (EVIDENCE/'CONTROLLER_STOP.json').write_text(json.dumps(dict(status='STOP_NO_RETRY',epoch=epoch,action=action,returncode=code,elapsed_seconds=time.time()-start),indent=2)+'\n')
            sys.exit(code)
    row=json.loads((RUN/f'metrics/epoch_{epoch:03d}.json').read_text())
    event=dict(epoch=epoch,validation=row['validation'],train_bce=row['train']['bce'],selection=row['selection'],seconds=time.time()-begin)
    events.append(event);print(json.dumps(event),flush=True)
    (EVIDENCE/'CONTROLLER_PROGRESS.json').write_text(json.dumps(events,indent=2)+'\n')
    if row['selection']['stop']:break
if (RUN/'study_result.json').exists():
    argv=[sys.executable,'scripts/train_assist_favor_plus_epochs.py','validate-best','--plan',str(PLAN),'--record',str(RECORD),'--execute-cuda']
    with (EVIDENCE/'best_validation_replay.console.log').open('xb') as stream:
        code=subprocess.run(argv,cwd=ROOT,env=os.environ,stdout=stream,stderr=subprocess.STDOUT).returncode
    if code:sys.exit(code)
print(json.dumps(dict(status='BOUNDED_SEQUENCE_ENDED',epochs_completed=1+len(events),elapsed_seconds=time.time()-start,test_evaluations=0)),flush=True)
