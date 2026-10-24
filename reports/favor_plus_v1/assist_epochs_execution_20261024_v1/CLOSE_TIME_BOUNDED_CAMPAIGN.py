"""CPU-only closure after the foreground controller declines another full epoch.
Requires actual time-stop evidence and all completed epochs' fresh recovery.
No optimizer, CUDA import, guard change, or failed-study recovery.
"""
from pathlib import Path
import hashlib,json,math,os,sys
ROOT=Path('/workspace/capstone');RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1'
EVIDENCE=ROOT/'reports/favor_plus_v1/assist_epochs_execution_20261024_v1'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def measured_selection(rows):
    assert rows and [r['epoch'] for r in rows]==list(range(1,len(rows)+1)) and len(rows)<=10
    assert all(r['status']=='PASS_FULL_EPOCH' and r['validation']['targets']==125800 and r['train']['targets']==566107 for r in rows)
    assert all(math.isfinite(r['validation']['roc_auc']) and math.isfinite(r['validation']['f1']) for r in rows)
    best=max(rows,key=lambda r:r['validation']['roc_auc']);f1=max(rows,key=lambda r:r['validation']['f1'])
    assert rows[-1]['selection']['best_epoch']==best['epoch'] and rows[-1]['selection']['best_auc']==best['validation']['roc_auc']
    assert rows[-1]['selection']['best_f1_epoch']==f1['epoch'] and rows[-1]['selection']['best_f1']==f1['validation']['f1']
    return best,f1

def close():
    stop=json.loads((EVIDENCE/'CONTROLLER_TIME_STOP.json').read_text())
    assert stop['status']=='STOP_BEFORE_NEXT_EPOCH' and not (RUN/'study_failure.json').exists() and not (RUN/'study_result.json').exists()
    rows=[json.loads(p.read_text()) for p in sorted((RUN/'metrics').glob('epoch_*.json'))]
    best,f1=measured_selection(rows);assert stop['epochs_completed']==len(rows) and not rows[-1]['selection']['stop']
    plan=json.loads((RUN/'prepared_plan.json').read_text())
    for row in rows:
        epoch=row['epoch'];checkpoint=RUN/f'checkpoints/epoch_{epoch:03d}.pt';proof=json.loads((RUN/f'recovery_{epoch:03d}/recovery_result.json').read_text())
        assert proof['status']=='PASS' and proof['optimizer_calls']==0 and proof['checkpoint_sha256']==digest(checkpoint)==row['checkpoint']['sha256']
        assert row['checkpoint']['optimizer_updates']==sum(len(plan['schedules']['train'][str(i)]) for i in range(1,epoch+1))
    checkpoint=RUN/f'checkpoints/epoch_{best["epoch"]:03d}.pt'
    result=dict(status='TIME_BOUNDED_EPOCH_TRAINING_COMPLETE',stop_reason='controller declined another complete epoch under unchanged45min campaign cap',epochs_completed=len(rows),max_epochs_authorized=10,early_stopping_patience=3,early_stopping_triggered=False,best_epoch=best['epoch'],best_validation_auc=best['validation']['roc_auc'],best_validation_f1=f1['validation']['f1'],best_validation_f1_epoch=f1['epoch'],auc_selected_checkpoint_f1=best['validation']['f1'],best_checkpoint_path=str(checkpoint),best_checkpoint_sha256=digest(checkpoint),optimizer_updates=rows[-1]['checkpoint']['optimizer_updates'],test_evaluations=0,no_automatic_more_training=True,time_stop_receipt_sha256=digest(EVIDENCE/'CONTROLLER_TIME_STOP.json'))
    with (RUN/'study_result.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    if '--self-test' in sys.argv:
        rows=[json.loads(p.read_text()) for p in sorted((RUN/'metrics').glob('epoch_*.json'))[:4]];best,f1=measured_selection(rows)
        assert best['epoch']==4 and f1['epoch']==2
        try:measured_selection([rows[1],rows[0]])
        except AssertionError:pass
        else:raise AssertionError('out-of-order/missing epoch accepted')
        print(json.dumps(dict(status='PASS_CPU_ONLY',measured_epochs=len(rows),best_epoch=best['epoch'],best_f1_epoch=f1['epoch'],writes=0,cuda_imports=0)))
    else:close()
