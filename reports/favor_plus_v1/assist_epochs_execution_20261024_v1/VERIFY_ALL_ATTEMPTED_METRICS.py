"""Independent CPU metric verification of saved full-validation predictions only."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,mean_squared_error
ROOT=Path('/workspace/capstone');RUN=ROOT/'experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1'
rows=[];reference_labels=None
for path in sorted(RUN.glob('epoch_*/full_validation_metrics.json')):
    epoch=int(path.parent.name.split('_')[1]);item={'validation':json.loads(path.read_text())['metrics']};prediction=path.parent/'full_validation_predictions.npz'
    with np.load(prediction) as z:
        y=z['label'].astype(np.float64);p=z['probability'].astype(np.float64);logits=z['logits'].astype(np.float64)
    assert y.shape==p.shape==logits.shape==(125800,)
    assert np.isfinite(p).all() and np.isfinite(logits).all() and set(np.unique(y))=={0.,1.}
    if reference_labels is None:reference_labels=y.copy()
    assert np.array_equal(y,reference_labels),'frozen validation labels/order changed'
    classes=p>=.5
    metrics=dict(roc_auc=float(roc_auc_score(y,p)),accuracy=float(accuracy_score(y,classes)),precision=float(precision_score(y,classes,zero_division=0)),recall=float(recall_score(y,classes,zero_division=0)),f1=float(f1_score(y,classes,zero_division=0)),mse=float(mean_squared_error(y,p)),bce=float(np.mean(np.logaddexp(0,logits)-y*logits)))
    delta={k:abs(v-float(item['validation'][k])) for k,v in metrics.items()}
    rows.append(dict(epoch=epoch,status='PASS' if max(delta.values())<=1e-6 else 'FAIL',metrics=metrics,abs_differences=delta,targets=len(y),positive_labels=int(y.sum()),prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(),confusion=dict(tp=int(((y==1)&classes).sum()),fp=int(((y==0)&classes).sum()),tn=int(((y==0)&~classes).sum()),fn=int(((y==1)&~classes).sum()))))
result=dict(status='PASS' if rows and all(x['status']=='PASS' for x in rows) else 'FAIL',reference='independent sklearn classification/rank metrics; NumPy float64 logaddexp BCE from saved logits',unchanged_tolerance=1e-6,epochs=rows,test_split_accessed=False)
(ROOT/'reports/favor_plus_v1/assist_epochs_execution_20261024_v1/INDEPENDENT_ALL_ATTEMPTED_METRIC_VERIFICATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status=result['status'],completed_epochs_verified=len(rows),max_abs_difference=max((max(x['abs_differences'].values()) for x in rows),default=None))))
assert result['status']=='PASS'
