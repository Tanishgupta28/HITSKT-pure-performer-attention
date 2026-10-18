#!/usr/bin/env python3
"""Fresh read-only checkpoint/selection checks, including incomplete DKT state."""
import gc
import os
from pathlib import Path
import numpy as np
import torch
from scripts.diagnose_hitskt_common import DATASETS,OLD,OUT,read,write,sha,now


def tensor_stats(t):
    f=t.detach().cpu();finite=torch.isfinite(f)
    return dict(dtype=str(f.dtype),elements=f.numel(),nan=int(torch.isnan(f).sum()),
        positive_inf=int(torch.isposinf(f).sum()),negative_inf=int(torch.isneginf(f).sum()),
        maximum_finite_absolute=float(f[finite].abs().max()) if finite.any() else None)


def checkpoint(path,details=False):
    c=torch.load(path,map_location='cpu',weights_only=False)
    names=list(c['state_dict']);bad=[];total=0
    for pid,state in c.get('optimizer_state_dict',{}).get('state',{}).items():
        for key,value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                info=tensor_stats(value);total+=info['nan']+info['positive_inf']+info['negative_inf']
                bad.append(dict(parameter=names[pid] if pid<len(names) else str(pid),state=key,**info))
    result=dict(path=str(path),sha256=sha(path),epoch=c['epoch'],validation_auc=c['validation_roc_auc'],
        model_tensors_all_finite=all(bool(torch.isfinite(t).all()) for t in c['state_dict'].values()),
        optimizer_nonfinite_elements=total,optimizer_nonfinite_states=bad)
    if details:
        result.update(optimizer_param_groups=c['optimizer_state_dict']['param_groups'],
            early_stopping=c['early_stopping'],early_stopping_state=c['early_stopping_state'],
            all_optimizer_tensor_stats={names[pid]:{key:tensor_stats(value) for key,value in state.items() if torch.is_tensor(value)}
                for pid,state in c['optimizer_state_dict']['state'].items()})
    del c;gc.collect();return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
    old=read(OLD/'benchmark_artifacts.json');records=[]
    for model in ['dkt','dkvmn','sakt','rkt','hitskt']:
        for dataset in DATASETS:
            root=Path('experiments')/model/dataset/'full';config=read(root/'config.json')
            log=[__import__('json').loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
            epochs=[r for r in log if r['event']=='epoch_complete'];selected=max(epochs,key=lambda x:x['validation']['roc_auc'])
            best=checkpoint(root/'best_model.pt');last=checkpoint(root/'last_model.pt',model=='dkt' and dataset=='ednet_kt1')
            prev=next(r for r in old['records'] if r['model']==model and r['dataset']==dataset)
            hashes_match=best['sha256']==prev['best_checkpoint']['sha256']
            complete=(root/'_SUCCESS').exists() and (root/'final_results.json').exists()
            record=dict(model=model,dataset=dataset,complete=complete,best=best,last=last,
                strict_validation_selected_epoch=selected['epoch'],strict_validation_selected_auc=selected['validation']['roc_auc'],
                checkpoint_matches_strict_selection=best['epoch']==selected['epoch'] and best['validation_auc']==selected['validation']['roc_auc'],
                best_matches_previous_audit_sha256=hashes_match,config_selected_epoch=config['best_epoch'],completed_epochs=len(epochs),
                fresh_baseline_test_replay='NOT VERIFIED' if model!='hitskt' else 'See independent full CPU replay manifests',
                stored_test_metrics=read(root/'final_results.json')['test'] if complete else None)
            assert hashes_match and record['checkpoint_matches_strict_selection'] and config['best_epoch']==best['epoch']
            records.append(record);print(model,dataset,'complete',complete,'selected',best['epoch'],'last optimizer bad',last['optimizer_nonfinite_elements'],flush=True)
    archived=[]
    for dirname in ['.inspection/transport_switch_20260922','.inspection/dkt_ednet_abandoned_before_restart_20261008']:
        for p in sorted(Path(dirname).glob('*model.pt')):archived.append(checkpoint(p))
    # Synthetic optimizer-state algebra only; never step a historical model.
    value=torch.tensor([float('inf'),1.],dtype=torch.float32)
    for _ in range(100):value=value*.999+(.001*torch.tensor([1.,1.]))
    synthetic=dict(positive_infinity_after_100_finite_second_moment_updates=bool(torch.isposinf(value[0])),
        finite_first_moment_divided_by_infinite_denominator=float(torch.tensor(1.)/(torch.sqrt(value[0])+1e-8)),
        scope='Synthetic Adam second-moment algebra, not resuming or modifying any model/checkpoint')
    write(OUT/'artifact_and_dkt_forensics.json',dict(utc=now(),records=records,archived_snapshots=archived,
        synthetic_optimizer_algebra=synthetic,implementation_sha256=sha(__file__),
        dkt_origin='Exact first overflowing gradient/step NOT VERIFIED; snapshots do not form a complete canonical step history',
        historical_vs_independent='Stored baseline scores remain historical; artifact consistency does not equal test replay'))


if __name__=='__main__':main()
