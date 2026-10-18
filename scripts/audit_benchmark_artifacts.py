#!/usr/bin/env python3
"""Inspect all canonical benchmark artifacts without replaying baseline tests."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import json
import os
import numpy as np
import torch
from scripts.audit_hitskt_replay import sha256,write_json


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    records=[]
    original_files=[]
    for model in ['dkt','dkvmn','sakt','rkt','hitskt']:
        for dataset in ['assist2017','junyi','ednet_kt1']:
            run=Path('experiments')/model/dataset/'full'
            config=json.loads((run/'config.json').read_text())
            row=dict(model=model,dataset=dataset,path=str(run),complete=(run/'_SUCCESS').exists(),
                seed=config.get('seed'),config=config,checks={},issues=[])
            for file in sorted(run.iterdir()):
                if file.is_file(): original_files.append(dict(path=str(file),bytes=file.stat().st_size,sha256=sha256(file)))
            events=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines() if line]
            epochs=[e for e in events if e.get('event')=='epoch_complete']
            row['checks']['contiguous_completed_epoch_prefix']=[e['epoch'] for e in epochs]==list(range(1,config['epochs_completed']+1))
            best=-float('inf'); best_epoch=None; counter=0; selection=True
            for event in epochs:
                auc=event['validation']['roc_auc']; improved=auc>best
                if improved: best=auc; best_epoch=event['epoch']; counter=0
                else: counter+=1
                selection &= event['checkpoint_saved']==improved
                selection &= event['consecutive_epochs_without_improvement']==counter
                event_limit=event.get('early_stopping_patience',config.get('early_stopping_patience_previous',config['early_stopping_patience']))
                selection &= event['early_stop']==(counter>=event_limit)
            row['checks']['strict_validation_auc_selection']=bool(selection and best_epoch==config['best_epoch'] and best==config['best_validation_auc'])
            row['non_improvement_count']=counter
            checkpoint=torch.load(run/'best_model.pt',map_location='cpu',weights_only=True)
            row['best_checkpoint']=dict(sha256=sha256(run/'best_model.pt'),epoch=checkpoint.get('epoch'),
                validation_auc=checkpoint.get('validation_roc_auc'),seed_in_payload=checkpoint.get('seed'),
                all_model_tensors_finite=all(torch.isfinite(v).all().item() for v in checkpoint['state_dict'].values()),
                config=checkpoint['config'])
            row['checks']['best_checkpoint_selection_matches']=(checkpoint.get('epoch')==best_epoch and checkpoint.get('validation_roc_auc')==best)
            nonfinite=[]
            for index,state in checkpoint.get('optimizer_state_dict',{}).get('state',{}).items():
                for name,tensor in state.items():
                    if torch.is_tensor(tensor) and not torch.isfinite(tensor).all().item():
                        nonfinite.append(dict(parameter_index=index,state_key=name,nonfinite_count=int((~torch.isfinite(tensor)).sum())))
            row['best_optimizer_nonfinite_states']=nonfinite
            if nonfinite: row['issues'].append('Nonfinite best-checkpoint optimizer tensors; model-weight replay may still be valid')
            if row['complete']:
                result=json.loads((run/'final_results.json').read_text())
                row['stored_test_metrics']=result['test']
                row['checks']['completion_metadata']=result['status']=='complete' and result['best_checkpoint_reloaded']
                row['checks']['result_selection_matches']=result['best_epoch']==best_epoch and result['best_validation_auc']==best
                rows=list(csv.DictReader((run/'metrics.csv').open()))
                test=[r for r in rows if r['split']=='test']
                row['checks']['csv_test_matches_json']=len(test)==1 and all(float(test[0][key])==value for key,value in result['test'].items())
                row['checks']['stopped_at_rule']=counter>=config['early_stopping_patience'] or config['epochs_completed']==config['epoch_ceiling']
                row['status']='PASS' if all(row['checks'].values()) and row['best_checkpoint']['all_model_tensors_finite'] else 'FAIL'
                row['fresh_baseline_metric_reproduction']='NOT VERIFIED in this audit; artifact consistency and relevant tests only' if model!='hitskt' else 'Separate independent replay manifests'
            else:
                row['status']='NOT VERIFIED'
                row['issues'].append('No completed final test result; excluded from reported gains')
            if model=='rkt':
                prepared=Path('data/processed')/dataset/'full/rkt_phi_full'
                metadata=json.loads((prepared/'metadata.json').read_text())
                row['rkt_cache_metadata_sha256']=sha256(prepared/'metadata.json')
                row['checks']['rkt_metadata_matches_config']=metadata==config['phi']
                row['rkt_full_cache_counts_reconstructed']='NOT VERIFIED: source/test inspection, metadata and complement algebra checked separately; not a raw-label rebuild of all cached pair counts'
            records.append(row)
    gains={}
    for dataset in ['assist2017','junyi','ednet_kt1']:
        rows=[r for r in records if r['dataset']==dataset and r['complete']]
        hitskt=next(r for r in rows if r['model']=='hitskt')
        baselines=[r for r in rows if r['model']!='hitskt']
        best=max(baselines,key=lambda r:r['stored_test_metrics']['roc_auc'])
        gains[dataset]=dict(hitskt_auc=hitskt['stored_test_metrics']['roc_auc'],
            best_available_baseline=best['model'],best_available_baseline_auc=best['stored_test_metrics']['roc_auc'],
            auc_difference=hitskt['stored_test_metrics']['roc_auc']-best['stored_test_metrics']['roc_auc'],
            completed_baseline_count=len(baselines),statistical_significance='NOT VERIFIED',
            primary_seed_count=1,primary_seed=42)
    write_json(Path('reports/independent_correctness_audit/benchmark_artifacts.json'),
        dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),scope='All 15 canonical experiment artifact directories; baseline metrics not replayed',
             records=records,recorded_gains=gains,original_artifact_hashes=original_files))
    print(json.dumps(dict(statuses=[(r['model'],r['dataset'],r['status']) for r in records],gains=gains),indent=2))


if __name__=='__main__': main()
