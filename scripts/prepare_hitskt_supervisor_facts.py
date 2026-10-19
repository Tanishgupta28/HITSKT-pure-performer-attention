#!/usr/bin/env python3
"""Recheck saved scores/configuration metadata without training or CUDA use."""
import gc
import os
from pathlib import Path
import subprocess
from datetime import datetime,timezone
import numpy as np
import torch
from scripts.audit_hitskt_replay import independent_metrics
from scripts.diagnose_hitskt_common import DATASETS,OLD,OUT as PREVIOUS,old_evaluation,read,write,sha,now

OUT=Path('reports/supervisor_verification')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
    memory=Path('/sys/fs/cgroup/memory')
    light=Path('/workspace/LightSAKT2/reports/progress.json');progress=read(light)
    gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=True)
    inventory=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,check=True)
    resource=dict(utc=now(),cuda_allocations=0,visible_inventory=inventory.stdout.strip(),
        gpu_observation='\n'.join(line.rstrip() for line in gpu.stdout.splitlines()),
        light_active_job_count=len(progress.get('running_jobs',[])),
        light_progress_mtime_utc=datetime.fromtimestamp(light.stat().st_mtime,timezone.utc).isoformat(),
        memory_limit_bytes=int((memory/'memory.limit_in_bytes').read_text()),
        observed_memory_usage_bytes=int((memory/'memory.usage_in_bytes').read_text()),
        oom_control=dict(line.split() for line in (memory/'memory.oom_control').read_text().splitlines()),
        cuda_verification_status='NOT VERIFIED',
        decision='No independent idle device established: occupied visible MIG and active protected training. Do not allocate CUDA or interpret a restricted empty process table as idle.')
    write(OUT/'resource_observation.json',resource)
    datasets={};inputs={}
    for name in ['HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md','HITSKT_RESPONSE_TIMING_ANALYSIS.md','HITSKT_RESEARCH_DECISION_REPORT.md']:
        p=Path('reports')/name;inputs[str(p)]=sha(p)
    for d in DATASETS:
        old=old_evaluation(d);run=Path('experiments/hitskt')/d/'full';result=read(run/'final_results.json');cfg=read(run/'config.json')
        assert result['test']==old['stored_metrics']
        assert cfg['best_epoch']==old['best_checkpoint_epoch']==result['best_epoch']
        assert sha(run/'best_model.pt')==old['historical_artifact_sha256']['best_model.pt']
        cache=Path(old['prediction_artifacts_local'])
        for name,value in old['prediction_sha256'].items():assert sha(cache/name)==value
        arrays={name:np.load(cache/f'{name}.npy',mmap_mode='r') for name in ['probability','label','logit']}
        new=independent_metrics(arrays['probability'],arrays['label'],arrays['logit'])
        for key,value in new.items():assert value==old['recalculated_metrics'][key],(d,key)
        data=read(OLD/f'{d}_data_manifest.json');lineage=data['source_lineage'];store=read(Path('data/processed')/d/'full/session_store/metadata.json')
        timing=read(PREVIOUS/f'{d}_timing_context.json')['actual_hitskt_15_session_context']['test']
        assert data['status']==data['store_contract']['status']=='PASS'
        datasets[d]=dict(historical_cuda_metrics=result['test'],independent_cpu_metrics=old['recalculated_metrics'],
            cached_cpu_metric_recalculation='PASS: all independent metrics exactly match previously captured CPU manifest',
            historical_to_cpu_metric_gate='FAIL',historical_to_cpu_auc_gate='PASS',absolute_tolerance=1e-6,
            selected_epoch=cfg['best_epoch'],checkpoint_sha256=sha(run/'best_model.pt'),
            config_sha256=sha(run/'config.json'),test_targets=old['expected_targets'],
            source_rows=lineage['raw_rows'],source_students=lineage['raw_students'],retained_students=lineage['retained_students'],
            retained_interactions=lineage['retained_rows'],sessions=store['sessions'],
            accepted_input_reconstruction='PASS',original_publisher_transform='NOT VERIFIED' if d!='ednet_kt1' else 'Official acquisition identity and full ZIP reconstruction PASS; no publisher-signed historical checksum',
            timing=timing,masking='PASS within tested production right-padding contract',cuda_replay='NOT VERIFIED',
            response_availability='NOT VERIFIED',recommendation='Retain historical artifacts provisionally; verify further, no established retraining requirement')
        for p in [OLD/f'{d}_data_manifest.json',run/'config.json',run/'final_results.json',PREVIOUS/f'{d}_timing_context.json']:
            inputs[str(p)]=sha(p)
        print(d,'full cached-score recalculation PASS',new['targets'],flush=True)
        del arrays,new;gc.collect()
    baselines=[];prior=read(PREVIOUS/'artifact_and_dkt_forensics.json')['records']
    for model in ['dkt','dkvmn','sakt','rkt']:
        for d in DATASETS:
            root=Path('experiments')/model/d/'full';cfg=read(root/'config.json')
            checkpoint=root/'best_model.pt';c=torch.load(checkpoint,map_location='cpu',weights_only=True)
            expected=next(v for v in prior if v['model']==model and v['dataset']==d)
            assert sha(checkpoint)==expected['best']['sha256']
            assert c['epoch']==cfg['best_epoch']==expected['strict_validation_selected_epoch']
            assert all(torch.isfinite(t).all() for t in c['state_dict'].values())
            for key in ['context_length','history_length']:
                if key in c['config']:assert c['config'][key]==cfg[key]
            complete=(root/'_SUCCESS').exists() and (root/'final_results.json').exists()
            baselines.append(dict(model=model,dataset=d,context_length=cfg['context_length'],history_length=cfg['history_length'],
                batch_size=cfg['batch_size'],selected_epoch=c['epoch'],complete=complete,
                stored_metrics=read(root/'final_results.json')['test'] if complete else None,
                fresh_full_metric_replay='NOT VERIFIED',best_sha256=sha(checkpoint),config_sha256=sha(root/'config.json'),
                checkpoint_history_length=c['config'].get('history_length'),all_best_model_tensors_finite=True))
            inputs[str(root/'config.json')]=sha(root/'config.json');del c;gc.collect()
    assert {b['history_length'] for b in baselines if b['model']=='dkvmn'}=={199}
    assert {b['history_length'] for b in baselines if b['model']=='rkt'}=={49}
    write(OUT/'baseline_context_correction.json',dict(utc=now(),status='VERIFIED DESCRIPTIVE CORRECTION',
        earlier_report='reports/HITSKT_RESEARCH_DECISION_REPORT.md',earlier_wording='DKVMN49, SAKT/RKT99',
        correction='DKT199, DKVMN199, SAKT99, RKT49 prior interactions, verified against every saved config and applicable best-checkpoint config',
        impact='Comparison description only; no result, model, data or training change. Earlier reports preserved unchanged.',
        source_sha256={p:sha(p) for p in ['ktbench/baseline_registry.py','ktbench/models/baselines.py','ktbench/rkt/phi.py','ktbench/rkt/data.py']},
        records=baselines,implementation_sha256=sha(__file__)))
    write(OUT/'supervisor_facts.json',dict(utc=now(),base_commit='e1065c9c6215f9c676b43e33ee0d5f689bbe3fbb',
        datasets=datasets,baselines=baselines,input_sha256=inputs,implementation_sha256=sha(__file__),
        scope='Fresh saved-score metric recalculation, input binding and baseline config/checkpoint inspection; not new full CPU/CUDA model inference or rerun of exhaustive raw reconstruction'))


if __name__=='__main__':main()
