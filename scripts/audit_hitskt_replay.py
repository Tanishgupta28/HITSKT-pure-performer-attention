#!/usr/bin/env python3
"""Fresh-process CPU-only replay with independently implemented test metrics.

Preserves experiment files. Production model/collation/batch order are retained
to replay the historical model; neither training _run_epoch nor metric helpers
are imported. Full raw-to-store auditing is a separate prerequisite/evidence file.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from ktbench.batching import collate_hitskt
from ktbench.data.session_store import RollingSessionDataset, SessionStore, make_session_dataloader
from ktbench.models.hitskt import HiTSKT

METRIC_ABSOLUTE_TOLERANCE = 1e-6  # declared before replay; FP32 CPU vs historical CUDA


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def independent_metrics(probabilities, labels, logits):
    """Confusion counts, tied-rank pair AUC, and logit BCE without sklearn."""
    y = np.asarray(labels, dtype=np.uint8)
    p = np.asarray(probabilities, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    assert set(np.unique(y)).issubset({0,1}) and np.isfinite(p).all() and np.isfinite(z).all()
    prediction = p >= .5
    positive = y == 1
    tp = int(np.count_nonzero(prediction & positive))
    fp = int(np.count_nonzero(prediction & ~positive))
    tn = int(np.count_nonzero(~prediction & ~positive))
    fn = int(np.count_nonzero(~prediction & positive))
    order = np.argsort(p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1])+1]
    size = np.diff(np.r_[starts, len(y)])
    positives = np.add.reduceat(sorted_y.astype(np.int64), starts)
    negatives = size-positives
    previous_negatives = np.cumsum(negatives, dtype=np.int64)-negatives
    auc = float(np.sum(positives*(previous_negatives+.5*negatives), dtype=np.float64)/(positive.sum()*(~positive).sum()))
    bce = float(np.mean(np.maximum(z,0)-z*y+np.log1p(np.exp(-np.abs(z))), dtype=np.float64))
    return dict(targets=len(y), roc_auc=auc, accuracy=(tp+tn)/len(y),
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                mse=float(np.mean((p-y)**2, dtype=np.float64)), bce_float64=bce,
                threshold=.5, confusion=dict(tp=tp,fp=fp,tn=tn,fn=fn))


def checkpoint_causality(model, dataset, collator):
    checks = 0
    maximum_difference = 0.
    # Every split's real checkpoint is exercised on genuine stored examples.
    indices = sorted(set([0, len(dataset)//3, len(dataset)//2, len(dataset)-1]))
    with torch.inference_mode():
        for index in indices:
            item = dataset[index]
            baseline = model(collator([item]))[0]
            for position in sorted(set([0,len(item.target)//2,len(item.target)-1])):
                correct = list(item.target.correct)
                correct[position:] = [1-int(y) for y in correct[position:]]
                changed = replace(item,target=replace(item.target,correct=correct))
                difference = float((baseline[:position+1]-model(collator([changed]))[0,:position+1]).abs().max())
                maximum_difference = max(maximum_difference,difference)
                checks += 1
    return dict(status='PASS' if maximum_difference <= 1e-5 else 'FAIL',
                raw_response_flip_checks=checks,maximum_earlier_logit_change=maximum_difference,
                absolute_tolerance=1e-5)


def replay(args, manifest):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Audit must not use LightSAKT2 GPU'
    assert not torch.cuda.is_available()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    root = Path.cwd()
    run = root/'experiments/hitskt'/args.dataset/'full'
    store_root = root/'data/processed'/args.dataset/'full/session_store'
    required = ['_SUCCESS','best_model.pt','config.json','dataset_stats.json','final_results.json','metrics.csv','training.jsonl']
    missing = [str(run/name) for name in required if not (run/name).exists()]
    if missing:
        raise FileNotFoundError(f'Required historical artifacts unavailable: {missing}')
    original_hashes = {name:sha256(run/name) for name in required}
    store_hashes = {p.name:sha256(p) for p in sorted(store_root.iterdir()) if p.is_file()}
    config = json.loads((run/'config.json').read_text())
    stored = json.loads((run/'final_results.json').read_text())
    stats = json.loads((run/'dataset_stats.json').read_text())
    assert config['seed'] == stored['seed'] == 42
    assert config['dataset'] == stored['dataset'] == args.dataset
    assert config['model'] == stored['model'] == 'HiTSKT'
    assert stored['status'] == 'complete' and stored['best_checkpoint_reloaded']
    checkpoint = torch.load(run/'best_model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == config['best_epoch'] == stored['best_epoch']
    assert checkpoint['validation_roc_auc'] == config['best_validation_auc'] == stored['best_validation_auc']
    assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
    model = HiTSKT.from_checkpoint(run/'best_model.pt',map_location='cpu').eval()
    for key,value in checkpoint['config'].items():
        if key not in ('num_questions','num_skills'):
            assert config[key] == value, key
    del checkpoint
    store = SessionStore(store_root)
    for key,meta in [('students','students'),('interactions','interactions'),('questions','num_questions'),('skills','num_skills')]:
        assert stats[key] == store.metadata[meta]
    dataset = RollingSessionDataset(store,'test',history_sessions=config['history_sessions'])
    expected = int(np.sum(store.session_offsets[dataset.targets+1]-store.session_offsets[dataset.targets]))
    assert expected == stats['target_counts']['test'] == stored['test']['targets']
    from functools import partial
    collator = partial(collate_hitskt,num_questions=store.metadata['num_questions'],num_skills=store.metadata['num_skills'])
    # Obtain the unchanged historical sampler; collate/iterate manually so IDs
    # can be independently checked against source store ranges for every batch.
    _,sampler = make_session_dataloader(dataset,token_budget=config['token_budget'],
        maximum_batch_size=config['maximum_batch_size'],bucket_size=config['bucket_size'],
        shuffle=False,seed=42,workers=0,pin_memory=False)
    sampler.set_epoch(0)
    manifest.update(historical_artifact_sha256=original_hashes,session_store_sha256=store_hashes,
                    historical_config=config,stored_metrics=stored['test'],expected_targets=expected,
                    best_checkpoint_epoch=config['best_epoch'],trained_checkpoint_causality=checkpoint_causality(model,dataset,collator))
    assert manifest['trained_checkpoint_causality']['status'] == 'PASS'
    write_json(args.output,manifest)
    cache = root/'.inspection/independent_correctness_audit'/f'replay_{args.dataset}'
    cache.mkdir(parents=True,exist_ok=False)
    arrays = {name:np.lib.format.open_memmap(cache/f'{name}.npy',mode='w+',dtype=dtype,shape=(expected,))
              for name,dtype in [('probability','float32'),('logit','float32'),('label','uint8'),('event_id','int64')]}
    seen = np.zeros(len(dataset),dtype=np.uint8)
    cursor,batches,weighted_loss=0,0,0.
    started,last_log=time.perf_counter(),time.perf_counter()
    with torch.inference_mode():
        for indices in sampler:
            assert not np.any(seen[indices]); seen[indices]=1
            batch=collator([dataset[i] for i in indices])
            ranges=[(int(store.session_offsets[dataset.targets[i]]),int(store.session_offsets[dataset.targets[i]+1])) for i in indices]
            expected_y=np.concatenate([store.correct[a:b] for a,b in ranges])
            mask=batch.target_metric_mask
            assert np.array_equal(batch.target_labels[mask].numpy(),expected_y)
            assert int(mask.sum()) == sum(b-a for a,b in ranges)
            logits=model(batch)
            valid=logits[mask]
            count=valid.numel()
            loss=F.binary_cross_entropy_with_logits(logits,batch.target_labels.to(logits.dtype),reduction='none')[mask].mean()
            weighted_loss+=float(loss)*count
            arrays['probability'][cursor:cursor+count]=torch.sigmoid(valid).numpy()
            arrays['logit'][cursor:cursor+count]=valid.numpy()
            arrays['label'][cursor:cursor+count]=expected_y
            arrays['event_id'][cursor:cursor+count]=np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in ranges])
            cursor+=count; batches+=1
            if time.perf_counter()-last_log >= 25:
                progress=dict(dataset=args.dataset,targets=cursor,total=expected,batches=batches,seconds=time.perf_counter()-started)
                print(json.dumps(progress),flush=True); write_json(cache/'progress.json',progress)
                last_log=time.perf_counter()
    assert cursor == expected and np.all(seen==1)
    for array in arrays.values(): array.flush()
    metrics=independent_metrics(arrays['probability'],arrays['label'],arrays['logit'])
    metrics['historical_batch_weighted_bce']=weighted_loss/cursor
    compare={}
    for key in ('roc_auc','accuracy','precision','recall','f1','mse'):
        actual,original=metrics[key],stored['test'][key]
        compare[key]=dict(stored=original,recalculated=actual,exact_equal=actual==original,
                          absolute_difference=abs(actual-original),within_tolerance=abs(actual-original)<=METRIC_ABSOLUTE_TOLERANCE)
    for key,actual in [('bce_float64',metrics['bce_float64']),('historical_batch_weighted_bce',metrics['historical_batch_weighted_bce'])]:
        original=stored['test']['loss']
        compare[key]=dict(stored=original,recalculated=actual,exact_equal=actual==original,
                          absolute_difference=abs(actual-original),within_tolerance=abs(actual-original)<=METRIC_ABSOLUTE_TOLERANCE)
    assert {name:sha256(run/name) for name in required} == original_hashes
    manifest.update(status='PASS' if all(c['within_tolerance'] for c in compare.values()) else 'FAIL',
        completion_utc=datetime.now(timezone.utc).isoformat(),runtime_seconds=time.perf_counter()-started,
        recalculated_metrics=metrics,comparison=compare,
        exact_equality_all_metrics=all(c['exact_equal'] for c in compare.values()),
        all_test_sessions_once=True,all_supervised_test_targets_once=True,
        original_artifacts_unchanged=True,prediction_artifacts_local=str(cache.relative_to(root)),
        prediction_sha256={p.name:sha256(p) for p in cache.glob('*.npy')})
    write_json(args.output,manifest)
    print(json.dumps(dict(dataset=args.dataset,status=manifest['status'],metrics=metrics,comparison=compare),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['assist2017','junyi','ednet_kt1'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(status='RUNNING',dataset=args.dataset,started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),fresh_process=True,device='CPU',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        evaluator_sha256=sha256(Path(__file__)),absolute_metric_tolerance=METRIC_ABSOLUTE_TOLERANCE,
        tolerance_rationale='Declared before replay: 1e-6 absolute for FP32 CPU vs historical CUDA; exact equality also recorded separately. No statistical tolerance or significance claim.',
        metric_implementation='Independent tied-pair AUC and NumPy confusion/MSE/logit-BCE; no training epoch or project metric function used')
    write_json(args.output,manifest)
    try:
        replay(args,manifest)
    except Exception as error:
        manifest.update(status='NOT VERIFIED' if isinstance(error,FileNotFoundError) else 'FAIL',
                        error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        write_json(args.output,manifest)
        raise


if __name__ == '__main__':
    main()
