#!/usr/bin/env python3
"""Parallel CPU replay of unchanged HiTSKT; each worker reloads the checkpoint.

Uses the same historical batch sequence, collected in sequence despite parallel
execution. This is evaluation orchestration, not an architecture/training change.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from ktbench.batching import collate_hitskt
from ktbench.data.session_store import RollingSessionDataset, SessionStore, make_session_dataloader
from ktbench.models.hitskt import HiTSKT
from scripts.audit_hitskt_replay import METRIC_ABSOLUTE_TOLERANCE, independent_metrics, sha256, write_json
from scripts.audit_memory_guard import admit

_model = _dataset = _collator = None


def initialize(dataset_name):
    global _model,_dataset,_collator
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
    # Prefer an audit worker as victim if unrelated jobs exhaust the container.
    try: Path('/proc/self/oom_score_adj').write_text('1000')
    except PermissionError: pass
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True,warn_only=True)
    run=Path('experiments/hitskt')/dataset_name/'full'
    config=json.loads((run/'config.json').read_text())
    store=SessionStore(Path('data/processed')/dataset_name/'full/session_store')
    _model=HiTSKT.from_checkpoint(run/'best_model.pt',map_location='cpu').eval()
    _dataset=RollingSessionDataset(store,'test',history_sessions=config['history_sessions'])
    _collator=partial(collate_hitskt,num_questions=store.metadata['num_questions'],num_skills=store.metadata['num_skills'])


def evaluate_batch(indices):
    batch=_collator([_dataset[i] for i in indices])
    ranges=[(int(_dataset.store.session_offsets[_dataset.targets[i]]),int(_dataset.store.session_offsets[_dataset.targets[i]+1])) for i in indices]
    truth=np.concatenate([_dataset.store.correct[a:b] for a,b in ranges])
    mask=batch.target_metric_mask
    assert np.array_equal(batch.target_labels[mask].numpy(),truth)
    assert int(mask.sum())==sum(b-a for a,b in ranges)
    with admit(batch,_model.config),torch.inference_mode():
        logits=_model(batch)
        valid=logits[mask]
        loss=F.binary_cross_entropy_with_logits(logits,batch.target_labels.to(logits.dtype),reduction='none')[mask].mean()
        return (indices,valid.numpy().copy(),torch.sigmoid(valid).numpy(),truth,
                np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in ranges]),float(loss))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['assist2017','junyi','ednet_kt1'])
    parser.add_argument('--precheck-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--processes',type=int,default=2)
    parser.add_argument('--resume-cache',type=Path)
    args=parser.parse_args()
    if not 1 <= args.processes <= 6: raise ValueError('Guarded audit supports 1–6 workers per replay')
    assert not args.output.exists()
    original=json.loads(args.precheck_manifest.read_text())
    run=Path('experiments/hitskt')/args.dataset/'full'
    store_root=Path('data/processed')/args.dataset/'full/session_store'
    manifest={**original,'status':'RUNNING','parallel_started_utc':datetime.now(timezone.utc).isoformat(),
        'pid':os.getpid(),'worker_processes':args.processes,'threads_per_worker':1,
        'parallel_evaluator_sha256':sha256(__file__),'precheck_manifest_sha256':sha256(args.precheck_manifest),
        'execution':'Fresh spawned CPU workers each reload actual best checkpoint; unchanged historical batches, ordered collection',
        'all_partial_previous_predictions_ignored':args.resume_cache is None}
    write_json(args.output,manifest)
    started=last=time.perf_counter()
    try:
        for name,digest in original['historical_artifact_sha256'].items(): assert sha256(run/name)==digest
        for name,digest in original['session_store_sha256'].items(): assert sha256(store_root/name)==digest
        initialize(args.dataset)
        config=original['historical_config']; expected=original['expected_targets']
        _,sampler=make_session_dataloader(_dataset,token_budget=config['token_budget'],
            maximum_batch_size=config['maximum_batch_size'],bucket_size=config['bucket_size'],
            shuffle=False,seed=42,workers=0,pin_memory=False)
        sampler.set_epoch(0)
        cache=Path('.inspection/independent_correctness_audit')/f'replay_{args.dataset}_guarded'
        cache.mkdir(exist_ok=False)
        arrays={name:np.lib.format.open_memmap(cache/f'{name}.npy',mode='w+',dtype=dtype,shape=(expected,))
            for name,dtype in [('probability','float32'),('logit','float32'),('label','uint8'),('event_id','int64')]}
        seen=np.zeros(len(_dataset),dtype=np.uint8)
        cursor=batches=0; weighted_loss=0.; max_worker_parity=0.
        remaining=iter(sampler)
        if args.resume_cache:
            progress=json.loads((args.resume_cache/'progress.json').read_text())
            assert progress['dataset']==args.dataset and progress['total']==expected
            previous={name:np.load(args.resume_cache/f'{name}.npy',mmap_mode='r') for name in arrays}
            for prefix_batch in range(progress['batches']):
                indices=next(remaining)
                assert not np.any(seen[indices]);seen[indices]=1
                batch=_collator([_dataset[i] for i in indices])
                ranges=[(int(_dataset.store.session_offsets[_dataset.targets[i]]),int(_dataset.store.session_offsets[_dataset.targets[i]+1])) for i in indices]
                labels=np.concatenate([_dataset.store.correct[a:b] for a,b in ranges])
                ids=np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in ranges])
                count=len(labels)
                assert np.array_equal(previous['label'][cursor:cursor+count],labels)
                assert np.array_equal(previous['event_id'][cursor:cursor+count],ids)
                if prefix_batch<3:
                    reference=evaluate_batch(indices)
                    assert np.array_equal(reference[1],previous['logit'][cursor:cursor+count])
                padded_logits=torch.zeros(batch.target_labels.shape,dtype=torch.float32)
                padded_logits[batch.target_metric_mask]=torch.from_numpy(np.array(previous['logit'][cursor:cursor+count],copy=True))
                loss=F.binary_cross_entropy_with_logits(padded_logits,batch.target_labels.float(),reduction='none')[batch.target_metric_mask].mean()
                weighted_loss+=float(loss)*count
                for name in arrays:arrays[name][cursor:cursor+count]=previous[name][cursor:cursor+count]
                cursor+=count;batches+=1
            assert cursor==progress['targets']
            manifest['preserved_prefix']=dict(cache=str(args.resume_cache),batches=batches,targets=cursor,
                independently_rechecked_all_event_ids_and_labels=True,first_three_batches_exact_logit_replay=True,
                original_prefix_not_modified=True,
                prefix_sha256={name:__import__('hashlib').sha256(previous[name][:cursor].tobytes()).hexdigest() for name in arrays})
            write_json(args.output,manifest)
            print(json.dumps(dict(stage='prefix_verified',dataset=args.dataset,targets=cursor,batches=batches)),flush=True)
        with mp.get_context('spawn').Pool(args.processes,initializer=initialize,initargs=(args.dataset,)) as pool:
            for indices,logits,probability,labels,ids,loss in pool.imap(evaluate_batch,remaining,chunksize=1):
                assert not np.any(seen[indices]); seen[indices]=1
                if batches<3:
                    reference=evaluate_batch(indices)
                    delta=float(np.max(np.abs(reference[1]-logits)))
                    max_worker_parity=max(max_worker_parity,delta)
                    assert delta==0.,'Spawned worker and direct CPU model disagree'
                count=len(labels)
                for name,values in [('probability',probability),('logit',logits),('label',labels),('event_id',ids)]:
                    arrays[name][cursor:cursor+count]=values
                cursor+=count; batches+=1; weighted_loss+=loss*count
                if time.perf_counter()-last>=25:
                    progress=dict(dataset=args.dataset,targets=cursor,total=expected,batches=batches,seconds=time.perf_counter()-started,processes=args.processes)
                    print(json.dumps(progress),flush=True); write_json(cache/'progress.json',progress)
                    last=time.perf_counter()
        assert cursor==expected and np.all(seen==1)
        for array in arrays.values(): array.flush()
        metrics=independent_metrics(arrays['probability'],arrays['label'],arrays['logit'])
        metrics['historical_batch_weighted_bce']=weighted_loss/cursor
        comparison={}
        for key in ['roc_auc','accuracy','precision','recall','f1','mse','bce_float64','historical_batch_weighted_bce']:
            old=original['stored_metrics']['loss' if 'bce' in key else key]
            value=metrics[key]
            comparison[key]=dict(stored=old,recalculated=value,exact_equal=old==value,
                absolute_difference=abs(old-value),within_tolerance=abs(old-value)<=METRIC_ABSOLUTE_TOLERANCE)
        for name,digest in original['historical_artifact_sha256'].items(): assert sha256(run/name)==digest
        manifest.update(status='PASS' if all(v['within_tolerance'] for v in comparison.values()) else 'FAIL',
            completion_utc=datetime.now(timezone.utc).isoformat(),runtime_seconds=time.perf_counter()-started,
            recalculated_metrics=metrics,comparison=comparison,exact_equality_all_metrics=all(v['exact_equal'] for v in comparison.values()),
            all_test_sessions_once=True,all_supervised_test_targets_once=True,original_artifacts_unchanged=True,
            first_three_batch_worker_direct_max_difference=max_worker_parity,
            prediction_artifacts_local=str(cache),prediction_sha256={p.name:sha256(p) for p in cache.glob('*.npy')})
        write_json(args.output,manifest)
        print(json.dumps(dict(dataset=args.dataset,status=manifest['status'],metrics=metrics,comparison=comparison),indent=2),flush=True)
    except Exception as error:
        manifest.update(status='FAIL',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        write_json(args.output,manifest)
        raise


if __name__=='__main__': main()
