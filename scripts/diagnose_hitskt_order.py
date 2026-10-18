#!/usr/bin/env python3
"""Regenerate every evaluation batch's target order without model inference."""
import hashlib
import os
from pathlib import Path
import numpy as np
from ktbench.data.session_store import SessionStore,RollingSessionDataset,make_session_dataloader
from scripts.diagnose_hitskt_common import DATASETS,OUT,old_evaluation,sha,now,write


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';records={}
    for dataset in DATASETS:
        old=old_evaluation(dataset);cfg=old['historical_config'];root=Path('data/processed')/dataset/'full/session_store'
        store=SessionStore(root);ds=RollingSessionDataset(store,'test',history_sessions=cfg['history_sessions'])
        _,sampler=make_session_dataloader(ds,token_budget=cfg['token_budget'],maximum_batch_size=cfg['maximum_batch_size'],
            bucket_size=cfg['bucket_size'],shuffle=False,seed=42,workers=0,pin_memory=False)
        sampler.set_epoch(0);cache=Path(old['prediction_artifacts_local'])
        ids=np.load(cache/'event_id.npy',mmap_mode='r');labels=np.load(cache/'label.npy',mmap_mode='r')
        batchhash=hashlib.sha256();cursor=batches=0
        for indices in sampler:
            ranges=[tuple(map(int,store.session_offsets[ds.targets[j]:ds.targets[j]+2])) for j in indices]
            expected=np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in ranges]);count=len(expected)
            assert np.array_equal(ids[cursor:cursor+count],expected)
            assert np.array_equal(labels[cursor:cursor+count],np.concatenate([store.correct[a:b] for a,b in ranges]))
            batchhash.update(np.asarray([count,*map(int,indices)],dtype='<i8').tobytes());cursor+=count;batches+=1
        assert cursor==old['expected_targets']==len(np.unique(ids))
        records[dataset]=dict(status='PASS',all_target_ids_labels_and_batch_order_match=True,targets=cursor,batches=batches,
            epoch=0,shuffle=False,seed=42,token_budget=cfg['token_budget'],maximum_batch_size=cfg['maximum_batch_size'],
            bucket_size=cfg['bucket_size'],batch_membership_count_sha256=batchhash.hexdigest(),
            cached_event_id_sha256=sha(cache/'event_id.npy'),
            limitation='Regenerated unchanged historical sampler and compared with independently captured CPU vectors; original CUDA per-target/batch logs were not persisted')
        print(dataset,records[dataset],flush=True)
    write(OUT/'evaluation_order.json',dict(utc=now(),records=records,implementation_sha256=sha(__file__)))


if __name__=='__main__':main()
