#!/usr/bin/env python3
"""Memory-bounded persisted RKT fold algebra; not a raw pair-count rebuild."""
from pathlib import Path
import os
import json
import shutil
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scripts.audit_hitskt_replay import sha256,write_json

CHUNK=250_000

def open_arrays(source,cache):
    destination=cache/source.stem
    destination.mkdir(parents=True,exist_ok=True)
    with ZipFile(source) as archive:
        for name in ['keys.npy','counts.npy']:
            path=destination/name
            if not path.exists():
                temporary=path.with_suffix('.tmp')
                with archive.open(name) as inp,temporary.open('wb') as out:
                    shutil.copyfileobj(inp,out,length=8*1024*1024)
                temporary.replace(path)
    return (np.load(destination/'keys.npy',mmap_mode='r'),np.load(destination/'counts.npy',mmap_mode='r'))


def close_arrays(arrays):
    for array in arrays:array._mmap.close()


def subset_for_chunk(own,keys):
    ownkeys,owncounts=own
    a=int(np.searchsorted(ownkeys,keys[0],side='left'))
    b=int(np.searchsorted(ownkeys,keys[-1],side='right'))
    indices=np.searchsorted(keys,ownkeys[a:b])
    assert np.all(indices<len(keys)) and np.array_equal(keys[indices],ownkeys[a:b])
    return indices,owncounts[a:b]


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    try:Path('/proc/self/oom_score_adj').write_text('1000')
    except PermissionError:pass
    records={}
    for dataset in ['assist2017','junyi','ednet_kt1']:
        root=Path('data/processed')/dataset/'full/rkt_phi_full'
        cache=Path('.inspection/independent_correctness_audit')/f'rkt_algebra_{dataset}'
        cache.mkdir(exist_ok=True)
        storemeta=json.loads((root.parent/'session_store/metadata.json').read_text())
        folds=pd.read_csv(root/'student_folds.csv')
        students=np.arange(1,storemeta['students']+1,dtype=np.int32)
        expected=np.empty(len(students),dtype=np.uint8)
        permutation=np.random.default_rng(42).permutation(len(students))
        expected[permutation]=np.arange(len(students))%5
        assert np.array_equal(folds.student_id,students) and np.array_equal(folds.fold,expected)
        allkeys,allcounts=open_arrays(root/'phi_all_train.npz',cache)
        own=[open_arrays(root/f'counts_fold_{fold}.npz',cache) for fold in range(5)]
        for a in range(0,len(allkeys),CHUNK):
            b=min(a+CHUNK,len(allkeys));keys=allkeys[a:b];counts=allcounts[a:b]
            assert np.all(np.diff(keys)>0) and (not a or allkeys[a-1]<keys[0])
            assert np.all(counts>=0)
            total=np.zeros_like(counts)
            for fold in own:
                indices,values=subset_for_chunk(fold,keys)
                assert np.all(values>=0)
                total[indices]+=values
            assert np.array_equal(total,counts)
        files=[dict(path=str(root/name),sha256=sha256(root/name)) for name in ['phi_all_train.npz','student_folds.csv','metadata.json']]
        for fold in range(5):
            complement=root/f'phi_excluding_fold_{fold}.npz'
            excluded=open_arrays(complement,cache)
            cursor=0
            for a in range(0,len(allkeys),CHUNK):
                b=min(a+CHUNK,len(allkeys));keys=allkeys[a:b]
                expected_counts=np.array(allcounts[a:b],copy=True)
                indices,values=subset_for_chunk(own[fold],keys)
                expected_counts[indices]-=values
                assert np.all(expected_counts>=0)
                keep=np.any(expected_counts!=0,axis=1)
                n=int(keep.sum())
                assert np.array_equal(excluded[0][cursor:cursor+n],keys[keep])
                assert np.array_equal(excluded[1][cursor:cursor+n],expected_counts[keep])
                cursor+=n
            assert cursor==len(excluded[0])
            close_arrays(excluded)
            # Only generated decompression scratch is removed, never a source cache.
            shutil.rmtree(cache/complement.stem)
            files.extend(dict(path=str(p),sha256=sha256(p)) for p in [root/f'counts_fold_{fold}.npz',complement])
            print(dataset,'excluded fold',fold,'all rows PASS',flush=True)
        records[dataset]=dict(status='PASS',all_cache_counts_equal_sum_of_five_folds=True,
            every_excluded_cache_equals_all_minus_own_fold=True,student_fold_assignment_seed42_exact=True,
            all_entries_checked=True,all_train_pairs=len(allkeys),cache_files=files,
            working_chunk_rows=CHUNK,execution='Compressed NPY members streamed to local scratch and memory-mapped; no full count-matrix RAM copies',
            raw_training_label_pair_counts_rebuilt='NOT VERIFIED: validates persisted cache algebra, not original pair-count lineage',
            source_selection_review='ktbench/rkt/phi.py:147 restricts construction to train sessions; own-fold complement for training, all-train cache for held-out rows')
        for arrays in own:close_arrays(arrays)
        close_arrays((allkeys,allcounts))
        print(dataset,'PASS: every persisted fold-complement entry checked',flush=True)
    write_json(Path('reports/independent_correctness_audit/rkt_cache_algebra.json'),
               dict(implementation_sha256=sha256(__file__),records=records))

if __name__=='__main__':main()
