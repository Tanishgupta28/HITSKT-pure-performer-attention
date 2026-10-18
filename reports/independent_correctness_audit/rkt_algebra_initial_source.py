#!/usr/bin/env python3
"""Validate persisted fold complements; deliberately not a raw Phi rebuild."""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
from scripts.audit_hitskt_replay import sha256,write_json

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
records={}
for dataset in ['assist2017','junyi','ednet_kt1']:
    root=Path('data/processed')/dataset/'full/rkt_phi_full'
    metadata=json.loads((root/'metadata.json').read_text())
    storemeta=json.loads((root.parent/'session_store/metadata.json').read_text())
    folds=pd.read_csv(root/'student_folds.csv')
    students=np.arange(1,storemeta['students']+1,dtype=np.int32)
    expected=np.empty(len(students),dtype=np.uint8)
    permutation=np.random.default_rng(42).permutation(len(students))
    expected[permutation]=np.arange(len(students))%5
    assert np.array_equal(folds.student_id,students) and np.array_equal(folds.fold,expected)
    with np.load(root/'phi_all_train.npz') as data:
        allkeys=data['keys']; allcounts=data['counts']
    assert np.all(np.diff(allkeys)>0) and np.all(allcounts>=0)
    total=np.zeros_like(allcounts)
    files=[]
    for fold in range(5):
        own=root/f'counts_fold_{fold}.npz'; complement=root/f'phi_excluding_fold_{fold}.npz'
        with np.load(own) as data: keys=data['keys'];counts=data['counts']
        assert np.all(np.diff(keys)>0) and np.all(counts>=0)
        positions=np.searchsorted(allkeys,keys)
        assert np.array_equal(allkeys[positions],keys)
        total[positions]+=counts
        expected_complement=allcounts.copy();expected_complement[positions]-=counts
        assert np.all(expected_complement>=0)
        keep=np.any(expected_complement!=0,axis=1)
        with np.load(complement) as data:
            assert np.array_equal(data['keys'],allkeys[keep])
            assert np.array_equal(data['counts'],expected_complement[keep])
        files.extend([dict(path=str(p),sha256=sha256(p)) for p in [own,complement]])
    assert np.array_equal(total,allcounts)
    files.extend(dict(path=str(root/name),sha256=sha256(root/name)) for name in ['phi_all_train.npz','student_folds.csv','metadata.json'])
    records[dataset]=dict(status='PASS',all_cache_counts_equal_sum_of_five_folds=True,
        every_excluded_cache_equals_all_minus_own_fold=True,student_fold_assignment_seed42_exact=True,
        all_entries_checked=True,all_train_pairs=len(allkeys),cache_files=files,
        raw_training_label_pair_counts_rebuilt='NOT VERIFIED: this check validates persisted cache algebra, not original pair-count lineage',
        source_selection_review='ktbench/rkt/phi.py:147 restricts construction to train sessions; repository lookup selects own-fold complement for training and all-train cache for held-out rows')
    print(dataset,'PASS: every persisted fold-complement entry checked',flush=True)
write_json(Path('reports/independent_correctness_audit/rkt_cache_algebra.json'),
           dict(implementation_sha256=sha256(__file__),records=records))
