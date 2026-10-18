#!/usr/bin/env python3
"""Close sparse-key ordering/range and provenance-counter checks for Phi caches."""
from pathlib import Path
import json
import os
from zipfile import ZipFile
import zlib
import numpy as np
from scripts.audit_hitskt_replay import sha256,write_json

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
try:Path('/proc/self/oom_score_adj').write_text('1000')
except PermissionError:pass
records={}
for dataset in ['assist2017','junyi','ednet_kt1']:
    root=Path('data/processed')/dataset/'full'
    cache=root/'rkt_phi_full'
    scratch=Path('.inspection/independent_correctness_audit')/f'rkt_algebra_{dataset}'
    meta=json.loads((cache/'metadata.json').read_text())
    store=json.loads((root/'session_store/metadata.json').read_text())
    summary=json.loads((root/'reports/summary.json').read_text())
    allkeys=np.load(scratch/'phi_all_train/keys.npy',mmap_mode='r')
    allcounts=np.load(scratch/'phi_all_train/counts.npy',mmap_mode='r')
    assert len(allkeys)==meta['all_train_unique_directed_pairs']
    assert meta['cache_students']==store['students']
    assert meta['training_targets_cross_fitted']==summary['split_interactions']['train']
    assert meta['seed']==42 and meta['fold_count']==5 and meta['history_length']==49
    contribution=0
    for a in range(0,len(allkeys),250_000):
        keys=allkeys[a:a+250_000]
        q=keys//(store['num_questions']+1);prior=keys%(store['num_questions']+1)
        assert np.all((q>=1)&(q<=store['num_questions'])&(prior>=1)&(prior<=store['num_questions']))
        contribution+=int(allcounts[a:a+250_000].sum(dtype=np.int64))
    assert contribution==meta['phi_pair_contributions']
    checks=[]
    for name in ['phi_all_train']+[f'counts_fold_{fold}' for fold in range(5)]:
        path=scratch/name/'keys.npy'
        crc=0
        with path.open('rb') as file:
            for block in iter(lambda:file.read(8*1024*1024),b''):crc=zlib.crc32(block,crc)
        with ZipFile(cache/(name+'.npz')) as archive:
            info=archive.getinfo('keys.npy')
            assert info.CRC==crc and info.file_size==path.stat().st_size
        values=np.load(path,mmap_mode='r')
        assert values[0]>=allkeys[0] and values[-1]<=allkeys[-1]
        for a in range(0,len(values),250_000):
            chunk=values[a:a+250_000]
            assert np.all(np.diff(chunk)>0) and (not a or values[a-1]<chunk[0])
        checks.append(dict(name=name,status='PASS',rows=len(values),decompressed_keys_crc_matches_original_member=True))
        values._mmap.close()
    records[dataset]=dict(status='PASS',keys=checks,all_pair_ids_in_valid_question_domain=True,
        no_fold_keys_outside_all_train_range=True,metadata_full_population_and_training_target_counts_match=True,
        pair_contributions_sum=contribution,metadata_pair_contributions_match=True)
    allkeys._mmap.close();allcounts._mmap.close()
    print(dataset,'PASS: sparse bounds/order, original-member CRC and full-population counters',flush=True)
write_json(Path('reports/independent_correctness_audit/rkt_cache_key_bounds.json'),
           dict(implementation_sha256=sha256(__file__),records=records))
