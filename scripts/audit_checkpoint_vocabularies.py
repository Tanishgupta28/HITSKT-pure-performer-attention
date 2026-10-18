#!/usr/bin/env python3
"""Check checkpoint dimensions and distinguish index bounds from distinct IDs."""
from pathlib import Path
import json
import os
import numpy as np
from scripts.audit_hitskt_replay import sha256,write_json

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
root=Path('reports/independent_correctness_audit')
original=json.loads((root/'benchmark_artifacts.json').read_text())['records']
records={}
for dataset in ['assist2017','junyi','ednet_kt1']:
    store=Path('data/processed')/dataset/'full/session_store'
    meta=json.loads((store/'metadata.json').read_text())
    experiment=next(row for row in original if row['model']=='hitskt' and row['dataset']==dataset)
    checkpoint=experiment['best_checkpoint']['config'];config=experiment['config']
    assert checkpoint['num_questions']==meta['num_questions']
    assert checkpoint['num_skills']==meta['num_skills']
    assert experiment['best_checkpoint']['seed_in_payload']==42
    for key,value in checkpoint.items():
        if key not in ['num_questions','num_skills']:assert config[key]==value,key
    record=dict(checkpoint_architecture_and_store_index_bounds_match=True,checkpoint_config=checkpoint,
        checkpoint_seed_metadata=42,vocabularies={})
    for name,field in [('question','num_questions'),('skill','num_skills')]:
        values=np.load(store/(name+'.npy'),mmap_mode='r');counts=np.zeros(meta[field]+1,dtype=np.int64)
        for a in range(0,len(values),1_000_000):
            part=np.asarray(values[a:a+1_000_000],dtype=np.int64)
            assert np.all(part>0) and np.all(part<=meta[field])
            counts+=np.bincount(part,minlength=len(counts))
        record['vocabularies'][name]=dict(metadata_index_upper_bound=meta[field],
            distinct_retained_ids=int(np.count_nonzero(counts)),
            unused_ids_within_index_bound=int(np.count_nonzero(counts[1:]==0)))
    records[dataset]=record
write_json(root/'checkpoint_vocabulary_consistency.json',dict(status='PASS',implementation_sha256=sha256(__file__),records=records))
print(json.dumps(records,indent=2))
