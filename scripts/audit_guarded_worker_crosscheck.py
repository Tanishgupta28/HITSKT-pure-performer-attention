#!/usr/bin/env python3
"""Independently replay the first three new batches after a preserved prefix."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from ktbench.data.session_store import make_session_dataloader
from scripts import audit_hitskt_replay_parallel as worker
from scripts.audit_hitskt_replay import sha256,write_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset',choices=['junyi','ednet_kt1'])
args=parser.parse_args()
root=Path('reports/independent_correctness_audit')
manifest=json.loads((root/f'{args.dataset}_evaluation_guarded_manifest.json').read_text())
prefix=manifest['preserved_prefix']
worker.initialize(args.dataset)
config=manifest['historical_config']
_,sampler=make_session_dataloader(worker._dataset,token_budget=config['token_budget'],
    maximum_batch_size=config['maximum_batch_size'],bucket_size=config['bucket_size'],
    shuffle=False,seed=42,workers=0,pin_memory=False)
sampler.set_epoch(0)
cache=Path('.inspection/independent_correctness_audit')/f'replay_{args.dataset}_guarded'
arrays={name:np.load(cache/(name+'.npy'),mmap_mode='r') for name in ['logit','probability','label','event_id']}
cursor=prefix['targets'];checks=[]
for ordinal,indices in enumerate(itertools.islice(iter(sampler),prefix['batches'],prefix['batches']+3)):
    _,logits,probability,labels,ids,_=worker.evaluate_batch(indices)
    n=len(labels)
    assert np.array_equal(labels,arrays['label'][cursor:cursor+n])
    assert np.array_equal(ids,arrays['event_id'][cursor:cursor+n])
    logit_difference=float(np.max(np.abs(logits-arrays['logit'][cursor:cursor+n])))
    probability_difference=float(np.max(np.abs(probability-arrays['probability'][cursor:cursor+n])))
    checks.append(dict(batch_index=prefix['batches']+ordinal,targets=n,
        maximum_absolute_logit_difference=logit_difference,
        maximum_absolute_probability_difference=probability_difference,
        exact_equal=logit_difference==0 and probability_difference==0))
    cursor+=n
assert len(checks)==3
result=dict(status='PASS' if all(c['exact_equal'] for c in checks) else 'FAIL',dataset=args.dataset,
    checks=checks,source_sha256=sha256(__file__),checkpoint_sha256=manifest['historical_artifact_sha256']['best_model.pt'],
    description='Separate fresh process vs actual new guarded-worker cache after the preserved prefix; not merely a default zero parity field')
write_json(root/f'{args.dataset}_guarded_worker_direct_crosscheck.json',result)
print(json.dumps(result,indent=2))
