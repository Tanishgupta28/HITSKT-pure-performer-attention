#!/usr/bin/env python3
"""Bounded real-checkpoint sensitivity diagnostic, not a replacement test set."""
from dataclasses import replace
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from ktbench.data.session_store import SessionStore,RollingSessionDataset
from ktbench.batching import collate_hitskt
from ktbench.models.hitskt import HiTSKT
from scripts.audit_hitskt_replay import sha256,write_json

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_available()
try:Path('/proc/self/oom_score_adj').write_text('1000')
except PermissionError:pass
torch.set_num_threads(1);torch.set_num_interop_threads(1)
records={}
for dataset in ['assist2017','ednet_kt1']:
    root=Path('data/processed')/dataset/'full'
    store=SessionStore(root/'session_store'); ds=RollingSessionDataset(store,'test')
    ends=None
    if dataset=='assist2017':
        ends=pd.read_csv('data/raw/assist2017/2017.csv',usecols=['endTime']).endTime.to_numpy()*1000
    candidates=[];seen_sessions=set();cursor=0
    for file in sorted((root/'events').glob('*.parquet')):
        columns=['student_id','timestamp','source_row','session_position','split']
        if dataset=='ednet_kt1':columns.append('elapsed_time')
        for batch in pq.ParquetFile(file).iter_batches(columns=columns,batch_size=262144,use_threads=False):
            values={key:batch.column(key).to_numpy(zero_copy_only=False) for key in columns}
            t=values['timestamp']
            end=t+values['elapsed_time'] if dataset=='ednet_kt1' else ends[values['source_row']-1]
            valid=(values['split'][1:]==2)&(values['session_position'][1:]>1)&(end[:-1]>t[1:])&(values['student_id'][1:]==values['student_id'][:-1])
            for local in np.flatnonzero(valid)+1:
                event=cursor+int(local)
                session=int(np.searchsorted(store.session_offsets,event,side='right')-1)
                if session in seen_sessions:continue
                index=int(np.searchsorted(ds.targets,session))
                a,b=ds._history_bounds(session)
                history_tokens=int(store.session_offsets[b]-store.session_offsets[a])
                if store.session_length(session)>256 or history_tokens>1024:continue
                candidates.append((index,int(event-store.session_offsets[session])))
                seen_sessions.add(session)
                if len(candidates)==8:break
            cursor+=len(t)
            if len(candidates)==8:break
        if len(candidates)==8:break
    assert len(candidates)==8
    net=HiTSKT.from_checkpoint(Path('experiments/hitskt')/dataset/'full/best_model.pt',map_location='cpu').eval()
    changes=[];own_changes=[]
    def batch(item):return collate_hitskt([item],num_questions=store.metadata['num_questions'],num_skills=store.metadata['num_skills'])
    with torch.inference_mode():
        for index,position in candidates:
            item=ds[index]; baseline=float(net(batch(item))[0,position])
            labels=list(item.target.correct);labels[position-1]=1-int(labels[position-1])
            changed=replace(item,target=replace(item.target,correct=labels))
            changes.append(abs(float(net(batch(changed))[0,position])-baseline))
            labels=list(item.target.correct);labels[position]=1-int(labels[position])
            changed=replace(item,target=replace(item.target,correct=labels))
            own_changes.append(abs(float(net(batch(changed))[0,position])-baseline))
    records[dataset]=dict(sample_size=8,sample_scope='Eight distinct test sessions with an overlapping previous response, target length<=256 and history<=1024; diagnostic only',
        unavailable_predecessor_flip_logit_absolute_changes=changes,
        own_label_flip_logit_absolute_changes=own_changes,
        earlier_index_label_sensitivity_observed=sum(v>1e-6 for v in changes),
        own_label_exclusion_status='PASS' if max(own_changes)<=1e-5 else 'FAIL',
        interpretation='Earlier event-index responses can affect predictions even where that response completes after the query presentation (EdNet completion uses timestamp+elapsed proxy). This differs from future-index/self-label leakage.',
        checkpoint_sha256=sha256(Path('experiments/hitskt')/dataset/'full/best_model.pt'))
    private=Path('.inspection/independent_correctness_audit')/f'{dataset}_availability_diagnostic_indices.json'
    private.write_text(json.dumps(candidates)+'\n')
write_json(Path('reports/independent_correctness_audit/unavailable_response_sensitivity.json'),
           dict(implementation_sha256=sha256(__file__),records=records))
print(json.dumps(records,indent=2))
