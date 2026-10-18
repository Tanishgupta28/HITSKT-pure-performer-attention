#!/usr/bin/env python3
"""Quantify answer-time ambiguity independently of sequence-index causality."""
from pathlib import Path
import argparse
import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scripts.audit_hitskt_replay import sha256,write_json

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():raise SystemExit('Refusing to overwrite prior timing evidence')
records={}
for dataset in ['assist2017','junyi','ednet_kt1']:
    processed=Path('data/processed')/dataset/'full'
    end=None
    if dataset!='ednet_kt1':
        source=Path('data/raw')/dataset/('2017.csv' if dataset=='assist2017' else 'Junyi.csv')
        end=pd.read_csv(source,usecols=['endTime']).endTime.to_numpy(dtype=np.float64)*1000
    by_split={name:dict(targets=0,previous_response_end_after_target_start=0,
        overlapping_targets_in_same_session=0,equal_start_to_previous=0,
        equal_start_in_same_solving_bundle=0) for name in ['train','validation','test']}
    previous=None
    for file in sorted((processed/'events').glob('*.parquet')):
        columns=['student_id','timestamp','source_row','session_id','session_position','split']
        columns+=['elapsed_time','solving_id'] if dataset=='ednet_kt1' else []
        for batch in pq.ParquetFile(file).iter_batches(columns=columns,batch_size=262144,use_threads=False):
            arrays={name:batch.column(name).to_numpy(zero_copy_only=False) for name in columns}
            starts=arrays['timestamp']
            ends=starts+arrays['elapsed_time'] if dataset=='ednet_kt1' else end[arrays['source_row']-1]
            p_student=np.r_[previous[0] if previous else -1,arrays['student_id'][:-1]]
            p_start=np.r_[previous[1] if previous else -1,starts[:-1]]
            p_end=np.r_[previous[2] if previous else -1,ends[:-1]]
            same=arrays['student_id']==p_student
            overlap=same & (p_end>starts)
            tied=same & (p_start==starts)
            within=arrays['session_position']>1
            bundle=np.zeros(len(starts),dtype=bool)
            if dataset=='ednet_kt1':
                p_bundle=np.r_[previous[3] if previous else -1,arrays['solving_id'][:-1]]
                bundle=same & (p_bundle==arrays['solving_id'])
            for split,name in enumerate(['train','validation','test']):
                selected=arrays['split']==split
                by_split[name]['targets']+=int(selected.sum())
                by_split[name]['previous_response_end_after_target_start']+=int(np.count_nonzero(selected&overlap))
                by_split[name]['overlapping_targets_in_same_session']+=int(np.count_nonzero(selected&overlap&within))
                by_split[name]['equal_start_to_previous']+=int(np.count_nonzero(selected&tied))
                by_split[name]['equal_start_in_same_solving_bundle']+=int(np.count_nonzero(selected&tied&bundle))
            previous=(int(arrays['student_id'][-1]),int(starts[-1]),float(ends[-1]),int(arrays['solving_id'][-1]) if dataset=='ednet_kt1' else -1)
    records[dataset]=dict(by_target_split=by_split,
        check_status='FAIL' if any(v['previous_response_end_after_target_start'] for v in by_split.values()) else 'NOT VERIFIED',
        interpretation=('The recorded endTime <= next start inequality fails. Whether correctness feedback actually arrived at endTime is NOT VERIFIED.'
            if dataset=='assist2017' else 'Timestamp ties and transformed source lack an independently established response-arrival order.'
            if dataset=='junyi' else 'timestamp+elapsed_time is a response-completion proxy from official field meanings, not a verified submission timestamp; overlap challenges immediate-feedback availability, especially bundles.'),
        sequence_index_causality='Separately tested; these observations do not assert an acausal attention graph',
        all_retained_event_rows_checked=True)
write_json(args.output,dict(implementation_sha256=sha256(__file__),records=records,
    scope='Recorded/proxy time inequalities and ties only; actual correctness feedback arrival is NOT VERIFIED for all datasets'))
print(json.dumps(records,indent=2))
