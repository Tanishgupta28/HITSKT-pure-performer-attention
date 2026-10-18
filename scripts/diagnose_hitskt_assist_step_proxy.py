#!/usr/bin/env python3
"""Conditional sensitivity count for ASSIST's step-duration field, not feedback."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ktbench.data.session_store import SessionStore
from scripts.diagnose_hitskt_timing import sweep,FIELDS
from scripts.diagnose_hitskt_common import OUT,now,sha,write


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    store=SessionStore('data/processed/assist2017/full/session_store');n=len(store.correct)
    frame=pd.read_csv('data/raw/assist2017/2017.csv',usecols=['startTime','endTime','timeTaken'])
    taken=frame.timeTaken.to_numpy(dtype='float64');end=frame.endTime.to_numpy(dtype='float64');start=frame.startTime.to_numpy(dtype='float64')
    values=start+taken;aligned=np.empty(n,dtype='float64');cursor=0
    for file in sorted(Path('data/processed/assist2017/full/events').glob('*.parquet')):
        for b in pq.ParquetFile(file).iter_batches(columns=['source_row'],batch_size=262144):
            rows=b.column(0).to_numpy()-1;aligned[cursor:cursor+len(rows)]=values[rows]*1000;cursor+=len(rows)
    assert cursor==n
    result=sweep(store.timestamp_ms,aligned,np.full(n,-1,dtype='int64'),store.session_offsets,store.student_offsets,store.session_split)
    write(OUT/'assist2017_step_duration_sensitivity.json',dict(utc=now(),status='COMPLETE CONDITIONAL DIAGNOSTIC',
        assumption='Interpret timeTaken as seconds and startTime+timeTaken as a possible step-end proxy; neither units-to-feedback mapping nor correctness release is independently proven',
        not_a_replacement='No protocol or original event/checkpoint changed',
        source_timeTaken_range=[float(taken.min()),float(taken.max())],source_timeTaken_negative=int(np.count_nonzero(taken<0)),
        source_step_proxy_before_recorded_end=int(np.count_nonzero(values<end)),source_step_proxy_after_recorded_end=int(np.count_nonzero(values>end)),
        by_target_split={name:dict(zip(FIELDS,map(int,row))) for name,row in zip(['train','validation','test'],result)},
        implementation_sha256=sha(__file__)))


if __name__=='__main__':main()
