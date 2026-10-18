#!/usr/bin/env python3
"""Aggregate recorded/proxy timing in actual 15-session HiTSKT contexts.

No claim that an end-time proxy equals correctness feedback. Learner-level
arrays stay in ignored local scratch; only aggregate counts enter evidence.
"""
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit
from ktbench.data.session_store import SessionStore
from scripts.diagnose_hitskt_common import DATASETS,OUT,now,sha,write

FIELDS=['targets','with_prior_visible_response','proxy_overlap_targets','proxy_overlap_edges',
        'cross_session_proxy_overlap_targets','equal_start_targets','equal_start_edges',
        'same_solving_episode_targets','same_solving_episode_edges']


@njit
def sweep(starts,ends,bundles,session_offsets,student_offsets,splits,history=15):
    result=np.zeros((3,9),np.int64)
    for student in range(len(student_offsets)-1):
        first=student_offsets[student];last=student_offsets[student+1]
        begin=session_offsets[first];stop=session_offsets[last]
        order=np.sort(ends[begin:stop]);tree=np.zeros(len(order)+1,np.int64)
        # Episode counts are valid only for the EdNet channel; -1 disables it.
        bundle_order=np.unique(bundles[begin:stop]);bundle_count=np.zeros(len(bundle_order),np.int64)
        removed=begin;added=begin
        for session in range(first,last):
            lower=session_offsets[max(first,session-history)]
            prior_max=-np.inf
            if session>first:prior_max=np.max(ends[lower:session_offsets[session]])
            for i in range(session_offsets[session],session_offsets[session+1]):
                while added<i:
                    rank=np.searchsorted(order,ends[added],side='left')+1
                    while rank<len(tree):tree[rank]+=1;rank+=rank&-rank
                    bundle_count[np.searchsorted(bundle_order,bundles[added])]+=1
                    added+=1
                while removed<lower:
                    rank=np.searchsorted(order,ends[removed],side='left')+1
                    while rank<len(tree):tree[rank]-=1;rank+=rank&-rank
                    bundle_count[np.searchsorted(bundle_order,bundles[removed])]-=1
                    removed+=1
                if session==first:continue # context-only first session is not supervised
                count_le=0;rank=np.searchsorted(order,starts[i],side='right')
                while rank>0:count_le+=tree[rank];rank-=rank&-rank
                overlap=i-lower-count_le
                prior_session_overlap=False
                if overlap>0:
                    # Cross-session availability can be decided using a range max.
                    prior_session_overlap=prior_max>starts[i]
                tied=i-max(lower,begin+np.searchsorted(starts[begin:stop],starts[i],side='left'))
                same_bundle=0
                if bundles[i]>=0:same_bundle=bundle_count[np.searchsorted(bundle_order,bundles[i])]
                s=splits[session]
                result[s,0]+=1;result[s,1]+=i>lower
                result[s,2]+=overlap>0;result[s,3]+=overlap
                result[s,4]+=prior_session_overlap
                result[s,5]+=tied>0;result[s,6]+=tied
                result[s,7]+=same_bundle>0;result[s,8]+=same_bundle
    return result


def diagnose(dataset):
    started=time.perf_counter();root=Path('data/processed')/dataset/'full';store=SessionStore(root/'session_store')
    n=len(store.correct);scratch=Path('.inspection/reproducibility_diagnostics')/(dataset+'_timing')
    scratch.mkdir(exist_ok=False)
    ends=np.lib.format.open_memmap(scratch/'recorded_or_proxy_end.npy',mode='w+',dtype='float64',shape=(n,))
    bundles=np.lib.format.open_memmap(scratch/'solving_episode.npy',mode='w+',dtype='int64',shape=(n,))
    bundles[:]=-1;source_end=None;extra={}
    if dataset!='ednet_kt1':
        source=Path('data/raw')/dataset/('2017.csv' if dataset=='assist2017' else 'Junyi.csv')
        cols=['startTime','endTime']+(['timeTaken'] if dataset=='assist2017' else [])
        frame=pd.read_csv(source,usecols=cols);source_end=frame.endTime.to_numpy(dtype=np.float64)*1000
        delta=frame.endTime.to_numpy()-frame.startTime.to_numpy()
        extra=dict(source_rows=len(frame),source_end_equal_start=int(np.count_nonzero(delta==0)),
            source_end_before_start=int(np.count_nonzero(delta<0)),source_positive_duration=int(np.count_nonzero(delta>0)))
        if dataset=='assist2017':
            taken=frame.timeTaken.to_numpy(dtype=np.float64)
            extra.update(timeTaken_nonfinite=int(np.count_nonzero(~np.isfinite(taken))),
                timeTaken_equals_end_minus_start=int(np.count_nonzero(taken==delta)),
                timeTaken_difference_maximum_seconds=float(np.max(np.abs(taken-delta))),
                interpretation='Official 2017 dictionary calls timeTaken time spent on current step; not an independent first-response or feedback timestamp')
        del frame,delta
    cursor=0;adj={name:dict(targets=0,previous_end_after_start=0,equal_start=0,same_solving_episode_previous=0) for name in ['train','validation','test']};previous=None
    for file in sorted((root/'events').glob('*.parquet')):
        cols=['student_id','timestamp','source_row','split']+(['elapsed_time','solving_id'] if dataset=='ednet_kt1' else [])
        for batch in pq.ParquetFile(file).iter_batches(batch_size=262144,columns=cols,use_threads=False):
            a={c:batch.column(c).to_numpy(zero_copy_only=False) for c in cols};count=len(a['timestamp']);stop=cursor+count
            assert np.array_equal(store.timestamp_ms[cursor:stop],a['timestamp'])
            end=a['timestamp']+a['elapsed_time'] if dataset=='ednet_kt1' else source_end[a['source_row']-1]
            assert np.all(np.isfinite(end))
            ends[cursor:stop]=end
            if dataset=='ednet_kt1':bundles[cursor:stop]=a['solving_id']
            same=a['student_id']==np.r_[previous[0] if previous else -1,a['student_id'][:-1]]
            pend=np.r_[previous[1] if previous else -1,end[:-1]];pstart=np.r_[previous[2] if previous else -1,a['timestamp'][:-1]]
            for split,name in enumerate(adj):
                m=a['split']==split;adj[name]['targets']+=int(m.sum())
                adj[name]['previous_end_after_start']+=int(np.count_nonzero(m&same&(pend>a['timestamp'])))
                adj[name]['equal_start']+=int(np.count_nonzero(m&same&(pstart==a['timestamp'])))
                if dataset=='ednet_kt1':
                    pb=np.r_[previous[3] if previous else -1,a['solving_id'][:-1]]
                    adj[name]['same_solving_episode_previous']+=int(np.count_nonzero(m&same&(pb==a['solving_id'])))
            previous=(a['student_id'][-1],end[-1],a['timestamp'][-1],a['solving_id'][-1] if dataset=='ednet_kt1' else -1)
            cursor=stop
        print(dataset,'aligned',cursor,'/',n,flush=True)
    assert cursor==n;ends.flush();bundles.flush();del source_end
    values=sweep(store.timestamp_ms,ends,bundles,store.session_offsets,store.student_offsets,store.session_split)
    result=dict(dataset=dataset,utc=now(),status='COMPLETE',retained_events=n,source_diagnostics=extra,
        all_retained_adjacent=adj,actual_hitskt_15_session_context={name:dict(zip(FIELDS,map(int,row))) for name,row in zip(['train','validation','test'],values)},
        scope='Every HiTSKT supervised target and every earlier response exposed by its actual 15-session context, including within-current-session prefix; no current or future indexed labels',
        caveat='Counts are inequalities against recorded endTime (ASSIST/Junyi) or timestamp+elapsed_time proxy (EdNet); not counts of proven unavailable correctness feedback',
        same_solving_episode_caveat='Logged bundle episode membership, not exact submission/feedback timing; -1 disables this channel outside EdNet',
        implementation_sha256=sha(__file__),runtime_seconds=time.perf_counter()-started,
        local_array_sha256={p.name:sha(p) for p in scratch.glob('*.npy')})
    write(OUT/f'{dataset}_timing_context.json',result)
    print(dataset,result['actual_hitskt_15_session_context'],flush=True)


if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    try:Path('/proc/self/oom_score_adj').write_text('1000')
    except PermissionError:pass
    for dataset in DATASETS:diagnose(dataset)
