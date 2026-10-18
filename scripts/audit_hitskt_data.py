#!/usr/bin/env python3
"""Read-only, exhaustive source -> Parquet -> session-store reconciliation.

No production preprocessing/session assignment functions are used. Real data
stays local; output contains aggregate checks and hashes, never interaction rows.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ktbench.data.session_store import RollingSessionDataset, SessionStore
from scripts.audit_hitskt_replay import sha256, write_json

EXPECTED_RAW = {
    'assist2017': ('data/raw/assist2017/2017.csv','7577287d4ead073fdb9393b586a260043151f27338246bd076ca740e28be8bbb'),
    'junyi': ('data/raw/junyi/Junyi.csv','8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a'),
    'ednet_kt1': ('data/raw/ednet/archives/EdNet-KT1.zip','0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535')}


class EventStream:
    def __init__(self, root):
        self.files=sorted((root/'events').glob('*.parquet'))
        self.iterator=(batch for file in self.files
            for batch in pq.ParquetFile(file).iter_batches(batch_size=262144,use_threads=False))
        self.current=None; self.offset=0; self.compared=0

    def compare(self, expected):
        remaining=len(expected['correct']); cursor=0
        while remaining:
            if self.current is None or self.offset==len(self.current['correct']):
                batch=next(self.iterator)
                self.current={name:batch.column(name).to_numpy(zero_copy_only=False) for name in batch.schema.names}
                self.offset=0
            take=min(remaining,len(self.current['correct'])-self.offset)
            for name,values in expected.items():
                if not np.array_equal(np.asarray(values)[cursor:cursor+take],self.current[name][self.offset:self.offset+take]):
                    raise AssertionError(f'Raw/Parquet mismatch: {name}, near retained offset {self.compared}')
            self.compared+=take; self.offset+=take; cursor+=take; remaining-=take

    def finish(self):
        assert self.current is not None and self.offset==len(self.current['correct'])
        assert next(self.iterator,None) is None, 'Extra processed rows'


def session_fields(students, timestamps, gap):
    """Independent vector session assignment, including student resets."""
    n=len(timestamps)
    first=np.r_[True,students[1:]!=students[:-1]]
    boundary=first | np.r_[False,np.diff(timestamps)>=gap]
    global_sessions=np.cumsum(boundary,dtype=np.int64)
    start=np.flatnonzero(first); lengths=np.diff(np.r_[start,n])
    number=global_sessions-np.repeat(global_sessions[start]-1,lengths)
    counts=np.repeat(number[np.r_[start[1:]-1,n-1]],lengths)
    holdout=counts//5
    split=(number>counts-2*holdout).astype(np.uint8)+(number>counts-holdout).astype(np.uint8)
    position=np.arange(n)-np.maximum.accumulate(np.where(boundary,np.arange(n),0))+1
    return number,counts,split,position


def availability(timestamps, end, students):
    valid=np.isfinite(end[:-1]) & (students[1:]==students[:-1])
    overlap=valid & (end[:-1]>timestamps[1:])
    ties=(students[1:]==students[:-1]) & (timestamps[1:]==timestamps[:-1])
    return dict(ordered_predecessor_ends_after_next_start=int(overlap.sum()),
                adjacent_equal_start_timestamps=int(ties.sum()),
                nonfinite_end_times=int((~np.isfinite(end)).sum()),
                ends_before_own_start=int(np.count_nonzero(np.isfinite(end)&(end<timestamps))),
                maximum_adjacent_overlap_native_units=float(np.max(np.where(valid,end[:-1]-timestamps[1:],0),initial=0)))


def audit_flat(root, dataset, processed, stream, store):
    source=root/EXPECTED_RAW[dataset][0]
    columns=['studentId','problemId','skill','correct','startTime','endTime']
    if dataset=='assist2017': columns.append('action_num')
    raw=pd.read_csv(source,usecols=columns)
    assert not raw[[c for c in columns if c!='endTime']].isna().any().any()
    assert raw.correct.isin([0,1]).all()
    raw['source_row']=np.arange(1,len(raw)+1,dtype=np.int32)
    questions=np.sort(raw.problemId.unique())
    qmap=pd.read_csv(processed/'mappings/questions.csv')
    assert np.array_equal(qmap.source_question_id,questions)
    assert np.array_equal(qmap.question_id,np.arange(1,len(questions)+1))
    raw['question_id']=np.searchsorted(questions,raw.problemId.to_numpy())+1
    skills=pd.read_csv(processed/'mappings/composite_skills.csv')
    if dataset=='assist2017':
        tags=np.sort(raw.skill.unique())
        assert skills.canonical_skill_key.astype(str).tolist()==[str(t) for t in tags]
        raw['skill_id']=np.searchsorted(tags,raw.skill.to_numpy())+1
        assert not raw.duplicated(['studentId','action_num']).any()
        order=np.lexsort((raw.source_row.to_numpy(),raw.action_num.to_numpy(),raw.startTime.to_numpy(),raw.studentId.to_numpy()))
    else:
        pairs=raw[['problemId','skill']].drop_duplicates()
        by_question={q:tuple(sorted(int(v) for v in group.skill)) for q,group in pairs.groupby('problemId',sort=False)}
        keys=sorted(set(by_question.values()))
        assert skills.canonical_skill_key.astype(str).tolist()==[';'.join(map(str,k)) for k in keys]
        ids={k:i+1 for i,k in enumerate(keys)}
        raw['skill_id']=raw.problemId.map({q:ids[k] for q,k in by_question.items()})
        order=np.lexsort((raw.source_row.to_numpy(),raw.startTime.to_numpy(),raw.studentId.to_numpy()))
    raw=raw.iloc[order].reset_index(drop=True)
    source_students=raw.studentId.nunique()
    num,count,split,pos=session_fields(raw.studentId.to_numpy(),raw.startTime.to_numpy(),36000)
    keep=count>=5
    retained=raw.loc[keep].reset_index(drop=True)
    retained_students=np.sort(retained.studentId.unique())
    smap=pd.read_csv(processed/'mappings/students.csv')
    assert np.array_equal(smap.source_student_id,retained_students)
    student=np.searchsorted(retained_students,retained.studentId.to_numpy())+1
    expected=dict(student_id=student,question_id=retained.question_id.to_numpy(),skill_id=retained.skill_id.to_numpy(),
        correct=retained.correct.to_numpy(),timestamp=retained.startTime.to_numpy()*1000,
        source_row=retained.source_row.to_numpy(),source_event_id=retained.action_num.to_numpy() if dataset=='assist2017' else np.full(len(retained),-1),
        session_id=num[keep],session_position=pos[keep],split=split[keep])
    for feature,column in [('question','question_id'),('skill','skill_id')]:
        groups=pd.DataFrame({'student':student,'id':expected[column]})
        expected[f'{feature}_attempt_no']=groups.groupby(['student','id'],sort=False).cumcount().to_numpy()+1
    assert len(np.unique(expected['source_row']))==len(retained)
    stream.compare(expected); stream.finish()
    for name,column in [('question','question_id'),('skill','skill_id'),('correct','correct'),('timestamp_ms','timestamp')]:
        assert np.array_equal(getattr(store,name),expected[column]),name
    first=np.r_[True,np.diff(expected['session_id'])!=0] | np.r_[True,np.diff(student)!=0]
    assert np.array_equal(store.session_offsets[:-1],np.flatnonzero(first))
    for name,column in [('session_student','student_id'),('session_number','session_id'),('session_split','split')]:
        assert np.array_equal(getattr(store,name),expected[column][first])
    available=availability(retained.startTime.to_numpy(dtype=np.float64),retained.endTime.to_numpy(dtype=np.float64),student)
    return dict(raw_rows=len(raw),raw_students=int(source_students),retained_rows=len(retained),
                retained_students=len(retained_students),excluded_students=int(source_students)-len(retained_students),
                raw_to_parquet_all_fields='PASS',parquet_to_store_all_events='PASS',
                supervised_source_row_unique=True,source_mapping_policy='row skill' if dataset=='assist2017' else 'full-source question tag-set metadata (transductive, label-free)',
                publisher_release_identity='NOT VERIFIED: accepted user-supplied transformed CSV; no publisher-signed checksum/version chain',
                answer_availability_native_unit='seconds; endTime',answer_availability=available)


def audit_ednet(root,processed,stream,store):
    contents=root/'data/raw/ednet/archives/EdNet-Contents.zip'
    assert sha256(contents)=='aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7'
    with ZipFile(contents) as archive:
        with archive.open('contents/questions.csv') as file:
            metadata=list(csv.DictReader(io.TextIOWrapper(file,encoding='utf-8')))
    metadata.sort(key=lambda row:int(row['question_id'][1:]))
    tags={row['question_id']:None if row['tags']=='-1' else tuple(sorted(set(map(int,row['tags'].split(';'))))) for row in metadata}
    keys=sorted(set(v for v in tags.values() if v is not None))
    skillids={None:1,**{key:i+2 for i,key in enumerate(keys)}}
    lookup={row['question_id']:(i+1,skillids[tags[row['question_id']]],row['correct_answer']) for i,row in enumerate(metadata)}
    qmap=pd.read_csv(processed/'mappings/questions.csv')
    for row in qmap.itertuples():
        assert (row.question_id,row.skill_id,row.correct_answer)==lookup[row.question_key]
        expected='<UNTAGGED>' if tags[row.question_key] is None else ';'.join(map(str,tags[row.question_key]))
        assert row.canonical_skill_key==expected
    smap=pd.read_csv(processed/'mappings/students.csv')
    retained_map=dict(zip(smap.original_student_id,smap.student_id))
    summary=json.loads((processed/'reports/summary.json').read_text())
    raw_rows=answered_rows=unanswered_rows=retained_rows=retained_students=0
    cursor=0
    available=dict(ordered_predecessor_ends_after_next_start=0,adjacent_equal_start_timestamps=0,
        nonfinite_end_times=0,ends_before_own_start=0,maximum_adjacent_overlap_native_units=0.)
    started=last=time.perf_counter()
    with ZipFile(root/EXPECTED_RAW['ednet_kt1'][0]) as archive:
        members=sorted((int(p[5:-4]),p) for p in archive.namelist() if p.startswith('KT1/u') and p.endswith('.csv'))
        assert len(members)==len(set(i for i,p in members))==784309
        for ordinal,(original,member) in enumerate(members,1):
            with archive.open(member) as file:
                reader=csv.reader(io.TextIOWrapper(file,encoding='utf-8'))
                assert next(reader)==['timestamp','solving_id','question_id','user_answer','elapsed_time']
                records=[]
                for source_row,row in enumerate(reader,1):
                    assert len(row)==5 and row[3] in ('','a','b','c','d')
                    q,s,answer=lookup[row[2]]
                    t,solving,elapsed=int(row[0]),int(row[1]),int(row[4])
                    assert t>=0 and solving>=1 and elapsed>=0
                    raw_rows+=1
                    if row[3]=='':
                        unanswered_rows+=1
                    else:
                        records.append((t,source_row,q,s,int(row[3]==answer),solving,elapsed))
            records.sort(key=lambda item:(item[0],item[1]))
            answered_rows+=len(records)
            if not records:
                assert original not in retained_map
                continue
            values=np.asarray(records,dtype=np.int64)
            num,count,split,pos=session_fields(np.zeros(len(values),dtype=np.int64),values[:,0],36000000)
            if count[0]<5:
                assert original not in retained_map
                continue
            retained_students+=1
            assert retained_map[original]==retained_students
            n=len(values); student=np.full(n,retained_students,dtype=np.int32)
            expected=dict(student_id=student,question_id=values[:,2],skill_id=values[:,3],correct=values[:,4],
                timestamp=values[:,0],source_row=values[:,1],solving_id=values[:,5],elapsed_time=values[:,6],
                session_id=num,session_position=pos,split=split)
            for feature,column in [('question','question_id'),('skill','skill_id')]:
                counters={}; attempts=[]
                for key in expected[column]:
                    counters[key]=counters.get(key,0)+1; attempts.append(counters[key])
                expected[f'{feature}_attempt_no']=attempts
            assert len(np.unique(expected['source_row']))==n
            stream.compare(expected)
            for name,column in [('question','question_id'),('skill','skill_id'),('correct','correct'),('timestamp_ms','timestamp')]:
                assert np.array_equal(getattr(store,name)[cursor:cursor+n],expected[column]),name
            startsession=int(store.student_offsets[retained_students-1]); endsession=int(store.student_offsets[retained_students])
            starts=np.flatnonzero(pos==1)
            assert np.array_equal(store.session_offsets[startsession:endsession],cursor+starts)
            assert np.array_equal(store.session_number[startsession:endsession],num[starts])
            assert np.array_equal(store.session_split[startsession:endsession],split[starts])
            assert np.all(store.session_student[startsession:endsession]==retained_students)
            a=availability(values[:,0],values[:,0]+values[:,6],student)
            for key,value in a.items():
                available[key]=max(available[key],value) if key.startswith('maximum') else available[key]+value
            cursor+=n; retained_rows+=n
            if time.perf_counter()-last>25:
                print(json.dumps(dict(stage='ednet_raw_reconciliation',students_read=ordinal,total_students=len(members),raw_rows=raw_rows,retained_rows=retained_rows,seconds=time.perf_counter()-started)),flush=True)
                last=time.perf_counter()
    stream.finish()
    assert raw_rows==summary['raw_interactions']
    assert answered_rows==summary['supervised_interactions']
    assert unanswered_rows==summary['unanswered_interactions']
    assert retained_rows==summary['retained_interactions'] and retained_students==summary['retained_students']
    auditrows=sum(pq.ParquetFile(p).metadata.num_rows for p in (processed/'audit/unanswered_events').glob('*.parquet'))
    assert auditrows==unanswered_rows
    return dict(raw_rows=raw_rows,raw_students=len(members),answered_rows=answered_rows,
        unanswered_rows=unanswered_rows,unanswered_audit_count_match=True,retained_rows=retained_rows,
        retained_students=retained_students,raw_to_parquet_all_fields='PASS',parquet_to_store_all_events='PASS',
        all_zip_student_members_read_with_crc=True,supervised_source_row_unique_within_student=True,
        contents_zip_sha256=sha256(contents),answer_availability_native_unit='milliseconds; timestamp + elapsed_time proxy',
        answer_availability=available,
        publisher_release_identity='Official repository acquisition chain documented; local archives match recorded acquisition SHA256. No publisher-signed checksum available.')


def store_contract(store):
    offsets=store.session_offsets; student_offsets=store.student_offsets
    assert offsets[0]==0 and offsets[-1]==len(store.correct)
    assert np.all(np.diff(offsets)>0)
    assert student_offsets[0]==0 and student_offsets[-1]==len(store.session_number)
    assert np.all(np.diff(student_offsets)>=5)
    assert set(np.unique(store.correct))=={0,1}
    train_students=set(); val_students=set(); test_students=set()
    expected_targets={}; histories={}; target_counts={}; baseline_counts={}
    for split,name in enumerate(['train','validation','test']):
        ds=RollingSessionDataset(store,name,history_sessions=15)
        wanted=np.flatnonzero((store.session_split==split)&(store.session_number>1))
        assert np.array_equal(ds.targets,wanted)
        expected_targets[name]=len(wanted)
        target_counts[name]=int(np.sum(offsets[wanted+1]-offsets[wanted]))
        sessions=np.flatnonzero(store.session_split==split)
        baseline_counts[name]=int(np.sum(offsets[sessions+1]-offsets[sessions]))
        same_split=0
        for target in wanted:
            first=int(student_offsets[int(store.session_student[target])-1])
            actual=ds._history_bounds(int(target))
            assert actual==(max(first,int(target)-15),int(target))
            a,b=actual
            assert a>=first and b==target and b>a
            same_split+=int(np.any(store.session_split[a:b]==split))
        histories[name]=dict(all_history_bounds_checked=True,targets=len(wanted),targets_with_earlier_same_split_history=same_split)
    for i,(a,b) in enumerate(zip(student_offsets[:-1],student_offsets[1:]),1):
        a,b=int(a),int(b); n=b-a
        assert np.all(store.session_student[a:b]==i)
        assert np.array_equal(store.session_number[a:b],np.arange(1,n+1))
        holdout=n//5
        assert np.array_equal(store.session_split[a:b],np.r_[np.zeros(n-2*holdout,dtype=np.uint8),np.ones(holdout,dtype=np.uint8),np.full(holdout,2,dtype=np.uint8)])
        # All original time order/session boundaries were independently rebuilt above.
        train_students.add(i); val_students.add(i); test_students.add(i)
    first_session_targets=int(np.sum(offsets[student_offsets[:-1]+1]-offsets[student_offsets[:-1]]))
    return dict(status='PASS',split_protocol='chronological WITHIN-STUDENT, not student-disjoint',
        student_overlap_train_validation_test=len(train_students & val_students & test_students),
        supervised_targets=target_counts,baseline_supervised_targets=baseline_counts,
        first_session_interactions_context_only_for_hitskt=first_session_targets,
        all_post_first_session_targets_exactly_once=True,history_visibility=histories,
        answer_visibility='Earlier response labels within validation/test are visible in rolling online histories; no label from a later event index is visible.',
        identical_test_population=target_counts['test']==baseline_counts['test'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=EXPECTED_RAW)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if args.output.exists(): raise FileExistsError(args.output)
    root=Path.cwd(); processed=root/'data/processed'/args.dataset/'full'
    result=dict(status='RUNNING',dataset=args.dataset,started_utc=datetime.now(timezone.utc).isoformat(),
        implementation_sha256=sha256(__file__),scope='All raw rows and all processed/store rows, not a sample')
    write_json(args.output,result)
    started=time.perf_counter()
    try:
        source=root/EXPECTED_RAW[args.dataset][0]
        assert sha256(source)==EXPECTED_RAW[args.dataset][1]
        result['raw_source_sha256']=sha256(source)
        files=[]
        for folder in ['events','mappings','reports','audit','session_store']:
            for p in sorted((processed/folder).rglob('*')):
                if p.is_file(): files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha256(p)))
        result['processed_file_checksums']=files
        publication=json.loads((root/'reports/datasets'/args.dataset/'published_artifacts.json').read_text())
        for item in publication['files']:
            p=processed/item['source']
            assert p.exists() and sha256(p)==item['sha256'],str(p)
        result['recorded_published_artifact_hashes']='PASS'
        stream=EventStream(processed); store=SessionStore(processed/'session_store')
        lineage=audit_ednet(root,processed,stream,store) if args.dataset=='ednet_kt1' else audit_flat(root,args.dataset,processed,stream,store)
        result.update(source_lineage=lineage,store_contract=store_contract(store),status='PASS',
            completion_utc=datetime.now(timezone.utc).isoformat(),runtime_seconds=time.perf_counter()-started)
        # PASS covers reconstruction, not the separately stated release-identity
        # or real-world response-availability limitations.
        write_json(args.output,result)
        print(json.dumps(dict(dataset=args.dataset,status=result['status'],lineage=lineage,store_contract=result['store_contract']),indent=2),flush=True)
    except Exception as error:
        result.update(status='NOT VERIFIED' if isinstance(error,FileNotFoundError) else 'FAIL',
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        write_json(args.output,result)
        raise


if __name__=='__main__': main()
