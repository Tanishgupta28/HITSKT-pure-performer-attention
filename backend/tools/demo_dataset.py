"""Generate a reproducible synthetic dataset; optionally seed a localhost-only demo.

No benchmark IDs, real learners, training claims, or private response logs.
"""
import argparse
import csv
import hashlib
import json
import random
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))

from app.catalog import BANK, CONCEPTS
from app.config import Settings
from app.content import original, key
from app.engine import concept_rows, initial_concept, recommendations, update_knowledge

SEED=42
VERSION='lumen-synthetic-v1'
STUDENT_COUNT=24
PASSWORD='Lumen12345'
DOMAIN='demo.lumen.example.com'
BASE=datetime(2026,11,17,tzinfo=timezone.utc)
FIELDS=['student_id','session_id','question_id','question_version','skill_id','topic_id','response','correct',
        'response_time_ms','attempt_timestamp','hints_used','mode','source','split']
NAMES=['Asha Shah','Rohan Mehta','Nia Patel','Omar Jain','Dev Malik','Ira Sen','Sana Roy','Neil Shah',
       'Arya Rao','Riya Bose','Kabir Das','Nikhil Joshi','Aditi Pai','Tanvi Shah','Esha Rao','Veer Jain',
       'Rehan Sen','Maya Patel','Diya Jain','Sia Roy','Arun Pai','Ishan Rao','Neel Das','Jiya Sen']


def uid(key): return str(uuid5(NAMESPACE_URL, f'https://lumen.example/{VERSION}/{key}'))


def question_set(j):
    return [original(BANK[f'{c["id"]}-{1+j%3}-{1+j*3+i}']) for c in CONCEPTS for i in range(3)]


def histories(anchor):
    rng=random.Random(SEED)
    for i,name in enumerate(NAMES):
        sessions=[]
        for j in range(4 if i<18 else 0):
            qs=question_set(j)
            start=anchor-timedelta(days=8-2*j,hours=i%5)
            items=[]
            for k,q in enumerate(qs):
                if j<2 and i in (0,1):
                    counts=([1,1,3,3] if i==0 else [3,3,1,1])[k//3]
                    correct=k%3<counts
                else:
                    # Simulated likelihood, not a pedagogical or model benchmark.
                    correct=rng.random() < min(.91,.42+((i+k//3)%5)*.085+.05*j)
                response=q['answer_index'] if correct else (q['answer_index']+rng.randint(1,3))%4
                latency=rng.randint(5500,39500)
                stamp=start+timedelta(seconds=43*k,milliseconds=latency)
                items.append((q,response,correct,latency,stamp,int(j>=2 and not correct and k%3==0)))
            sessions.append((j,start,items))
        yield i,name,sessions


def dataset(out):
    out.mkdir(parents=True,exist_ok=True)
    csv_path=out/'interactions.csv'
    split_counts={"train":0,"validation":0,"test":0}
    students=set(); sessions=set()
    with csv_path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS,lineterminator='\n'); writer.writeheader()
        for i,_,records in histories(BASE):
            split='train' if i<12 else 'validation' if i<15 else 'test'
            for j,_,items in records:
                student=f'synthetic-{i+1:03d}'; session=f'{student}-session-{j+1}'
                students.add(student); sessions.add(session)
                for q,response,correct,latency,stamp,hints in items:
                    writer.writerow(dict(student_id=student,session_id=session,question_id=q['id'],question_version=q['content_version'],
                        skill_id=q['skill_id'],topic_id=q['concept_id'],response=response,correct=int(correct),response_time_ms=latency,
                        attempt_timestamp=stamp.isoformat(),hints_used=hints,mode='diagnostic' if j==0 else 'fixed_paper' if j==1 else 'practice',
                        source='synthetic',split=split))
                    split_counts[split]+=1
    with (out/'questions.csv').open('w',newline='',encoding='utf-8') as file:
        fields=['question_id','topic_id','skill_id','difficulty','prompt','option_a','option_b','option_c','option_d','answer_index','explanation','version']
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n'); writer.writeheader()
        for bank in (question_set(j) for j in range(4)):
            for q in bank:
                writer.writerow(dict(zip(fields,[q['id'],q['topic_id'],q['skill_id'],q['difficulty'],q['prompt'],*q['choices'],q['answer_index'],q['explanation'],q['content_version']])))
    manifest={'version':VERSION,'seed':SEED,'source':'synthetic','reference_timestamp':BASE.isoformat(),
              'learners_with_attempts':len(students),'sessions':len(sessions),'attempts':sum(split_counts.values()),
              'attempts_by_split':split_counts,'split_unit':'student','fields':FIELDS,
              'sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('interactions.csv','questions.csv')},
              'usage':'Demo and pipeline validation only; no learner data, learning outcome claim, benchmark mapping, or model evidence.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def seed_database(settings):
    from pymongo import MongoClient
    from pwdlib import PasswordHash
    if not settings.demo_enabled or urlparse(settings.mongodb_uri).hostname not in {'127.0.0.1','localhost','::1'} or urlparse(settings.frontend_origin).hostname not in {'127.0.0.1','localhost','::1'}:
        raise SystemExit('Set DEMO_ENABLED=true and use loopback MongoDB and frontend addresses to seed the local demo.')
    anchor=datetime.now(timezone.utc)
    secret=PasswordHash.recommended().hash(PASSWORD)
    other=PasswordHash.recommended().hash(secrets.token_urlsafe(28))
    with MongoClient(settings.mongodb_uri,serverSelectionTimeoutMS=5000,tz_aware=True) as connection:
        db=connection[settings.mongodb_database]; db.command('ping')
        db.users.create_index('email',unique=True)
        teacher=uid('teacher'); admin=uid('admin'); cls=uid('class-8b'); paper=uid('exam-paper')
        def account(id,name,handle,role,allowed):
            address=f'{handle}@{DOMAIN}'
            row=db.users.find_one({'email':address},{'_id':1,'demo':1})
            if row and (not row.get('demo') or row['_id']!=id): raise SystemExit(f'Reserved demo email overlaps an existing account: {address}.')
            db.users.update_one({'_id':id},{'$setOnInsert':{'name':name,'email':address,'role':role,'demo':True,
                 'created_at':anchor,'onboarding_complete':True,'password_hash':secret if allowed else other,
                 'profile':{'grade_level':'Grade 8','curriculum':'CBSE','weekly_question_target':18},'profile_source':'sample'}},upsert=True)
        account(teacher,'Mira Rao','teacher','teacher',True); account(admin,'Lumen Content','admin','admin',True)
        members={}
        for i,name,sessions in histories(anchor):
            id=uid(f'student-{i}'); members[id]={'status':'active','joined_at':anchor-timedelta(days=14)}
            account(id,name,'student' if i==0 else 'rohan' if i==1 else f'learner{i+1}','student',i<2)
            state={'_id':f'{id}:mathematics','user_id':id,'demo':True,'subject_id':'mathematics','version':0,
                   'history':[],'fixed':{},'assessments':[],'concepts':{c['id']:initial_concept() for c in CONCEPTS}}
            for j,start,items in sessions:
                if i<2 and j>1: continue
                aid=uid(f'learner-{i}-session-{j}'); mode='diagnostic' if j==0 else 'exam' if j==1 else 'practice'
                rows=[]; answers={}; shown=[]
                for k,(q,response,correct,latency,t,hints) in enumerate(items):
                    c=q['concept_id']; stamp=t.isoformat(); shown.append(q)
                    state['concepts'][c]=update_knowledge(state['concepts'][c],correct,q['difficulty'],stamp)
                    record={'question_id':q['id'],'assessment_id':aid,'concept_id':c,'skill_id':q['skill_id'],'content_version':q['content_version'],
                        'difficulty':q['difficulty'],'correct':correct,'choice_index':response,'answered_at':stamp,'response_time_ms':latency,
                        'hints_used':hints,'source':'synthetic','mode':mode,'request_id':uid(f'{aid}-{k}'),
                        'prediction':{'provider':'not_scored','probability':None}}
                    rows.append(record); answers[str(k)]={'value':response,'saved_at':stamp,'response_time_ms':latency}
                state['history']+=rows
                finished=items[-1][4].isoformat()
                if j<2:
                    state['fixed'][aid]={'id':aid,'mode':mode,'title':'Mathematics diagnostic' if j==0 else 'Grade 8 unit check · released',
                        'assignment_id':None if j==0 else paper,'questions':shown,'started_at':start.isoformat(),'submitted_at':finished,
                        'deadline_at':None,'index':11,'answers':answers,'opened':{},'incorporated':True,
                        'rules':{'review_allowed':True,'skips_allowed':True,'show_explanations':True},
                        'summary':{'concepts':concept_rows(state),'recommendations':recommendations(state)}}
                else:
                    state['assessments'].append({'id':aid,'mode':'practice','status':'completed','started_at':start.isoformat(),'completed_at':finished,
                        'responses':rows,'pending':None,'total':len(rows),'summary':{'accuracy':sum(r['correct'] for r in rows)/len(rows),
                        'average_mastery':sum(c['mastery'] for c in state['concepts'].values())/len(CONCEPTS)}})
            db.learning.update_one({'_id':state['_id']},{'$setOnInsert':{k:v for k,v in state.items() if k!='_id'}},upsert=True)
        for id,name,code,roster in ((cls,'8B Mathematics','LUMEN08B',members),(uid('class-8a'),'8A Mathematics','LUMEN08A',{})):
            db.classes.update_one({'_id':id},{'$setOnInsert':{'name':name,'teacher_id':teacher,'demo':True,'join_code':code,'members':roster,
                 'grade_level':'Grade 8','curriculum':'CBSE','created_at':anchor,'subject_id':'mathematics'}},upsert=True)
        for id,status,title,items in ((paper,'released','Grade 8 unit check · released',question_set(1)),
              (uid('exam-new'),'assigned','Grade 8 unit check',question_set(2)),(uid('exam-draft'),'draft','Percentage and fractions review',question_set(0)[:6])):
            row={'teacher_id':teacher,'class_id':cls,'class_name':'8B Mathematics','demo':True,'status':status,'title':title,
                 'revision':0,'question_ids':[q['id'] for q in items],'questions':items if status!='draft' else [],'duration_minutes':20,
                 'review_allowed':True,'skips_allowed':True,'show_explanations':True,'created_at':anchor,'assigned_at':anchor,
                 'extra_minutes':{},'opens_at':None,'closes_at':None if status=='draft' else anchor+timedelta(days=30),
                 'closed_at':anchor-timedelta(days=5) if status=='released' else None}
            db.papers.update_one({'_id':id},{'$setOnInsert':row},upsert=True)
        for id,prompt,choices,correct,explanation in (
          ('FRA-018','What is 1/2 + 1/3?',['2/3','2/5','5/6','1/6'],2,'The denominators describe part size. Rewrite 1/2 as 3/6 and 1/3 as 2/6. Add the numerators: 3/6 + 2/6 = 5/6.'),
          ('FRA-019','What is 1/3 + 1/4?',['2/7','1/12','7/12','5/12'],2,'Rewrite 1/3 as 4/12 and 1/4 as 3/12. Add the numerators: 4/12 + 3/12 = 7/12.')):
            q={'id':id,'subject_id':'mathematics','topic_id':'fractions','concept_id':'fractions','skill_id':'fractions-foundations',
               'question_type':'mcq','difficulty':1,'choices':choices,'answer_index':correct,'prompt':prompt,'explanation':explanation,
               'hint':'Find a common denominator shared by both fractions.', 'status':'published','version':1,'content_version':f'{id}:v1','origin':'authored'}
            db.questions.update_one({'_id':key(True,id)},{'$setOnInsert':{'demo':True,'id':id,'status':'published','version':1,'revision':0,'live':q}},upsert=True)
    print(f'Local demo ready: student, rohan, teacher, admin @{DOMAIN}. Existing accounts and attempts preserved.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'backend'/'data'/'lumen-synthetic-v1')
    parser.add_argument('--seed-local',action='store_true')
    parser.add_argument('--seed-only',action='store_true',help='Seed MongoDB without rewriting dataset files.')
    args=parser.parse_args()
    if not args.seed_only:
        m=dataset(args.output); print(f'{m["attempts"]} synthetic attempts / {m["sessions"]} sessions / {m["learners_with_attempts"]} learners')
    if args.seed_local or args.seed_only: seed_database(Settings())
