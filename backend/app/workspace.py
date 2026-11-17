"""Role boundaries, classroom operations, research replay, and pseudonymous exports."""
import csv
import hashlib
import hmac
import io
import json
import secrets
import smtplib
from collections import Counter
from datetime import timedelta
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import urlencode, urlparse
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import EmailStr, Field
from pymongo.errors import DuplicateKeyError
from starlette.concurrency import run_in_threadpool

from app.catalog import BANK, CONCEPTS
from app.config import ROOT
from app.content import mount_content, realm, scope
from app.engine import bkt_probability, initial_concept, update_knowledge
from app.lab import DEFAULT_CHECKPOINT, run_hitskt_fixture
from app.papers import Strict, clock, grade, mount_papers, parse, result_for

BENCH = ROOT / 'reports' / 'final_results.csv'
SMOKE = ROOT / 'reports' / 'platform' / 'inference_smoke.json'
AUDIT = ROOT / 'reports' / 'supervisor_verification' / 'supervisor_facts.json'
EXPORT_FIELDS = ['student_id','session_id','question_id','question_version','skill_id','topic_id','response',
                 'correct','response_time_ms','attempt_timestamp','hints_used','mode','source']


class ClassDraft(Strict):
    name: str = Field(min_length=2,max_length=100)
    grade_level: str = Field(min_length=1,max_length=40)
    curriculum: str = Field(default='',max_length=80)


class Enroll(Strict):
    email: EmailStr


class Join(Strict):
    code: str = Field(pattern=r'^[A-Za-z0-9]{8}$')


class Recovery(Strict):
    email: EmailStr


class Reset(Strict):
    token: str = Field(min_length=30,max_length=120)
    password: str = Field(min_length=10,max_length=128)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def class_view(c):
    return {k:c[k] for k in ('_id','name','grade_level','curriculum','join_code')} | {
        'student_count':sum(m['status']=='active' for m in c['members'].values())}


def grade_record(state,since):
    fixed = [a for a in state.get('fixed',{}).values() if a['mode']=='diagnostic' and a['submitted_at'] and parse(a['started_at'])>=since]
    if fixed:
        return result_for(sorted(fixed,key=lambda a:a['submitted_at'])[-1],False)
    legacy = [a for a in state.get('assessments',[]) if a['mode']=='diagnostic' and a['status']=='completed' and parse(a['started_at'])>=since]
    if not legacy: return None
    a=legacy[-1]; counts=Counter(); good=Counter()
    for r in a['responses']: counts[r['concept_id']]+=1; good[r['concept_id']]+=r['correct']
    return {'score':sum(good.values()),'total':a['total'],'topics':[{'concept_id':k,'correct':good[k],'shown':v,'answered':v} for k,v in counts.items()]}


def field_error(r):
    if not r.get('skill_id',r.get('concept_id')): return 'Missing skill ID'
    if not r.get('assessment_id') or not r.get('question_id'): return 'Missing question or session ID'
    if not isinstance(r.get('correct'),bool): return 'Invalid correctness'
    ms=r.get('response_time_ms')
    if not isinstance(ms,int) or ms < 0: return 'Invalid duration'
    try:
        if not parse(r.get('answered_at')) or not parse(r['answered_at']).tzinfo: return 'Invalid attempt timestamp'
    except (ValueError, TypeError): return 'Invalid attempt timestamp'
    return None


def dataset_row(state,r,secret):
    def alias(prefix,value):
        return prefix+'-'+hmac.new(secret,str(value).encode(),'sha256').hexdigest()[:24]
    value=r.get('choice_index')
    # Encode numeric/option values as JSON; prefix spreadsheet formula-like strings.
    value=json.dumps(value,ensure_ascii=False)
    return {'student_id':alias('p',state['user_id']), 'session_id':alias('s',r.get('assessment_id')),
            'question_id':r.get('question_id'),'question_version':r.get('content_version','unknown'),
            'skill_id':r.get('skill_id',r.get('concept_id')),'topic_id':r.get('concept_id'),'response':value,
            'correct':r.get('correct'),'response_time_ms':r.get('response_time_ms'),'attempt_timestamp':r.get('answered_at'),
            'hints_used':r.get('hints_used',0),'mode':r.get('mode','legacy'),'source':'synthetic' if state.get('demo') else 'live'}


def collected_interactions(st):
    yield from st.get('history',[])
    # Held results remain invisible to students; admins may inspect submitted
    # records by pseudonym before release without writing them to a BKT history.
    for a in st.get('fixed',{}).values():
        if not a.get('submitted_at') or a.get('incorporated'):
            continue
        for i,saved in a['answers'].items():
            if 'value' not in saved: continue
            q=a['questions'][int(i)]
            yield {'assessment_id':a['id'],'question_id':q['id'],'concept_id':q['concept_id'],'skill_id':q.get('skill_id',q['concept_id']),
                   'content_version':q['content_version'],'choice_index':saved['value'],'correct':grade(q,saved['value']),
                   'response_time_ms':saved['response_time_ms'],'answered_at':saved['saved_at'],'hints_used':0,'mode':a['mode']}


def smtp_send(settings,address,token):
    msg=EmailMessage()
    msg['Subject']='Reset your Lumen password'; msg['From']=settings.smtp_from; msg['To']=address
    msg.set_content('Open this link to reset your password within 30 minutes. If you did not request this, ignore the email.\n\n'+
                    settings.frontend_origin.rstrip('/')+'/reset?'+urlencode({'token':token}))
    try:
        with smtplib.SMTP(settings.smtp_host,settings.smtp_port,timeout=10) as server:
            if settings.smtp_starttls: server.starttls()
            if settings.smtp_user: server.login(settings.smtp_user,settings.smtp_password or '')
            server.send_message(msg)
    except (OSError,smtplib.SMTPException):
        # Avoid echoing credentials, recipient addresses, or reset tokens into logs.
        import logging
        logging.getLogger(__name__).warning('Password recovery email could not be delivered.')


def mount_workspace(app,settings,auth,student,learning,save,limit_auth,passwords):
    lab_secret=secrets.token_bytes(32)
    async def teacher(user=Depends(auth)):
        if user.get('role')!='teacher': raise HTTPException(403,'Teacher access is required.')
        return user

    async def admin(user=Depends(auth)):
        if user.get('role')!='admin': raise HTTPException(403,'Administrator access is required.')
        return user

    async def get_class(class_id,user):
        c=await app.state.db.classes.find_one({'_id':class_id,'teacher_id':user['_id'],**scope(user)})
        if not c: raise HTTPException(404,'Class not found.')
        return c

    @app.get('/api/config')
    async def public_config():
        return {'demo_enabled':settings.demo_enabled and urlparse(settings.frontend_origin).hostname in {'localhost','127.0.0.1','::1'},
                'email_recovery':bool(settings.smtp_host and settings.smtp_from)}

    @app.post('/api/auth/recovery')
    async def recovery(data:Recovery,request:Request,tasks:BackgroundTasks):
        await limit_auth(request)
        if not settings.smtp_host or not settings.smtp_from:
            raise HTTPException(503,'Email recovery is not set up. Contact your administrator for help signing in.')
        token=secrets.token_urlsafe(36)
        user=await app.state.db.users.find_one_and_update({'email':str(data.email).lower(),'inactive':{'$ne':True},'demo':{'$ne':True}},
            {'$set':{'reset_hash':digest(token),'reset_until':clock()+timedelta(minutes=30)}})
        if user: tasks.add_task(smtp_send,settings,user['email'],token)
        return {'message':'If the account exists, a reset link will be sent.'}

    @app.post('/api/auth/reset')
    async def reset(data:Reset,request:Request):
        await limit_auth(request)
        password_hash=await run_in_threadpool(passwords.hash,data.password)
        db=app.state.db
        user=await db.users.find_one_and_update({'reset_hash':digest(data.token),'reset_until':{'$gt':clock()},'inactive':{'$ne':True}},
            {'$set':{'password_hash':password_hash},'$unset':{'reset_hash':'','reset_until':''}})
        if not user: raise HTTPException(422,'This reset link is no longer valid. Request a new link.')
        await db.sessions.delete_many({'user_id':user['_id']})
        return {'message':'Password reset. Sign in with your new password.'}

    @app.get('/api/teacher/classes')
    async def classes(user=Depends(teacher)):
        return [class_view(c) for c in await app.state.db.classes.find({'teacher_id':user['_id'],**scope(user)}).sort('name',1).to_list(200)]

    @app.post('/api/teacher/classes',status_code=201)
    async def create_class(body:ClassDraft,user=Depends(teacher)):
        row={**body.model_dump(),'_id':str(uuid4()),'teacher_id':user['_id'],'demo':realm(user),'subject_id':'mathematics',
             'members':{},'created_at':clock()}
        for _ in range(4):
            row['join_code']=secrets.token_hex(4).upper()
            try:
                await app.state.db.classes.insert_one(row)
                return class_view(row)
            except DuplicateKeyError: pass
        raise HTTPException(503,'Could not allocate a class code. Try again.')

    async def enroll(c,learner):
        id=learner['_id']; db=app.state.db
        if len(c['members'])>=250 and id not in c['members']: raise HTTPException(409,'This class is full.')
        active=f'members.{id}.status'
        result=await db.classes.update_one({'_id':c['_id'],active:{'$ne':'active'}},
            {'$set':{f'members.{id}':{'status':'active','joined_at':clock()}}})
        if not result.modified_count: raise HTTPException(409,'This student is already enrolled.')
        return {'class_id':c['_id'],'name':c['name']}

    @app.post('/api/teacher/classes/{class_id}/enroll')
    async def add_student(class_id:str,data:Enroll,user=Depends(teacher)):
        c=await get_class(class_id,user)
        learner=await app.state.db.users.find_one({'email':str(data.email).lower(),'inactive':{'$ne':True},
                    'role':{'$in':['student',None]},**scope(user)})
        if not learner: raise HTTPException(404,'No active student account matches that email.')
        return await enroll(c,learner)

    @app.post('/api/student/classes/join')
    async def join(data:Join,request:Request,user=Depends(student)):
        await limit_auth(request)
        c=await app.state.db.classes.find_one({'join_code':data.code.upper(),**scope(user)})
        if not c: raise HTTPException(404,'Class code not found. Ask your teacher to check it.')
        return await enroll(c,user)

    @app.get('/api/student/classes')
    async def student_classes(user=Depends(student)):
        result=await app.state.db.classes.find({f'members.{user["_id"]}.status':'active',**scope(user)},{'name':1,'grade_level':1}).to_list(250)
        return result

    @app.post('/api/teacher/classes/{class_id}/remove/{student_id}')
    async def remove(class_id:str,student_id:str,user=Depends(teacher)):
        c=await get_class(class_id,user)
        if student_id not in c['members']: raise HTTPException(404,'Enrollment not found.')
        await app.state.db.classes.update_one({'_id':class_id},{'$set':{f'members.{student_id}.status':'inactive'}})
        return {'status':'inactive'}

    @app.post('/api/student/classes/{class_id}/leave')
    async def leave(class_id:str,user=Depends(student)):
        key=f'members.{user["_id"]}.status'
        result=await app.state.db.classes.update_one({'_id':class_id,key:'active',**scope(user)},{'$set':{key:'inactive'}})
        if not result.modified_count: raise HTTPException(404,'Enrollment not found.')
        return {'status':'inactive'}

    def learner_data(user, st, member):
        diag=grade_record(st,member['joined_at']) if member['status']=='active' else None
        history=[r for r in st.get('history',[]) if parse(r['answered_at'])>=member['joined_at']] if member['status']=='active' else []
        return {'id':user['_id'],'name':user['name'],'membership':member['status'],
                'diagnostic':{k:diag[k] for k in ('score','total','topics')} if diag else None,
                'attempts':len(history),'last_activity':max((r['answered_at'] for r in history),default=None)}

    @app.get('/api/teacher/classes/{id}')
    async def roster(id:str,user=Depends(teacher)):
        c=await get_class(id,user); db=app.state.db
        learners=await db.users.find({'_id':{'$in':list(c['members'])},**scope(user)},{'name':1}).sort('name',1).to_list(250)
        states={s['user_id']:s for s in await db.learning.find({'user_id':{'$in':list(c['members'])}}).to_list(250)}
        rows=[learner_data(s,states.get(s['_id'],{}),c['members'][s['_id']]) for s in learners]
        completed=[s for s in rows if s['membership']=='active' and s['diagnostic']]
        counts=Counter(); correct=Counter(); coverage=Counter()
        for s in completed:
            diag=s['diagnostic']; coverage[diag['score']]+=1
            for t in diag['topics']: counts[t['concept_id']]+=t['answered']; correct[t['concept_id']]+=t['correct']
        return {'class':class_view(c),'students':rows,'diagnostic_count':len(completed),
                'topics':[{'concept_id':k,'correct':correct[k],'total':n} for k,n in counts.items()],
                'distribution':[{'low':n,'high':min(n+2,12),'count':sum(coverage[i] for i in range(n,n+3))} for n in (0,3,6,9,12)]}

    @app.get('/api/teacher/classes/{class_id}/students/{student_id}')
    async def learner_detail(class_id:str,student_id:str,user=Depends(teacher)):
        c=await get_class(class_id,user)
        member=c['members'].get(student_id)
        if not member or member['status']!='active': raise HTTPException(404,'Active enrollment not found.')
        db=app.state.db
        learner=await db.users.find_one({'_id':student_id,**scope(user)},{'name':1})
        if not learner: raise HTTPException(404,'Student not found.')
        st=await db.learning.find_one({'_id':f'{student_id}:mathematics'}) or {}
        joined=member['joined_at']
        history=[r for r in st.get('history',[]) if parse(r['answered_at'])>=joined]
        sessions={}
        for r in history:
            v=sessions.setdefault(r['assessment_id'],{'id':r['assessment_id'],'attempts':0,'correct':0,'hints':0,'mode':r.get('mode','learning'),'date':r['answered_at']})
            v['attempts']+=1; v['correct']+=r['correct']; v['hints']+=r.get('hints_used',0)
        return {**learner_data(learner,st,member),'history':sorted(sessions.values(),key=lambda s:s['date'],reverse=True)[:50]}

    def stream_states(user):
        return app.state.db.learning.find(scope(user),{'user_id':1,'history':1,'assessments.id':1,'assessments.started_at':1,'fixed':1,'demo':1})

    @app.get('/api/admin/interactions')
    async def interactions(source:str='all',limit:int=Query(50,ge=1,le=100),user=Depends(admin)):
        history=[]; counts=Counter(); sessions=set(); secret=secrets.token_bytes(32)
        async for st in stream_states(user):
            for a in st.get('assessments',[]): sessions.add(a['id'])
            for id in st.get('fixed',{}): sessions.add(id)
            for r in collected_interactions(st):
                err=field_error(r); counts['attempts']+=1; counts['invalid']+=int(bool(err)); counts['correct']+=int(r.get('correct') is True); counts['incorrect']+=int(r.get('correct') is False)
                if source=='invalid' and not err: continue
                history.append({**dataset_row(st,r,secret),'validation':err or 'Valid'})
                if len(history)>limit*2:
                    history=sorted(history,key=lambda r:r['attempt_timestamp'] or '',reverse=True)[:limit]
        students=await app.state.db.users.count_documents({'role':{'$in':['student',None]},**scope(user)})
        return {'students':students,'sessions':len(sessions),'attempts':counts['attempts'],'correct':counts['correct'],
                'incorrect':counts['incorrect'],'invalid':counts['invalid'],'demo':realm(user),
                'rows':sorted(history,key=lambda r:r['attempt_timestamp'] or '',reverse=True)[:limit], 'collection_status':'active'}

    @app.get('/api/admin/interactions/schema')
    async def manifest(user=Depends(admin)):
        return {'version':'lumen-interactions-v1','fields':EXPORT_FIELDS,'source':'synthetic' if realm(user) else 'live',
                'privacy':'Student and session aliases rotate per export. Only valid records are exported. No direct identifiers.',
                'research_note':'Platform IDs are not dataset question IDs for a trained benchmark checkpoint.'}

    @app.post('/api/admin/interactions/export')
    async def export(days:int=Query(30,ge=1,le=365),user=Depends(admin)):
        secret=secrets.token_bytes(32); start=clock()-timedelta(days=days)
        await app.state.db.export_audit.insert_one({'actor_id':user['_id'],'at':clock(),'days':days,'demo':realm(user),'schema':'lumen-interactions-v1'})
        async def generate():
            buf=io.StringIO(); writer=csv.DictWriter(buf,fieldnames=EXPORT_FIELDS)
            writer.writeheader(); yield buf.getvalue(); buf.seek(0); buf.truncate(0)
            async for st in stream_states(user):
                for r in collected_interactions(st):
                    if field_error(r) or parse(r['answered_at'])<start: continue
                    writer.writerow(dataset_row(st,r,secret))
                    yield buf.getvalue(); buf.seek(0); buf.truncate(0)
        return StreamingResponse(generate(),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="lumen-{ "synthetic" if realm(user) else "interactions"}.csv"'})

    @app.get('/api/admin/model')
    async def model_overview(user=Depends(admin)):
        rows=[]
        audit=[]
        if BENCH.is_file():
            with BENCH.open(encoding='utf-8',newline='') as src:
                rows=[{k:r.get(k) for k in ('dataset','model','test_auc','test_accuracy','test_precision','test_recall','best_epoch')} for r in csv.DictReader(src)]
        fixture=SMOKE.is_file()
        if AUDIT.is_file():
            facts=json.loads(AUDIT.read_text(encoding='utf-8'))
            audit=[{'dataset':key,'cpu_auc':row['independent_cpu_metrics']['roc_auc'],
                    'stored_auc':row['historical_cuda_metrics']['roc_auc'],'auc_gate':row['historical_to_cpu_auc_gate'],
                    'metrics_gate':row['historical_to_cpu_metric_gate'],'cuda_replay':row['cuda_replay']}
                   for key,row in facts['datasets'].items()]
        return {'bkt':{'status':'live_practice','version':'lumen-bkt-v1'},'hitskt':{'status':'research_only','saved_fixture':fixture,
                 'checkpoint_present': Path(settings.lab_checkpoint or DEFAULT_CHECKPOINT).is_file(), 'message':'Requires reviewed item mappings and matching data for live use.'},
                 'evaluation':{'source':'historical repository benchmark','rows':rows,'audit':audit,'evaluation_date':None,'live_pilot':False}}

    def learner_ref(id):
        return hmac.new(lab_secret,str(id).encode(),'sha256').hexdigest()[:22]

    async def lab_histories(user):
        if not realm(user): return []
        return await app.state.db.learning.find({'demo':True,'history.12':{'$exists':True}},
              {'user_id':1,'history':1}).sort('user_id',1).to_list(50)

    @app.get('/api/admin/model/replay-learners')
    async def replay_learners(user=Depends(admin)):
        return [{'ref':learner_ref(s['user_id']),'label':f'Synthetic learner {i+1:02d}','attempts':len(s['history'])}
                for i,s in enumerate(await lab_histories(user))]

    async def replay_data(model,ref,user):
        if ref:
            if model!='bkt': raise HTTPException(422,'Synthetic Lumen histories are available for BKT replay only.')
            states=await lab_histories(user)
            learner=next((s for s in states if secrets.compare_digest(learner_ref(s['user_id']),ref)),None)
            if not learner: raise HTTPException(404,'This sequence is not available.')
            history=sorted(learner['history'],key=lambda r:r['answered_at'])
            target=history[-1]; previous=history[:-1][-12:]
            sessions={}
            def row(r):
                session=sessions.setdefault(r['assessment_id'],len(sessions)+1)
                return {'session':session,'question_id':r['question_id'],'skill_id':r.get('skill_id',r['concept_id']),
                        'correct':r['correct'],'concept_id':r['concept_id'],'difficulty':r['difficulty']}
            return {'source':'Recorded Lumen demo history · synthetic','rows':[row(r) for r in previous],'next':row(target)}
        if model=='hitskt':
            if not SMOKE.is_file(): raise HTTPException(503,'No verified HiTSKT fixture is available.')
            return {'source':'EdNet-KT1 synthetic parity fixture','rows':[{'session':1,'question_id':'1','skill_id':'2','correct':True},
                     {'session':2,'question_id':'2','skill_id':'75','correct':False}], 'next':{'session':2,'question_id':'3','skill_id':'65','correct':True}}
        if model!='bkt': raise HTTPException(404,'Model not found.')
        return {'source':'Lumen synthetic model fixture','rows':[{'session':1,'question_id':'fractions-1-1','skill_id':'fractions','concept_id':'fractions','difficulty':1,'correct':False},
                 {'session':2,'question_id':'fractions-1-2','skill_id':'fractions','concept_id':'fractions','difficulty':1,'correct':True}],
                 'next':{'session':2,'question_id':'fractions-2-1','skill_id':'fractions','concept_id':'fractions','difficulty':2,'correct':True}}

    def receipt(user,model,ref,seq):
        key=f'{user["_id"]}:{model}:{ref}:'+json.dumps(seq,sort_keys=True)
        return hmac.new(lab_secret,key.encode(),'sha256').hexdigest()

    @app.get('/api/admin/model/sequence/{model}')
    async def sequence(model:str,ref:str=Query('',max_length=35),user=Depends(admin)):
        seq=await replay_data(model,ref,user)
        return {**seq,'next':{k:v for k,v in seq['next'].items() if k!='correct'}}

    @app.post('/api/admin/model/sequence/{model}/predict')
    async def predict(model:str,ref:str=Query('',max_length=35),user=Depends(admin)):
        seq=await replay_data(model,ref,user)
        authorize=receipt(user,model,ref,seq)
        if model=='hitskt':
            with SMOKE.open(encoding='utf-8') as src: evidence=json.load(src)
            path=Path(settings.lab_checkpoint or DEFAULT_CHECKPOINT)
            if path.is_file():
                try:
                    p=await run_in_threadpool(run_hitskt_fixture,path,evidence['checkpoint_sha256'])
                    return {'probability':p,'status':'computed_from_verified_checkpoint','checkpoint':evidence['checkpoint_sha256'],'source':seq['source'],'reveal_key':authorize}
                except (ImportError,OSError,ValueError,RuntimeError):
                    pass
            return {'probability':evidence['probability'],'status':'saved_verified_checkpoint_output','checkpoint':evidence['checkpoint_sha256'],
                    'source':seq['source'],'reveal_key':authorize}
        estimates={}
        for row in seq['rows']:
            c=row['concept_id']; estimates[c]=update_knowledge(estimates.get(c,initial_concept()),row['correct'],row['difficulty'],'fixture')
        return {'probability':bkt_probability(seq['next'],estimates),'status':'computed_now','source':seq['source'],'reveal_key':authorize}

    @app.get('/api/admin/model/sequence/{model}/outcome')
    async def reveal(model:str,ref:str=Query('',max_length=35),reveal_key:str=Query('',max_length=64),user=Depends(admin)):
        seq=await replay_data(model,ref,user)
        if not secrets.compare_digest(receipt(user,model,ref,seq),reveal_key):
            raise HTTPException(409,'Run the prediction on the current sequence before revealing the outcome.')
        return {'correct':seq['next']['correct'],'source':'synthetic target label','note':'This labels a synthetic response, not a real learner outcome.'}

    mount_content(app,admin,teacher,auth)
    mount_papers(app,student,teacher,learning,save,get_class)
