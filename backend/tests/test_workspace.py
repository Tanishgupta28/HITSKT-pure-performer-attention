import csv
import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from pymongo import MongoClient

from app.catalog import BANK
from app.main import passwords

TEST_PASSWORD='lumen-test-42'


def as_role(client,settings,role='teacher',suffix='one'):
    email=f'{role}-{suffix}@example.com'
    with MongoClient(settings.mongodb_uri) as mongo:
        mongo[settings.mongodb_database].users.insert_one({'_id':str(uuid4()),'name':f'Test {role}','email':email,'role':role,
            'created_at':datetime.now(timezone.utc),'password_hash':passwords.hash(TEST_PASSWORD)})
    client.cookies.clear()
    r=client.post('/api/auth/login',json={'email':email,'password':TEST_PASSWORD})
    assert r.status_code==200, r.text
    assert r.json()['role']==role
    return dict(client.cookies)


def restore(client,cookies):
    client.cookies.clear()
    client.cookies.update(cookies)


def test_diagnostic_resume_skip_and_idempotent_submit(student):
    response=student.post('/api/student/fixed/diagnostic')
    assert response.status_code==201, response.text
    a=response.json(); key=a['id']; url=f'/api/student/fixed/{key}'
    assert a['total']==12 and a['question']['concept_id']=='fractions'
    assert 'answer_index' not in a['question'] and 'explanation' not in a['question']
    assert student.post('/api/student/fixed/diagnostic').json()['id']==key
    r=student.post(url+'/item',json={'action':'skip','index':0}).json()
    assert r['positions'][0]=='skipped' and not r['result']
    r=student.post(url+'/item',json={'action':'open','index':1}).json()
    j=BANK[r['question']['id']]['answer_index']
    for _ in range(2):
        v=student.post(url+'/item',json={'action':'answer','index':1,'choice':j}).json()
        assert v['saved_choice']==j and v['positions'][1]=='answered' and v['result'] is None
    assert student.get(url).json()['saved_choice']==j
    assert student.get('/api/learning/mathematics').json()['stats']['answers']==0
    output=student.post(url+'/submit').json()
    assert output['released'] and output['result']['score']==1 and output['result']['accuracy']==1
    assert {r['shown'] for r in output['result']['topics']}=={3}
    assert [r['status'] for r in output['result']['questions'][:3]]==['skipped','correct','unanswered']
    assert student.post(url+'/submit').json()['result']==output['result']
    assert student.get('/api/learning/mathematics').json()['stats']['answers']==1


def test_exam_lock_release_rls_archival_and_pseudonymous_export(student,settings):
    learner=dict(student.cookies)
    original=student.get('/api/auth/me').json()
    assert student.get('/api/admin/model').status_code==403
    assert student.get('/api/teacher/classes').status_code==403
    assert student.post('/api/auth/register',json={'name':'Escalation','email':'test@example.com','password':TEST_PASSWORD,'role':'admin'}).status_code==422
    teacher=as_role(student,settings)
    created=student.post('/api/teacher/classes',json={'name':'Grade 8A Maths','grade_level':'Grade 8'})
    assert created.status_code==201,created.text
    cls=created.json(); cid=cls['_id']
    assert student.post(f'/api/teacher/classes/{cid}/enroll',json={'email':original['email']}).status_code==200
    assert student.post(f'/api/teacher/classes/{cid}/enroll',json={'email':original['email']}).status_code==409
    other=as_role(student,settings,suffix='other')
    assert student.get(f'/api/teacher/classes/{cid}').status_code==404
    restore(student,teacher)
    qid='fractions-1-1'
    paper=student.post('/api/teacher/papers',json={'title':'Common unit paper','class_id':cid,'question_ids':[qid],
                      'duration_minutes':10,'show_explanations':False}).json(); pid=paper['_id']
    pub=student.post(f'/api/teacher/papers/{pid}/assign');assert pub.status_code==200,pub.text
    assert student.post(f'/api/teacher/papers/{pid}/release').status_code==409
    assert 'questions' not in pub.json()
    admin=as_role(student,settings,'admin')
    assert student.post(f'/api/admin/questions/{qid}/archive').status_code==200
    restore(student,learner)
    start=student.post(f'/api/student/papers/{pid}/start');assert start.status_code==201,start.text
    attempt=start.json(); key=attempt['id']; route=f'/api/student/fixed/{key}'
    assert attempt['question']['id']==qid and attempt['deadline_at']
    assert student.post(f'/api/student/papers/{pid}/start').json()['id']==key
    j=BANK[qid]['answer_index']
    assert student.post(route+'/item',json={'action':'answer','index':0,'choice':j}).status_code==200
    v=student.post(route+'/submit').json();assert not v['released'] and v['result'] is None
    assert student.get('/api/student/history').json()[0]['score'] is None
    assert student.get('/api/learning/mathematics').json()['stats']['answers']==0
    restore(student,admin)
    pre_release=student.get('/api/admin/interactions').json()
    assert pre_release['attempts']==1 and pre_release['rows'][0]['student_id']!=original['id']
    restore(student,teacher)
    graded=student.get(f'/api/teacher/papers/{pid}/results').json()
    assert graded['submissions'][0]['score']==1
    assert student.post(f'/api/teacher/papers/{pid}/close').status_code==200
    assert student.post(f'/api/teacher/papers/{pid}/release').status_code==200
    restore(student,learner)
    report=student.get(route).json()
    assert report['released'] and report['result']['score']==1
    assert 'answer_key' not in report['result']['questions'][0]
    assert student.get('/api/learning/mathematics').json()['stats']['answers']==1
    assert student.get('/api/learning/mathematics').json()['stats']['answers']==1
    restore(student,admin)
    result=student.post('/api/admin/interactions/export?days=30')
    assert result.status_code==200 and original['email'] not in result.text and original['id'] not in result.text
    rows=list(csv.DictReader(io.StringIO(result.text)))
    assert len(rows)==1 and rows[0]['correct']=='True' and rows[0]['question_id']==qid and rows[0]['source']=='live'


def test_publication_requires_mapping_and_answers(student,settings):
    as_role(student,settings,'admin')
    route='/api/admin/questions'
    q={'id':'NEW-FRACTION','topic_id':'fractions','skill_id':'missing','difficulty':1,'prompt':'What is 1/2 + 1/3?',
       'choices':['1/3','5/6','2/5','1/6'],'answer_index':1,'explanation':'Find sixths before adding.'}
    assert student.post(route,json=q).status_code==201
    assert student.post(route,json=q).status_code==409
    assert student.post(route+'/NEW-FRACTION/publish').status_code==422
    skill=student.post('/api/admin/taxonomy',json={'id':'add-unlike','name':'Add unlike fractions','kind':'skill','topic_id':'fractions'})
    assert skill.status_code==201
    assert student.post(route+'/NEW-FRACTION/edit',json={**q,'skill_id':'add-unlike'}).status_code==200
    assert student.post(route+'/NEW-FRACTION/publish').json()['version']==1
    assert student.delete(route+'/NEW-FRACTION').status_code==409
    assert student.post(route+'/NEW-FRACTION/archive').status_code==200
    assert student.get(route+'?status=archived').json()['questions'][0]['id']=='NEW-FRACTION'
    assert student.post(route,json={'id':'UNUSED-DRAFT'}).status_code==201
    assert student.delete(route+'/UNUSED-DRAFT').status_code==204


def test_recovery_is_single_use_and_does_not_send_email(client,settings,monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app
    received=[]
    monkeypatch.setattr('app.workspace.smtp_send',lambda _,address,token: received.append((address,token)))
    config=settings.model_copy(update={'smtp_host':'mail.example.com','smtp_from':'help@example.com'})
    with TestClient(create_app(config)) as api:
        student={'name':'Recruit Test','email':'recovery@example.com','password':TEST_PASSWORD}
        assert api.post('/api/auth/register',json=student).status_code==201
        r=api.post('/api/auth/recovery',json={'email':student['email']});assert r.status_code==200
        assert received and received[0][0]==student['email']
        token=received[0][1]
        assert token not in r.text
        assert api.post('/api/auth/reset',json={'token':token,'password':'renewed-password-42'}).status_code==200
        assert api.get('/api/auth/me').status_code==401
        assert api.post('/api/auth/reset',json={'token':token,'password':'anything-else-42'}).status_code==422
        assert api.post('/api/auth/login',json={'email':student['email'],'password':'renewed-password-42'}).status_code==200


def test_synthetic_dataset_reproducible_and_split_by_learner():
    path=Path(__file__).resolve().parents[1]/'data'/'lumen-synthetic-v1'
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    for name,signature in manifest['sha256'].items():
        assert hashlib.sha256((path/name).read_bytes()).hexdigest()==signature
    with (path/'interactions.csv').open(newline='',encoding='utf-8') as f: records=list(csv.DictReader(f))
    assert len(records)==manifest['attempts']==864
    learners={}
    for row in records:
        assert row['source']=='synthetic' and int(row['correct'])==int(int(row['response'])==BANK[row['question_id']]['answer_index'])
        learners.setdefault(row['student_id'],set()).add(row['split'])
    assert len(learners)==18 and all(len(x)==1 for x in learners.values())


def test_exam_expiry_locks_confirmed_answers_without_releasing(student,settings):
    learner=dict(student.cookies)
    identity=student.get('/api/auth/me').json()
    as_role(student,settings)
    c=student.post('/api/teacher/classes',json={'name':'Time test class','grade_level':'Grade 8'}).json()
    student.post(f'/api/teacher/classes/{c["_id"]}/enroll',json={'email':identity['email']}).raise_for_status()
    qs=['fractions-1-1','fractions-1-2']
    p=student.post('/api/teacher/papers',json={'title':'Timed unit check','class_id':c['_id'],'question_ids':qs,
                          'skips_allowed':False, 'review_allowed':False}).json()
    student.post(f'/api/teacher/papers/{p["_id"]}/assign').raise_for_status()
    restore(student,learner)
    attempt=student.post(f'/api/student/papers/{p["_id"]}/start').json(); id=attempt['id']; base=f'/api/student/fixed/{id}'
    assert student.post(base+'/item',json={'action':'skip','index':0}).status_code==409
    assert student.post(base+'/item',json={'action':'open','index':1}).status_code==409
    student.post(base+'/item',json={'action':'answer','index':0,'choice':BANK[qs[0]]['answer_index']}).raise_for_status()
    student.post(base+'/item',json={'action':'open','index':1}).raise_for_status()
    assert student.post(base+'/item',json={'action':'open','index':0}).status_code==409
    with MongoClient(settings.mongodb_uri) as mongo:
        mongo[settings.mongodb_database].learning.update_one({'user_id':identity['id']},{'$set':{f'fixed.{id}.deadline_at':(datetime.now(timezone.utc)-timedelta(milliseconds=100)).isoformat()}})
    late=student.post(base+'/item',json={'action':'answer','index':1,'choice':BANK[qs[1]]['answer_index']}).json()
    assert late['status']=='submitted' and late['submission_reason']=='deadline'
    assert late['positions']==['answered','unanswered'] and late['result'] is None


def test_replay_keeps_the_target_hidden_until_receipt(client,settings):
    as_role(client,settings,'admin')
    base='/api/admin/model/sequence/bkt'
    assert 'correct' not in client.get(base).json()['next']
    assert client.get(base+'/outcome').status_code==409
    pred=client.post(base+'/predict').json()
    assert 0<pred['probability']<1 and pred['status']=='computed_now'
    assert client.get(base+'/outcome',params={'reveal_key':pred['reveal_key']}).json()['source']=='synthetic target label'
    assert client.get('/api/admin/model/replay-learners').json()==[]
