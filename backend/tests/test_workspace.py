import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pymongo import MongoClient

from app.catalog import BANK
from app.main import passwords

TEST_PASSWORD='lumen-test-42'


def as_role(client,settings,role='admin',suffix='one'):
    email=f'{role}-{suffix}@example.com'
    with MongoClient(settings.mongodb_uri) as mongo:
        mongo[settings.mongodb_database].users.insert_one({'_id':str(uuid4()),'name':f'Test {role}','email':email,'role':role,
            'created_at':datetime.now(timezone.utc),'password_hash':passwords.hash(TEST_PASSWORD)})
    client.cookies.clear()
    r=client.post('/api/auth/login',json={'email':email,'password':TEST_PASSWORD})
    assert r.status_code==200, r.text
    assert r.json()['role']==role
    return dict(client.cookies)


def test_retired_routes_and_legacy_diagnostic_are_unavailable(student, settings):
    identity = student.get('/api/auth/me').json()
    old_id = str(uuid4())
    legacy = {'id': old_id, 'mode': 'diagnostic', 'status': 'active', 'responses': []}
    student.get('/api/learning/mathematics').raise_for_status()
    with MongoClient(settings.mongodb_uri) as mongo:
        mongo[settings.mongodb_database].learning.update_one(
            {'user_id': identity['id']}, {'$push': {'assessments': legacy}},
        )

    for path in ('/api/teacher/classes', '/api/teacher/papers', '/api/student/papers',
                 '/api/student/classes', '/api/student/fixed/' + old_id):
        assert student.get(path).status_code == 404
    assert student.post('/api/student/fixed/diagnostic').status_code == 404
    assert student.post('/api/assessments', json={'mode': 'adaptive'}).status_code == 404
    assert student.post('/api/practice', json={'mode': 'diagnostic'}).status_code == 422
    assert student.get('/api/learning/mathematics').json()['active_assessment'] is None
    assert student.get('/api/student/history').json() == []
    assert student.get(f'/api/practice/{old_id}').status_code == 404
    assert student.post('/api/practice', json={'mode': 'adaptive'}).status_code == 201
    with MongoClient(settings.mongodb_uri) as mongo:
        state = mongo[settings.mongodb_database].learning.find_one({'user_id': identity['id']})
    assert legacy in state['assessments']


def test_retired_teacher_session_and_login_are_disabled(student, settings):
    identity = student.get('/api/auth/me').json()
    assert student.get('/api/admin/model').status_code == 403
    assert student.post('/api/auth/register', json={
        'name': 'Escalation', 'email': 'test@example.com', 'password': TEST_PASSWORD, 'role': 'admin',
    }).status_code == 422
    with MongoClient(settings.mongodb_uri) as mongo:
        mongo[settings.mongodb_database].users.update_one({'_id': identity['id']}, {'$set': {'role': 'teacher'}})
    assert student.get('/api/auth/me').status_code == 401
    assert student.post('/api/auth/login', json={
        'email': identity['email'], 'password': 'thoughtful-practice-42',
    }).status_code == 403


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


def test_replay_keeps_the_target_hidden_until_receipt(client,settings):
    as_role(client,settings,'admin')
    base='/api/admin/model/sequence/bkt'
    assert 'correct' not in client.get(base).json()['next']
    assert client.get(base+'/outcome').status_code==409
    pred=client.post(base+'/predict').json()
    assert 0<pred['probability']<1 and pred['status']=='computed_now'
    assert client.get(base+'/outcome',params={'reveal_key':pred['reveal_key']}).json()['source']=='synthetic target label'
    assert client.get('/api/admin/model/replay-learners').json()==[]
