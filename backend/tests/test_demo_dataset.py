import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from pymongo import MongoClient

from app.catalog import BANK
from app.papers import result_for
from tools.demo_dataset import BASE, VERSION, attempt_id, dataset, seed_database, uid


def test_v2_dataset_is_reproducible_and_has_distinct_sessions(tmp_path):
    manifest = dataset(tmp_path)
    published = Path(__file__).resolve().parents[1] / 'data' / VERSION
    assert manifest == json.loads((published / 'manifest.json').read_text(encoding='utf-8'))
    for name, signature in manifest['sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == signature
        assert (tmp_path / name).read_bytes() == (published / name).read_bytes()
    with (tmp_path / 'interactions.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    with (tmp_path / 'questions.csv').open(newline='', encoding='utf-8') as file:
        questions = {q['question_id']: q for q in csv.DictReader(file)}
    assert len(rows) == manifest['attempts']
    assert len(questions) == manifest['question_versions']
    splits = {}
    modes_by_split = {}
    last = {}
    lengths = Counter()
    for r in rows:
        question = questions[r['question_id']]
        assert r['correct'] == str(int(int(r['response']) == BANK[r['question_id']]['answer_index']))
        assert r['question_version'] == question['version'] and r['skill_id'] == question['skill_id']
        assert r['source'] == 'synthetic'
        s = r['session_id']
        stamp = datetime.fromisoformat(r['attempt_timestamp'])
        assert stamp.tzinfo and stamp < BASE and (s not in last or stamp > last[s])
        last[s] = stamp
        lengths[s] += 1
        splits.setdefault(r['student_id'], set()).add(r['split'])
        modes_by_split.setdefault(r['split'], set()).add(r['mode'])
    assert len(splits) == 18 and all(len(value) == 1 for value in splits.values())
    assert set(lengths.values()) == {6, 11, 12}
    assert len(last) == manifest['sessions']
    assert all(modes == {'diagnostic', 'exam', 'practice', 'adaptive'} for modes in modes_by_split.values())
    assert all(r['hints_used'] == '0' for r in rows if r['mode'] in {'diagnostic', 'exam'})


def test_demo_seed_refresh_preserves_user_attempt_and_held_scores(settings):
    config = settings.model_copy(update={'demo_enabled': True})
    with MongoClient(config.mongodb_uri, tz_aware=True) as client:
        db = client[config.mongodb_database]
        anchor = datetime.now(timezone.utc) - timedelta(days=1)
        anchor = anchor.replace(microsecond=0)
        teacher = uid('teacher')
        student = uid('student-0')
        db.users.insert_one({'_id': teacher, 'email': 'teacher@demo.lumen.example.com',
                             'demo': True, 'created_at': anchor, 'password_hash': 'unchanged'})
        old = attempt_id(0, 0)
        written = str(uuid4())
        response = {'request_id': str(uuid4()), 'assessment_id': written, 'question_id':'fractions-1-1',
                    'source': 'synthetic', 'concept_id': 'fractions', 'correct': True, 'difficulty': 1,
                    'answered_at': (anchor + timedelta(hours=1)).isoformat()}
        user_attempt = {'id': written, 'mode': 'practice', 'started_at': response['answered_at'],
                        'status': 'completed', 'responses': [response]}
        db.learning.insert_one({'_id': f'{student}:mathematics', 'user_id': student, 'demo': True,
                                'version': 6, 'history': [{**response, 'assessment_id': old,
                                'request_id': uid(f'{old}-0')}, response], 'fixed': {old: {'id': old}},
                                'assessments': [user_attempt]})
        db.learning.insert_one({'_id': 'real:mathematics', 'demo': False, 'version': 2, 'history': []})
        seed_database(config)
        after = db.learning.find_one({'_id': f'{student}:mathematics'})
        rohan = db.learning.find_one({'_id': f'{uid("student-1")}:mathematics'})
        assert after['version'] == 7 and after['demo_fixture_version'] == VERSION
        assert after['assessments'][-1] == user_attempt and after['history'][-1] == response
        assert len(after['history']) == 31 and len(rohan['history']) == 42
        assert sum(c['attempts'] for c in after['concepts'].values()) == len(after['history'])
        assert [result_for(s['fixed'][attempt_id(i, 0)])['score'] for i, s in enumerate([after, rohan])] == [6, 9]
        assert after['assessments'][0]['responses'][0]['feedback']['explanation']
        assert {'practice', 'adaptive'} == {a['mode'] for a in rohan['assessments']}
        assert db.users.find_one({'_id': teacher})['password_hash'] == 'unchanged'
        assert db.learning.find_one({'_id': 'real:mathematics'})['version'] == 2
        paper = db.papers.find_one({'_id': uid('exam-paper')})
        exam = datetime.fromisoformat(after['fixed'][attempt_id(0, 1)]['submitted_at'])
        assert paper['created_at'] <= paper['assigned_at'] < exam < paper['closed_at'] < paper['released_at']
        nia = db.learning.find_one({'_id': f'{uid("student-2")}:mathematics'})
        held = nia['fixed'][attempt_id(2, 4)]
        assert held['submitted_at'] and not held['incorporated']
        assert all(r['assessment_id'] != held['id'] for r in nia['history'])
        seed_database(config)
        assert db.learning.find_one({'_id': after['_id']}) == after
