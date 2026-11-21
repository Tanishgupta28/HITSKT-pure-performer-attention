import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from pymongo import MongoClient

from app.catalog import BANK
from tools.demo_dataset import BASE, VERSION, dataset, seed_database, uid


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


def test_demo_seed_only_adds_missing_practice_histories(settings):
    config = settings.model_copy(update={'demo_enabled': True})
    with MongoClient(config.mongodb_uri, tz_aware=True) as client:
        db = client[config.mongodb_database]
        teacher_id = uid('teacher')
        student_id = uid('student-0')
        db.users.insert_one({'_id': teacher_id, 'email': 'teacher@demo.lumen.example.com',
                             'role': 'teacher', 'demo': True, 'password_hash': 'unchanged'})
        previous = {'_id': f'{student_id}:mathematics', 'user_id': student_id, 'demo': True,
                    'version': 6, 'fixed': {'old-diagnostic': {'id': 'old-diagnostic'}},
                    'assessments': [{'id': 'old-practice', 'mode': 'practice'}]}
        db.learning.insert_one(previous)
        db.learning.insert_one({'_id': 'real:mathematics', 'demo': False, 'version': 2, 'history': []})

        seed_database(config)

        rohan = db.learning.find_one({'_id': f'{uid("student-1")}:mathematics'})
        assert rohan['demo_fixture_version'] == VERSION
        assert rohan['history']
        assert {'practice', 'adaptive'} == {a['mode'] for a in rohan['assessments']}
        assert all(r['feedback']['explanation'] for r in rohan['history'])
        assert db.learning.find_one({'_id': previous['_id']}) == previous
        assert db.learning.find_one({'_id': 'real:mathematics'})['version'] == 2
        assert db.users.count_documents({'role': 'teacher'}) == 1
        assert db.users.find_one({'_id': teacher_id})['password_hash'] == 'unchanged'
        assert 'papers' not in db.list_collection_names() and 'classes' not in db.list_collection_names()

        seed_database(config)
        assert db.learning.find_one({'_id': rohan['_id']}) == rohan
