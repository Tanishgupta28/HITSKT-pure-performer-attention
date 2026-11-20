"""Generate fictional learning histories and install them in a loopback demo."""

import argparse
import csv
import hashlib
import json
import random
import secrets
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from app.catalog import BANK, CONCEPTS
from app.config import Settings
from app.content import key, original, safe_question
from app.engine import concept_rows, initial_concept, recommendations, update_knowledge

SEED = 42
VERSION = 'lumen-synthetic-v2'
ID_NAMESPACE = 'lumen-synthetic-v1'  # Keep accounts, class codes and existing report links.
PASSWORD = 'Lumen12345'
DOMAIN = 'demo.lumen.example.com'
BASE = datetime(2026, 11, 20, tzinfo=timezone.utc)
FIELDS = ['student_id', 'session_id', 'question_id', 'question_version', 'skill_id',
          'topic_id', 'response', 'correct', 'response_time_ms', 'attempt_timestamp',
          'hints_used', 'mode', 'source', 'split']
NAMES = ['Asha Shah', 'Rohan Mehta', 'Nia Patel', 'Omar Jain', 'Dev Malik', 'Ira Sen',
         'Sana Roy', 'Neil Shah', 'Arya Rao', 'Riya Bose', 'Kabir Das', 'Nikhil Joshi',
         'Aditi Pai', 'Tanvi Shah', 'Esha Rao', 'Veer Jain', 'Rehan Sen', 'Maya Patel',
         'Diya Jain', 'Sia Roy', 'Arun Pai', 'Ishan Rao', 'Neel Das', 'Jiya Sen']
CONCEPT_IDS = [c['id'] for c in CONCEPTS]
RULES = {'review_allowed': True, 'skips_allowed': True, 'show_explanations': True}
DIAGNOSTIC_PROFILES = {
    # Fractions, percentages, equations, geometry. The first two drive the demo.
    0: [0, 1, 3, 2], 1: [3, 3, 1, 2], 2: [3, 3, 3, 3],
    4: [2, 0, 1, 1], 7: [3, 2, 3, 2], 14: [1, 0, 1, 1], 15: [2, 3, 2, 3],
}


@dataclass(frozen=True)
class Item:
    question: dict
    response: int | None
    elapsed_ms: int
    stamp: datetime
    hints: int

    @property
    def correct(self):
        return self.response == self.question['answer_index']


@dataclass(frozen=True)
class Session:
    slot: int
    mode: str
    start: datetime
    items: list[Item]
    assignment: str | None = None
    focus: str | None = None

    @property
    def held(self):
        return self.assignment == 'exam-new'


def uid(name):
    return str(uuid5(NAMESPACE_URL, f'https://lumen.example/{ID_NAMESPACE}/{name}'))


def attempt_id(index, slot):
    return uid(f'learner-{index}-session-{slot}')


def question_set(slot):
    return [original(BANK[f'{concept}-{1 + slot % 3}-{1 + slot * 3 + i}'])
            for concept in CONCEPT_IDS for i in range(3)]


def focused_questions(concept, variant):
    start = 13 + (variant % 3) * 6
    return [original(BANK[f'{concept}-{1 + i % 3}-{start + i}']) for i in range(6)]


def sample_session(index, slot, start, questions, mode, *, counts=None, pattern=None,
                   assignment=None, focus=None, skipped=()):
    rng = random.Random(SEED + 101 * index + slot)
    clock = start
    items = []
    for i, question in enumerate(questions):
        if counts is not None:
            correct = i % 3 >= 3 - counts[i // 3]
        elif pattern is not None:
            correct = pattern[i]
        else:
            topic = CONCEPT_IDS.index(question['concept_id'])
            chance = 0.34 + ((index + topic * 2) % 7) * 0.07 + (slot - 1) * 0.03
            correct = rng.random() < chance
        answer = question['answer_index']
        response = None if i in skipped else answer if correct else (answer + rng.randint(1, 3)) % 4
        elapsed = rng.randint(6_000 + index % 4 * 2_000, 29_000 + index % 4 * 4_000)
        clock += timedelta(milliseconds=elapsed)
        hints = int(mode in {'practice', 'adaptive'} and not correct and rng.random() < 0.55)
        items.append(Item(question, response, elapsed, clock, hints))
        clock += timedelta(seconds=2)
    return Session(slot, mode, start, items, assignment, focus)


def histories(anchor, open_assignment=None):
    """Generate distinct, reproducible journeys around a common diagnostic."""
    for index, name in enumerate(NAMES):
        if index >= 18:
            yield index, name, []
            continue
        offset = timedelta(hours=index % 5)
        exam = ([1, 2, 3, 2], [3, 3, 2, 2])
        sessions = [sample_session(index, 0, anchor - timedelta(days=12) + offset,
                                   question_set(0), 'diagnostic', counts=DIAGNOSTIC_PROFILES.get(index))]
        if index in {0, 1, 2, 3, 5, 6, 8, 9, 12, 15, 16, 17}:
            sessions.append(sample_session(index, 1, anchor - timedelta(days=8) + offset,
                                           question_set(1), 'exam', assignment='exam-paper',
                                           counts=exam[index] if index < 2 else None,
                                           skipped=(4,) if index in {3, 9} else ()))
        if index in {0, 1, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16}:
            focus = 'fractions' if index == 0 else 'equations' if index == 1 else CONCEPT_IDS[index % 4]
            pattern = ([True, True, False, True, False, False], [True, False, False, True, True, False])
            sessions.append(sample_session(index, 2, anchor - timedelta(days=3) + offset,
                                           focused_questions(focus, index), 'practice', focus=focus,
                                           pattern=pattern[index] if index < 2 else None))
        if index in {1, 3, 6, 9, 11, 14, 16}:
            sessions.append(sample_session(index, 3, anchor - timedelta(days=2) + offset,
                                           question_set(3), 'adaptive', counts=[3, 3, 0, 3] if index == 1 else None))
        if index in {2, 12, 17}:
            start = anchor - timedelta(days=1) + offset
            if open_assignment:
                start = max(start, open_assignment + timedelta(minutes=20 + index))
            sessions.append(sample_session(index, 4, start, question_set(2), 'exam',
                                           assignment='exam-new', skipped=(5,) if index == 12 else ()))
        yield index, name, sessions


def dataset(out):
    out.mkdir(parents=True, exist_ok=True)
    splits = Counter()
    modes = Counter()
    sessions_by_mode = Counter()
    learners = Counter()
    catalog = {}
    with (out / 'interactions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        for index, _, sessions in histories(BASE):
            if not sessions:
                continue
            split = 'train' if index < 12 else 'validation' if index < 15 else 'test'
            student = f'synthetic-{index + 1:03d}'
            learners[split] += 1
            for session in sessions:
                sessions_by_mode[session.mode] += 1
                for item in session.items:
                    q = item.question
                    catalog[q['id']] = q
                    if item.response is None:
                        continue
                    writer.writerow(dict(student_id=student, session_id=f'{student}-session-{session.slot + 1}',
                        question_id=q['id'], question_version=q['content_version'], skill_id=q['skill_id'],
                        topic_id=q['concept_id'], response=item.response, correct=int(item.correct),
                        response_time_ms=item.elapsed_ms, attempt_timestamp=item.stamp.isoformat(),
                        hints_used=item.hints, mode=session.mode, source='synthetic', split=split))
                    splits[split] += 1
                    modes[session.mode] += 1
    with (out / 'questions.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ['question_id', 'topic_id', 'skill_id', 'difficulty', 'prompt', 'option_a',
                  'option_b', 'option_c', 'option_d', 'answer_index', 'explanation', 'version']
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for q in sorted(catalog.values(), key=lambda row: row['id']):
            writer.writerow(dict(zip(fields, [q['id'], q['topic_id'], q['skill_id'], q['difficulty'],
                q['prompt'], *q['choices'], q['answer_index'], q['explanation'], q['content_version']])))
    manifest = {'version': VERSION, 'seed': SEED, 'source': 'synthetic',
                'reference_timestamp': BASE.isoformat(), 'learners_with_attempts': sum(learners.values()),
                'sessions': sum(sessions_by_mode.values()), 'attempts': sum(splits.values()),
                'question_versions': len(catalog), 'learners_by_split': dict(learners),
                'attempts_by_split': dict(splits), 'sessions_by_mode': dict(sessions_by_mode),
                'attempts_by_mode': dict(modes), 'split_unit': 'student', 'fields': FIELDS,
                'sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                           for name in ('interactions.csv', 'questions.csv')},
                'usage': 'Synthetic demonstration and pipeline validation only. No real learner records or benchmark mappings.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def paper_seed(anchor, name, title, status, items):
    assigned = None if status == 'draft' else anchor - timedelta(days=9 if status == 'released' else 3)
    close = anchor - timedelta(days=6) if status == 'released' else None
    return {'_id': uid(name), 'teacher_id': uid('teacher'), 'class_id': uid('class-8b'),
            'class_name': '8B Mathematics', 'demo': True, 'status': status, 'title': title,
            'revision': 0, 'question_ids': [q['id'] for q in items],
            'questions': [] if status == 'draft' else items, 'duration_minutes': 20, **RULES,
            'extra_minutes': {}, 'created_at': assigned or anchor, 'assigned_at': assigned,
            'opens_at': assigned, 'closes_at': close or (anchor + timedelta(days=30) if assigned else None),
            'closed_at': close, 'released_at': close + timedelta(hours=1) if close else None}


def seed_papers(db, anchor):
    seeds = [paper_seed(anchor, 'exam-paper', 'Grade 8 unit check', 'released', question_set(1)),
             paper_seed(anchor, 'exam-new', 'Next unit check', 'assigned', question_set(2)),
             paper_seed(anchor, 'exam-draft', 'Percentage and fractions review', 'draft', question_set(0)[:6])]
    for seed in seeds:
        id_ = seed['_id']
        db.papers.update_one({'_id': id_}, {'$setOnInsert': {**seed, 'demo_fixture_version': VERSION}}, upsert=True)
        original_seed = {'_id': id_, 'demo': True, 'status': seed['status'], 'revision': 0,
                         'question_ids': seed['question_ids'], 'created_at': anchor,
                         'assigned_at': anchor, 'demo_fixture_version': {'$exists': False}}
        # Correct the old generated timeline, retaining shared question IDs and existing attempts.
        patch = {k: seed[k] for k in ('title', 'created_at', 'assigned_at', 'closed_at', 'released_at')}
        if seed['status'] == 'released':
            patch.update(opens_at=seed['opens_at'], closes_at=seed['closes_at'])
        elif seed['status'] == 'assigned':
            # The open paper already exists and may have active timers. Keep its availability.
            patch.update(created_at=anchor, assigned_at=anchor)
        db.papers.update_one(original_seed, {'$set': {**patch, 'demo_fixture_version': VERSION}})
    return {seed['_id']: db.papers.find_one({'_id': seed['_id'], 'demo': True}) for seed in seeds}


def legacy_session_ids(index):
    slots = range(2 if index < 2 else 4 if index < 18 else 0)
    return {attempt_id(index, slot) for slot in slots}


def learning_seed(index, student_id, sessions, papers, previous, now):
    """Replace only recognized v1 fixtures; replay evidence around preserved attempts."""
    legacy = legacy_session_ids(index)
    legacy_rows = {uid(f'{session_id}-{i}') for session_id in legacy for i in range(12)}
    previous = previous or {}
    data = {'_id': f'{student_id}:mathematics', 'user_id': student_id, 'demo': True,
            'subject_id': 'mathematics', 'version': 0, **previous}
    fixed = {k: session for k, session in previous.get('fixed', {}).items() if k not in legacy}
    practice = [session for session in previous.get('assessments', []) if session['id'] not in legacy]
    retained = [row for row in previous.get('history', []) if row.get('request_id') not in legacy_rows]
    existing_papers = {session.get('assignment_id') for session in fixed.values()}
    concepts = {concept: initial_concept() for concept in CONCEPT_IDS}
    history = []
    state = {**data, 'fixed': fixed, 'assessments': practice, 'history': history, 'concepts': concepts,
             'demo_fixture_version': VERSION}

    def observe(record):
        concept = record['concept_id']
        concepts[concept] = update_knowledge(concepts[concept], record['correct'], record['difficulty'], record['answered_at'])
        history.append(record)

    for session in sessions:
        sid = attempt_id(index, session.slot)
        started = session.start.isoformat()
        finished = session.items[-1].stamp.isoformat()
        assignment = uid(session.assignment) if session.assignment else None
        paper = papers.get(assignment)
        if assignment and (not paper or assignment in existing_papers):
            continue
        if session.held and (paper['status'] != 'assigned' or session.items[-1].stamp > now
                             or (paper.get('closes_at') and session.items[-1].stamp > paper['closes_at'])):
            continue
        if session.mode in {'diagnostic', 'exam'}:
            title = paper['title'] if paper else 'Mathematics diagnostic'
            answers = {str(i): {'value': item.response, 'saved_at': item.stamp.isoformat(),
                       'response_time_ms': item.elapsed_ms} if item.response is not None else {'skipped': True}
                       for i, item in enumerate(session.items)}
            opened = {str(i): (item.stamp - timedelta(milliseconds=item.elapsed_ms)).isoformat()
                      for i, item in enumerate(session.items)}
            fixed[sid] = {'id': sid, 'mode': session.mode, 'title': title, 'assignment_id': assignment,
                          'questions': [item.question for item in session.items], 'started_at': started,
                          'submitted_at': finished, 'submission_reason': 'submitted',
                          'deadline_at': (session.start + timedelta(minutes=paper['duration_minutes'])).isoformat() if paper else None,
                          'index': len(session.items) - 1, 'answers': answers, 'opened': opened,
                          'incorporated': not session.held, 'rules': RULES}
        if session.held:
            continue
        responses = []
        for i, item in enumerate(session.items):
            if item.response is None:
                continue
            q = item.question
            record = {'question_id': q['id'], 'assessment_id': sid, 'concept_id': q['concept_id'],
                      'skill_id': q['skill_id'], 'content_version': q['content_version'], 'difficulty': q['difficulty'],
                      'correct': item.correct, 'choice_index': item.response, 'answered_at': item.stamp.isoformat(),
                      'response_time_ms': item.elapsed_ms, 'hints_used': item.hints, 'source': 'synthetic',
                      'mode': session.mode, 'request_id': uid(f'{sid}-{i}'),
                      'prediction': {'provider': 'not_scored', 'probability': None}}
            observe(record)
            if session.mode in {'practice', 'adaptive'}:
                concept = q['concept_id']
                record['question_snapshot'] = safe_question(q)
                record['feedback'] = {'correct': item.correct, 'answer_index': q['answer_index'],
                                      'explanation': q['explanation'], 'mastery': concepts[concept]['mastery'],
                                      'concept_name': next(c['name'] for c in CONCEPTS if c['id'] == concept)}
            responses.append(record)
        summary = {'concepts': concept_rows(state), 'recommendations': recommendations(state)}
        if sid in fixed:
            fixed[sid]['summary'] = summary
        else:
            practice.append({'id': sid, 'mode': session.mode, 'concept_id': session.focus, 'status': 'completed',
                             'started_at': started, 'completed_at': finished, 'responses': responses,
                             'pending': None, 'total': len(session.items),
                             'summary': {**summary, 'accuracy': sum(r['correct'] for r in responses) / len(responses),
                                         'average_mastery': sum(c['mastery'] for c in concepts.values()) / len(CONCEPTS)}})
    for record in retained:
        observe(record)
    practice.sort(key=lambda a: a['started_at'])
    return state


def seed_database(settings):
    from pymongo import MongoClient
    from pwdlib import PasswordHash

    local = {'127.0.0.1', 'localhost', '::1'}
    if not settings.demo_enabled or urlparse(settings.mongodb_uri).hostname not in local or urlparse(settings.frontend_origin).hostname not in local:
        raise SystemExit('Set DEMO_ENABLED=true and use loopback MongoDB and frontend addresses to seed the local demo.')
    now = datetime.now(timezone.utc)
    now = now.replace(microsecond=now.microsecond // 1000 * 1000)
    secret = PasswordHash.recommended().hash(PASSWORD)
    other = PasswordHash.recommended().hash(secrets.token_urlsafe(28))
    migrated = inserted = 0
    with MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000, tz_aware=True) as connection:
        db = connection[settings.mongodb_database]
        db.command('ping')
        db.users.create_index('email', unique=True)
        teacher = uid('teacher')

        def account(id_, name, handle, role, allowed):
            address = f'{handle}@{DOMAIN}'
            row = db.users.find_one({'$or': [{'_id': id_}, {'email': address}]}, {'email': 1, 'demo': 1})
            if row and (not row.get('demo') or row['_id'] != id_ or row['email'] != address):
                raise SystemExit(f'Reserved demo identity overlaps an existing account: {address}.')
            db.users.update_one({'_id': id_}, {'$setOnInsert': {'name': name, 'email': address, 'role': role,
                'demo': True, 'created_at': now, 'onboarding_complete': True, 'password_hash': secret if allowed else other,
                'profile': {'grade_level': 'Grade 8', 'curriculum': 'CBSE', 'weekly_question_target': 18},
                'profile_source': 'sample'}}, upsert=True)

        account(teacher, 'Mira Rao', 'teacher', 'teacher', True)
        account(uid('admin'), 'Lumen Content', 'admin', 'admin', True)
        anchor = db.users.find_one({'_id': teacher}, {'created_at': 1})['created_at']
        papers = seed_papers(db, anchor)
        assigned = papers[uid('exam-new')]
        members = {}
        for index, name, sessions in histories(anchor, assigned.get('assigned_at') if assigned else None):
            id_ = uid(f'student-{index}')
            members[id_] = {'status': 'active', 'joined_at': anchor - timedelta(days=14)}
            handle = 'student' if index == 0 else 'rohan' if index == 1 else f'learner{index + 1}'
            account(id_, name, handle, 'student', index < 2)
            record_id = f'{id_}:mathematics'
            previous = db.learning.find_one({'_id': record_id})
            if previous and (not previous.get('demo') or previous.get('demo_fixture_version')):
                continue
            fixture = learning_seed(index, id_, sessions, papers, previous, now)
            if previous:
                fixture['version'] = previous['version'] + 1
                result = db.learning.replace_one({'_id': record_id, 'demo': True, 'version': previous['version'],
                                                  'demo_fixture_version': {'$exists': False}}, fixture)
                migrated += result.modified_count
            else:
                result = db.learning.update_one({'_id': record_id}, {'$setOnInsert': fixture}, upsert=True)
                inserted += int(result.upserted_id is not None)

        for name, title, code, roster in (('class-8b', '8B Mathematics', 'LUMEN08B', members),
                                          ('class-8a', '8A Mathematics', 'LUMEN08A', {})):
            db.classes.update_one({'_id': uid(name)}, {'$setOnInsert': {'name': title, 'teacher_id': teacher,
                'demo': True, 'join_code': code, 'members': roster, 'grade_level': 'Grade 8', 'curriculum': 'CBSE',
                'created_at': anchor - timedelta(days=14), 'subject_id': 'mathematics'}}, upsert=True)
            db.classes.update_one({'_id': uid(name), 'teacher_id': teacher, 'demo': True, 'created_at': anchor},
                                  {'$set': {'created_at': anchor - timedelta(days=14)}})

        for id_, prompt, choices, correct, explanation in (
            ('FRA-018', 'What is 1/2 + 1/3?', ['2/3', '2/5', '5/6', '1/6'], 2,
             'The denominators describe part size. Rewrite 1/2 as 3/6 and 1/3 as 2/6. Add the numerators: 3/6 + 2/6 = 5/6.'),
            ('FRA-019', 'What is 1/3 + 1/4?', ['2/7', '1/12', '7/12', '5/12'], 2,
             'Rewrite 1/3 as 4/12 and 1/4 as 3/12. Add the numerators: 4/12 + 3/12 = 7/12.'),
        ):
            q = {'id': id_, 'subject_id': 'mathematics', 'topic_id': 'fractions', 'concept_id': 'fractions',
                 'skill_id': 'fractions-foundations', 'question_type': 'mcq', 'difficulty': 1, 'choices': choices,
                 'answer_index': correct, 'prompt': prompt, 'explanation': explanation,
                 'hint': 'Find a common denominator shared by both fractions.', 'status': 'published', 'version': 1,
                 'content_version': f'{id_}:v1', 'origin': 'authored'}
            db.questions.update_one({'_id': key(True, id_)}, {'$setOnInsert': {'demo': True, 'id': id_,
                                    'status': 'published', 'version': 1, 'revision': 0, 'live': q}}, upsert=True)
    print(f'Local demo ready: {inserted} new and {migrated} refreshed sample histories. Existing accounts and submitted user attempts preserved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'backend' / 'data' / VERSION)
    parser.add_argument('--seed-local', action='store_true')
    parser.add_argument('--seed-only', action='store_true', help='Seed MongoDB without rewriting dataset files.')
    args = parser.parse_args()
    if not args.seed_only:
        manifest = dataset(args.output)
        print(f'{manifest["attempts"]} synthetic attempts / {manifest["sessions"]} sessions / {manifest["learners_with_attempts"]} learners')
    if args.seed_local or args.seed_only:
        seed_database(Settings())
