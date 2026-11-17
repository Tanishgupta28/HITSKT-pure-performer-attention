"""Versioned authoring. Existing papers and served questions hold immutable copies."""
import re
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from pymongo.errors import DuplicateKeyError

from app.catalog import BANK, CONCEPTS

CONCEPT_IDS = {c['id'] for c in CONCEPTS}
ID = r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$'
DEFAULT_TOPICS = [{"id": c["id"], "name": c["name"], "concept_id": c["id"], "kind": "topic", "subject_id": "mathematics"} for c in CONCEPTS]
DEFAULT_SKILLS = [{"id": f'{c["id"]}-foundations', "name": f'{c["name"]} foundations', "topic_id": c["id"], "concept_id": c["id"], "kind": "skill"} for c in CONCEPTS]


def realm(user):
    return bool(user.get("demo", False))


def scope(user):
    return {"demo": True} if realm(user) else {"demo": {"$ne": True}}


def key(demo, id):
    return f"{'demo' if demo else 'live'}:{id}"


def original(q):
    return {**q, "topic_id": q["concept_id"], "skill_id": f'{q["concept_id"]}-foundations',
            "question_type": "mcq", "version": 1, "status": "published", "origin": "original"}


async def available_questions(db, demo=False):
    result = {k: original(q) for k, q in BANK.items()}
    for row in await db.questions.find({"demo": demo}, {"_id": 0, "id": 1, "live": 1}).to_list(5000):
        if row.get("live"):
            result[row['id']] = row['live']
        else:
            result.pop(row['id'], None)
    return result


async def taxonomy(db, demo):
    rows = await db.taxonomy.find({"demo": demo}, {"_id": 0, "demo": 0}).sort("name", 1).to_list(2500)
    return DEFAULT_TOPICS + DEFAULT_SKILLS + rows


class QuestionDraft(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    id: str = Field(pattern=ID)
    topic_id: str = Field(default='', max_length=80)
    skill_id: str = Field(default='', max_length=80)
    difficulty: int = Field(default=1, ge=1, le=3)
    question_type: Literal['mcq', 'numeric'] = 'mcq'
    prompt: str = Field(default='', max_length=1500)
    choices: list[str] = Field(default_factory=list, max_length=4)
    answer_index: int | None = Field(default=None, ge=0, le=3)
    numeric_answer: str = Field(default='', max_length=100)
    explanation: str = Field(default='', max_length=1800)
    hint: str = Field(default='', max_length=700)
    practice_allowed: bool = True


class TaxonomyDraft(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    id: str = Field(pattern=ID)
    name: str = Field(min_length=2, max_length=100)
    kind: Literal['topic', 'skill']
    topic_id: str = Field(default='', max_length=80)
    concept_id: str = Field(default='', max_length=80)
    prerequisite: str = Field(default='', max_length=80)


def validated_question(draft, rows, version):
    skill = next((item for item in rows if item['id'] == draft['skill_id'] and item['kind'] == 'skill'), None)
    if not skill or skill['topic_id'] != draft['topic_id']:
        raise HTTPException(422, 'Choose a skill that belongs to the selected topic before publishing.')
    if not draft['prompt'] or not draft['explanation']:
        raise HTTPException(422, 'Question text and an explanation are required.')
    if draft['question_type'] == 'mcq':
        choices = [v.strip() for v in draft['choices']]
        if len(choices) != 4 or any(not c or len(c) > 300 for c in choices) or len(set(choices)) != 4 or draft['answer_index'] is None:
            raise HTTPException(422, 'Provide four different answer choices and select the answer key.')
        draft['choices'] = choices
    else:
        from fractions import Fraction
        try:
            Fraction(draft['numeric_answer'])
        except (ValueError, ZeroDivisionError):
            raise HTTPException(422, 'Enter a valid number or fraction as the answer key.')
    return {**draft, 'version': version, 'concept_id': skill['concept_id'], 'subject_id': 'mathematics',
            'origin': 'authored', 'status': 'published', 'content_version': f'{draft["id"]}:v{version}'}


def safe_question(q):
    return {k: q.get(k) for k in ('id','prompt','choices','concept_id','topic_id','skill_id','difficulty','question_type','version')}


def mount_content(app, admin, teacher, auth):
    @app.get('/api/content/taxonomy')
    async def list_taxonomy(user=Depends(auth)):
        return await taxonomy(app.state.db, realm(user))

    @app.post('/api/admin/taxonomy', status_code=201)
    async def add_taxonomy(body: TaxonomyDraft, user=Depends(admin)):
        db = app.state.db
        rows = await taxonomy(db, realm(user))
        if any(r['id'] == body.id for r in rows):
            raise HTTPException(409, 'This topic or skill ID already exists.')
        data = body.model_dump()
        if body.kind == 'topic' and body.concept_id not in CONCEPT_IDS:
            raise HTTPException(422, 'Link this topic to an existing BKT foundation.')
        if body.kind == 'skill':
            topic = next((t for t in rows if t['id'] == body.topic_id and t['kind'] == 'topic'), None)
            if not topic:
                raise HTTPException(422, 'Choose an existing topic.')
            data['concept_id'] = topic['concept_id']
            if body.prerequisite and not any(r['kind']=='skill' and r['id']==body.prerequisite for r in rows):
                raise HTTPException(422, 'Choose an existing prerequisite skill.')
        try:
            await db.taxonomy.insert_one({**data, '_id': key(realm(user), body.id), 'demo': realm(user), 'subject_id': 'mathematics'})
        except DuplicateKeyError:
            raise HTTPException(409, 'This topic or skill ID already exists.')
        return data

    @app.post('/api/admin/taxonomy/{id}/rename')
    async def rename_taxonomy(id: str, body: TaxonomyDraft, user=Depends(admin)):
        if id != body.id:
            raise HTTPException(422, 'Taxonomy IDs cannot be changed.')
        result = await app.state.db.taxonomy.update_one({'_id':key(realm(user),id),'kind':body.kind},
                                                        {'$set':{'name':body.name}})
        if not result.matched_count:
            raise HTTPException(404, 'Built-in topics and skills are stable references. Only added labels can be renamed.')
        return {'id':id,'name':body.name}

    @app.get('/api/teacher/question-options')
    async def eligible_questions(user=Depends(teacher)):
        bank = await available_questions(app.state.db, realm(user))
        return [safe_question(q) for q in bank.values()]

    @app.get('/api/admin/questions')
    async def list_questions(search: str=Query('', max_length=100), topic: str='', status: str='',
                             page: int=Query(1, ge=1), user=Depends(admin)):
        rows = {k: original(q) for k, q in BANK.items()}
        for row in await app.state.db.questions.find({'demo': realm(user)}, {'_id': 0}).to_list(5000):
            rows[row['id']] = {**(row.get('working') or row.get('live') or row.get('last_version') or {}),
                               'id': row['id'], 'status': row['status'], 'version': row['version'],
                               'has_published_version': bool(row.get('live'))}
        subset = sorted((q for q in rows.values() if (not topic or q.get('topic_id') == topic) and
                         (not status or q['status'] == status) and search.lower() in f"{q['id']} {q.get('prompt','')}".lower()),
                        key=lambda q: (q['status'] == 'published', q['id']))
        return {'questions': subset[(page-1)*20:page*20], 'total': len(subset), 'page': page}

    @app.post('/api/admin/questions', status_code=201)
    async def new_question(body: QuestionDraft, user=Depends(admin)):
        if body.id in BANK:
            raise HTTPException(409, 'Question ID already exists. Open it to draft a new version.')
        try:
            await app.state.db.questions.insert_one({'_id': key(realm(user), body.id), 'id': body.id,
                'status': 'draft', 'version': 0, 'revision': 0, 'working': body.model_dump(), 'demo': realm(user)})
        except DuplicateKeyError:
            raise HTTPException(409, 'Question ID already exists.')
        return {'id': body.id}

    async def question_record(id, user):
        db = app.state.db
        k = key(realm(user), id)
        row = await db.questions.find_one({'_id': k})
        if not row and id in BANK:
            base = original(BANK[id])
            row = {'_id': k, 'id': id, 'version': 1, 'revision': 0, 'status': 'published', 'demo': realm(user), 'live': base}
            try:
                await db.questions.insert_one(row)
            except DuplicateKeyError:
                row = await db.questions.find_one({'_id': k})
        if not row:
            raise HTTPException(404, 'Question not found.')
        return row

    async def change_question(row, changes):
        db = app.state.db
        result = await db.questions.update_one({'_id': row['_id'], 'revision': row['revision']},
            {'$set': changes, '$inc': {'revision': 1}})
        if not result.modified_count:
            raise HTTPException(409, 'Question changed in another tab. Reload before editing.')

    @app.post('/api/admin/questions/{id}/edit')
    async def edit_question(id: str, body: QuestionDraft, user=Depends(admin)):
        if id != body.id:
            raise HTTPException(422, 'The question ID cannot be changed.')
        row = await question_record(id, user)
        await change_question(row, {'working': body.model_dump(), 'status': 'draft'})
        return {'id': id}

    @app.post('/api/admin/questions/{id}/publish')
    async def publish_question(id: str, user=Depends(admin)):
        row = await question_record(id,user)
        working = row.get('working')
        if not working:
            raise HTTPException(409, 'Save a draft before publishing a new version.')
        version = row['version'] + 1
        record = validated_question(working, await taxonomy(app.state.db, realm(user)), version)
        await change_question(row, {'live': record, 'last_version': record, 'working': None, 'version': version,
                                    'status': 'published', 'published_at': datetime.now(timezone.utc)})
        return {'id': id, 'version': version}

    @app.post('/api/admin/questions/{id}/archive')
    async def archive_question(id: str, user=Depends(admin)):
        row = await question_record(id,user)
        if not row['version']:
            raise HTTPException(422, 'Unused drafts can be deleted instead.')
        await change_question(row, {'live': None, 'last_version': row.get('live') or row.get('last_version'), 'status': 'archived'})
        return {'id': id, 'status': 'archived'}

    @app.delete('/api/admin/questions/{id}', status_code=204)
    async def delete_question(id: str, user=Depends(admin)):
        row = await question_record(id,user)
        if row['version'] or row['status'] != 'draft':
            raise HTTPException(409, 'Published or previously used questions must be archived.')
        result = await app.state.db.questions.delete_one({'_id': row['_id'], 'revision': row['revision'], 'version': 0})
        if not result.deleted_count:
            raise HTTPException(409, 'Question changed in another tab.')
