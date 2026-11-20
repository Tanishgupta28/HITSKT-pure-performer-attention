"""Common papers: versioned items, owner-scoped drafts, explicit grading and release."""
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from typing import Literal
from uuid import uuid4

from fastapi import Depends, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from pymongo import ReturnDocument

from app.catalog import BANK, CONCEPTS
from app.content import available_questions, original, realm, safe_question, scope
from app.engine import concept_rows, initial_concept, recommendations, update_knowledge


def clock():
    return datetime.now(timezone.utc)


def parse(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def grade(q, response):
    if q.get('question_type') == 'numeric':
        try:
            return Fraction(response) == Fraction(q['numeric_answer'])
        except (TypeError, ValueError, ZeroDivisionError):
            return False
    return type(response) is int and response == q['answer_index']


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class PaperDraft(Strict):
    title: str = Field(min_length=3, max_length=120)
    class_id: str = Field(min_length=1, max_length=80)
    question_ids: list[str] = Field(default_factory=list, max_length=60)
    duration_minutes: int = Field(default=20, ge=2, le=180)
    opens_at: AwareDatetime | None = None
    closes_at: AwareDatetime | None = None
    review_allowed: bool = True
    skips_allowed: bool = True
    show_explanations: bool = True
    extra_minutes: dict[str, int] = Field(default_factory=dict, max_length=250)


class ItemAction(Strict):
    action: Literal['answer','skip','open']
    index: int = Field(ge=0, le=59)
    choice: int | str | None = None


def paper_public(row):
    return {k: v for k,v in row.items() if k not in ('questions','demo','teacher_id')}


def result_for(a, review=True):
    rows = {}
    details = []
    for i, q in enumerate(a['questions']):
        saved = a['answers'].get(str(i), {})
        answered = 'value' in saved
        correct = grade(q, saved.get('value')) if answered else False
        topic = q['concept_id']
        record = rows.setdefault(topic, {'concept_id': topic, 'shown': 0, 'answered':0, 'correct':0})
        record['shown'] += 1
        record['answered'] += int(answered)
        record['correct'] += int(correct)
        detail = {**safe_question(q), 'response': saved.get('value'), 'status': 'correct' if correct else
                  'incorrect' if answered else 'skipped' if saved.get('skipped') else 'unanswered'}
        if review:
            detail.update(answer_key=q['numeric_answer'] if q.get('question_type')=='numeric' else q.get('answer_index'),
                          explanation=q['explanation'])
        details.append(detail)
    answered = sum(r['answered'] for r in rows.values())
    correct = sum(r['correct'] for r in rows.values())
    return {'score': correct, 'total': len(a['questions']), 'answered': answered,
            'accuracy': correct/answered if answered else None, 'topics': list(rows.values()),
            'questions': details, 'elapsed_seconds': max(0, int((parse(a['submitted_at'])-parse(a['started_at'])).total_seconds()))}


def effective_deadline(a, paper=None):
    dates = [parse(a.get('deadline_at'))]
    if paper:
        dates += [paper.get('closed_at'), paper.get('closes_at')]
    return min((x for x in dates if x), default=None)


def start_fixed(state, questions, title, mode, assignment_id=None, deadline=None, **rules):
    if len(state['history']) + len(questions) > 5000 or len(state.get('fixed', {})) >= 80:
        raise HTTPException(409, 'This account has reached its learning history capacity.')
    stamp = clock().isoformat()
    a = {'id':str(uuid4()), 'assignment_id': assignment_id, 'mode':mode, 'title':title, 'questions':questions,
         'answers':{}, 'index':0, 'opened':{'0':stamp}, 'started_at':stamp, 'deadline_at':deadline.isoformat() if deadline else None,
         'submitted_at':None, 'rules':rules}
    state.setdefault('fixed', {})[a['id']] = a
    return a


def finalize(state, a, stamp, reason='submitted'):
    if a.get('submitted_at'):
        return
    a['submitted_at'] = stamp.isoformat()
    a['submission_reason'] = reason
    if a['mode'] == 'diagnostic':
        incorporate(state, a, stamp)


def incorporate(state, a, stamp):
    if a.get('incorporated'):
        return
    touches = sorted((r['saved_at'], int(i), r) for i,r in a['answers'].items() if 'value' in r)
    for answered_at,i,item in touches:
        q = a['questions'][i]
        concept = q['concept_id']
        correct = grade(q, item['value'])
        obs = update_knowledge(state['concepts'].get(concept, initial_concept()), correct, q['difficulty'], stamp.isoformat())
        state['concepts'][concept] = obs
        state['history'].append({'question_id':q['id'],'assessment_id':a['id'],'concept_id':concept,
            'skill_id':q.get('skill_id',concept),'content_version':q['content_version'],'difficulty':q['difficulty'],
            'correct':correct,'choice_index':item['value'],'answered_at':answered_at,
            'response_time_ms':item['response_time_ms'],'hints_used':0,'request_id':f'{a["id"]}:{i}',
            'mode':a['mode'],'source':'synthetic' if state.get('demo') else 'live',
            'prediction':{'provider':'not_scored','probability':None}})
    a['summary'] = {'concepts':concept_rows(state), 'recommendations':recommendations(state)}
    a['incorporated'] = True


def fixed_view(a, paper):
    submitted = bool(a.get('submitted_at'))
    released = a['mode']=='diagnostic' or bool(paper and paper['status']=='released')
    statuses = ['answered' if 'value' in a['answers'].get(str(i),{}) else
                'skipped' if a['answers'].get(str(i),{}).get('skipped') else 'unanswered' for i in range(len(a['questions']))]
    current = a['index']
    deadline = effective_deadline(a, paper)
    return {'id':a['id'],'assignment_id':a['assignment_id'],'title':a['title'],'mode':a['mode'],'status':'submitted' if submitted else 'active',
            'index':current,'total':len(a['questions']),'started_at':a['started_at'],'deadline_at':deadline,
            'submitted_at':a['submitted_at'],'submission_reason':a.get('submission_reason'),'server_time':clock(),'positions':statuses,'rules':a['rules'],
            'visited_until':max((int(i) for i in a['opened']),default=current),
            'question':None if submitted else safe_question(a['questions'][current]),
            'saved_choice':None if submitted else a['answers'].get(str(current),{}).get('value'),
            'released':submitted and released,'result':result_for(a,a['rules'].get('show_explanations',True)) if submitted and released else None,
            'summary':a.get('summary') if submitted and released else None}


def mount_papers(app, student, teacher, learning, save, get_class):
    async def reconcile(state):
        pending = [a for a in state.get('fixed',{}).values() if a['assignment_id'] and not a.get('incorporated')]
        if not pending:
            return
        found = {p['_id']:p for p in await app.state.db.papers.find({'_id':{'$in':[a['assignment_id'] for a in pending]},'status':'released'}).to_list(80)}
        changed = False
        for a in pending:
            p = found.get(a['assignment_id'])
            if p:
                deadline = effective_deadline(a,p)
                if not a['submitted_at'] and deadline and deadline <= clock(): finalize(state,a,deadline,'deadline')
                if a['submitted_at']:
                    incorporate(state,a,parse(a['submitted_at'])); changed = True
        if changed: await save(state)
    app.state.reconcile_papers = reconcile
    async def visible_paper(paper_id,user,assigner=False):
        db=app.state.db
        q={'_id':paper_id,**scope(user)}
        if assigner: q['teacher_id']=user['_id']
        else: q['status']={'$in':['assigned','closed','released']}
        row=await db.papers.find_one(q)
        if not row:
            raise HTTPException(404,'Assessment not found.')
        return row

    @app.get('/api/teacher/papers')
    async def list_papers(user=Depends(teacher)):
        rows=await app.state.db.papers.find({'teacher_id':user['_id'],**scope(user)}).sort('created_at',-1).to_list(200)
        return [paper_public(row) for row in rows]

    async def validate_draft(body,user):
        cls=await get_class(body.class_id, user)
        if len(body.question_ids)!=len(set(body.question_ids)):
            raise HTTPException(422,'Select each question only once.')
        if body.opens_at and body.closes_at and body.opens_at>=body.closes_at:
            raise HTTPException(422,'The closing time must follow the opening time.')
        if any(v<0 or v>120 for v in body.extra_minutes.values()) or not set(body.extra_minutes)<=set(cls['members']):
            raise HTTPException(422,'Accommodations must match students in this class (0–120 minutes).')
        return cls

    @app.post('/api/teacher/papers',status_code=201)
    async def create_paper(body:PaperDraft,user=Depends(teacher)):
        cls=await validate_draft(body,user)
        data={**body.model_dump(),'_id':str(uuid4()),'class_name':cls['name'],'teacher_id':user['_id'],'demo':realm(user),
              'status':'draft','revision':0,'created_at':clock()}
        await app.state.db.papers.insert_one(data)
        return paper_public(data)

    @app.post('/api/teacher/papers/{paper_id}/edit')
    async def edit_paper(paper_id:str,body:PaperDraft,user=Depends(teacher)):
        row=await visible_paper(paper_id,user,True)
        cls=await validate_draft(body,user)
        edited=await app.state.db.papers.find_one_and_update({'_id':paper_id,'status':'draft','revision':row['revision']},
            {'$set':{**body.model_dump(),'class_name':cls['name']},'$inc':{'revision':1}},return_document=ReturnDocument.AFTER)
        if not edited: raise HTTPException(409,'Only the current draft can be edited.')
        return paper_public(edited)

    @app.post('/api/teacher/papers/{paper_id}/assign')
    async def assign(paper_id:str,user=Depends(teacher)):
        row=await visible_paper(paper_id,user,True)
        cls=await get_class(row['class_id'],user)
        if not any(m['status']=='active' for m in cls['members'].values()):
            raise HTTPException(422,'Add students to the class before assigning a paper.')
        bank=await available_questions(app.state.db,realm(user))
        if not row['question_ids'] or not set(row['question_ids'])<=bank.keys():
            raise HTTPException(422,'Select published questions. An archived or missing item must be replaced.')
        if row.get('closes_at') and row['closes_at']<=clock(): raise HTTPException(422,'Set a future deadline.')
        found=await app.state.db.papers.find_one_and_update({'_id':paper_id,'status':'draft','revision':row['revision']},
            {'$set':{'status':'assigned','questions':[bank[i] for i in row['question_ids']],'assigned_at':clock()},'$inc':{'revision':1}},
            return_document=ReturnDocument.AFTER)
        if not found: raise HTTPException(409,'Draft changed or was already assigned.')
        return paper_public(found)

    async def expire_if_needed(state,a,paper):
        deadline=effective_deadline(a,paper)
        if not a.get('submitted_at') and deadline and deadline<=clock():
            finalize(state,a,deadline,'deadline')
            await save(state)

    @app.post('/api/teacher/papers/{paper_id}/close')
    async def close(paper_id:str,user=Depends(teacher)):
        row=await visible_paper(paper_id,user,True)
        if row['status']!='assigned': raise HTTPException(409,'Only an open assignment can be closed.')
        await app.state.db.papers.update_one({'_id':paper_id,'status':'assigned'}, {'$set':{'status':'closed','closed_at':clock()}})
        return {'status':'closed'}

    @app.post('/api/teacher/papers/{paper_id}/release')
    async def release(paper_id:str,user=Depends(teacher)):
        row=await visible_paper(paper_id,user,True)
        if row['status']!='closed' and not (row['status']=='assigned' and row.get('closes_at') and row['closes_at']<=clock()):
            raise HTTPException(409,'Close the paper before releasing answers and scores.')
        await app.state.db.papers.update_one({'_id':paper_id,'status':row['status']}, {'$set':{'status':'released','released_at':clock()}})
        return {'status':'released'}

    @app.get('/api/teacher/papers/{paper_id}/results')
    async def results(paper_id:str,user=Depends(teacher)):
        p=await visible_paper(paper_id,user,True)
        cls=await get_class(p['class_id'],user)
        students=await app.state.db.users.find({'_id':{'$in':list(cls['members'])},**scope(user)},{'name':1}).to_list(250)
        submissions=[]
        aggregate=Counter(); denominator=Counter()
        for learner in students:
            state=await app.state.db.learning.find_one({'_id':f'{learner["_id"]}:mathematics'})
            a=next((a for a in (state or {}).get('fixed',{}).values() if a['assignment_id']==paper_id), None)
            if not a and cls['members'][learner['_id']]['status']!='active': continue
            if a: await expire_if_needed(state,a,p)
            result=result_for(a,False) if a and a['submitted_at'] else None
            if result:
                for t in result['topics']:
                    aggregate[t['concept_id']]+=t['correct']; denominator[t['concept_id']]+=t['shown']
            submissions.append({'student_id':learner['_id'],'name':learner['name'],
                'status':'submitted' if result else 'in_progress' if a else 'not_started','score':result['score'] if result else None,
                'total':result['total'] if result else len(p['question_ids']), 'submitted_at':a['submitted_at'] if a else None,
                'elapsed_seconds':result['elapsed_seconds'] if result else None})
        return {'paper':paper_public(p),'submissions':submissions,'topics':[{'concept_id':k,'correct':aggregate[k],'total':v} for k,v in denominator.items()]}

    @app.get('/api/student/papers')
    async def assigned_list(user=Depends(student)):
        db=app.state.db
        state=await learning(user)
        cls=await db.classes.find({f'members.{user["_id"]}.status':'active',**scope(user)},{'name':1}).to_list(250)
        by_assignment={a['assignment_id']:a for a in state.get('fixed',{}).values() if a['assignment_id']}
        rows=await db.papers.find({**scope(user),'status':{'$in':['assigned','closed','released']},
            '$or':[{'class_id':{'$in':[c['_id'] for c in cls]}},{'_id':{'$in':list(by_assignment)}}]}).sort('assigned_at',-1).to_list(250)
        results = []
        for paper in rows:
            attempt = by_assignment.get(paper['_id'])
            if attempt:
                await expire_if_needed(state, attempt, paper)
            results.append({k:paper.get(k) for k in ('_id','title','class_name','status','duration_minutes','opens_at','closes_at')} |
                           {'question_count':len(paper['questions']), 'review_allowed':paper['review_allowed'],
                            'skips_allowed':paper['skips_allowed'], 'extra_minutes':paper.get('extra_minutes',{}).get(user['_id'],0),
                            'attempt_id':attempt['id'] if attempt else None,
                            'attempt_status':('submitted' if attempt['submitted_at'] else 'in_progress') if attempt else None})
        return results

    @app.post('/api/student/fixed/diagnostic',status_code=201)
    async def diagnostic(user=Depends(student)):
        state=await learning(user); state['demo']=realm(user)
        active=next((a for a in state.get('fixed',{}).values() if a['mode']=='diagnostic' and not a['submitted_at']),None)
        if active:return fixed_view(active,None)
        bank=await available_questions(app.state.db,realm(user))
        seen={r['question_id'] for r in state['history']}
        qs=[]
        for concept in CONCEPTS:
            pool=sorted((q for q in bank.values() if q['concept_id']==concept['id'] and q.get('practice_allowed',True)),key=lambda q:(q['id'] in seen,q['difficulty'],q['id']))
            if len(pool)<3: raise HTTPException(409,'A required topic has too few published questions. Contact your teacher.')
            qs+=pool[:3]
        a=start_fixed(state,qs,'Mathematics diagnostic','diagnostic',review_allowed=True,skips_allowed=True,show_explanations=True)
        await save(state)
        return fixed_view(a,None)

    @app.post('/api/student/papers/{paper_id}/start',status_code=201)
    async def start(paper_id:str,user=Depends(student)):
        p=await visible_paper(paper_id,user)
        state=await learning(user); state['demo']=realm(user)
        a=next((a for a in state.get('fixed',{}).values() if a['assignment_id']==paper_id),None)
        if a:
            await expire_if_needed(state,a,p)
            return fixed_view(a,p)
        c=await app.state.db.classes.find_one({'_id':p['class_id'],f'members.{user["_id"]}.status':'active',**scope(user)})
        if not c: raise HTTPException(404,'This assessment is not assigned to you.')
        now=clock()
        if p['status']!='assigned' or p.get('closes_at') and p['closes_at']<=now:
            raise HTTPException(409,'This assessment is closed.')
        if p.get('opens_at') and p['opens_at']>now: raise HTTPException(409,'This assessment has not opened yet.')
        duration=p['duration_minutes']+p.get('extra_minutes',{}).get(user['_id'],0)
        deadline=min(now+timedelta(minutes=duration), p.get('closes_at') or now+timedelta(minutes=duration))
        a=start_fixed(state,p['questions'],p['title'],'exam',paper_id,deadline,
            review_allowed=p['review_allowed'],skips_allowed=p['skips_allowed'],show_explanations=p['show_explanations'])
        await save(state)
        return fixed_view(a,p)

    async def own_attempt(id,user):
        state=await learning(user)
        a=state.get('fixed',{}).get(id)
        if not a: raise HTTPException(404,'Assessment attempt not found.')
        p=await visible_paper(a['assignment_id'],user) if a['assignment_id'] else None
        await expire_if_needed(state,a,p)
        if p and not a['submitted_at'] and not await app.state.db.classes.find_one({'_id':p['class_id'],f'members.{user["_id"]}.status':'active'}):
            raise HTTPException(403,'This class assignment is no longer available to you.')
        return state,a,p

    @app.get('/api/student/fixed/{id}')
    async def view(id:str,user=Depends(student)):
        _,a,p=await own_attempt(id,user)
        return fixed_view(a,p)

    @app.post('/api/student/fixed/{id}/item')
    async def item(id:str,body:ItemAction,user=Depends(student)):
        state,a,p=await own_attempt(id,user)
        if a['submitted_at']: return fixed_view(a,p)
        n=body.index
        if n>=len(a['questions']): raise HTTPException(422,'Question is out of range.')
        curr=a['index']; now=clock(); stamp=now.isoformat()
        if body.action=='open':
            farthest=max((int(i) for i in a['opened']),default=curr)
            if n>farthest+1 or (not a['rules']['review_allowed'] and n<curr): raise HTTPException(409,'Navigation for this paper is restricted.')
            if not a['rules']['skips_allowed'] and n>curr and 'value' not in a['answers'].get(str(curr),{}): raise HTTPException(409,'Save an answer before moving on.')
            a['index']=n; a['opened'].setdefault(str(n),stamp)
        else:
            if n!=curr: raise HTTPException(409,'Reload this attempt before saving a different question.')
            qs=a['questions'][n]; k=str(n)
            if body.action=='skip':
                if not a['rules']['skips_allowed']: raise HTTPException(409,'Skipping is disabled for this paper.')
                if 'value' not in a['answers'].get(k,{}): a['answers'][k]={'skipped':True}
            else:
                value=body.choice
                if qs.get('question_type')=='numeric':
                    try:
                        if not isinstance(value,str) or not value or len(value)>100: raise ValueError()
                        Fraction(value)
                    except (ValueError, ZeroDivisionError): raise HTTPException(422,'Enter a number or fraction.')
                elif type(value) is not int or value<0 or value>=len(qs['choices']):
                    raise HTTPException(422,'Select one answer.')
                if a['answers'].get(k,{}).get('value')!=value:
                    a['answers'][k]={'value':value,'saved_at':stamp,'response_time_ms':max(0,int((now-parse(a['opened'].get(k,stamp))).total_seconds()*1000))}
        await save(state)
        return fixed_view(a,p)

    @app.post('/api/student/fixed/{id}/submit')
    async def submit(id:str,user=Depends(student)):
        state,a,p=await own_attempt(id,user)
        if not a['submitted_at']:
            finalize(state,a,clock()); await save(state)
        return fixed_view(a,p)

    @app.get('/api/student/history')
    async def history(user=Depends(student)):
        state=await learning(user)
        topic_names = {c['id']: c['name'] for c in CONCEPTS}
        def session_title(assessment):
            if assessment['mode'] == 'diagnostic':
                return 'Diagnostic'
            focus = topic_names.get(assessment.get('concept_id'))
            return f'{focus} practice' if focus else 'Mixed practice'

        items=[{'id':a['id'],'mode':a['mode'],'status':a['status'],
                'title':session_title(a),
                'answered':len(a['responses']),'score':sum(r['correct'] for r in a['responses']),'total':a['total'],
                'date':a.get('completed_at',a['started_at']),'elapsed_seconds':max(0,int((parse(a.get('completed_at',a['started_at']))-parse(a['started_at'])).total_seconds())),
                'concept_id':a.get('concept_id'),'fixed':False} for a in state['assessments']]
        for a in state.get('fixed',{}).values():
            p=await visible_paper(a['assignment_id'],user) if a['assignment_id'] else None
            await expire_if_needed(state,a,p)
            v=fixed_view(a,p)
            items.append({'id':a['id'],'mode':a['mode'],'status':v['status'],'title':a['title'],'fixed':True,
              'answered':v['positions'].count('answered'),'score':v['result']['score'] if v['result'] else None,'total':v['total'],
              'date':a.get('submitted_at') or a['started_at'],'elapsed_seconds':v['result']['elapsed_seconds'] if v['result'] else None})
        return sorted(items,key=lambda a:a['date'],reverse=True)
