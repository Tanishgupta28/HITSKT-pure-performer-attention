import hashlib
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from pymongo import AsyncMongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError
from pwdlib import PasswordHash
from starlette.concurrency import run_in_threadpool

from app.catalog import BANK, CONCEPTS, SUBJECT, public_question
from app.config import Settings
from app.engine import concept_rows, initial_concept, recommendations, select_question, update_knowledge
from app.inference import Predictor

passwords = PasswordHash.recommended()
DUMMY_HASH = passwords.hash("dummy-password-for-constant-work")
COOKIE = "hitskt_session"


def now():
    return datetime.now(timezone.utc)


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class Registration(Credentials):
    name: str = Field(min_length=2, max_length=80, pattern=r".*\S.*")


class StartAssessment(BaseModel):
    subject_id: Literal["mathematics"] = "mathematics"
    mode: Literal["diagnostic", "adaptive", "practice"] = "adaptive"
    concept_id: str | None = None


class Answer(BaseModel):
    question_id: str = Field(max_length=100)
    choice_index: int = Field(ge=0, le=3)
    request_id: UUID


def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        client = AsyncMongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000, tz_aware=True)
        db = client[settings.mongodb_database]
        await db.command("ping")
        await db.users.create_index("email", unique=True)
        await db.sessions.create_index("expires_at", expireAfterSeconds=0)
        await db.rate_limits.create_index("expires_at", expireAfterSeconds=0)
        app.state.db = db
        app.state.predictor = await run_in_threadpool(Predictor, settings)
        yield
        await client.close()

    app = FastAPI(title="HiTSKT Learning API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_origin], allow_credentials=True,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def origin_guard(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != settings.frontend_origin.rstrip("/"):
                return JSONResponse({"detail": "Request origin is not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    async def auth(request: Request):
        token = request.cookies.get(COOKIE)
        session = await app.state.db.sessions.find_one({"_id": token_hash(token)}) if token else None
        if not session or session["expires_at"] <= now():
            raise HTTPException(401, "Please sign in to continue")
        user = await app.state.db.users.find_one({"_id": session["user_id"]})
        if not user:
            raise HTTPException(401, "Account is unavailable")
        return user

    async def limit_auth(request):
        # Persistent across workers. Never trust a client-supplied forwarded IP.
        stamp = now()
        ip = request.client.host if request.client else "unknown"
        key = token_hash(f"{ip}:{stamp.strftime('%Y%m%d%H%M')}")
        document = await app.state.db.rate_limits.find_one_and_update(
            {"_id": key}, {"$inc": {"count": 1}, "$setOnInsert": {"expires_at": stamp + timedelta(minutes=2)}},
            upsert=True, return_document=ReturnDocument.AFTER)
        if document["count"] > 20:
            raise HTTPException(429, "Too many attempts. Try again in a minute.")

    async def issue_session(user, response):
        token = secrets.token_urlsafe(48)
        await app.state.db.sessions.insert_one({"_id": token_hash(token), "user_id": user["_id"],
                                                "expires_at": now() + timedelta(days=settings.session_days)})
        response.set_cookie(COOKIE, token, httponly=True, secure=settings.cookie_secure, samesite="lax",
                            max_age=settings.session_days * 86400, path="/")
        return {"id": user["_id"], "name": user["name"], "email": user["email"]}

    async def learning(user, subject_id="mathematics"):
        if subject_id != SUBJECT["id"]:
            raise HTTPException(404, "Subject not found")
        key = f"{user['_id']}:{subject_id}"
        state = await app.state.db.learning.find_one({"_id": key})
        if state is None:
            state = {"_id": key, "user_id": user["_id"], "subject_id": subject_id, "version": 0,
                     "concepts": {c["id"]: initial_concept() for c in CONCEPTS}, "history": [], "assessments": []}
            try:
                await app.state.db.learning.insert_one(state)
            except DuplicateKeyError:
                state = await app.state.db.learning.find_one({"_id": key})
        return state

    async def save(state):
        original = state["version"]
        state["version"] += 1
        result = await app.state.db.learning.replace_one({"_id": state["_id"], "version": original}, state)
        if not result.modified_count:
            raise HTTPException(409, "Learning state changed in another tab. Reload and try again.")

    def assessment_view(assessment):
        pending = assessment.get("pending")
        return {"id": assessment["id"], "mode": assessment["mode"], "status": assessment["status"],
                "concept_id": assessment.get("concept_id"), "answered": len(assessment["responses"]),
                "total": assessment["total"], "started_at": assessment["started_at"],
                "correct": sum(r["correct"] for r in assessment["responses"]),
                "question": public_question(BANK[pending["question_id"]]) if pending else None,
                "prediction": pending["prediction"] if pending else None,
                "selection_reason": pending["selection_reason"] if pending else None,
                "summary": assessment.get("summary"), "latest_feedback": assessment.get("latest_feedback")}

    async def next_question(state, assessment):
        state["inference_at"] = now().isoformat()
        pending = await run_in_threadpool(select_question, state, assessment, app.state.predictor)
        if pending:
            pending["served_at"] = now().isoformat()
        return pending

    @app.get("/api/health")
    async def health():
        await app.state.db.command("ping")
        return {"status": "ok", "database": "mongodb", "model": app.state.predictor.status()}

    @app.post("/api/auth/register", status_code=201)
    async def register(data: Registration, request: Request, response: Response):
        await limit_auth(request)
        user = {"_id": str(uuid4()), "name": data.name.strip(), "email": str(data.email).lower(),
                "password_hash": await run_in_threadpool(passwords.hash, data.password), "created_at": now()}
        try:
            await app.state.db.users.insert_one(user)
        except DuplicateKeyError:
            raise HTTPException(409, "This email is already registered")
        return await issue_session(user, response)

    @app.post("/api/auth/login")
    async def login(data: Credentials, request: Request, response: Response):
        await limit_auth(request)
        user = await app.state.db.users.find_one({"email": str(data.email).lower()})
        valid = await run_in_threadpool(passwords.verify, data.password, user["password_hash"] if user else DUMMY_HASH)
        if not user or not valid:
            raise HTTPException(401, "Email or password is incorrect")
        return await issue_session(user, response)

    @app.get("/api/auth/me")
    async def me(user=Depends(auth)):
        return {"id": user["_id"], "name": user["name"], "email": user["email"]}

    @app.post("/api/auth/logout", status_code=204)
    async def logout(request: Request, response: Response):
        token = request.cookies.get(COOKIE)
        if token:
            await app.state.db.sessions.delete_one({"_id": token_hash(token)})
        response.delete_cookie(COOKIE, path="/")

    @app.get("/api/subjects")
    async def subjects(user=Depends(auth)):
        return [SUBJECT]

    @app.get("/api/learning/{subject_id}")
    async def dashboard(subject_id: str, user=Depends(auth)):
        state = await learning(user, subject_id)
        completed = [a for a in state["assessments"] if a["status"] == "completed"]
        active = next((a for a in state["assessments"] if a["status"] == "active"), None)
        history = state["history"]
        return {"subject": SUBJECT, "concepts": concept_rows(state), "recommendations": recommendations(state),
                "stats": {"answers": len(history), "accuracy": sum(r["correct"] for r in history) / len(history) if history else None,
                          "completed_assessments": len(completed), "study_days": len({r["answered_at"][:10] for r in history})},
                "active_assessment": assessment_view(active) if active else None,
                "diagnostic_complete": any(a["mode"] == "diagnostic" for a in completed),
                "progress": [{"id": a["id"], "mode": a["mode"], "completed_at": a["completed_at"],
                              "accuracy": a["summary"]["accuracy"], "mastery": a["summary"]["average_mastery"]} for a in completed[-20:]],
                "recent_answers": [{k: r[k] for k in ("question_id", "concept_id", "correct", "answered_at", "difficulty", "prediction")} for r in history[-8:]][::-1],
                "model": app.state.predictor.status()}

    @app.post("/api/assessments", status_code=201)
    async def start(data: StartAssessment, user=Depends(auth)):
        state = await learning(user, data.subject_id)
        if data.concept_id and data.concept_id not in state["concepts"]:
            raise HTTPException(422, "Unknown concept")
        if data.mode == "diagnostic" and data.concept_id:
            raise HTTPException(422, "Diagnostics must cover every concept")
        active = next((a for a in state["assessments"] if a["status"] == "active"), None)
        if active:
            return assessment_view(active)
        total = 6 if data.mode == "practice" else 12
        if len(state["history"]) + total > 5000:
            raise HTTPException(409, "Learning archive capacity reached. Contact the administrator to archive this history.")
        assessment = {"id": str(uuid4()), "mode": data.mode, "concept_id": data.concept_id,
                      "status": "active", "total": total,
                      "responses": [], "started_at": now().isoformat()}
        assessment["pending"] = await next_question(state, assessment)
        state["assessments"].append(assessment)
        await save(state)
        return assessment_view(assessment)

    @app.get("/api/assessments/{assessment_id}")
    async def get_assessment(assessment_id: UUID, user=Depends(auth)):
        state = await learning(user)
        assessment = next((a for a in state["assessments"] if a["id"] == str(assessment_id)), None)
        if assessment is None:
            raise HTTPException(404, "Assessment not found")
        return assessment_view(assessment)

    @app.post("/api/assessments/{assessment_id}/answers")
    async def answer(assessment_id: UUID, data: Answer, user=Depends(auth)):
        state = await learning(user)
        assessment = next((a for a in state["assessments"] if a["id"] == str(assessment_id)), None)
        if assessment is None:
            raise HTTPException(404, "Assessment not found")
        duplicate = next((r for a in state["assessments"] for r in a["responses"] if r["request_id"] == str(data.request_id)), None)
        if duplicate:
            if duplicate["question_id"] != data.question_id or duplicate["choice_index"] != data.choice_index or duplicate["assessment_id"] != str(assessment_id):
                raise HTTPException(409, "Request ID was already used for another answer")
            return {"assessment": assessment_view(assessment), "feedback": duplicate["feedback"], "replayed": True}
        if assessment["status"] != "active":
            raise HTTPException(409, "Assessment is already completed")
        pending = assessment["pending"]
        if not pending or pending["question_id"] != data.question_id:
            raise HTTPException(409, "This is not the current question. Reload the assessment.")
        question = BANK[data.question_id]
        correct = data.choice_index == question["answer_index"]
        timestamp = now().isoformat()
        concept_id = question["concept_id"]
        updated = update_knowledge(state["concepts"][concept_id], correct, question["difficulty"], timestamp)
        feedback = {"correct": correct, "answer_index": question["answer_index"], "explanation": question["explanation"],
                    "concept_name": next(c["name"] for c in CONCEPTS if c["id"] == concept_id), "mastery": updated["mastery"]}
        record = {"question_id": question["id"], "assessment_id": assessment["id"], "concept_id": concept_id,
                  "difficulty": question["difficulty"], "content_version": question["content_version"],
                  "correct": correct, "choice_index": data.choice_index, "answered_at": timestamp,
                  "response_time_ms": max(0, int((datetime.fromisoformat(timestamp) - datetime.fromisoformat(pending["served_at"])).total_seconds() * 1000)),
                  "request_id": str(data.request_id), "prediction": pending["prediction"], "feedback": feedback}
        state["concepts"][concept_id] = updated
        state["history"].append(record)
        assessment["responses"].append(record)
        assessment["latest_feedback"] = feedback
        if len(assessment["responses"]) >= assessment["total"]:
            assessment["status"] = "completed"
            assessment["pending"] = None
            assessment["completed_at"] = timestamp
            assessment["summary"] = {"accuracy": sum(r["correct"] for r in assessment["responses"]) / len(assessment["responses"]),
                                     "average_mastery": sum(c["mastery"] for c in state["concepts"].values()) / len(CONCEPTS),
                                     "concepts": concept_rows(state), "recommendations": recommendations(state)}
        else:
            assessment["pending"] = await next_question(state, assessment)
        # One MongoDB atomic compare-and-swap persists response, mastery and next
        # question together; no partial updates or replica-set transaction needed.
        await save(state)
        return {"assessment": assessment_view(assessment), "feedback": feedback, "replayed": False}

    return app


app = create_app()
