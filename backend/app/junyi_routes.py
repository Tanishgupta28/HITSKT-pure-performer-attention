"""Account-scoped Junyi assessment, response history, and study workflows."""

import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from time import time_ns
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from pymongo import DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError
from starlette.concurrency import run_in_threadpool

from app.junyi_bank import digest
from app.junyi_model import GAP, ModelUnavailable
from app.junyi_paper import BankShortage, candidates_for, freeze_items, prompt_candidates, select_paper
from app.junyi_resources import STUDY_RESOURCE_TTL
from app.junyi_study_policy import FOREST_POLICY, rule_checklist


logger = logging.getLogger(__name__)
LEASE_MS = 30_000
STUDY_LOOKUP_LEASE_MS = 120_000
STUDY_RECHECK_MS = 120_000
IN_PROGRESS = ("open", "locked")


def _now():
    return time_ns() // 1_000_000


class AnswerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(min_length=1, max_length=100)
    choice_id: str = Field(min_length=1, max_length=100)
    request_id: UUID


class StudyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=8, max_length=2500)


class OverrideInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=5, max_length=500, pattern=r".*\S.*")


def matches_account(user, configured_id):
    return bool(configured_id) and user["_id"] == configured_id


async def setup_junyi(db):
    await db.junyi_papers.create_index("owner", unique=True, partialFilterExpression={"active": True}, name="one_active_junyi_paper")
    await db.junyi_papers.create_index([("owner", 1), ("sequence", 1)], unique=True)
    await db.junyi_optional_study.create_index([("owner", 1), ("tag", 1)], unique=True)


def _public_skills(snapshot):
    return [{key: value for key, value in row.items() if key != "probes"} for row in snapshot["skills"]] if snapshot else []


def _study_required(requirement):
    return (requirement["status"] == "pending" and requirement["lookup"] == "verified"
            and bool(requirement["resources"]))


def _paper_status(paper):
    if paper["status"] in IN_PROGRESS:
        return "locked" if any(map(_study_required, paper.get("requirements", ()))) else "open"
    return paper["status"]


def _study_lookup_due(requirement, now):
    return requirement["status"] == "pending" and (
        requirement["lookup"] == "waiting"
        or (requirement["lookup"] == "loading" and requirement.get("lease_until", 0) <= now)
        or (requirement["lookup"] == "unavailable" and requirement.get("retry_after", 0) <= now))


def _requirement_view(requirement):
    view = {key: requirement[key] for key in ("tag", "topic", "status", "lookup", "resources", "message") if key in requirement}
    view["required"] = _study_required(requirement)
    view["can_retry"] = requirement["status"] == "pending" and (
        requirement["lookup"] in {"waiting", "unavailable"}
        or (requirement["lookup"] == "loading" and requirement.get("lease_until", 0) <= _now()))
    return view


def _question(paper):
    item = paper["items"][len(paper["responses"])]
    return {"id": item["id"], "topic": item["topic"], "prompt": item["question"], "options": item["options"]}


def _view(paper, *, operator=False):
    status = _paper_status(paper)
    done = status == "completed"
    result_snapshot = paper.get("result_snapshot") if done else None
    return {"id": paper["_id"], "sequence": paper["sequence"], "revision": paper["version"],
            "status": status, "stage": paper["stage"], "total": paper["total"],
            "answered": len(paper["responses"]), "requirements": list(map(_requirement_view, paper["requirements"])),
            "message": paper.get("message"), "created_at": paper["created_at"],
            "question": _question(paper) if status == "open" and not operator else None,
            "results": paper.get("results") if done else None, "analysis_status": paper.get("analysis_status") if done else None,
            "analysis_retry": done and paper.get("analysis_status") in {"pending", "unavailable"} and paper.get("analysis_lease_until", 0) <= _now(),
            "answers_used": result_snapshot["app_responses"] if result_snapshot and not operator else None,
            "skills": _public_skills(result_snapshot) if not operator else [],
            "can_retry": status == "preparing" and paper.get("generation_lease_until", 0) <= _now()}


def _history_order(answer):
    return answer["timestamp_ms"], answer["paper_sequence"], answer["position"]


def _answer_proxy(mappings, tag):
    if mappings is not None:
        for item in mappings.representatives.get(tag, ()):
            source_id = mappings.source_ids.get(item.question_id)
            if (source_id and mappings.pairs.get(item.question_id) == item
                    and tag in mappings.composite_tags.get(item.skill_id, ())):
                return {"label": "PROXY", "question_id": item.question_id, "skill_id": item.skill_id,
                        "source_question_id": source_id, "mapping_version": mappings.version}
    return None


class JunyiService:
    def __init__(self, db, runtime, stage, allow_reuse):
        self.papers = db.junyi_papers
        self.optional = db.junyi_optional_study
        self.runtime, self.stage, self.allow_reuse = runtime, stage, allow_reuse
        self.resource_slots = asyncio.Semaphore(3)

    async def _keep_lease(self, match, field):
        while True:
            await asyncio.sleep(10)
            result = await self.papers.update_one(match, {"$set": {field: _now() + LEASE_MS}, "$inc": {"version": 1}})
            if not result.matched_count:
                return

    async def get(self, id, owner):
        paper = await self.papers.find_one({"_id": id, "owner": owner})
        if not paper:
            raise HTTPException(404, "Assessment was not found")
        return paper

    async def change(self, id, owner, operation):
        for _ in range(5):
            paper = await self.get(id, owner)
            previous = paper["version"]
            changed = operation(paper)
            if not changed:
                return paper
            paper["version"] += 1
            result = await self.papers.replace_one({"_id": id, "owner": owner, "version": previous}, paper)
            if result.modified_count:
                return paper
        raise HTTPException(409, "The assessment changed in another tab. Reload and try again.")

    async def _history(self, owner, sequence):
        answers = []
        async for paper in self.papers.find({"owner": owner, "sequence": {"$lte": sequence}}, {"responses": 1}):
            answers.extend(paper["responses"])
        return sorted(answers, key=_history_order)

    async def start(self, owner):
        active = await self.papers.find_one({"owner": owner, "active": True})
        if active:
            return active
        if not self.runtime.ready:
            raise HTTPException(503, " ".join(self.runtime.issues) or "The Junyi assessment is being configured.")
        count = sum(self.stage is None or item.stage == self.stage for item in self.runtime.bank.items.values())
        quotas = self.runtime.quotas.copy()
        total = sum(quotas.values())
        if count < total:
            raise HTTPException(409, str(BankShortage(count, total, "Choose a stage with sufficient coverage.")))
        latest = await self.papers.find_one({"owner": owner}, {"sequence": 1}, sort=[("sequence", DESCENDING)])
        sequence = latest["sequence"] + 1 if latest else 1
        paper = {"_id": str(uuid4()), "owner": owner, "sequence": sequence, "active": True, "status": "preparing",
                 "stage": self.stage, "total": total, "created_at": _now(), "version": 0,
                 "quotas": quotas, "allow_reuse": self.allow_reuse,
                 "generation_lease_until": 0, "seed": secrets.token_hex(24), "items": [], "requirements": [], "responses": []}
        try:
            await self.papers.insert_one(paper)
        except DuplicateKeyError:
            existing = await self.papers.find_one({"owner": owner, "sequence": sequence}) or await self.papers.find_one({"owner": owner, "active": True})
            if existing:
                return existing
            raise HTTPException(409, "The assessment changed in another tab. Reload and try again.") from None
        return paper

    async def prepare(self, id, owner):
        job = str(uuid4())
        paper = await self.papers.find_one_and_update(
            {"_id": id, "owner": owner, "status": "preparing", "generation_lease_until": {"$lte": _now()}},
            {"$set": {"generation_job": job, "generation_lease_until": _now() + LEASE_MS}, "$inc": {"version": 1}},
            return_document=ReturnDocument.AFTER)
        if not paper:
            return
        match = {"_id": id, "owner": owner, "status": "preparing", "generation_job": job}
        heartbeat = asyncio.create_task(self._keep_lease(match, "generation_lease_until"))
        try:
            answers = await self._history(owner, paper["sequence"] - 1)
            previous = await self.papers.find_one({"owner": owner, "sequence": {"$lt": paper["sequence"]}, "status": "completed"},
                                                 {"result_snapshot": 1}, sort=[("sequence", DESCENDING)])
            snapshot = previous.get("result_snapshot") if previous else None
            as_of = _now()
            app_history_version = digest(json.dumps(answers, sort_keys=True).encode())
            if (not snapshot or snapshot["app_responses"] != len(answers)
                    or snapshot.get("app_history_version") != app_history_version
                    or snapshot["mapping_version"] != self.runtime.mappings.version
                    or snapshot["checkpoint"] != self.runtime.estimator.checkpoint_id
                    or snapshot["history_version"] != self.runtime.estimator.baseline.version
                    or snapshot["thresholds"] != {"weak": self.runtime.estimator.weak, "strong": self.runtime.estimator.strong}
                    or answers and (snapshot["as_of"] - answers[-1]["timestamp_ms"] < GAP) != (as_of - answers[-1]["timestamp_ms"] < GAP)):
                snapshot = await run_in_threadpool(self.runtime.estimator.snapshot, answers, as_of)
            if not any(row["score"] is not None for row in snapshot["skills"]):
                raise ModelUnavailable("A valid HiTSKT skill estimate is unavailable for this history.")
            quotas = paper.get("quotas", self.runtime.quotas)
            allow_reuse = paper.get("allow_reuse", self.allow_reuse)
            draft = candidates_for(self.runtime.bank, snapshot, paper["stage"], {r["bank_question_id"] for r in answers}, allow_reuse)
            if len(draft) < paper["total"]:
                raise BankShortage(len(draft), paper["total"], "Available unseen questions are exhausted under this paper's rules.")
            proposed = prompt_candidates(draft, quotas, paper["seed"])
            suggestions = await run_in_threadpool(self.runtime.gemini.choose, proposed, quotas)
            chosen, allocation = select_paper(draft, quotas, paper["seed"], suggestions, proposed)
            items = freeze_items(self.runtime.bank, chosen, paper["seed"])
            if self.runtime.study_policy is None:
                requirements, checklist_audit = rule_checklist(items, snapshot, self.runtime.study_policy_issue)
            else:
                requirements, checklist_audit = await run_in_threadpool(
                    self.runtime.study_policy.requirements, self.runtime, answers, as_of, items, snapshot,
                )
            for requirement in requirements:
                tag = requirement["tag"]
                requirement.update(topic=self.runtime.bank.topics[tag], stage=self.runtime.bank.stages[tag])
            result = await self.papers.update_one(match, {"$set": {
                "status": "open", "snapshot": snapshot, "bank_version": self.runtime.bank.version,
                "items": items, "requirements": requirements, "checklist_audit": checklist_audit,
                "allocation": allocation, "quotas": quotas, "allow_reuse": allow_reuse,
            }, "$inc": {"version": 1}})
            if result.modified_count and requirements:
                await asyncio.gather(*(self.lookup(id, owner, requirement["tag"]) for requirement in requirements))
        except (BankShortage, ModelUnavailable) as exc:
            await self.papers.update_one(match, {"$set": {"status": "shortage" if isinstance(exc, BankShortage) else "unavailable",
                                                           "message": str(exc), "active": False}, "$inc": {"version": 1}})
        except Exception:
            logger.error("Junyi assessment preparation failed")
            await self.papers.update_one(match, {"$set": {"status": "unavailable", "active": False,
                                                           "message": "We couldn’t prepare this paper. Please try again."}, "$inc": {"version": 1}})
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    @staticmethod
    def requirement(paper, tag):
        for record in paper["requirements"]:
            if record["tag"] == tag:
                return record
        raise HTTPException(404, "This skill is not required for the paper")

    async def lookup(self, id, owner, tag, retry=False):
        ticket = str(uuid4())
        async with self.resource_slots:
            now = _now()
            unavailable = {"lookup": "unavailable"}
            if not retry:
                unavailable["$or"] = [{"retry_after": {"$exists": False}}, {"retry_after": {"$lte": now}}]
            eligible = [{"lookup": "waiting"}, {"lookup": "loading", "lease_until": {"$lte": now}}, unavailable]
            identity = {"_id": id, "owner": owner, "status": {"$in": IN_PROGRESS}}
            paper = await self.papers.find_one_and_update(
                {**identity, "requirements": {"$elemMatch": {"tag": tag, "status": "pending", "$or": eligible}}},
                {"$set": {"requirements.$.lookup": "loading", "requirements.$.ticket": ticket,
                          "requirements.$.lease_until": now + STUDY_LOOKUP_LEASE_MS}, "$inc": {"version": 1}},
                return_document=ReturnDocument.AFTER)
            if not paper:
                return
            requirement = self.requirement(paper, tag)
            stage = requirement["stage"]
            try:
                links, message = await run_in_threadpool(self.runtime.recommender.lookup, tag, requirement["topic"], stage, "en")
            except Exception:
                logger.error("Junyi study lookup failed")
                links, message = [], "The resource search couldn’t finish. Please retry."

            changes = {"lookup": "verified" if links else "unavailable", "resources": links,
                       "message": message, "ticket": None, "lease_until": 0,
                       "retry_after": 0 if links else _now() + STUDY_RECHECK_MS}
            updated_requirements = {"$map": {
                "input": "$requirements", "as": "row",
                "in": {"$cond": [
                    {"$eq": ["$$row.tag", tag]},
                    {"$mergeObjects": ["$$row", {"$literal": changes}]},
                    "$$row",
                ]},
            }}
            required = {"$filter": {
                "input": "$requirements", "as": "row",
                "cond": {"$and": [
                    {"$eq": ["$$row.status", "pending"]},
                    {"$eq": ["$$row.lookup", "verified"]},
                    {"$ne": ["$$row.resources", []]},
                ]},
            }}
            # Save the link and gate together without replacing concurrent answers or study updates.
            await self.papers.update_one(
                {**identity, "requirements": {"$elemMatch": {"tag": tag, "status": "pending", "ticket": ticket}}},
                [
                    {"$set": {"requirements": updated_requirements}},
                    {"$set": {
                        "status": {"$cond": [{"$gt": [{"$size": required}, 0]}, "locked", "open"]},
                        "version": {"$add": ["$version", 1]},
                    }},
                ],
            )

    async def recheck_study(self, id, owner, tags):
        await asyncio.gather(*(self.lookup(id, owner, tag) for tag in tags))

    async def study_done(self, id, owner, tag, url=None, override=None, actor=None):
        def complete(paper):
            record = self.requirement(paper, tag)
            if record["status"] != "pending":
                return False
            if paper["status"] not in IN_PROGRESS:
                raise HTTPException(409, "The assessment no longer accepts study updates")
            if override is not None:
                if record["lookup"] != "unavailable":
                    raise HTTPException(409, "Overrides apply after a resource search is unavailable")
                record.update(status="overridden", audit={"actor": actor, "at": _now(), "reason": override})
            else:
                if not _study_required(record) or url not in {link["url"] for link in record["resources"]}:
                    raise HTTPException(422, "Choose a verified resource for this skill")
                record.update(status="satisfied", completion={"url": url, "at": _now()})
            paper["status"] = _paper_status(paper)
            return True

        return await self.change(id, owner, complete)

    async def record(self, id, owner, data):
        request_id = str(data.request_id)
        save_token = str(uuid4())

        def submit(paper):
            for old in paper["responses"]:
                if old["request_id"] == request_id:
                    if old["bank_question_id"] != data.question_id or old["choice_id"] != data.choice_id:
                        raise HTTPException(409, "This submission ID already belongs to a different answer")
                    return False
            paper["status"] = _paper_status(paper)
            if paper["status"] != "open":
                raise HTTPException(423 if paper["status"] == "locked" else 409, "The assessment is not open for answers")
            index = len(paper["responses"])
            item = paper["items"][index]
            if data.question_id != item["id"]:
                raise HTTPException(409, "This is not the current question. Reload your progress.")
            if data.choice_id not in {option["id"] for option in item["options"]}:
                raise HTTPException(422, "Choose one of the four displayed options")
            proxy = _answer_proxy(self.runtime.mappings, item["tag"])
            stamp = _now()
            if paper["responses"] and stamp < paper["responses"][-1]["timestamp_ms"]:
                raise HTTPException(503, "The assessment clock is unavailable. Retry this answer.")
            row = {"request_id": request_id, "save_token": save_token, "account_id": owner, "paper_id": id, "paper_sequence": paper["sequence"],
                   "bank_question_id": item["id"], "tag": item["tag"], "choice_id": data.choice_id,
                   "correct": item["answer_id"] == data.choice_id, "timestamp_ms": stamp, "position": index,
                   "attempt": 1, "bank_version": paper["bank_version"], "proxy": proxy,
                   "mapping_issue": None if proxy else "There is no valid representative for this question skill."}
            paper["responses"].append(row)
            if len(paper["responses"]) == paper["total"]:
                correct = sum(row["correct"] for row in paper["responses"])
                paper.update(status="completed", active=False, completed_at=stamp, analysis_status="pending", analysis_lease_until=0,
                             results={"correct": correct, "total": paper["total"], "accuracy": correct / paper["total"]})
            return True

        paper = await self.change(id, owner, submit)
        saved = next(row for row in paper["responses"] if row["request_id"] == request_id)
        return paper, saved["save_token"] != save_token

    async def analyze(self, id, owner):
        job = str(uuid4())
        paper = await self.papers.find_one_and_update(
            {"_id": id, "owner": owner, "status": "completed", "analysis_status": {"$in": ["pending", "unavailable"]},
             "analysis_lease_until": {"$lte": _now()}},
            {"$set": {"analysis_job": job, "analysis_lease_until": _now() + LEASE_MS, "analysis_status": "pending"}, "$inc": {"version": 1}},
            return_document=ReturnDocument.AFTER)
        if not paper:
            return
        match = {"_id": id, "owner": owner, "analysis_job": job, "analysis_status": "pending"}
        heartbeat = asyncio.create_task(self._keep_lease(match, "analysis_lease_until"))
        try:
            answers = await self._history(owner, paper["sequence"])
            snapshot = await run_in_threadpool(self.runtime.estimator.snapshot, answers, paper["completed_at"])
            changes = {"result_snapshot": snapshot,
                       "analysis_status": "ready" if any(row["score"] is not None for row in snapshot["skills"]) else "unavailable"}
        except Exception:
            logger.error("Junyi model analysis failed")
            changes = {"analysis_status": "unavailable"}
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        await self.papers.update_one(match, {"$set": {**changes, "analysis_lease_until": 0}, "$inc": {"version": 1}})

    async def dashboard(self, owner):
        active = await self.papers.find_one({"owner": owner, "active": True}, {"sequence": 1, "status": 1, "snapshot": 1, "requirements": 1})
        complete = await self.papers.find({"owner": owner, "status": "completed"}, {"results": 1, "completed_at": 1, "stage": 1, "sequence": 1})\
            .sort("sequence", DESCENDING).limit(12).to_list(12)
        latest = await self.papers.find_one({"owner": owner, "status": "completed", "result_snapshot": {"$exists": True}},
                                           {"result_snapshot": 1}, sort=[("sequence", DESCENDING)])
        snapshot = (active or {}).get("snapshot") or (latest or {}).get("result_snapshot")
        return {"assigned": True, "ready": self.runtime.ready, "issues": self.runtime.issues, "stage": self.stage,
                "active": {"id": active["_id"], "status": _paper_status(active), "sequence": active["sequence"],
                           "required_tags": [row["tag"] for row in active["requirements"] if _study_required(row)]} if active else None,
                "history": [{"id": paper["_id"], "sequence": paper["sequence"], "completed_at": paper["completed_at"],
                             "stage": paper["stage"], **paper["results"]} for paper in complete],
                "answers_used": snapshot["app_responses"] if snapshot else None,
                "skills": _public_skills(snapshot)}


def mount_junyi(app, settings, student, auth):
    def service():
        return app.state.junyi

    async def owner(user=Depends(student)):
        if not matches_account(user, settings.junyi_student_account_id):
            raise HTTPException(404, "Junyi is available to the assigned prototype learner")
        return user["_id"]

    async def operator(user=Depends(auth)):
        if user.get("role") != "admin" or not matches_account(user, settings.junyi_operator_account_id):
            raise HTTPException(403, "This operation requires the designated assessment operator")
        return user["_id"]

    @app.get("/api/junyi/dashboard")
    async def dashboard(user=Depends(student)):
        if not matches_account(user, settings.junyi_student_account_id):
            return {"assigned": False}
        return await service().dashboard(user["_id"])

    @app.post("/api/assessments", status_code=202)
    async def start(tasks: BackgroundTasks, account=Depends(owner)):
        paper = await service().start(account)
        if paper["status"] == "preparing":
            tasks.add_task(service().prepare, paper["_id"], account)
        return _view(paper)

    @app.get("/api/assessments/{id}")
    async def read(id: UUID, tasks: BackgroundTasks, account=Depends(owner)):
        instance = service()
        paper = await instance.get(str(id), account)
        if paper["status"] in IN_PROGRESS:
            now = _now()
            tags = [row["tag"] for row in paper["requirements"] if _study_lookup_due(row, now)]
            if tags:
                tasks.add_task(instance.recheck_study, str(id), account, tags)
        return _view(paper)

    @app.get("/api/assessments/{id}/question")
    async def question(id: UUID, account=Depends(owner)):
        paper = await service().get(str(id), account)
        status = _paper_status(paper)
        if status != "open":
            raise HTTPException(423 if status == "locked" else 409, "The assessment is not open")
        return _question(paper)

    @app.post("/api/assessments/{id}/answers")
    async def answer(id: UUID, data: AnswerInput, tasks: BackgroundTasks, account=Depends(owner)):
        paper, replayed = await service().record(str(id), account, data)
        if paper["status"] == "completed" and paper.get("analysis_status") == "pending":
            tasks.add_task(service().analyze, str(id), account)
        return {"paper": _view(paper), "replayed": replayed}

    @app.post("/api/assessments/{id}/analysis", status_code=202)
    async def retry_analysis(id: UUID, tasks: BackgroundTasks, account=Depends(owner)):
        paper = await service().get(str(id), account)
        if paper["status"] != "completed":
            raise HTTPException(409, "Complete the assessment to analyze its results")
        tasks.add_task(service().analyze, str(id), account)
        return _view(paper)

    @app.post("/api/assessments/{id}/study/{tag}/retry", status_code=202)
    async def retry_study(id: UUID, tag: int, tasks: BackgroundTasks, account=Depends(owner)):
        paper = await service().get(str(id), account)
        service().requirement(paper, tag)
        tasks.add_task(service().lookup, str(id), account, tag, True)
        return _view(paper)

    @app.post("/api/assessments/{id}/study/{tag}/complete")
    async def complete_study(id: UUID, tag: int, data: StudyInput, account=Depends(owner)):
        return _view(await service().study_done(str(id), account, tag, url=data.url))

    @app.post("/api/junyi/study/{tag}")
    async def optional_study(tag: int, account=Depends(owner)):
        instance = service()
        dashboard = await instance.dashboard(account)
        skill = next((row for row in dashboard["skills"] if row["tag"] == tag and row["category"] == "weak"), None)
        if not skill:
            raise HTTPException(404, "This skill is unavailable for optional study")
        saved = await instance.optional.find_one({"owner": account, "tag": tag})
        fresh_after = (datetime.now(timezone.utc) - STUDY_RESOURCE_TTL).isoformat()
        if saved and saved["resources"] and all(
            link.get("source") == "curated" and link.get("verified_at", "") > fresh_after for link in saved["resources"]
        ):
            return {"resources": saved["resources"], "message": saved["message"]}
        async with instance.resource_slots:
            links, message = await run_in_threadpool(instance.runtime.recommender.lookup, tag, skill["topic"], skill["stage"], "en")
        fields = {"resources": links, "message": message, "updated_at": datetime.now(timezone.utc)}
        await instance.optional.update_one({"owner": account, "tag": tag}, {"$set": fields}, upsert=True)
        return {"resources": links, "message": message}

    @app.get("/api/admin/junyi")
    async def admin_status(_actor=Depends(operator)):
        instance = service()
        papers = await instance.papers.find({"active": True, "owner": settings.junyi_student_account_id}).sort("sequence", DESCENDING).limit(5).to_list(5)
        return {"ready": instance.runtime.ready, "issues": instance.runtime.issues,
                "bank_questions": len(instance.runtime.bank.items) if instance.runtime.bank else 0,
                "study_policy": {"name": FOREST_POLICY if instance.runtime.study_policy else "original_rule",
                                 "issue": instance.runtime.study_policy_issue},
                "active": [_view(paper, operator=True) for paper in papers]}

    @app.post("/api/admin/junyi/{id}/study/{tag}/override")
    async def override(id: UUID, tag: int, data: OverrideInput, actor=Depends(operator)):
        paper = await service().papers.find_one({"_id": str(id), "owner": settings.junyi_student_account_id})
        if not paper:
            raise HTTPException(404, "Assessment was not found")
        return _view(await service().study_done(str(id), paper["owner"], tag, override=data.reason.strip(), actor=actor), operator=True)
