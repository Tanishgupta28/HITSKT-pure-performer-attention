"""Question allocation and immutable paper assembly for Junyi assessments."""

import hashlib
import random
from collections import Counter
from dataclasses import dataclass

from app.junyi_bank import JunyiBank


DEFAULT_QUOTAS = {"weak": 8, "unattempted": 6, "review": 4, "broad": 2}
BUCKETS = tuple(DEFAULT_QUOTAS)


class BankShortage(ValueError):
    def __init__(self, eligible: int, required: int, reason: str):
        self.eligible, self.required = eligible, required
        super().__init__(f"Only {eligible} eligible questions are available for a {required}-question paper. {reason}")


def _primary(row):
    if row["unattempted"]:
        return "unattempted"
    if row["category"] == "weak":
        return "weak"
    if row["category"] in ("intermediate", "strong"):
        return "review"
    return "broad"


@dataclass(frozen=True)
class Candidate:
    id: str
    tag: int
    topic: str
    stage: str
    primary: str
    score: float
    template: str
    seen: bool


def candidates_for(bank: JunyiBank, snapshot, stage, seen, allow_reuse):
    scores = {row["tag"]: row for row in snapshot["skills"]}
    return [Candidate(item.id, item.tag, item.topic, item.stage, _primary(scores[item.tag]),
                      scores[item.tag]["score"] if scores[item.tag]["score"] is not None else 1.0,
                      item.template, item.id in seen)
            for item in bank.items.values() if (stage is None or item.stage == stage) and (allow_reuse or item.id not in seen)]


def _groups(candidates):
    return {bucket: [item for item in candidates if item.primary == bucket or bucket == "broad"] for bucket in BUCKETS}


def _tie(seed, id):
    return hashlib.sha256(f"{seed}:{id}".encode()).digest()


def _priority(item, bucket, tie, tags, templates):
    return (item.seen, tags[item.tag], templates[item.template], item.score if bucket == "weak" else 0,
            tie[item.id], item.id)


def prompt_candidates(candidates, quotas, seed):
    tie = {item.id: _tie(seed, item.id) for item in candidates}
    result = []
    for bucket, pool in _groups(candidates).items():
        tags = Counter()
        pool = sorted((item for item in pool if not item.seen), key=lambda item: _priority(item, bucket, tie, tags, Counter()))
        for item in pool:
            if tags[item.tag]:
                continue
            result.append({"bank_question_id": item.id, "tag": item.tag, "topic": item.topic, "stage": item.stage, "bucket": bucket})
            tags[item.tag] += 1
            if len(tags) == max(12, 3 * quotas[bucket]):
                break
    return result


def select_paper(candidates, quotas, seed, proposals=(), offered=()):
    required = sum(quotas.values())
    if len(candidates) < required:
        raise BankShortage(len(candidates), required, "Stage and repeat settings were applied.")
    groups = _groups(candidates)
    items = {item.id: item for item in candidates}
    order = {item.id: _tie(seed, item.id) for item in candidates}
    allowed = {(item["bank_question_id"], item["bucket"]) for item in offered}
    preferred = {bucket: {} for bucket in BUCKETS}
    proposed_ids = set()
    for row in proposals:
        if not isinstance(row, dict):
            continue
        id, bucket, reason = (row.get(key) for key in ("bank_question_id", "bucket", "reason"))
        if not isinstance(id, str) or not isinstance(bucket, str) or (id, bucket) not in allowed or id in proposed_ids:
            continue
        if id in items and (bucket == "broad" or items[id].primary == bucket) and isinstance(reason, str) and reason.strip():
            preferred[bucket][id] = reason.strip()[:180]
            proposed_ids.add(id)

    used, tags, templates, counts, chosen = set(), Counter(), Counter(), Counter(), []

    def pick(bucket, *, unseen):
        pool = [item for item in groups[bucket] if item.id not in used and (not unseen or not item.seen)]
        if not pool:
            return False

        def priority(item):
            seen, tag_count, template_count, score, tie, id = _priority(item, bucket, order, tags, templates)
            return seen, tag_count, template_count, id not in preferred[bucket], score, tie, id

        item = min(pool, key=priority)
        reason = preferred[bucket].get(item.id)
        used.add(item.id)
        tags[item.tag] += 1
        templates[item.template] += 1
        counts[bucket] += 1
        chosen.append({"id": item.id, "bucket": bucket, "provider": "gemini" if reason else "rules",
                       "reason": reason or "Selected by stage, new-skill coverage and repeat rules", "reused": item.seen})
        return True

    for bucket in BUCKETS:
        for _ in range(quotas[bucket]):
            if not pick(bucket, unseen=True):
                break
    deficits = {bucket: quotas[bucket] - counts[bucket] for bucket in BUCKETS if counts[bucket] < quotas[bucket]}
    for unseen in (True, False):
        for bucket in BUCKETS:
            while len(chosen) < required and pick(bucket, unseen=unseen):
                pass
    if len(chosen) != required:
        raise BankShortage(len(chosen), required, "The requested allocation could not be filled.")
    return chosen, {"counts": dict(counts), "initial_deficits": deficits, "reallocated": sum(deficits.values()),
                    "reused_items": sum(item["reused"] for item in chosen)}


def freeze_items(bank, selected, seed):
    materialized = []
    for item in selected:
        question = bank.items[item["id"]]
        options = [{"id": _tie(seed, f"{question.id}:choice:{index}").hex()[:32], "text": text}
                   for index, text in enumerate(question.choices)]
        answer_id = options[0]["id"]
        random.Random(f"{seed}:{question.id}").shuffle(options)
        materialized.append({**item, "tag": question.tag, "topic": question.topic, "stage": question.stage,
                             "question": question.prompt, "options": options, "answer_id": answer_id})
    random.Random(f"{seed}:paper").shuffle(materialized)
    return materialized


def study_requirements(items, snapshot):
    weak = {row["tag"] for row in snapshot["skills"] if row["category"] == "weak" and not row["unattempted"]}
    return [{"tag": tag, "status": "pending", "lookup": "waiting", "resources": []}
            for tag in sorted(weak.intersection(item["tag"] for item in items))]
