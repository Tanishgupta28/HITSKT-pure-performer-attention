"""Build legal Junyi contexts and call the original, unchanged HiTSKT implementation."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from app.junyi_bank import JunyiBank, JunyiMappings, ModelItem, digest


CHECKPOINT_SHA256 = "2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd"
STORE_SHA256 = "44ef8746d5addda0f44401aaeb87a50818910d250049dd733dbd65e925cae9fb"
GAP = 36_000_000  # milliseconds, including the exact 10-hour boundary
MAX_EARLIER_SESSIONS = 15
# Keep CPU serving batches smaller than the research batch budget on the platform host.
TOKEN_BUDGET = 8_192


class ModelUnavailable(ValueError):
    pass


def timestamp_ms(value) -> int:
    if type(value) is int:
        return value
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Junyi timestamps require a timezone")
    return (stamp.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)) // _MILLISECOND


_MILLISECOND = timedelta(milliseconds=1)


@dataclass(frozen=True)
class Event:
    item: ModelItem | None
    correct: int
    time: int


@dataclass(frozen=True)
class Baseline:
    sessions: tuple[tuple[Event, ...], ...]
    cutoff: int
    version: str


def _field(record, *keys):
    for key in keys:
        if key in record:
            return record[key]
    raise ValueError(f"Missing approved history field: {keys[0]}")


def _events(session, mappings):
    if isinstance(session, dict) and not ("events" in session or "interactions" in session):
        columns = [_field(session, *names) for names in (
            ("question_ids", "question", "question_id"), ("skill_ids", "skill", "skill_id"),
            ("correct", "correctness", "is_correct"), ("timestamp_ms", "timestamps_ms", "timestamps"),
        )]
        records = zip(*columns, strict=True)
    else:
        if isinstance(session, dict):
            session = _field(session, "events", "interactions")
        records = ((
            _field(event, "model_question_id", "processed_question_id", "question_id", "question"),
            _field(event, "composite_skill_id", "model_skill_id", "skill_id", "skill"),
            _field(event, "correct", "correctness", "is_correct"),
            _field(event, "timestamp_ms", "timestamp"),
        ) for event in session)
    events = []
    for q, s, correct, timestamp in records:
        item = ModelItem(int(q), int(s))
        if mappings.pairs.get(item.question_id) != item or correct not in (0, 1):
            raise ValueError("Approved history contains an invalid question/skill pair or outcome")
        events.append(Event(item, int(correct), timestamp_ms(timestamp)))
    return tuple(events)


def read_baseline(path: Path, expected_hash: str, mappings: JunyiMappings) -> Baseline:
    data = path.read_bytes()
    if not expected_hash or digest(data) != expected_hash.lower():
        raise ValueError("Student history does not match the verified private member hash")
    root = json.loads(data)
    metadata = {**(root.get("provenance") or {}), **root}
    raw = _field(metadata, "raw_student_id", "source_student_id", "original_student_id", "original_raw_student_id")
    processed = _field(metadata, "processed_student_id", "model_student_id", "student_index", "student_id")
    if int(raw) != 1 or int(processed) != 1:
        raise ValueError("History is outside the approved research learner scope")
    source = metadata.get("source_session_store_sha256", metadata.get("session_store_sha256"))
    if source is not None and source != STORE_SHA256:
        raise ValueError("History was extracted from a different Junyi SessionStore")
    cutoff = _field(metadata, "first_test_timestamp_ms", "first_test_session_timestamp_ms", "first_excluded_test_timestamp_ms",
                    "cutoff_timestamp_ms", "cutoff_ms", "cutoff")
    if isinstance(cutoff, dict):
        cutoff = _field(cutoff, "first_test_timestamp_ms", "first_excluded_test_timestamp_ms", "timestamp_ms")
    cutoff = timestamp_ms(cutoff)
    sessions = tuple(_events(session, mappings) for session in _field(root, "sessions", "pre_test_sessions"))
    if len(sessions) != 9 or sum(map(len, sessions)) != 179 or any(not session for session in sessions):
        raise ValueError("Student history must contain exactly 9 earlier sessions and 179 interactions")
    previous = None
    for session in sessions:
        for index, event in enumerate(session):
            if event.time >= cutoff or previous is not None and (event.time < previous or (event.time - previous >= GAP) != (index == 0)):
                raise ValueError("History is outside the approved cutoff or session boundaries")
            previous = event.time
    return Baseline(sessions, cutoff, digest(data))


def session_context(baseline: Baseline, responses: list[dict], mappings: JunyiMappings, as_of: int):
    sessions = list(baseline.sessions)
    new_sessions: list[list[Event]] = []
    previous = baseline.sessions[-1][-1].time
    for answer in sorted(responses, key=lambda row: (row["timestamp_ms"], row["paper_sequence"], row["position"])):
        time = answer["timestamp_ms"]
        if time > as_of or time < previous or time < baseline.cutoff:
            raise ModelUnavailable("App history is outside the legal inference window")
        proxy = answer.get("proxy")
        item = ModelItem(proxy["question_id"], proxy["skill_id"]) if proxy else None
        if item is not None and (proxy.get("mapping_version") != mappings.version or mappings.pairs.get(item.question_id) != item
                                 or answer["tag"] not in mappings.composite_tags[item.skill_id]):
            item = None
        if not new_sessions or time - previous >= GAP:
            new_sessions.append([])
        new_sessions[-1].append(Event(item, int(answer["correct"]), time))
        previous = time

    prefix: tuple[Event, ...] = ()
    if new_sessions and as_of - previous < GAP:
        prefix = tuple(new_sessions.pop())
    sessions.extend(tuple(session) for session in new_sessions)
    earlier = tuple(sessions[-MAX_EARLIER_SESSIONS:])
    if any(event.item is None for session in (*earlier, prefix) for event in session):
        raise ModelUnavailable("An unmapped answer leaves the retained Junyi context incomplete")
    return earlier, prefix


def score_class(score: float | None, weak: float, strong: float) -> str | None:
    if score is None:
        return None
    if score < weak:
        return "weak"
    return "intermediate" if score < strong else "strong"


class OriginalHiTSKT:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self.model = None

    def load(self):
        if self.model is None:
            from app.research import hitskt_types
            with self.checkpoint.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != CHECKPOINT_SHA256:
                    raise ModelUnavailable("The supplied checkpoint is not the verified Junyi release checkpoint")
            model, _ = hitskt_types()
            loaded = model.from_checkpoint(self.checkpoint, map_location="cpu").eval()
            if (loaded.config.num_questions, loaded.config.num_skills) != (25_630, 1_521):
                raise ModelUnavailable("Checkpoint vocabulary does not match Junyi")
            self.model = loaded

    def predict(self, earlier, prefix, candidates, placeholder=0):
        import torch
        from ktbench.batching import Session, SessionExample, collate_hitskt, example_shape, padded_token_cost

        self.load()

        def session(events):
            return Session([e.item.question_id for e in events], [e.item.skill_id for e in events], [e.correct for e in events])

        if not candidates:
            return {}
        history = tuple(session(events) for events in earlier)
        position = len(prefix)

        def example(item):
            return SessionExample(history, session((*prefix, Event(item, placeholder, 0))), split=2)

        cost = padded_token_cost([example_shape(example(candidates[0]))])
        if cost > TOKEN_BUDGET:
            raise ModelUnavailable("The full retained context exceeds the serving token budget")
        size = max(1, min(64, TOKEN_BUDGET // cost))
        output = {}
        with torch.inference_mode():
            for start in range(0, len(candidates), size):
                chunk = candidates[start:start + size]
                batch = collate_hitskt([example(item) for item in chunk], num_questions=self.model.config.num_questions,
                                       num_skills=self.model.config.num_skills)
                if (batch.target_question[:, position].tolist() != [item.question_id for item in chunk]
                        or batch.target_skill[:, position].tolist() != [item.skill_id for item in chunk]
                        or not batch.target_metric_mask[:, position].all()):
                    raise ModelUnavailable("Model candidate position does not match the requested IDs")
                probabilities = torch.sigmoid(self.model(batch)[:, position]).tolist()
                if any(not math.isfinite(value) for value in probabilities):
                    raise ModelUnavailable("The checkpoint produced a non-finite prediction")
                output.update(zip(chunk, probabilities, strict=True))
        return output


class SkillEstimator:
    checkpoint_id = CHECKPOINT_SHA256

    def __init__(self, baseline, bank: JunyiBank, mappings: JunyiMappings, predictor, weak=.61, strong=.86):
        self.baseline, self.bank, self.mappings, self.predictor = baseline, bank, mappings, predictor
        self.weak, self.strong = weak, strong
        self._original = None
        self._lock = Lock()
        self._candidates = sorted({item for tag in self.bank.by_tag for item in mappings.representatives[tag]})

    def snapshot(self, responses: list[dict], as_of: int):
        with self._lock:
            if self._original is None:
                self._original = self.predictor.predict(self.baseline.sessions, (), self._candidates)
            original = self._original
            unavailable = None
            try:
                earlier, prefix = session_context(self.baseline, responses, self.mappings, as_of)
                current = self.predictor.predict(earlier, prefix, self._candidates) if responses else original
            except ModelUnavailable as exc:
                unavailable, current = str(exc), {}

        original_attempts, original_correct = Counter(), Counter()
        app_attempts, app_correct = Counter(), Counter()
        for session in self.baseline.sessions:
            for event in session:
                for tag in self.mappings.composite_tags[event.item.skill_id]:
                    original_attempts[tag] += 1
                    original_correct[tag] += event.correct
        for answer in responses:
            app_attempts[answer["tag"]] += 1
            app_correct[answer["tag"]] += int(answer["correct"])

        rows = []
        for tag in sorted(self.bank.by_tag):
            probes = self.mappings.representatives[tag]
            baseline_scores = [original[item] for item in probes if item in original]
            current_scores = [current[item] for item in probes if item in current]
            score = math.fsum(current_scores) / len(current_scores) if len(current_scores) == len(probes) and probes else None
            original_score = math.fsum(baseline_scores) / len(baseline_scores) if len(baseline_scores) == len(probes) and probes else None
            attempts = original_attempts[tag] + app_attempts[tag]
            correct = original_correct[tag] + app_correct[tag]
            rows.append({"tag": tag, "topic": self.bank.topics[tag], "stage": self.bank.stages[tag],
                         "score": score, "original_score": original_score, "category": score_class(score, self.weak, self.strong),
                         "unattempted": attempts == 0, "unavailable_reason": None if score is not None else unavailable or "Missing representative prediction",
                         "actual_rate": correct / attempts if attempts else None, "original_attempts": original_attempts[tag],
                         "original_correct": original_correct[tag], "app_attempts": app_attempts[tag], "app_correct": app_correct[tag],
                         "probe_min": min(current_scores, default=None), "probe_max": max(current_scores, default=None),
                         "probes": [{"question_id": item.question_id, "source_question_id": self.mappings.source_ids[item.question_id],
                                     "skill_id": item.skill_id, "original": original.get(item), "current": current.get(item)} for item in probes]})
        return {"as_of": as_of, "history_cutoff": self.baseline.cutoff, "history_version": self.baseline.version,
                "mapping_version": self.mappings.version, "checkpoint": self.checkpoint_id,
                "app_responses": len(responses), "app_history_version": digest(json.dumps(responses, sort_keys=True).encode()),
                "thresholds": {"weak": self.weak, "strong": self.strong}, "skills": rows}
