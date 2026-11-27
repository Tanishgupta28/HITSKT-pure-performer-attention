"""Rank study topics on a frozen Junyi paper using the reviewed forest."""

import csv
import hashlib
import io
import logging
import math
from collections import Counter, defaultdict

from app.junyi_bank import ModelItem, RANKING_SHA256, digest
from app.junyi_model import ModelUnavailable, session_context
from app.junyi_paper import study_requirements


logger = logging.getLogger(__name__)
FOREST_POLICY = "gmm_forest_v1"
FOREST_SHA256 = "10eaccae7c8e55091cc42e5580c0b288dbc19c9cdb4dcd9b9d78951fc413b0c5"
FEATURES = ("score", "past_attempts", "past_accuracy", "days_since")
RISK_THRESHOLD = 0.33
MAX_TOPICS = 3


def rule_checklist(items, snapshot, reason=None):
    return study_requirements(items, snapshot), {
        "policy": "original_rule_fallback" if reason else "original_rule",
        "reason": reason or "study_forest_not_configured",
    }


class PaperRecommender:
    def __init__(self, forest, probes, mappings):
        self.forest = forest
        self.probes = probes
        self.composites_by_tag = defaultdict(list)
        for skill in probes:
            for tag in mappings.composite_tags[skill]:
                self.composites_by_tag[tag].append(skill)

    @classmethod
    def load(cls, path, ranking_file, mappings):
        from joblib import load

        ranking = ranking_file.read_bytes()
        if digest(ranking) != RANKING_SHA256:
            raise ValueError("Study probe rankings do not match the reviewed handoff")
        ranked = defaultdict(list)
        for row in csv.DictReader(io.StringIO(ranking.decode("utf-8-sig")), strict=True):
            item = ModelItem(int(row["question_id"]), int(row["composite_skill_id"]))
            if mappings.pairs.get(item.question_id) != item:
                raise ValueError("Study probe does not match the trained item mapping")
            ranked[item.skill_id].append((int(row["rank"]), item))
        probes = {skill: tuple(item for _, item in sorted(group)) for skill, group in ranked.items()}

        # joblib deserializes Python objects. Accept only the reviewed forest's exact bytes.
        with path.open("rb") as model:
            if hashlib.file_digest(model, "sha256").hexdigest() != FOREST_SHA256:
                raise ValueError("Study forest does not match the reviewed handoff")
            model.seek(0)
            forest = load(model)
        if tuple(getattr(forest, "feature_names_in_", ())) != FEATURES:
            raise ValueError("Study forest feature schema does not match the adapter")
        return cls(forest, probes, mappings)

    def requirements(self, runtime, answers, as_of, items, snapshot):
        attempted = {row["tag"] for row in snapshot["skills"] if not row["unattempted"]}
        tags = sorted(attempted.intersection(item["tag"] for item in items))
        if not tags:
            return [], {"policy": FOREST_POLICY, "reason": "no_attempted_paper_topics"}

        try:
            mapped, risks = self._risks(runtime, answers, as_of, tags)
        except ModelUnavailable as exc:
            return rule_checklist(items, snapshot, str(exc))
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Junyi study forest could not score the paper (%s)", type(exc).__name__)
            return rule_checklist(items, snapshot, "study_forest_prediction_failed")

        flagged = sorted((tag for tag in tags if risks[mapped[tag]] >= RISK_THRESHOLD),
                         key=lambda tag: (-risks[mapped[tag]], mapped[tag], tag))[:MAX_TOPICS]
        requirements = [{"tag": tag, "status": "pending", "lookup": "waiting", "resources": []} for tag in flagged]
        audit = {
            "policy": FOREST_POLICY, "risk_threshold": RISK_THRESHOLD,
            "topics": [{"tag": tag, "composite": skill, "risk": risks[skill], "selected": tag in flagged}
                       for tag, skill in mapped.items()],
        }
        return requirements, audit

    def _risks(self, runtime, answers, as_of, tags):
        from pandas import DataFrame

        earlier, prefix = session_context(runtime.estimator.baseline, answers, runtime.mappings, as_of)
        events = [event for session in earlier for event in session]
        if prefix or len(earlier) < 4 or len(events) < 20:
            raise ModelUnavailable("History outside the evaluated complete-session setting")
        if any(tag not in self.composites_by_tag for tag in tags):
            raise ModelUnavailable("A paper topic has no eligible composite mapping")

        attempts, correct, last = Counter(), Counter(), {}
        for event in events:
            skill = event.item.skill_id
            attempts[skill] += 1
            correct[skill] += event.correct
            last[skill] = event.time

        def priority(skill):
            return -attempts[skill], len(runtime.mappings.composite_tags[skill]), skill

        mapped = {tag: min(self.composites_by_tag[tag], key=priority) for tag in tags}
        skills = sorted(set(mapped.values()))
        probes = [item for skill in skills for item in self.probes[skill]]
        probabilities = runtime.estimator.predictor.predict(earlier, (), probes)
        rows = []
        for skill in skills:
            values = [probabilities.get(item) for item in self.probes[skill]]
            if any(value is None or not 0 <= value <= 1 for value in values):
                raise ModelUnavailable("Incomplete composite forecast")
            count = attempts[skill]
            rows.append((math.fsum(values) / len(values), count, correct[skill] / count if count else -1,
                         (events[-1].time - last[skill]) / 86_400_000 if count else -1))
        risks = tuple(map(float, self.forest.predict(DataFrame(rows, columns=FEATURES))))
        if len(risks) != len(skills) or any(not 0 <= risk <= 1 for risk in risks):
            raise ModelUnavailable("Invalid study forest risks")
        return mapped, dict(zip(skills, risks, strict=True))
