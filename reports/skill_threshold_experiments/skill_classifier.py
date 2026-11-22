"""Apply frozen Junyi composite-skill thresholds to HiTSKT probabilities."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from itertools import repeat
from numbers import Integral
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MAPPING = HERE.parents[1] / "data/processed/junyi/full/mappings/question_skills.csv"


@lru_cache(maxsize=1)
def _question_to_skill():
    with DEFAULT_MAPPING.open(newline="") as stream:
        return {int(row["question_id"]): int(row["skill_id"]) for row in csv.DictReader(stream)}


def classify_skills(question_ids, skill_ids, probabilities, *, student_ids=None):
    """Return scores by (student, composite skill), with no correctness inputs.

    Pass a scalar student ID for a single learner or an ID per score for multiple
    learners. For the validated use case, score the published distinct question
    bank at BOS with the same fixed history for every question from that learner.
    """
    thresholds = json.loads((HERE / "optimized_thresholds.json").read_text())
    t1, t2 = thresholds["T1"], thresholds["T2"]
    if not 0 < t1 < t2 < 1:
        raise ValueError("Invalid frozen skill thresholds")
    mapping = _question_to_skill()
    items = zip(question_ids, skill_ids, probabilities, strict=True)
    if student_ids is None or isinstance(student_ids, (str, Integral)):
        records = zip(repeat(student_ids), items)
    else:
        records = zip(student_ids, items, strict=True)
    groups = defaultdict(list)
    for student, (question, skill, probability) in records:
        if (not isinstance(question, Integral) or isinstance(question, bool)
                or not isinstance(skill, Integral) or isinstance(skill, bool)
                or mapping.get(question) != skill):
            raise ValueError(f"Question {question!r} does not map to composite skill {skill!r}")
        value = float(probability)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("HiTSKT probabilities must be finite values in [0, 1]")
        groups[(student, int(skill))].append(value)
    result = []
    for (student, skill), values in groups.items():
        score = math.fsum(values) / len(values)
        label = "Weak" if score < t1 else "Intermediate" if score < t2 else "Strong"
        result.append({"student_id": student, "composite_skill_id": skill, "skill_score": score,
                       "question_count": len(values), "classification": label})
    return result
