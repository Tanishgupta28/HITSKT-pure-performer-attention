"""Interpretable cold-start KT, distinct from HiTSKT answer predictions."""
from app.catalog import BANK, CONCEPTS


def initial_concept():
    return {"mastery": 0.25, "attempts": 0, "correct": 0, "last_practiced": None}


def update_knowledge(previous, correct, difficulty, timestamp):
    """Bayesian knowledge tracing posterior + learning transition.

    Fixed product defaults; these parameters are not benchmark-trained.
    Difficulty adjusts slip/guess in the observation model.
    """
    p = previous["mastery"]
    slip, guess = {1: (0.08, 0.30), 2: (0.15, 0.25), 3: (0.23, 0.18)}[difficulty]
    if correct:
        posterior = p * (1 - slip) / (p * (1 - slip) + (1 - p) * guess)
    else:
        posterior = p * slip / (p * slip + (1 - p) * (1 - guess))
    return {"mastery": min(0.99, posterior + (1 - posterior) * 0.08),
            "attempts": previous["attempts"] + 1, "correct": previous["correct"] + int(correct),
            "last_practiced": timestamp}


def bkt_probability(question, concepts):
    p = concepts.get(question["concept_id"], initial_concept())["mastery"]
    slip, guess = {1: (0.08, 0.30), 2: (0.15, 0.25), 3: (0.23, 0.18)}[question["difficulty"]]
    return p * (1 - slip) + (1 - p) * guess


def concept_rows(state):
    rows = []
    for concept in CONCEPTS:
        stats = state["concepts"].get(concept["id"], initial_concept())
        n = stats["attempts"]
        rows.append({**concept, **stats, "confidence": "building" if n < 5 else "developing" if n < 12 else "established",
                     "status": "unexplored" if not n else "strong" if stats["mastery"] >= 0.75 else "developing" if stats["mastery"] >= 0.45 else "needs_practice",
                     "evidence_count": n})
    return rows


def recommendations(state):
    rows = concept_rows(state)
    by_id = {row["id"]: row for row in rows}
    ranked = sorted(rows, key=lambda c: (c["mastery"], c["attempts"]))
    output, seen = [], set()
    for row in ranked:
        missing = [by_id[p] for p in row["prerequisites"] if by_id[p]["mastery"] < 0.55]
        target = missing[0] if missing else row
        if target["id"] in seen:
            continue
        seen.add(target["id"])
        reason = (f"Build this foundation before {row['name'].lower()}." if missing else
                  "Take a few questions to build an initial estimate." if not target["attempts"] else
                  "Your recent answers suggest this is a useful area to revisit." if target["mastery"] < 0.55 else
                  "Strengthen this concept with a more challenging practice set.")
        output.append({"concept_id": target["id"], "name": target["name"], "reason": reason,
                       "mastery": target["mastery"], "lesson": target["lesson"], "minutes": 5})
    return output[:3]


def select_question(state, assessment, predictor):
    answered = {r["question_id"] for r in assessment["responses"]}
    candidates = [q for q in BANK.values() if q["id"] not in answered]
    if assessment.get("concept_id"):
        candidates = [q for q in candidates if q["concept_id"] == assessment["concept_id"]]
    if not candidates:
        return None
    counts = {c["id"]: sum(r["concept_id"] == c["id"] for r in assessment["responses"]) for c in CONCEPTS}
    # A diagnostic first covers every concept, and continues balancing coverage.
    if assessment["mode"] == "diagnostic":
        minimum = min(counts.values())
        candidates = [q for q in candidates if counts[q["concept_id"]] == minimum]
    predictions = predictor.predict_many(state, candidates)
    seen = {r["question_id"] for r in state["history"]}
    def score(q):
        concept = state["concepts"].get(q["concept_id"], initial_concept())
        predicted = predictions[q["id"]]["probability"]
        desired = 1 if concept["mastery"] < 0.4 else 2 if concept["mastery"] < 0.7 else 3
        # Prefer a suitable difficulty, ~70% success, weak concepts, fresh items.
        return (0.36 * abs(q["difficulty"] - desired) + abs(predicted - 0.70)
                + (0 if assessment["mode"] == "diagnostic" else 0.25 * concept["mastery"])
                + (0.20 if q["id"] in seen else 0), q["id"])
    chosen = min(candidates, key=score)
    return {"question_id": chosen["id"], "prediction": predictions[chosen["id"]],
            "selection_reason": "Balanced diagnostic coverage" if assessment["mode"] == "diagnostic" else "Selected from your current concept estimate and previous answers"}
