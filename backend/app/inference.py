"""Serving adapter for the unchanged, pure-Performer research model."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.engine import bkt_probability


class Predictor:
    def __init__(self, settings):
        self.model = None
        self.mapping = {}
        self.dataset = None
        if bool(settings.hitskt_checkpoint) != bool(settings.hitskt_manifest):
            raise ValueError("Set both HITSKT_CHECKPOINT and HITSKT_MANIFEST")
        if settings.hitskt_checkpoint:
            import torch
            from app.research import hitskt_types
            HiTSKT, _ = hitskt_types()
            checkpoint = Path(settings.hitskt_checkpoint).resolve()
            manifest = json.loads(Path(settings.hitskt_manifest).read_text(encoding="utf-8"))
            with checkpoint.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if digest != manifest["checkpoint_sha256"]:
                raise ValueError("Checkpoint SHA256 does not match reviewed mapping manifest")
            if manifest.get("format_version") != 1 or not manifest.get("dataset"):
                raise ValueError("Unsupported mapping manifest")
            self.model = HiTSKT.from_checkpoint(checkpoint, map_location="cpu").eval()
            self.mapping, self.dataset = manifest["questions"], manifest["dataset"]
            for value in self.mapping.values():
                if not (1 <= value["question_id"] <= self.model.config.num_questions and
                        1 <= value["skill_id"] <= self.model.config.num_skills):
                    raise ValueError("Manifest IDs outside checkpoint vocabulary")
            self.checkpoint_sha256 = digest

    def status(self):
        return {"hitskt_loaded": self.model is not None, "dataset": self.dataset,
                "mapped_questions": len(self.mapping), "cold_start_provider": "bayesian_knowledge_tracing",
                "knowledge_estimator": "bayesian_knowledge_tracing",
                "note": "HiTSKT predicts correctness for mapped questions after an earlier 10-hour session; concept estimates use an interpretable Bayesian update."}

    def _sessions(self, history):
        sessions = []
        last = None
        for response in history:
            stamp = datetime.fromisoformat(response["answered_at"])
            if last is None or (stamp - last).total_seconds() >= 36000:
                sessions.append([])
            sessions[-1].append(response)
            last = stamp
        return sessions

    def predict_many(self, state, candidates):
        fallback = {q["id"]: {"probability": bkt_probability(q, state["concepts"]), "provider": "bayesian_knowledge_tracing",
                              "reason": "checkpoint_not_configured" if self.model is None else "unmapped_content_or_cold_start"} for q in candidates}
        if self.model is None:
            return fallback
        # Never fabricate an earlier session or silently remap a new question.
        sessions = self._sessions(state["history"])
        if not sessions:
            return fallback
        now = datetime.fromisoformat(state.get("inference_at", state["history"][-1]["answered_at"]))
        last = datetime.fromisoformat(state["history"][-1]["answered_at"])
        if (now - last).total_seconds() >= 36000:
            history_sessions, current = sessions[-15:], []
        else:
            history_sessions, current = sessions[:-1][-15:], sessions[-1]
        if not history_sessions:
            return fallback
        used = [r for s in history_sessions for r in s] + current
        if any(r["question_id"] not in self.mapping for r in used):
            return fallback
        import torch
        from ktbench.batching import Session, SessionExample, collate_hitskt
        def make_session(rows):
            return Session([self.mapping[r["question_id"]]["question_id"] for r in rows],
                           [self.mapping[r["question_id"]]["skill_id"] for r in rows],
                           [int(r["correct"]) for r in rows])
        earlier = [make_session(rows) for rows in history_sessions]
        mapped = [q for q in candidates if q["id"] in self.mapping]
        # Full sessions retained. Bound product serving context with a visible fallback.
        if sum(len(s) for s in history_sessions) + len(current) > 4096:
            for q in mapped:
                fallback[q["id"]]["reason"] = "serving_context_capacity"
            return fallback
        # Small candidate batches bound memory without altering the research model.
        for start in range(0, len(mapped), 8):
            chunk = mapped[start:start + 8]
            examples = []
            for q in chunk:
                # Placeholder candidate label cannot enter its own shifted input.
                target = make_session(current + [{"question_id": q["id"], "correct": False}])
                examples.append(SessionExample(earlier, target, split=2))
            batch = collate_hitskt(examples, num_questions=self.model.config.num_questions, num_skills=self.model.config.num_skills)
            with torch.inference_mode():
                probabilities = torch.sigmoid(self.model(batch)[:, len(current)]).tolist()
            for q, probability in zip(chunk, probabilities):
                fallback[q["id"]] = {"probability": probability, "provider": "hitskt", "dataset": self.dataset,
                                      "checkpoint_sha256": self.checkpoint_sha256}
        return fallback
