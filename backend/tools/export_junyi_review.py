"""Package a completed Junyi result and its following paper's redacted metadata locally."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import ROOT, Settings
from app.junyi_bank import digest, load_bank, load_mappings
from app.junyi_model import CHECKPOINT_SHA256, read_baseline
from tools.export_junyi_history import (
    HISTORY_COLUMNS, _completed_answers, _history_rows, _publish_export, _timestamp, _write_csv, export_completed,
)


_ANALYSIS_COLUMNS = ("tag", "topic", "stage", "category", "unattempted", "unavailable_reason", "score", "original_score",
                     "actual_rate", "original_attempts", "original_correct", "app_attempts", "app_correct", "probe_min", "probe_max")
_PREDICTION_COLUMNS = ("tag", "rank", "question_id", "source_question_id", "skill_id", "original", "current")


def _json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _paper_metadata(paper):
    return {"sequence": paper["sequence"], "status": paper["status"], "stage": paper["stage"], "bank_version": paper["bank_version"],
            "selected_count": len(paper["items"]), "answers_saved": len(paper["responses"]), "quotas": paper["quotas"],
            "allocation": paper["allocation"], "selection_sources": dict(Counter(row["provider"] for row in paper["items"])),
            "required_skill_count": len(paper["requirements"]), "answers_used_for_selection": paper["snapshot"]["app_responses"],
            "frozen_items_sha256": digest(json.dumps(paper["items"], sort_keys=True).encode())}


def export_review(settings, sequence, directory):
    if not settings.junyi_student_account_id or not settings.junyi_history_sha256:
        raise ValueError("Set the pilot account and verified history hash in the backend environment.")
    with MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000) as client:
        collection = client[settings.mongodb_database].junyi_papers
        records = list(collection.find({"owner": settings.junyi_student_account_id, "sequence": {"$lte": sequence + 1}}).sort("sequence", 1))
    completed = [paper for paper in records if paper["status"] == "completed" and paper["sequence"] <= sequence]
    next_paper = next((paper for paper in records if paper["sequence"] == sequence + 1 and paper.get("items")), None)
    if not completed or completed[-1]["sequence"] != sequence or completed[-1].get("analysis_status") != "ready" or not next_paper:
        raise ValueError("The review requires a completed result with ready analysis and its saved following paper.")

    mappings = load_mappings(ROOT / "reports" / "datasets" / "junyi", settings.junyi_ranking_file)
    baseline = read_baseline(settings.junyi_history_file, settings.junyi_history_sha256, mappings)
    bank = load_bank(settings.junyi_bank_file, mappings)
    with settings.junyi_checkpoint.open("rb") as checkpoint:
        if hashlib.file_digest(checkpoint, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("The Junyi checkpoint does not match the approved release.")
    answers = _completed_answers(completed)
    snapshot = completed[-1]["result_snapshot"]
    next_snapshot = next_paper["snapshot"]
    fingerprint = digest(json.dumps(answers, sort_keys=True).encode())
    for selected_snapshot in (snapshot, next_snapshot):
        if (selected_snapshot["app_history_version"] != fingerprint or selected_snapshot["app_responses"] != len(answers)
                or selected_snapshot["history_version"] != baseline.version or selected_snapshot["mapping_version"] != mappings.version
                or selected_snapshot["checkpoint"] != CHECKPOINT_SHA256):
            raise ValueError("The reviewed result or paper snapshot does not match the verified history and inputs.")
    skills = snapshot["skills"]
    if {row["tag"] for row in skills} != set(bank.topics) or len(skills) != len(bank.topics) or next_paper["bank_version"] != bank.version:
        raise ValueError("The saved review does not match the verified bank's tag coverage or version.")

    history_source, _ = export_completed(settings, sequence, directory)
    output = directory / f"review-through-paper-{sequence:02d}"
    with TemporaryDirectory(prefix=".junyi-review-", dir=directory) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        for filename in ("assessment_results.csv", "updated_student_history.csv"):
            copyfile(history_source / filename, staged / filename)
        _write_csv(staged / "student1_history.csv", HISTORY_COLUMNS, _history_rows(baseline, mappings, ()))
        _write_csv(staged / "student1_skill_analysis.csv", _ANALYSIS_COLUMNS,
                   (tuple(row[field] for field in _ANALYSIS_COLUMNS) for row in skills))
        _write_csv(staged / "student1_skill_predictions.csv", _PREDICTION_COLUMNS,
                   ((row["tag"], rank, probe["question_id"], probe["source_question_id"], probe["skill_id"], probe["original"], probe["current"])
                    for row in skills for rank, probe in enumerate(row["probes"], 1)))
        _write_csv(staged / "question_bank_verified.csv", ("bank_question_id", "original_tag", "topic", "stage", "bank_sha256"),
                   ((row.id, row.tag, row.topic, row.stage, bank.version) for row in bank.items.values()))
        _json(staged / "generated_paper.json", _paper_metadata(next_paper))
        _json(staged / "recommended_resources.json", {"paper_sequence": next_paper["sequence"], "required_study": [
            {field: row[field] for field in ("tag", "topic", "stage", "lookup", "status", "resources")}
            for row in next_paper["requirements"]]})
        _json(staged / "review_manifest.json", {
            "completed_through": sequence, "original_sessions": len(baseline.sessions), "original_events": sum(map(len, baseline.sessions)),
            "app_answers": len(answers), "history_cutoff_utc": _timestamp(baseline.cutoff), "analysis_cutoff_utc": _timestamp(snapshot["as_of"]),
            "baseline_sha256": baseline.version, "app_history_sha256": fingerprint, "mapping_sha256": mappings.version,
            "bank_sha256": bank.version, "checkpoint_sha256": CHECKPOINT_SHA256, "thresholds": snapshot["thresholds"],
            "skill_count": len(skills), "score_classes": dict(Counter(row["category"] or "unavailable" for row in skills)),
            "unattempted": sum(row["unattempted"] for row in skills),
            "files": {path.name: digest(path.read_bytes()) for path in sorted(staged.iterdir())},
        })
        _publish_export(staged, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, required=True, help="Last completed paper to review")
    args = parser.parse_args()
    if args.through < 1:
        parser.error("--through must be a positive paper number")
    try:
        directory = ROOT / "reports" / "adaptive_junyi_student1" / "private_exports"
        output = export_review(Settings(), args.through, directory)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Wrote the private Junyi review to {output}")


if __name__ == "__main__":
    main()
